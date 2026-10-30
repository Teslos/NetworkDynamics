# ude_xor_implicit.jl
# XOR on a per-edge-coupled Kuramoto network, trained with the IMPLICIT
# (fixed-point) gradient instead of backpropagation through the trajectory.
#
# Why: src/models/Wang-Kuramoto.jl trains by differentiating through RK4 over a
# time window. Scored on t = 5..20 it optimises a transient that passes through
# the target and then leaves it; scored on t = 50..200 the long unrolled
# gradient trains poorly. EP-trained XY networks solve XOR on every seed, and EP
# takes its gradient from the settled state alone. This script does the same
# thing exactly: relax to the fixed point φ*, then
#     dL/dθ = -λᵀ ∂f/∂θ,   with   Jᵀ λ = ∂L/∂φ,   J = ∂f/∂φ at φ*,
# the implicit-function-theorem gradient that EP approximates.
#
# Network: N oscillators, inputs 1,2 clamped to ±π/2, hidden 3..N-1, output N.
#   dφ_i/dt = Σ_j K_ij sin(φ_j - φ_i) - [i == N] h sin(φ_i - ψ)     (i free)
# with K symmetrised (K_ij = K_ji, EP's coupling) unless --directed=1, and one
# learned output field (h, ψ) shared by all four patterns.
#
# Usage: julia --project=. scripts/ude_xor_implicit.jl --seed=1 [--N=5] [--directed=1]

using OrdinaryDiffEq, LinearAlgebra, StableRNGs, Printf

function argval(name, default)
    for a in ARGS
        startswith(a, "--$name=") && return parse(Int, split(a, "=")[2])
    end
    return default
end
const N        = argval("N", 5)
const SEED     = argval("seed", 1)
const DIRECTED = argval("directed", 0) == 1
const ITERS    = argval("iters", 2000)
# `--fields=all` gives every free oscillator a learned field (h_i, ψ_i), as the
# EP-trained XY network has; the default gives one to the output only.
const FIELDS_ALL = argval("fields", 0) == 1
const FREE     = 3:N
const NF       = length(FREE)
const PSI_IN   = [-π/2 -π/2; -π/2 π/2; π/2 -π/2; π/2 π/2]
const TARGETS  = [-π/2, π/2, π/2, -π/2]
const LABELS   = ["FF", "FT", "TF", "TT"]
const HIDDEN_U0 = 0.5 .* randn(StableRNG(SEED + 1000), max(N - 3, 0))

# θ = [vec(K); h_free; ψ_free], one field per free oscillator. Without
# --fields=all the hidden entries stay at zero (their gradient is masked).
coupling(θ) = (K = reshape(θ[1:N*N], N, N); DIRECTED ? K : (K + K') / 2)
fields(θ) = (θ[N*N .+ (1:NF)], θ[N*N + NF .+ (1:NF)])
const FIELD_MASK = vcat(trues(N * N), [FIELDS_ALL || a == NF for a in 1:NF],
                        [FIELDS_ALL || a == NF for a in 1:NF])

full_state(φF, k) = vcat(PSI_IN[k, 1], PSI_IN[k, 2], φF)

function rhs(φF, θ, k)
    Ks, (h, ψ) = coupling(θ), fields(θ)
    φ = full_state(φF, k)
    return [sum(Ks[i, j] * sin(φ[j] - φ[i]) for j in 1:N if j != i) -
            h[a] * sin(φ[i] - ψ[a]) for (a, i) in enumerate(FREE)]
end

# J[a, b] = ∂f_{FREE[a]} / ∂φ_{FREE[b]}
function jac_phi(φF, θ, k)
    Ks, (h, ψ) = coupling(θ), fields(θ)
    φ = full_state(φF, k)
    J = zeros(NF, NF)
    for (a, i) in enumerate(FREE), (b, m) in enumerate(FREE)
        if m != i
            J[a, b] = Ks[i, m] * cos(φ[m] - φ[i])
        else
            J[a, b] = -sum(Ks[i, j] * cos(φ[j] - φ[i]) for j in 1:N if j != i) -
                      h[a] * cos(φ[i] - ψ[a])
        end
    end
    return J
end

# gθ = Σ_a λ_a ∂f_a/∂θ, for the adjoint vector λ
function vjp_theta(φF, θ, k, λ)
    h, ψ = fields(θ)
    φ = full_state(φF, k)
    gKs = zeros(N, N)                       # gradient w.r.t. the coupling actually used
    for (a, i) in enumerate(FREE), j in 1:N
        j != i && (gKs[i, j] += λ[a] * sin(φ[j] - φ[i]))
    end
    gK = DIRECTED ? gKs : (gKs + gKs') / 2  # chain rule through the symmetrisation
    gh = [-λ[a] * sin(φ[i] - ψ[a]) for (a, i) in enumerate(FREE)]
    gψ = [ λ[a] * h[a] * cos(φ[i] - ψ[a]) for (a, i) in enumerate(FREE)]
    return vcat(vec(gK), gh, gψ) .* FIELD_MASK
end

function relax(θ, k, φF0; T=1000.0)
    prob = ODEProblem((u, _, _) -> rhs(u, θ, k), φF0, (0.0, T))
    return solve(prob, Tsit5(); reltol=1e-10, abstol=1e-10, save_everystep=false).u[end]
end

u0_fixed() = vcat(HIDDEN_U0, 0.0)
wang_cost(x) = -log(max(1 + cos(x), 1e-12))
wang_dcost(x) = sin(x) / max(1 + cos(x), 1e-12)

function loss_and_grad(θ)
    L, g, maxres = 0.0, zeros(length(θ)), 0.0
    for k in 1:4
        φs = relax(θ, k, u0_fixed())
        maxres = max(maxres, norm(rhs(φs, θ, k)))
        d = φs[end] - TARGETS[k]
        L += wang_cost(d)
        gφ = zeros(NF); gφ[end] = wang_dcost(d)
        J = jac_phi(φs, θ, k)
        λ = transpose(J) \ gφ
        g .-= vjp_theta(φs, θ, k, λ)
    end
    return L, g, maxres
end

loss_only(θ) = sum(wang_cost(relax(θ, k, u0_fixed())[end] - TARGETS[k]) for k in 1:4)

# ── Gradient check against central finite differences of the settled loss ──
θ = vcat(0.1 .* randn(StableRNG(SEED), N * N),
         [FIELDS_ALL || a == NF ? 0.1 : 0.0 for a in 1:NF], zeros(NF))
let (_, g, _) = loss_and_grad(θ), ϵ = 1e-6
    gfd = [(loss_only(θ .+ ϵ .* (1:length(θ) .== p)) -
            loss_only(θ .- ϵ .* (1:length(θ) .== p))) / 2ϵ for p in eachindex(θ)]
    @printf("gradient check: cos(implicit, finite-diff) = %.6f, |g|/|gfd| = %.4f\n",
            dot(g, gfd) / (norm(g) * norm(gfd) + 1e-30), norm(g) / (norm(gfd) + 1e-30))
end

# ── Adam on the implicit gradient ──
let m = zero(θ), v = zero(θ), lr = 0.01, β1 = 0.9, β2 = 0.999
    for it in 1:ITERS
        L, g, res = loss_and_grad(θ)
        g = clamp.(g, -1.0, 1.0)
        m .= β1 .* m .+ (1 - β1) .* g
        v .= β2 .* v .+ (1 - β2) .* g .^ 2
        θ .-= lr .* (m ./ (1 - β1^it)) ./ (sqrt.(v ./ (1 - β2^it)) .+ 1e-8)
        it % 200 == 0 && @printf("iter %5d  loss %.6f  max residual %.1e\n", it, L, res)
    end
end

# ── Evaluation on the settled state ──
correct(φ, t) = cos(φ - t) > cos(φ + t) + 1e-6
wrapdev(φ, t) = abs(rem2pi(φ - t, RoundNearest))

function evaluate(θ, title)
    println("── $title")
    n, md = 0, 0.0
    for k in 1:4
        out = relax(θ, k, u0_fixed())[end]
        ok = correct(out, TARGETS[k]); n += ok; md = max(md, wrapdev(out, TARGETS[k]))
        @printf("  %s  φ_out=%+.3fπ  target=%+.3fπ  %s\n", LABELS[k],
                rem2pi(out, RoundNearest) / π, TARGETS[k] / π, ok ? "✓" : "✗")
    end
    return n, md
end

n, maxdev = evaluate(θ, "trained, fixed initial state")
# EP's protocol: free cells start at 0.1·randn, 20 draws per pattern
robust, rmax = let rng = StableRNG(SEED + 5000), c = 0, m = 0.0
    for k in 1:4, _ in 1:20
        out = relax(θ, k, 0.1 .* randn(rng, NF))[end]
        c += correct(out, TARGETS[k]); m = max(m, wrapdev(out, TARGETS[k]))
    end
    c / 80, m
end
θc = copy(θ); θc[1:N*N] .= 0.0                 # control: learned coupling removed
nc, _ = evaluate(θc, "control: zero coupling, output field kept")
@printf("RESULT N=%d variant=implicit_%s%s seed=%d final_loss=%.6f accuracy=%d/4 maxdev=%.3fpi robust=%.3f robust_maxdev=%.3fpi control=%d/4\n",
        N, DIRECTED ? "pair" : "sympair", FIELDS_ALL ? "_allfields" : "", SEED, loss_only(θ), n, maxdev / π,
        robust, rmax / π, nc)

# ── Figure (`--plot=1`): free relaxation from the fixed initial state ──
# Same style as src/models/Wang-Kuramoto.jl: large fonts for a full-width figure.
if argval("plot", 0) == 1
    using CairoMakie
    tplot = 60.0
    fig = Figure(size=(1300, 900), fontsize=30)
    for k in 1:4
        row, col = divrem(k - 1, 2)
        ax = Axis(fig[row+1, col+1]; xlabel = row == 1 ? "Time" : "",
                  ylabel = col == 0 ? "Phase / π" : "", title = LABELS[k], titlesize=32,
                  yticks = -1:0.5:1, xgridvisible=false, ygridvisible=false)
        prob = ODEProblem((u, _, _) -> rhs(u, θ, k), u0_fixed(), (0.0, tplot))
        sol = solve(prob, Tsit5(); reltol=1e-10, abstol=1e-10, saveat=0.1)
        traj = reduce(hcat, [full_state(u, k) for u in sol.u])
        wrapped = rem2pi.(traj, RoundNearest) ./ π
        for i in 1:N, s in 2:length(sol.t)            # break lines at the ±π wrap
            abs(wrapped[i, s] - wrapped[i, s-1]) > 1 && (wrapped[i, s] = NaN)
        end
        for i in 3:N-1
            lines!(ax, sol.t, wrapped[i, :]; linewidth=2, color=(:gray50, 0.6),
                   label = i == 3 ? "hidden" : nothing)
        end
        lines!(ax, sol.t, wrapped[1, :]; linewidth=4, color=:steelblue, label="input 1")
        lines!(ax, sol.t, wrapped[2, :]; linewidth=4, color=:darkorange, linestyle=:dashdot,
               label="input 2")
        lines!(ax, sol.t, wrapped[N, :]; linewidth=5, color=:firebrick, label="output")
        hlines!(ax, [TARGETS[k] / π]; linestyle=:dash, linewidth=3, color=:black, label="target")
        ylims!(ax, -1.05, 1.05)
        row == 0 && hidexdecorations!(ax; grid=false, ticks=false)
        col == 1 && hideydecorations!(ax; grid=false, ticks=false)
    end
    Legend(fig[3, 1:2], content(fig[1, 1]); orientation=:horizontal,
           framevisible=false, labelsize=30)
    figdir = joinpath(dirname(Base.active_project()), "results", "figures")
    mkpath(figdir)
    tag = "N$(N)_implicit_$(DIRECTED ? "pair" : "sympair")$(FIELDS_ALL ? "_allfields" : "")_seed$(SEED)"
    for ext in ("png", "pdf")
        save(joinpath(figdir, "kuramoto_xor_ude_$(tag).$ext"), fig)
    end
    println("Saved: results/figures/kuramoto_xor_ude_$(tag).{png,pdf}")
end
