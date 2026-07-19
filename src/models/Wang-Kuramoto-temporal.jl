# Wang-Kuramoto-temporal.jl
# Temporal XOR: inputs A and B are presented sequentially.
#   Phase 1 [0, T_sw]:  node 1 clamped to A (node 2 free)
#   Phase 2 [T_sw, T]:  node 2 clamped to B (node 1 released)
#   At time T: output node N should encode XOR(A, B).
#
# Requires working memory: hidden nodes must retain A's encoding across the
# switch so the output can compute XOR(A, B).
#
# Findings (see also the wang-kuramoto-temporal-xor project note):
#   * 4/4 IS achievable at the small original config (N=5, H=16), but it is
#     SEED-DEPENDENT — most inits settle into a 3/4 local optimum (which pattern
#     fails varies by seed). We therefore sweep seeds and keep the best by
#     accuracy, early-stopping on the first 4/4.
#   * Larger capacity (more oscillators / wider NN) trains WORSE, not better
#     (stiffer, divergence-prone landscape) — so N=5/H=16 is the sweet spot.
#   * Gradients use Enzyme reverse-mode (~9x faster than ForwardDiff here and
#     independent of parameter count); the RHS is a scalar non-allocating loop
#     (a Dual/AD-friendly matmul RHS is a severe compile trap). NOTE: Enzyme's
#     first autodiff call per process compiles its pipeline (~5 min, one-time).
#
# Run: julia --project=. src/models/Wang-Kuramoto-temporal.jl
#   (define TEMPXOR_SKIP_RUN = true before include() to only load definitions)

using ComponentArrays, LinearAlgebra
using Random, StableRNGs
using Printf, JLD2
import Enzyme

# ── Task / integration constants ──────────────────────────────────────────────
const N        = 5        # node1=input_A, node2=input_B, 3..N-1=hidden, N=output
const H        = 16       # coupling-NN hidden width
const OUT_NODE = N
const T_SW     = 5.0      # switch time: A clamped → B clamped
const T_TRAIN  = 10.0     # training horizon
const T_EVAL   = 40.0     # evaluation horizon (longer for clean convergence)
const DT       = 0.1

const PHI_0 = -π/2        # phase encoding: False
const PHI_1 =  π/2        # phase encoding: True

# 4 sequential XOR patterns — columns: [ψ_A, ψ_B, target]
const SEQ_XOR = Float64[
    PHI_0  PHI_0  PHI_0;   # 0 XOR 0 = 0
    PHI_0  PHI_1  PHI_1;   # 0 XOR 1 = 1
    PHI_1  PHI_0  PHI_1;   # 1 XOR 0 = 1
    PHI_1  PHI_1  PHI_0    # 1 XOR 1 = 0
]
const TARGETS = SEQ_XOR[:, 3]
const LABELS  = ["0⊕0", "0⊕1", "1⊕0", "1⊕1"]

# ── Training hyperparameters ──────────────────────────────────────────────────
const SEEDS = [13, 42, 1, 7, 2]   # seed 13 solves 4/4; swept in this order
const M1, LR1 = 300, 0.006        # stage 1 (Adam)
const M2, LR2 = 300, 0.0015       # stage 2 (Adam, refine)
const CLIP    = 8.0               # gradient-norm clip (prevents divergence)

# ── Model (scalar, non-allocating; Enzyme-reverse friendly) ───────────────────
build_edges(n) = begin
    dst = Int[]; src = Int[]
    for i in 1:n, j in 1:n
        if j != i; push!(dst, i); push!(src, j); end
    end
    (dst, src)
end
const DST, SRC = build_edges(N)

build_nn_init(rng) = ComponentArray(
    W1 = randn(rng, Float64, H*4) .* 0.1,
    b1 = zeros(Float64, H),
    W2 = randn(rng, Float64, H)   .* 0.1,
    b2 = zeros(Float64, 1),
)

# coupling weight w_ij = W2·tanh(W1·[sin φj, cos φj, sin φi, cos φi] + b1) + b2
function rhs(φ, nn_p, h_vec, psi_vec)
    Tp  = promote_type(eltype(φ), eltype(nn_p))
    W1  = reshape(nn_p.W1, H, 4)
    b1  = nn_p.b1; W2 = nn_p.W2; b2 = nn_p.b2[1]
    out = Vector{Tp}(undef, N)
    @inbounds for i in 1:N
        si = sin(φ[i]); ci = cos(φ[i])
        # i-side pre-activation W1[:,3]·sinφi + W1[:,4]·cosφi + b1 depends only on i;
        # hoist it out of the edge loop (~1.17× faster Enzyme gradient, math unchanged).
        bi = Vector{Tp}(undef, H)
        for h in 1:H
            bi[h] = W1[h,3]*si + W1[h,4]*ci + b1[h]
        end
        acc = -h_vec[i] * sin(φ[i] - psi_vec[i])
        for j in 1:N
            j == i && continue
            sj = sin(φ[j]); cj = cos(φ[j])
            w = b2
            for h in 1:H
                z = W1[h,1]*sj + W1[h,2]*cj + bi[h]
                w += W2[h] * tanh(z)
            end
            acc += w * sin(φ[j] - φ[i])
        end
        out[i] = acc
    end
    out
end

function rk4_phase(φ, nn_p, h_vec, psi_vec, nsteps)
    for _ in 1:nsteps
        k1 = rhs(φ,               nn_p, h_vec, psi_vec)
        k2 = rhs(φ .+ 0.5*DT.*k1, nn_p, h_vec, psi_vec)
        k3 = rhs(φ .+ 0.5*DT.*k2, nn_p, h_vec, psi_vec)
        k4 = rhs(φ .+    DT.*k3,  nn_p, h_vec, psi_vec)
        φ  = φ .+ (DT/6.0) .* (k1 .+ 2.0.*k2 .+ 2.0.*k3 .+ k4)
    end
    φ
end

# Two-phase temporal integration — split at T_sw to avoid the clamp discontinuity
function rk4_temporal(psi_A, psi_B, nn_p; T=T_TRAIN)
    φ0 = zeros(eltype(nn_p), N); φ0[1] = psi_A
    h1 = zeros(N); h1[1] = 0.5;  psi1 = zeros(N); psi1[1] = psi_A
    φ  = rk4_phase(φ0, nn_p, h1, psi1, round(Int, T_SW/DT))
    h2 = zeros(N); h2[2] = 0.5;  psi2 = zeros(N); psi2[2] = psi_B
    φ  = rk4_phase(φ,  nn_p, h2, psi2, round(Int, (T-T_SW)/DT))
    φ
end

wang_cost(φ, φt) = -log(max(1.0 + cos(φ - φt), 1e-8))

function loss_temporal(nn_p)
    loss = zero(eltype(nn_p))
    for k in 1:4
        φf = rk4_temporal(SEQ_XOR[k,1], SEQ_XOR[k,2], nn_p)
        loss += wang_cost(φf[OUT_NODE], TARGETS[k])
    end
    loss
end

enz_grad(nn) = Enzyme.gradient(Enzyme.Reverse, loss_temporal, nn)[1]

# accuracy: cos(φ_out − target) > 0  (angular distance < π/2, winding-safe)
function evaluate(nn)
    nc = 0; phis = Float64[]
    for k in 1:4
        φf = rk4_temporal(SEQ_XOR[k,1], SEQ_XOR[k,2], nn; T=T_EVAL)
        φo = φf[OUT_NODE]; push!(phis, φo)
        (cos(φo - TARGETS[k]) > 0) && (nc += 1)
    end
    (nc, phis)
end

# one training run (manual Adam + grad clipping); returns (acc, loss, nn, phis)
function train_seed(seed; verbose=false)
    nn = build_nn_init(StableRNG(seed))
    β1=0.9; β2=0.999; ϵ=1e-8; m = zero(nn); v = zero(nn); t = 0
    stage(lr, iters) = for it in 1:iters
        g  = enz_grad(nn)
        gn = norm(vec(g)); gn > CLIP && (g .*= CLIP/gn)
        t += 1
        @. m = β1*m + (1-β1)*g
        @. v = β2*v + (1-β2)*g*g
        @. nn = nn - lr * (m/(1-β1^t)) / (sqrt(v/(1-β2^t)) + ϵ)
        verbose && it % 100 == 0 && @printf("      it %4d  loss=%.4f\n", t, loss_temporal(nn))
    end
    stage(LR1, M1); stage(LR2, M2)
    acc, phis = evaluate(nn)
    (acc, loss_temporal(nn), nn, phis)
end

# full trajectory (φ at every step) for plotting
function rk4_temporal_traj(psi_A, psi_B, nn_p; T=T_EVAL)
    φ = zeros(N); φ[1] = psi_A
    h1 = zeros(N); h1[1] = 0.5; psi1 = zeros(N); psi1[1] = psi_A
    h2 = zeros(N); h2[2] = 0.5; psi2 = zeros(N); psi2[2] = psi_B
    nsteps = round(Int, T/DT); n1 = round(Int, T_SW/DT)
    ts = collect(range(0.0, T, length=nsteps+1))
    M  = Matrix{Float64}(undef, N, nsteps+1); M[:,1] = φ
    for step in 1:nsteps
        hv, pv = step <= n1 ? (h1, psi1) : (h2, psi2)
        k1 = rhs(φ,               nn_p, hv, pv)
        k2 = rhs(φ .+ 0.5*DT.*k1, nn_p, hv, pv)
        k3 = rhs(φ .+ 0.5*DT.*k2, nn_p, hv, pv)
        k4 = rhs(φ .+    DT.*k3,  nn_p, hv, pv)
        φ  = φ .+ (DT/6.0).*(k1 .+ 2.0.*k2 .+ 2.0.*k3 .+ k4)
        M[:, step+1] = φ
    end
    (ts, M)
end

# Sweep seeds, keep the best by accuracy (early-stop on 4/4), save params + figure.
# Wrapped in a function so loop-variable scoping is well-defined.
function run_temporal_xor()
    println("Temporal XOR — seed sweep (N=$N, H=$H); Enzyme first call compiles ~5 min…")
    best = (acc=-1, loss=Inf, seed=0, nn=build_nn_init(StableRNG(0)), phis=Float64[])
    for s in SEEDS
        t0 = time()
        acc, l, nn, phis = train_seed(s)
        @printf("  seed %-3d  acc=%d/4  loss=%.4f  (%.0fs)\n", s, acc, l, time()-t0)
        if acc > best.acc || (acc == best.acc && l < best.loss)
            best = (acc=acc, loss=l, seed=s, nn=nn, phis=phis)
        end
        if best.acc == 4
            println("  → 4/4 reached, stopping sweep.")
            break
        end
    end

    trained_nn = best.nn
    @printf("\nBEST: seed=%d  accuracy=%d/4  loss=%.4f  (theoretical min %.4f)\n",
            best.seed, best.acc, best.loss, -4*log(2))
    for k in 1:4
        ok = cos(best.phis[k] - TARGETS[k]) > 0
        @printf("  %s  φ_out=%+.3fπ  target=%+.3fπ  %s\n",
                LABELS[k], best.phis[k]/π, TARGETS[k]/π, ok ? "✓" : "✗")
    end

    FIGDIR = joinpath(@__DIR__, "..", "..", "results", "figures")
    isdir(FIGDIR) || mkpath(FIGDIR)

    # persist the winning parameters (gitignored under results/figures)
    JLD2.jldsave(joinpath(FIGDIR, "kuramoto_temporal_xor_params.jld2");
                 nn=collect(trained_nn), N=N, H=H, seed=best.seed, acc=best.acc)

    fig = Figure(size=(1200, 800))
    for k in 1:4
        row, col = divrem(k-1, 2)
        ax = CairoMakie.Axis(fig[row+1, col+1]; xlabel="Time", ylabel="Phase / π",
            title="$(LABELS[k])  →  XOR=$(TARGETS[k] > 0 ? 1 : 0)")
        ts, M = rk4_temporal_traj(SEQ_XOR[k,1], SEQ_XOR[k,2], trained_nn)
        for (i, lbl) in [(1,"input A"), (2,"input B"), (OUT_NODE,"output")]
            lines!(ax, ts, M[i,:] ./ π; linewidth=2, label=lbl)
        end
        vlines!(ax, [T_SW]; linestyle=:dash, color=:gray, label="A→B switch")
        hlines!(ax, [TARGETS[k]/π]; linestyle=:dot, color=:black, label="target")
        axislegend(ax; position=:rc)
    end
    save(joinpath(FIGDIR, "kuramoto_temporal_xor.png"), fig)
    println("\nSaved: results/figures/kuramoto_temporal_xor.png")
    println("Saved: results/figures/kuramoto_temporal_xor_params.jld2")
    return best.acc
end

# ── Script entry point ────────────────────────────────────────────────────────
if !@isdefined(TEMPXOR_SKIP_RUN)
    using CairoMakie
    run_temporal_xor()
end
