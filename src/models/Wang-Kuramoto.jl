# Wang-Kuramoto.jl
# Part 1: Physical Kuramoto simulation via NetworkDynamics (reference trajectories).
# Part 2: UDE training — neural network learns oscillator coupling to solve XOR.
using ComponentArrays, DiffEqFlux, NetworkDynamics, Lux, OrdinaryDiffEq, LinearAlgebra
using Graphs
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using CairoMakie
using Random, StableRNGs
using LaTeXStrings, Printf
using Distributions
using JLD2
# Network size and seed: `--N=4 --seed=2` on the command line, or WK_N / WK_SEED
# in ENV when the file is include()d from a warm session.
function _wk_arg(name, envname, default)
    for a in ARGS
        startswith(a, "--$name=") && return parse(Int, split(a, "=")[2])
    end
    return parse(Int, get(ENV, envname, string(default)))
end
const SEED = _wk_arg("seed", "WK_SEED", 1)
# Coupling mask A_ij: `--topology=complete` (default) or `--topology=ws`
# (Watts–Strogatz, mean degree 4, rewiring 0.25, as in Figure 1 of the paper).
const TOPOLOGY = let t = get(ENV, "WK_TOPOLOGY", "complete")
    for a in ARGS
        startswith(a, "--topology=") && (t = split(a, "=")[2])
    end
    Symbol(t)
end
# EP-comparison switches, both off by default: `--clamp=1` holds the inputs at
# their pattern phases (as the EP-trained XY network does); `--outfield=1` gives
# the output a learned field (h, ψ), one pair shared by all four patterns.
const CLAMP    = _wk_arg("clamp", "WK_CLAMP", 0) == 1
const OUTFIELD = _wk_arg("outfield", "WK_OUTFIELD", 0) == 1
# `--hidfields=1` also gives every hidden oscillator its own learned field, as
# the EP-trained XY network has on all free cells.
const HIDFIELDS = _wk_arg("hidfields", "WK_HIDFIELDS", 0) == 1
# Coupling form: `--coupling=nn` (default; one NN of the two phases, shared by
# all edges), `pair` (a learned weight per directed edge, no NN), `sympair`
# (the same, symmetrised so K_ij = K_ji, as in EP), or `nnid`
# (the shared NN also given one-hot labels of the edge's two nodes, so it can
# tell edges apart). `--hidden=` sets the NN width (default 16).
const COUPLING = let c = get(ENV, "WK_COUPLING", "nn")
    for a in ARGS
        startswith(a, "--coupling=") && (c = split(a, "=")[2])
    end
    Symbol(c)
end
COUPLING in (:nn, :pair, :sympair, :nnid) || error("--coupling must be nn, pair, sympair or nnid")
const HID = _wk_arg("hidden", "WK_HIDDEN", 16)
# Weight of the settling penalty |dφ/dt|² at the end of the loss window
# (`--settle=100`; default 0 = off)
const SETTLE = _wk_arg("settle", "WK_SETTLE", 0)
# default rng
rng = StableRNG(SEED)

function plot_rect_map(N::Int, M::Int, data::Vector{Float64}, ax)
    #ax = Axis(f[1, 1])
    centers_x = 1:N
    centers_y = 1:M
    data = reshape(abs.(data), N, M)
    GLMakie.heatmap!(ax, centers_x, centers_y, data, colormap = :viridis)
end

function plot_phases(N::Int, M::Int, u::Array{Float64,2}, t::Array{Float64,1}, x0::Array{Float64,1} ,forcing_period::Float64, tspan)
    # Find the index of the value in t that is closest to t0 time
    for t0 in forcing_period:2:tspan[2]
        f = Figure()
        ax = Axis(f[1, 1], xlabel = "Node", ylabel = L"\phi", title = "XOR gate network at time $t0")
        # set y-axis to be in the range of -π to π
        ax.yticks = (0:π/2:2π, ["0", "π/2", "π", "3π/2", "2π"])
        index_closest_to_t = findmin(abs.(t .- t0))[2]
        state_vector_at_t = [mod2pi((u[i,index_closest_to_t]-u[1,index_closest_to_t])) for i in 1:N*M]
        #state_vector_at_t = [mod2pi((u[i,index_closest_to_t]-x0[i]) ) for i in 1:N*M]
        #plot_rect_map(N, M, state_vector_at_t, ax)
        lines!(ax, 1:N, state_vector_at_t, linewidth=2, label="Oscillator")
        # record the frames
        GLMakie.save("./figs/xor_network_sol_phase_t$(t0).png",f, px_per_unit = 4)
    end
end

function create_graph(N::Int=5)
    # create all to all graph
    g = Graphs.complete_graph(N)
    edge_weights = ones(length(edges(g)))
    g_weighted = SimpleDiGraph(g)
    g_directed = SimpleDiGraph(g_weighted)
    return g_directed, edge_weights
end

function xor_gate(u::Vector{Int64})
    return u[1] ⊻ u[2]
end

N = _wk_arg("N", "WK_N", 3)   # 2 inputs + (N-3) hidden + 1 output
labels_xor = ["FF", "FT", "TF", "TT"]

# ── Part 1: NetworkDynamics reference simulation (skipped unless --nd passed) ─
if "--nd" in ARGS

g, edge_weights = create_graph(N)

# Functions for edges and vertices
Base.Base.@propagate_inbounds function kiedge!(e, v_s, v_d, (w,σ), t)
    if t < forcing_period
        #e .= -w*sin.(v_s .- v_d) # no coupling in the forcing period 
        e .= 0.0
    else
        e .= -w*sin.(v_s .- v_d) * σ
    end
    nothing
end

Base.Base.@propagate_inbounds function ki_force_vertex!(dv, v, esum, (h, ψ, β, τ), t)
    #println("h: ", h, " ψ: ", ψ)
    beta = 0.4
    if t < forcing_period
        dv .= -β * h*sin.(v .- ψ ) 
    else
        #dv .= -h*sin.(v .- ψ) - beta * sin.(v .- τ)/(1 .+ cos.(v .- τ)) # forcing term from cost function
        #dv .= 0.0
        dv .= -h*sin.(v .- ψ)
    end
    dv .+= esum[1]
    nothing
end

# generate random values from normal distribution for parameters of the edges
w_ij = randn(rng, length(edges(g)))
# generate random values from uniform distribution for parameters of the vertices :h and :ψ
uniform_ψ = Uniform(-π, π)
uniform_h = Uniform(-0.5, 0.5)
# generate vertices values Mdata = 4 for all possible combinations of XOR gate
# both values are equivalent to the bias the first value is h and the second is ψ
ξ_0 = [[0.5,-π/2], [0.5,-π/2], [0,+π/2], [0,0.5], [0.5,-π/2]]
ξ_1 = [[0.5,-π/2], [0.5,π/2], [0,+π/2], [0,0.5], [0.5,π/2]]
ξ_2 = [[0.5,π/2], [0.5,-π/2], [0,+π/2], [0,0.5], [0.5,π/2]]
ξ_3 = [[0.5,π/2], [0.5,π/2], [0,+π/2], [0,0.5], [0.5,-π/2]]
all_solutions = []
solutions = []
forcing_period = 50.0
# Initial conditions
ϕ0 = randn(rng, nv(g))
ϕ0[1] = +π/2
ϕ0[2] = +π/2

tspan = (0.0, 500.0)
tsteps = range(tspan[1], tspan[2], length=1000)
nd_vertex = VertexModel(; f=ki_force_vertex!, g=StateMask(1:1), dim=1, sym=[:v], psym=[:h, :ψ, :β=>20.0, :τ])
nd_edge = EdgeModel(; g=AntiSymmetric(kiedge!), outdim=1, psym=[:weight,:σ=>1.0])
vertex_list = [nd_vertex for i in vertices(g)]
edge_list = [nd_edge for i in edges(g)]
nd! = Network(g, vertex_list, edge_list)
p_nd = NWParameter(nd!)
p_nd.e[:, :weight] = w_ij
p_nd.v[:, :h] = rand(uniform_h, N)
p_nd.v[:, :ψ] .= -π/2
nothing

# all the cases of the XOR gate
pars = [ξ_0, ξ_1, ξ_2, ξ_3]
#u0s = randn(rng, 4, N)
u0s = zeros(Float64, 4, N)
x0 = randn(rng, N)
#x0 = zeros(Float64, N)
ode_prob = ODEProblem(nd!, ϕ0, tspan)

# solve ensamble problem for all possible combinations of XOR gate
Φ = [-π/2 -π/2 π/2 0 -π/2; 
     -π/2 π/2  π/2 0 π/2; 
      π/2 -π/2 π/2 0 π/2; 
      π/2 π/2  π/2 0 -π/2]
Φr = rand(uniform_ψ, 4, 5)

# solve ensamble problem for all possible combinations of XOR gate
function prob_func(prob, i, repeat)
    new_p = deepcopy(p_nd)
    new_p.v[:,:ψ] = Φ[i,:] # prescribed values for ψ 
    new_p.v[:,:τ] = zeros(Float64, N)
    new_p.v[5,:τ] = Φ[i,5]
    
    prob = remake(prob; u0 = u0s[i,:], p = pflat(new_p))
    return prob
end

ens_prob = EnsembleProblem(ode_prob; prob_func=prob_func)
ens_sol = solve(ens_prob, Tsit5(), EnsembleThreads();
                trajectories=4, saveat=tsteps, tstops=[forcing_period])
all_solutions = Array(ens_sol)

# plot the solutions
fig = Figure(size=(900, 600))
for (i,sol) in enumerate(ens_sol)
    row, col = divrem(i-1, 2)
    ax = CairoMakie.Axis(fig[row+1, col+1]; xlabel="Time", ylabel="u", title="XOR gate $(labels_xor[i])")
    t = sol.t
    u = sol(sol.t)[1:N,:]
    for j in [1,2,5]
        lines!(ax, t, u[j,:], linewidth=2, label="Oscillator $j")
    end
    axislegend(ax, position = :rt)
end
let figdir = joinpath(@__DIR__, "../../results/figures")
    isdir(figdir) || mkpath(figdir)
    save(joinpath(figdir, "kuramoto_xor_nd.png"), fig)
    println("Saved: results/figures/kuramoto_xor_nd.png")
end

end # if "--nd" in ARGS


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — UDE training: neural network learns Kuramoto coupling for XOR gate
#
# Design:
#   - Vertex dynamics: dφᵢ = -hᵢ·sin(φᵢ - ψᵢ)  (ψᵢ encodes the XOR input pattern)
#   - Edge coupling:   Σⱼ NN([sin φⱼ, cos φⱼ, sin φᵢ, cos φᵢ]) · sin(φⱼ - φᵢ)
#   - The NN is shared across all edges and is the only thing optimized.
#   - Loss: Wang et al. (2024) cost C = -log(1 + cos(φ_out - φ_target)) summed
#     over all 4 XOR input patterns. More stable than 1-cos near ±π.
#   - Deterministic initial conditions — input nodes start at their pattern values.
#   - Only the two input nodes carry a field. An earlier version set the output
#     node's ψ to the XOR answer with h = 0.5, which pulls the output onto the
#     target with no coupling at all; every N then "solved" XOR trivially.
# ═══════════════════════════════════════════════════════════════════════════════

# Plain Julia neural network — bypasses LuxLib.Utils._return_type bug that
# affects all activations in this Lux/LuxLib version under Zygote AD.
# Architecture: NN_IN → HID (tanh) → 1 (linear).  Parameters stored flat so
# ComponentArray gives a single gradient vector for the optimizer.
const NN_IN = COUPLING == :nnid ? 4 + 2N : 4
const nn_init = let base = COUPLING in (:pair, :sympair) ?
        (K = randn(rng, Float64, N * N) .* Float64(0.1),) :
        (W1 = randn(rng, Float64, HID * NN_IN) .* Float64(0.1),
         b1 = zeros(Float64, HID),
         W2 = randn(rng, Float64, HID) .* Float64(0.1),
         b2 = zeros(Float64, 1))
    out = OUTFIELD ? (h_out = [0.1], psi_out = [0.0]) : (;)
    hid = HIDFIELDS ? (h_hid = fill(0.1, N - 3), psi_hid = zeros(N - 3)) : (;)
    ComponentArray(; base..., out..., hid...)
end

function apply_nn(nn_p, x::AbstractVector)
    W1 = reshape(nn_p.W1, HID, NN_IN)
    h  = tanh.(W1 * x .+ nn_p.b1)
    return dot(nn_p.W2, h) + nn_p.b2[1]
end

# Weight of the j → i edge under the chosen coupling form
function edge_weight(nn_p, φ, i, j)
    COUPLING == :pair && return nn_p.K[(j - 1) * N + i]
    # Symmetric K_ij = K_ji, as in EP: the free dynamics are then a gradient flow
    COUPLING == :sympair && return (nn_p.K[(j - 1) * N + i] + nn_p.K[(i - 1) * N + j]) / 2
    inp = [sin(φ[j]), cos(φ[j]), sin(φ[i]), cos(φ[i])]
    if COUPLING == :nnid
        onehot = zeros(2N); onehot[i] = 1.0; onehot[N + j] = 1.0
        inp = vcat(inp, onehot)
    end
    return apply_nn(nn_p, inp)
end

# XOR input phases — rows: [FF, FT, TF, TT], columns: [in1, in2]
const Ψ_IN = Float64[
    -π/2  -π/2;
    -π/2   π/2;
     π/2  -π/2;
     π/2   π/2
]
# Field phases per node for pattern k; hidden and output ψ are unused (h = 0)
psi_xor(k::Int) = vcat(Ψ_IN[k, :], zeros(N - 2))
const H_NODES     = vcat(0.5, 0.5, zeros(N - 2))  # field on inputs only
const XOR_TARGETS = Float64[-π/2, π/2, π/2, -π/2] # target phase for output node
const OUTPUT_NODE = N
# Fixed, distinct hidden initial phases: with a shared NN, identical hidden
# initial states would stay identical forever and act as one oscillator.
const HIDDEN_U0 = 0.5 .* randn(StableRNG(SEED + 1000), max(N - 3, 0))
const UDE_TSPAN   = (0.0, 200.0)
const UDE_TSTEPS  = range(UDE_TSPAN[1], UDE_TSPAN[2], length=400)
# The cost is averaged over these times, not taken at t=5 alone: scoring only
# t=5 rewards passing through the target, and many such networks drift off it
# by T=200. Averaging over the settling window keeps the optimum at -4·log(2).
const T_END       = _wk_arg("tend", "WK_TEND", 20)   # `--tend=200` scores the settled state
const SCORE_T     = Tuple(T_END .* (0.25, 0.5, 0.75, 1.0))

# Fixed topology mask; only the shared NN coupling is learned
const ADJ = if TOPOLOGY == :complete
    trues(N, N)
elseif TOPOLOGY == :ws
    Matrix(adjacency_matrix(watts_strogatz(N, 4, 0.25; rng=StableRNG(SEED + 2000)))) .!= 0
else
    error("--topology must be complete or ws")
end

# UDE ODE — p = ComponentArray(nn=<Lux params>, psi=<vertex ψ for this pattern>)
# Pure functional form (no mutation) so Zygote can differentiate through it.
function kuramoto_rhs(φ, nn_p, psi)
    return [begin
        s = -H_NODES[i] * sin(φ[i] - psi[i])
        for j in 1:N
            if j != i && ADJ[i, j]
                w_ij = edge_weight(nn_p, φ, i, j)
                s = s + w_ij * sin(φ[j] - φ[i])
            end
        end
        if OUTFIELD && i == OUTPUT_NODE
            s = s - nn_p.h_out[1] * sin(φ[i] - nn_p.psi_out[1])
        end
        if HIDFIELDS && 3 <= i < OUTPUT_NODE
            s = s - nn_p.h_hid[i - 2] * sin(φ[i] - nn_p.psi_hid[i - 2])
        end
        # zero(s) rather than 0.0 keeps the element type (Dual under AD)
        CLAMP && i <= 2 ? zero(s) : s
    end for i in 1:N]
end

# Manual RK4 integrator — fully differentiable by Zygote.
# Uses immutable rebinding (φ = φ .+ ...) so no mutation hits Zygote's tape.
# float.() rather than Float64.() so a Dual state can be integrated onward.
function rk4_integrate(φ0, nn_p, psi; T=20.0, dt=0.1)
    φ     = float.(φ0)
    nsteps = round(Int, T / dt)
    for _ in 1:nsteps
        k1 = kuramoto_rhs(φ,               nn_p, psi)
        k2 = kuramoto_rhs(φ .+ 0.5*dt.*k1, nn_p, psi)
        k3 = kuramoto_rhs(φ .+ 0.5*dt.*k2, nn_p, psi)
        k4 = kuramoto_rhs(φ .+    dt.*k3,  nn_p, psi)
        φ  = φ .+ (dt/6.0) .* (k1 .+ 2.0.*k2 .+ 2.0.*k3 .+ k4)
    end
    return φ
end

# Wang et al. (2024) phase cost — numerically guarded near anti-phase
wang_cost(φ::Real, φt::Real) = -log(max(1.0 + cos(φ - φt), 1e-8))

# Initial condition builder — inputs start at the pattern phases, output at 0
function make_u0(k::Int)
    return vcat(Ψ_IN[k, 1], Ψ_IN[k, 2], HIDDEN_U0, 0.0)
end

# Correct when the output is strictly nearer its target than the opposite
# target (wrap-safe). A strict comparison matters: an untouched output at 0 is
# equidistant, and `cos(φ - target) > 0` passes it on rounding (cos(π/2) ≈ 6e-17).
xor_correct(φ_out, target) = cos(φ_out - target) > cos(φ_out + target) + 1e-6

# Loss: Wang cost averaged over SCORE_T, summed over all 4 XOR patterns.
# Uses manual RK4 so AD differentiates through integration directly —
# no ODE solver adjoint, no FunctionWrappers, no _return_type errors.
function loss_xor(nn_p)
    loss = zero(eltype(nn_p))
    for k in 1:4
        φ, t_prev = make_u0(k), 0.0
        for t in SCORE_T
            φ = rk4_integrate(φ, nn_p, psi_xor(k); T=t - t_prev)
            loss  += wang_cost(φ[OUTPUT_NODE], XOR_TARGETS[k]) / length(SCORE_T)
            t_prev = t
        end
        # Settling penalty: the window cost alone rewards a trajectory that only
        # passes through the target; penalising |dφ/dt|² at the window's end asks
        # for a state that is at rest there, which is what EP trains on.
        if SETTLE > 0
            loss += SETTLE * sum(abs2, kuramoto_rhs(φ, nn_p, psi_xor(k)))
        end
    end
    return loss
end

# ── Optimization ──────────────────────────────────────────────────────────────
p_init  = nn_init                            # Float64 ComponentArray, matches apply_nn
adtype  = Optimization.AutoForwardDiff()
optf    = Optimization.OptimizationFunction((x, _) -> loss_xor(x), adtype)
optprob = Optimization.OptimizationProblem(optf, p_init)

iter_count = Ref(0)
cb = (p, l) -> begin
    iter_count[] += 1
    iter_count[] % 20 == 0 && @printf("Iter %4d  loss = %.6f\n", iter_count[], l)
    return false
end

# Stage 1: Adam — broad exploration
result_ude = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.01);
    callback=cb, maxiters=500)

# Stage 2: AdamW with lower learning rate — fine-tuning
optprob2    = remake(optprob; u0=result_ude.minimizer)
result_ude2 = Optimization.solve(optprob2, OptimizationOptimisers.AdamW(0.001);
    callback=cb, maxiters=300)

# Save trained NN parameters as a plain vector: the optimizer result holds
# closures that JLD2 cannot rebuild outside this script.
const VARIANT = (TOPOLOGY == :complete ? "" : "_$(TOPOLOGY)") *
                (CLAMP ? "_clamp" : "") * (OUTFIELD ? "_outfield" : "") * (HIDFIELDS ? "_hidfields" : "") *
                (COUPLING == :nn ? "" : "_$(COUPLING)") *
                (!(COUPLING in (:pair, :sympair)) && HID != 16 ? "_h$(HID)" : "") *
                (SETTLE > 0 ? "_settle$(SETTLE)" : "") *
                (T_END != 20 ? "_tend$(T_END)" : "")
const RUN_TAG = "N$(N)$(VARIANT)_seed$(SEED)"
let outdir = joinpath(@__DIR__, "../../results/models")
    isdir(outdir) || mkpath(outdir)
    nn_params = collect(result_ude2.minimizer)
    @save joinpath(outdir, "Wang-Kuramoto-ude-xor-$(RUN_TAG).jld2") nn_params
    println("Saved: results/models/Wang-Kuramoto-ude-xor-$(RUN_TAG).jld2")
end

# ── Evaluation ────────────────────────────────────────────────────────────────
# Reference solve of the actual ODE: adaptive Tsit5 at tight tolerance. The
# fixed-step RK4 used for training is not trusted for evaluation, because a
# network trained through it can give different answers at different dt.
function solve_reference(nn_p, k; T=UDE_TSPAN[2], saveat=nothing)
    prob = ODEProblem((φ, _, _) -> kuramoto_rhs(φ, nn_p, psi_xor(k)),
                      make_u0(k), (0.0, T))
    return isnothing(saveat) ?
        solve(prob, Tsit5(); reltol=1e-10, abstol=1e-10, save_everystep=false) :
        solve(prob, Tsit5(); reltol=1e-10, abstol=1e-10, saveat=saveat)
end

function evaluate_xor(nn_p, title; integrator=:reference)
    println("\n── XOR evaluation ($title, $integrator) ─────────────────")
    n_correct, maxdev = 0, 0.0
    for k in 1:4
        φ_final = integrator == :reference ? solve_reference(nn_p, k).u[end] :
                  rk4_integrate(make_u0(k), nn_p, psi_xor(k); T=UDE_TSPAN[2])
        φ_out   = rem2pi(φ_final[OUTPUT_NODE], RoundNearest)
        target  = XOR_TARGETS[k]
        ok      = xor_correct(φ_out, target)
        ok && (n_correct += 1)
        maxdev  = max(maxdev, abs(rem2pi(φ_out - target, RoundNearest)))
        @printf("  %s  φ_out=%+.3fπ  target=%+.3fπ  %s\n",
                labels_xor[k], φ_out/π, target/π, ok ? "✓" : "✗")
    end
    @printf("Accuracy: %d/4 = %.0f%%   worst deviation from target %.3fπ\n",
            n_correct, 100.0 * n_correct / 4, maxdev / π)
    return n_correct, maxdev
end

trained_nn = result_ude2.minimizer
n_correct, maxdev = evaluate_xor(trained_nn, "trained UDE")
n_rk4, _   = evaluate_xor(trained_nn, "trained UDE"; integrator=:rk4)
# Control: everything except the learned coupling. With no output field the
# output must stay at its initial phase; with one, a single field shared by all
# patterns cannot separate them. Above 2/4 here would mean the target leaks in.
control_nn = let c = zero(trained_nn)
    if OUTFIELD
        c.h_out .= trained_nn.h_out; c.psi_out .= trained_nn.psi_out
    end
    if HIDFIELDS
        c.h_hid .= trained_nn.h_hid; c.psi_hid .= trained_nn.psi_hid
    end
    c
end
n_control, _ = evaluate_xor(control_nn, "zero-coupling control")
final_loss = loss_xor(trained_nn)
@printf("RESULT N=%d variant=%s seed=%d final_loss=%.6f accuracy=%d/4 maxdev=%.3fpi rk4_dt0.1=%d/4 control=%d/4\n\n",
        N, isempty(VARIANT) ? "complete" : VARIANT[2:end], SEED, final_loss,
        n_correct, maxdev / π, n_rk4, n_control)

# ── Plot trained dynamics ─────────────────────────────────────────────────────
# Sized for a full-width paper figure: large fonts, thick lines, one shared legend.
fig_ude = Figure(size=(1300, 900), fontsize=30)
for k in 1:4
    row, col = divrem(k-1, 2)
    ax = CairoMakie.Axis(fig_ude[row+1, col+1];
              xlabel = row == 1 ? "Time" : "",
              ylabel = col == 0 ? "Phase / π" : "",
              title = labels_xor[k], titlesize=32,
              yticks = -1:0.5:1, xgridvisible=false, ygridvisible=false)
    # Plot the same reference solve that the evaluation scores
    sol  = solve_reference(trained_nn, k; saveat=0.2)
    ts   = sol.t
    # Phases folded to (-π, π] for display; a NaN at each wrap breaks the line
    # instead of drawing a vertical jump across the panel
    wrapped = rem2pi.(reduce(hcat, sol.u), RoundNearest) ./ π
    for i in 1:N, s in 2:length(ts)
        abs(wrapped[i, s] - wrapped[i, s-1]) > 1 && (wrapped[i, s] = NaN)
    end
    for i in 3:N-1
        lines!(ax, ts, wrapped[i, :]; linewidth=2, color=(:gray50, 0.6),
               label = i == 3 ? "hidden" : nothing)
    end
    lines!(ax, ts, wrapped[1, :]; linewidth=4, color=:steelblue, label="input 1")
    lines!(ax, ts, wrapped[2, :]; linewidth=4, color=:darkorange, linestyle=:dashdot,
           label="input 2")
    lines!(ax, ts, wrapped[OUTPUT_NODE, :]; linewidth=5, color=:firebrick, label="output")
    hlines!(ax, [XOR_TARGETS[k]/π]; linestyle=:dash, linewidth=3, color=:black,
            label="target")
    ylims!(ax, -1.05, 1.05)
    row == 0 && hidexdecorations!(ax; grid=false, ticks=false)
    col == 1 && hideydecorations!(ax; grid=false, ticks=false)
end
Legend(fig_ude[3, 1:2], content(fig_ude[1, 1]); orientation=:horizontal,
       framevisible=false, labelsize=30)
let figdir = joinpath(@__DIR__, "../../results/figures")
    isdir(figdir) || mkpath(figdir)
    for ext in ("png", "pdf")
        f = joinpath(figdir, "kuramoto_xor_ude_$(RUN_TAG).$ext")
        save(f, fig_ude)
        println("Saved: ", f)
    end
end
