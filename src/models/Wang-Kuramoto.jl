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
# default rng
rng = StableRNG(1)

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

N = 3   # 2 inputs + 1 output (no hidden)
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
#   - Deterministic initial conditions — input nodes clamped to pattern values.
# ═══════════════════════════════════════════════════════════════════════════════

# Plain Julia neural network — bypasses LuxLib.Utils._return_type bug that
# affects all activations in this Lux/LuxLib version under Zygote AD.
# Architecture: 4 → 16 (tanh) → 1 (linear).  Parameters stored flat so
# ComponentArray gives a single gradient vector for the optimizer.
const nn_init = ComponentArray(
    W1 = randn(rng, Float64, 16 * 4) .* Float64(0.1),
    b1 = zeros(Float64, 16),
    W2 = randn(rng, Float64, 16) .* Float64(0.1),
    b2 = zeros(Float64, 1)
)

function apply_nn(nn_p, x::AbstractVector)
    W1 = reshape(nn_p.W1, 16, 4)
    h  = tanh.(W1 * x .+ nn_p.b1)
    return dot(nn_p.W2, h) + nn_p.b2[1]
end

# XOR truth table as target phases — rows: [FF, FT, TF, TT]
# Columns: [in1, in2, hid1, hid2, out]
const Ψ_XOR = Float64[
    -π/2  -π/2  -π/2;   # 0 XOR 0 = 0
    -π/2   π/2   π/2;   # 0 XOR 1 = 1
     π/2  -π/2   π/2;   # 1 XOR 0 = 1
     π/2   π/2  -π/2    # 1 XOR 1 = 0
]
const H_NODES     = Float64[0.5, 0.5, 0.5]        # bias strength per node
const XOR_TARGETS = Float64[-π/2, π/2, π/2, -π/2] # target phase for output node
const OUTPUT_NODE = 3
const UDE_TSPAN   = (0.0, 200.0)
const UDE_TSTEPS  = range(UDE_TSPAN[1], UDE_TSPAN[2], length=400)
const TRAIN_T     = 5.0   # short horizon for Zygote AD (50 steps); evaluate at T=200

# UDE ODE — p = ComponentArray(nn=<Lux params>, psi=<vertex ψ for this pattern>)
# Pure functional form (no mutation) so Zygote can differentiate through it.
function kuramoto_rhs(φ, nn_p, psi)
    return [begin
        s = -H_NODES[i] * sin(φ[i] - psi[i])
        for j in 1:N
            if j != i
                inp = [sin(φ[j]), cos(φ[j]), sin(φ[i]), cos(φ[i])]
                w_ij = apply_nn(nn_p, inp)
                s = s + w_ij * sin(φ[j] - φ[i])
            end
        end
        s
    end for i in 1:N]
end

# Manual RK4 integrator — fully differentiable by Zygote.
# Uses immutable rebinding (φ = φ .+ ...) so no mutation hits Zygote's tape.
# Training uses T=20 (fast); evaluation/plotting can pass a larger T.
function rk4_integrate(φ0, nn_p, psi; T=20.0, dt=0.1)
    φ     = Float64.(φ0)
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

# Trajectory integrator for plotting — mutating, efficient, no gradient needed.
function rk4_trajectory(φ0::Vector{Float64}, nn_p, psi; T=200.0, dt=0.2, save_every=5)
    φ      = copy(φ0)
    nsteps = round(Int, T / dt)
    nsaved = nsteps ÷ save_every + 1
    ts     = Vector{Float64}(undef, nsaved)
    traj   = Matrix{Float64}(undef, N, nsaved)
    ts[1]  = 0.0
    traj[:, 1] = φ
    si = 2
    for step in 1:nsteps
        k1 = kuramoto_rhs(φ,               nn_p, psi)
        k2 = kuramoto_rhs(φ .+ 0.5*dt.*k1, nn_p, psi)
        k3 = kuramoto_rhs(φ .+ 0.5*dt.*k2, nn_p, psi)
        k4 = kuramoto_rhs(φ .+    dt.*k3,  nn_p, psi)
        φ .= φ .+ (dt/6.0) .* (k1 .+ 2.0.*k2 .+ 2.0.*k3 .+ k4)
        if step % save_every == 0 && si <= nsaved
            ts[si]     = step * dt
            traj[:, si] = φ
            si += 1
        end
    end
    return ts, traj
end

# Wang et al. (2024) phase cost — numerically guarded near anti-phase
wang_cost(φ::Real, φt::Real) = -log(max(1.0 + cos(φ - φt), 1e-8))

# Initial condition builder — input nodes clamped to XOR pattern phases
function make_u0(k::Int)
    return [Ψ_XOR[k, 1], Ψ_XOR[k, 2], 0.0]
end

# Loss: Wang cost on final state (T=20) summed over all 4 XOR patterns.
# Uses manual RK4 so Zygote differentiates through integration directly —
# no ODE solver adjoint, no FunctionWrappers, no _return_type errors.
function loss_xor(nn_p)
    loss = zero(eltype(nn_p))
    for k in 1:4
        φ_final = rk4_integrate(make_u0(k), nn_p, Ψ_XOR[k, :]; T=TRAIN_T)
        loss   += wang_cost(φ_final[OUTPUT_NODE], XOR_TARGETS[k])
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

# Save trained NN parameters
let outdir = joinpath(@__DIR__, "../../results/models")
    isdir(outdir) || mkpath(outdir)
    @save joinpath(outdir, "Wang-Kuramoto-result-neuralode-N$(N).jld2") result_ude2
    println("Saved: results/models/Wang-Kuramoto-result-neuralode-N$(N).jld2")
end

# ── Evaluation ────────────────────────────────────────────────────────────────
println("\n── XOR evaluation (trained UDE) ─────────────────")
trained_nn = result_ude2.minimizer
let n_correct = 0
    for k in 1:4
        φ_final = rk4_integrate(make_u0(k), trained_nn, Ψ_XOR[k, :]; T=UDE_TSPAN[2])
        φ_out   = φ_final[OUTPUT_NODE]
        target  = XOR_TARGETS[k]
        ok      = sign(φ_out) == sign(target)
        ok && (n_correct += 1)
        @printf("  %s  φ_out=%+.3fπ  target=%+.3fπ  %s\n",
                labels_xor[k], φ_out/π, target/π, ok ? "✓" : "✗")
    end
    @printf("Accuracy: %d/4 = %.0f%%\n\n", n_correct, 100.0 * n_correct / 4)
end

# ── Plot trained dynamics ─────────────────────────────────────────────────────
fig_ude = Figure(size=(1000, 700))
for k in 1:4
    row, col = divrem(k-1, 2)
    ax = CairoMakie.Axis(fig_ude[row+1, col+1]; xlabel="Time", ylabel="Phase / π",
              title="UDE XOR $(labels_xor[k])")
    ts, traj = rk4_trajectory(make_u0(k), trained_nn, Ψ_XOR[k, :]; T=UDE_TSPAN[2])
    for i in [1, 2, OUTPUT_NODE]
        lbl = i == OUTPUT_NODE ? "Node $i (output)" : "Node $i (input)"
        lines!(ax, ts, traj[i, :] ./ π; linewidth=2, label=lbl)
    end
    hlines!(ax, [XOR_TARGETS[k]/π]; linestyle=:dash, color=:black, label="target")
    axislegend(ax; position=:rc)
end
let figdir = joinpath(@__DIR__, "../../results/figures")
    isdir(figdir) || mkpath(figdir)
    save(joinpath(figdir, "kuramoto_xor_ude_N$(N).png"), fig_ude)
    println("Saved: results/figures/kuramoto_xor_ude_N$(N).png")
end
