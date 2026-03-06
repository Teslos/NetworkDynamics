# generate simple graph and edge weights to test lyapunov spectrum calculation
include("../src/models/chaos_fhn.jl")
using .ChaosFHN
using NetworkDynamics
using Graphs
using Random
using LinearAlgebra
using SimpleWeightedGraphs
using Interpolations
using Distributions
using Dierckx
using OrdinaryDiffEq
using CairoMakie

# Create a random directed graph
function create_complete_graph(N::Int=5)
    # create all to all graph
    g = Graphs.complete_graph(N)
    edge_weights = ones(length(edges(g)))
    g_weighted = SimpleDiGraph(g)
    g_directed = SimpleDiGraph(g_weighted)
    return g_directed, edge_weights
end

g_directed, edge_weights = create_complete_graph(10)

@inline Base.@propagate_inbounds function fhn_electrical_vertex_simple!(dv, v, edges, p, t)
    g = p
    e_s, e_d = edges
    dv[1] = g(t) + v[1] - v[1]^3 / 3 - v[2]
    dv[2] = (g(t) .* R0 + v[1] - a) * ϵ
    for e in e_s
        dv[1] -= e[1]
    end
    for e in e_d
        dv[1] += e[1]
    end
    nothing
end

@inline Base.@propagate_inbounds function electrical_edge_simple!(e, v_s, v_d, p, t)
    e[1] =  p * (v_s[1] - v_d[1]) # * σ
    nothing
end

odeelevertex = NetworkDynamics.ODEVertex(; f=fhn_electrical_vertex_simple!, dim=2, sym=[:u, :v])
odeeleedge = StaticEdge(; f=electrical_edge_simple!, dim=2, coupling=:directed)

fhn_network! = network_dynamics(odeelevertex, odeeleedge, g_directed)
σ = 0.7
a = 0.5
ϵ = 0.05
R0 = 0.5
N = nv(g_directed)
x0 = rand(Float64, 2*N)
g_input = [Spline1D(1:32, randn(32), k=2) for _ in 1:nv(g_directed)]
tspan = (0.0, 100.0)
w_ij = [pdf(Normal(), x) for x in range(-1, 1, length=ne(g_directed))]
# 1. Solve system
N = nv(g_directed)
p = (g_input, σ * w_ij)
prob = ODEProblem(fhn_network!, x0, (0.0, 2500.0), p)
sol = solve(prob, Tsit5(), saveat=0.1)
# Chaos criterion: λ_max > 0
λs = ChaosFHN.calculate_lyapunov_spectrum(g_directed, w_ij, g_input, tspan, x0, σ, fhn_network!)
println("Lyapunov exponents: ", λs)

println("\n2. PHASE SPACE ANALYSIS")
fig_phase = ChaosFHN.analyze_phase_space(sol, 1:5)
CairoMakie.save("phase_space_5_osc_FHN.png", fig_phase)
