using DelimitedFiles

# adjust the load path for your system
G = readdlm(joinpath(@__DIR__, "./Norm_G_DTI.txt"),',', Float64, '\n')

using SimpleWeightedGraphs, Graphs

# first we need to create a weighted, directed graph
g_weighted = SimpleWeightedDiGraph(G)

# For later use, we extract the edge.weight attributes
# . is the broadcast operator and gets the attribute :weight for every edge 
edge_weights = getfield.(collect(edges(g_weighted)), :weight)

# we promote the g_weighted as directed graph (weights of the edges are included in parameters)
g_directed = SimpleDiGraph(g_weighted)

using NetworkDynamics

Base.@propagate_inbounds function duffing_vertex!(dv, v, edges, p, t)
    dv[1] = v[2]
    # Duffing oscillator
    omega = ω - rand()*0.5
    dv[2] = -v[1] - β * v[1]^3 - d*v[2] + f*cos(omega*t)
    for e in edges
        dv[1] += e[1]
    end
    nothing
end

Base.Base.@propagate_inbounds function duffing_edge!(e, v_s, v_d, p, t)
    e[1] = p*(v_s[1] - v_d[1]) # *σ
    nothing
end

odeelevertex = ODEVertex(; f=duffing_vertex!, dim=2, sym=[:u, :v])
odeeleedge = StaticEdge(; f=duffing_edge!, dim=1, coupling=:directed)

duffing_network! = network_dynamics(odeelevertex, odeeleedge, g_directed)

# Parameter handling
N = 90 # Number of nodes
const ϵ = 0.05 # global variables that are accessed several times should be declared as constants
const a = 0.5
const σ = 10.0
const f = 0.1
const β = 20.0
const ω = 1.0
const d = 0.1


# Tuple of parameters for nodes and edges
p = (nothing, σ * edge_weights)
#Initial conditions
x0 = randn(2N)
x0[1] = 1.0

# Solving the ODE
using OrdinaryDiffEq

tspan = (0.0, 400.0)
prob = ODEProblem(duffing_network!, x0, tspan, p)
sol = solve(prob, AutoTsit5(TRBDF2()))

# Plotting the solution
using Plots
plot(sol, vars=idx_containing(duffing_network!, :u), legend=false, ylim = (-5,5), xlim =(0,100), fmt=:png)

using GraphMakie, WGLMakie
using GraphMakie.NetworkLayout

fig, ax, p = graphplot(g_directed)
hidedecorations!(ax); hidespines!(ax); hide