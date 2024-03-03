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

Base.@propagate_inbounds function fhn_electrical_vertex!(dv, v, edges, p, t)
    dv[1] = (v[1] - v[1]^3/3 - v[2])
    dv[2] = (v[1] + a)*ϵ
    for e in edges
        dv[1] += e[1]
    end
    nothing
end

Base.Base.@propagate_inbounds function electrical_edge!(e, v_s, v_d, p, t)
    e[1] = p*(v_s[1] - v_d[1]) # *σ
    nothing
end

odeelevertex = ODEVertex(; f=fhn_electrical_vertex!, dim=2, sym=[:u, :v])
odeeleedge = StaticEdge(; f=electrical_edge!, dim=1, coupling=:directed)

fhn_network! = network_dynamics(odeelevertex, odeeleedge, g_directed)

# Parameter handling
N = 90 # Number of nodes
const ϵ = 0.05 # global variables that are accessed several times should be declared as constants
const a = 0.5
const σ = 0.5


# Tuple of parameters for nodes and edges
p = (nothing, σ/ϵ * edge_weights)
#Initial conditions
x0 = randn(2N)*5

# Solving the ODE
using OrdinaryDiffEq

tspan = (0.0, 200.0)
prob = ODEProblem(fhn_network!, x0, tspan, p)
sol = solve(prob, AutoTsit5(TRBDF2()))

# Plotting the solution
using Plots
plot(sol, vars=idx_containing(fhn_network!, :u), legend=false, ylim = (-5,5), fmt=:png)