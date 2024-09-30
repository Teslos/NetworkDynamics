using DelimitedFiles

# adjust the load path for your filesystem!
G = readdlm(joinpath(@__DIR__, "./Norm_G_DTI.txt"), ',', Float64, '\n')

using Graphs

function create_graph(G)
    # First we construct a weighted, directed graph
    g_weighted = SimpleWeightedDiGraph(G)

    # For later use we extract the edge.weight attributes
    # . is the broadcasting operator and gets the attribute :weight for every edge
    edge_weights = getfield.(collect(edges(g_weighted)), :weight)
    # edge_weights = 0.001*ones(length(edges(g_weighted)))
    # edge_weights = 0.001*2*(rand(length(edges(g_weighted))).-0.5) 
    edge_weights = 0.001 * [pdf(Normal(),x) for x in range(-1,1, length(edge_weights))]
    g_directed = SimpleDiGraph(g_weighted)
    return g_directed, edge_weights
end


using NetworkDynamics

@inline Base.@propagate_inbounds function fhn_electrical_vertex!(dv, v, edges, p, t)
    e_s, e_d = edges
    dv[1] = v[1] - v[1]^3 / 3 - v[2]
    dv[2] = (v[1] - a) * ϵ
    for e in e_s
        dv[1] -= e[1]
    end
    for e in e_d
        dv[1] += e[1]
    end
    nothing
end

@inline Base.@propagate_inbounds function electrical_edge!(e, v_s, v_d, p, t)
    e[1] =  p * (v_s[1] - v_d[1]) # * σ
    nothing
end
g_directed, edge_weights = create_graph(G)
odeelevertex = ODEVertex(; f=fhn_electrical_vertex!, dim = 2, sym=[:u, :v])
electricaledge = StaticEdge(; f=electrical_edge!, dim = 1)

fhn_network! = network_dynamics(odeelevertex, electricaledge, g_directed)

# Defining global parameters

N = 90         # number of nodes
const ϵ = 0.05 # global variables that are accessed several times should be declared `const`
const a = .5
const σ = .6

# Tuple of parameters for nodes and edges

p = (nothing, σ * edge_weights)

# Initial conditions

x0 = randn(2N)*5

using OrdinaryDiffEq

tspan = (0., 300.)
prob  = ODEProblem(fhn_network!, x0, tspan, p)
sol   = solve(prob, AutoTsit5(TRBDF2()))
nothing # hide

using GLMakie
fig = Figure()
ax = GLMakie.Axis(fig[1, 1], xlabel = "Time", ylabel = "u", title = "FitzHugh-Nagumo network")
t= sol.t

for i in 1:N
    #lines!(ax, t, sol[i,:], label="Oscillator $i")
    # plot heatmap of the network
    GLMakie.heatmap!(ax, t, i*ones(length(t)), sol[i,:], colormap = :viridis)
    #text!(ax, t[end], u[i,end]+0.1, text=string("Oscillator ", i), align=(:right, :center))
end
#axislegend(ax, position = :rt)
fig
using Plots

Plots.plot(sol, vars = idx_containing(fhn_network!, :u), legend = false, ylim=(-5, 5))
savefig("./fhnsync.svg") # hide