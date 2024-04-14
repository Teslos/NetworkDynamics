using DelimitedFiles

# adjust the load path for your system
G = readdlm(joinpath(@__DIR__, "./Norm_G_DTI.txt"),',', Float64, '\n')

using SimpleWeightedGraphs, Graphs

function create_graph()
    # first we need to create a weighted, directed graph
    g_weighted = SimpleWeightedDiGraph(G)

    # For later use, we extract the edge.weight attributes
    # . is the broadcast operator and gets the attribute :weight for every edge 
    edge_weights = getfield.(collect(edges(g_weighted)), :weight)

    # we promote the g_weighted as directed graph (weights of the edges are included in parameters)
    g_directed = SimpleDiGraph(g_weighted)
    return g_directed, edge_weights
end

function create_barabasi_albert_graph()
    g = barabasi_albert(100, 3, is_directed=true)
    return g
end


#g_directed, edge_weights = create_graph()
g_directed = create_barabasi_albert_graph()
edge_weights = ones(length(edges(g_directed)))

using NetworkDynamics

Base.@propagate_inbounds function duffing_vertex!(dv, v, edges, p, t)
    dv[1] = v[2]
    # Duffing oscillator
    omega = ω - rand()*0.05
    # we are setting the frequency to be constant 
    # omega = ω
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
N = 100 # Number of nodes
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

# Plotting the solution in Makie
#using Plots
#plot(sol, vars=idx_containing(duffing_network!, :u), legend=false, ylim = (-5,5), xlim =(0,100), fmt=:png)
#savefig("duffing_barabasi_albert.png")
using GraphMakie, GLMakie
using GraphMakie.NetworkLayout
fig1 = Figure()
ax = Axis(fig1[1, 1], xlabel = "Time", ylabel = "u", title = "Duffing network")
t = sol.t
y = sol[1:N,:]
for i in 1:N
    lines!(ax, t, y[i,:], color = (:blue, 0.1))
end
save("duffing_barabasi_albert.png",fig1, px_per_unit = 4)

fig, ax, p = graphplot(g_directed)
hidedecorations!(ax); hidespines!(ax);
save("barabasi_albert.png",fig)