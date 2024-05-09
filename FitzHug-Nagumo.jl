using Pkg
Pkg.activate(".")
using DelimitedFiles
using Graphs

# adjust the load path for your system
G = readdlm(joinpath(@__DIR__, "./Norm_G_DTI.txt"),',', Float64, '\n')

using SimpleWeightedGraphs, Graphs

# first we need to create a weighted, directed graph
g_weighted = SimpleWeightedDiGraph(G)

# For later use, we extract the edge.weight attributes
# . is the broadcast operator and gets the attribute :weight for every edge 
#edge_weights = getfield.(collect(edges(g_weighted)), :weight)

# we promote the g_weighted as directed graph (weights of the edges are included in parameters)
#g_directed = SimpleDiGraph(g_weighted)

function create_graph(N::Int=8, M::Int=10)
    g = Graphs.grid([N, M])
    edge_weights = ones(length(edges(g)))
    g_weighted = SimpleWeightedDiGraph(g)
    g_directed = SimpleDiGraph(g_weighted)
    return g_directed, edge_weights
end

g_directed, edge_weights = create_graph(4, 4)
println("Number of nodes: ", nv(g_directed))
println("Number of edges: ", size(edge_weights))
println("Number of edges: ", ne(g_directed))
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
N = nv(g_directed) # Number of nodes
const ϵ = 0.05 # global variables that are accessed several times should be declared as constants
const a = 0.5
const σ = 0.5


# Tuple of parameters for nodes and edges
p = (nothing, σ/ϵ * ones(ne(g_directed)))
#Initial conditions
x0 = randn(2N)

# Solving the ODE
using OrdinaryDiffEq

tspan = (0.0, 200.0)
datasize = 100
tsteps = range(tspan[1], tspan[2], length=datasize)
prob = ODEProblem(fhn_network!, x0, tspan, p)
sol = solve(prob, AutoTsit5(TRBDF2()),saveat=tsteps)
diff_data = Array(sol)

# Plotting the solution
#using Plots
#plot(sol, vars=idx_containing(fhn_network!, :u), legend=false, ylim = (-5,5), fmt=:png)

# using the new plotting package CairoMakie
#=
using CairoMakie
fig = Figure()
ax = CairoMakie.Axis(fig[1, 1], xlabel = "Time", ylabel = "u", title = "FitzHugh-Nagumo network")
t= sol.t
u = sol(sol.t)[1:N,:]
for i in 1:N
    lines!(ax, t, u[i,:], color = (:blue, 0.1))
end
fig
=#
using Plots
using Random
using Lux
using Optimization
using OptimizationOptimJL
using OptimizationOptimisers
using ComponentArrays
using DiffEqFlux

# Learning the coupling strength
rng = Random.default_rng()
ann_fhn = Chain(Dense(2, 20, tanh),
    Dense(20, 1))
p, st = Lux.setup(rng, ann_fhn)
@inline function fhn_edge!(e, v_s, v_d, p, t)
    in = [v_s[1], v_d[1]]
    e[1] = Lux.apply(ann_fhn, in, p, st)[1][1]
    nothing
end

ann_fhn_edge = StaticEdge(; f=fhn_edge!, dim=1, coupling=:directed)
fhn_network! = network_dynamics(odeelevertex, ann_fhn_edge, g_directed)

prob_neuralode = ODEProblem(fhn_network!, x0, tspan, p)

function predict_neuralode(p)
    prob = remake(prob_neuralode, p=p)
    Array(solve(prob, Tsit5(), saveat=tsteps,sensealg = ForwardDiffSensitivity()))
end


function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, diff_data .- pred)
    return loss, pred
end

callback = function (p, l, pred; doplot = false)
    println("Current loss is: ", l)
    # plot current prediction against data
    if doplot
        plt = scatter(tsteps, diff_data[1, :]; label = "data")
        scatter!(plt, tsteps, pred[1, :]; label = "prediction")
        display(plot(plt))
    end
    return false
end
pinit = ComponentArray(p)
callback(pinit, loss_neuralode(pinit)...; doplot=true)

# use Optimization.jl to solve the problem
adtype = Optimization.AutoZygote()

optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pinit)

result_neuralode = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.05); callback = callback,
    maxiters = 10)
optprob2 = remake(optprob; u0 = result_neuralode.u)

result_neuralode2 = Optimization.solve(optprob2, Optim.BFGS(; initial_stepnorm = 0.01);
    callback, allow_f_increases = false)

callback(result_neuralode2.u, loss_neuralode(result_neuralode2.u)...; doplot=true)
savefig("FitzHug-Nagumo-opt.png")
