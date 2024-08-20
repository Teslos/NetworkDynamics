using Pkg
Pkg.activate(".")
using DelimitedFiles
using Graphs
using Interpolations
using Distributions
using FileIO
using Images
using ScikitLearn
@sk_import datasets: load_digits
include("spikerate.jl")
include("mnist.jl")

# read the data
function load_image_as_array(file_path::String)
    img_array = zeros(Float64, 8, 8, 2)
    for i in 1:2
        file = joinpath(file_path,"digit-$(i).jpg")
        img = FileIO.load(file)
        print(img)
        img = Gray.(img)
        img_array[:,:,i] = convert(Array{Float64,2}, img)
    end
    return img_array
end

image_digits = load_image_as_array("digits/")
digits = load_digits()
pl_digits = reshape(digits["data"][1:10,:]',8,8,1,10)
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
function create_complete_graph(N::Int=5)
    # create all to all graph
    g = Graphs.complete_graph(N)
    edge_weights = ones(length(edges(g)))
    g_weighted = SimpleDiGraph(g)
    g_directed = SimpleDiGraph(g_weighted)
    return g_directed, edge_weights
end

function create_graph(N::Int=8, M::Int=10)
    g = Graphs.grid([N, M])
    edge_weights = ones(length(edges(g)))
    g_weighted = SimpleWeightedDiGraph(g)
    g_directed = SimpleDiGraph(g_weighted)
    return g_directed, edge_weights
end

g_directed, edge_weights = create_complete_graph(8*8)
println("Number of nodes: ", nv(g_directed))
println("Number of edges: ", size(edge_weights))
println("Number of edges: ", ne(g_directed))
using NetworkDynamics

# function to generate the spike train
# rate: the rate of the pulse
# duration: the duration of the signal
# Tp: the duration of the pulse
# example: 0.5 rate we will have 0.5 * duration pulses

function pulse_generator(Tp, rate, duration)
    # Calculate the number of pulses
    num_pulses = Int(floor(duration * rate))
    
    # Initialize the current signal array
    signal = zeros(Float64, duration)
    
    # Generate the pulses
    for i in 1:num_pulses
        start_index = (i - 1) * Int(floor(1 / rate)) + 1
        end_index = min(start_index + Tp - 1, duration)
        signal[start_index:end_index] .= 1
    end
    
    return signal
end
x = pl_digits[:,:,1,1] ./ 16.0 
spike_train = spikerate.rate(x, 350)
# convert the spike train to a Float32 array and (time, color, 1)
spike_train = reshape(Float32.(spike_train), 350, :, 1)

#j0 = zeros(Float64, 350)
g0 = interpolate(spike_train[:,1], BSpline(Quadratic(Line(OnCell()))))
R0 = 0.01

g0v = [interpolate(spike_train[:,i], BSpline(Quadratic(Line(OnCell())))) for i in 1:nv(g_directed)] # repeat the signal for all nodes
e0v = [g0 .* R0 for _ in 1:ne(g_directed)] # repeat the signal for all edges
Base.@propagate_inbounds function fhn_electrical_vertex!(dv, v, edges, p, t)
    # add external input j0
    g = p

    #println("g:",g)
    # adding external input current
    if t < 0.5
        dv[1] = (v[1] + v[1]^3/3 - v[2])
    else
        dv[1] = (g[t] + v[1] - v[1]^3/3 - v[2])
    end
    # adding the external input voltage
    if t < 0.5
        dv[2] = (v[1] + a)*ϵ
    else
        dv[2] = R0 .* g[t] + (v[1] + a)*ϵ
    end
    
    for e in edges
        dv[1] += e[1]
        #dv[2] += e[2]
    end
    nothing
end

Base.Base.@propagate_inbounds function electrical_edge!(e, v_s, v_d, p, t)
    e[1] = p*(v_s[1] - v_d[1]) # *σ  - edge coupling for current
    e[2] = p*(v_s[2] - v_d[2]) # *σ  - edge coupling for voltage
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

# different weights for edges, because the resitivity of the edges are always positive
w_ij = [pdf(Normal(), x) for x in range(-1, 1, length=ne(g_directed))]
#w_ij = σ/ϵ * ones(ne(g_directed))

# Tuple of parameters for nodes and edges
p = (g0v, w_ij)
#Initial conditions
x0 = randn(2N)

# Solving the ODE
using OrdinaryDiffEq

tspan = (0.0, 300.0)
datasize = 400
tsteps = range(tspan[1], tspan[2], length=datasize)
prob = ODEProblem(fhn_network!, x0, tspan, p)
sol = solve(prob, Tsit5(), saveat=tsteps)
#sol = solve(prob, Tsit5())
diff_data = Array(sol)

# Plotting the solution
#using Plots
#plot(sol, vars=idx_containing(fhn_network!, :u), legend=false, ylim = (-5,5), fmt=:png)

# using the new plotting package CairoMakie

using GLMakie
fig = Figure()
ax = GLMakie.Axis(fig[1, 1], xlabel = "Time", ylabel = "u", title = "FitzHugh-Nagumo network")
t= sol.t
u = sol(sol.t)[1:N,:]
for i in 1:N
    lines!(ax, t, u[i,:], label="Oscillator $i")
    #text!(ax, t[end], u[i,end]+0.1, text=string("Oscillator ", i), align=(:right, :center))
end
#axislegend(ax, position = :rt)
fig
GLMakie.save("FitzHug-Nagumo.png", fig)
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
ann_fhn = Lux.Chain(Lux.Dense(2, 20, tanh),
    Lux.Dense(20, 1))
nn_pp, st = Lux.setup(rng, ann_fhn)
@inline function fhn_edge!(e, v_s, v_d, p, t)
    in = [v_s[1], v_d[1]]
    e[1] = Lux.apply(ann_fhn, in, p, st)[1][1]
    nothing
end
@inline function fhn_vertex!(dv, v, edges, p, t)
    # add external input j0
    g = p

    #println("g:",g)
    # adding external input current
    if t < 0.5
        dv[1] = (v[1] + v[1]^3/3 - v[2])
    else
        dv[1] = (g[t] + v[1] - v[1]^3/3 - v[2])
    end
    # adding the external input voltage
    if t < 0.5
        dv[2] = (v[1] + a)*ϵ
    else
        dv[2] = R0 .* g[t] + (v[1] + a)*ϵ
    end
    
    for e in edges
        dv[1] += e[1]
        #dv[2] += e[2]
    end
    nothing
end
ann_fhn_vertex = ODEVertex(; f=fhn_vertex!, dim=2, sym=[:u, :v])
ann_fhn_edge = StaticEdge(; f=fhn_edge!, dim=1, coupling=:directed)
fhn_network! = network_dynamics(ann_fhn_vertex, ann_fhn_edge, g_directed)

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
        plt = Plots.scatter(tsteps, diff_data[1, :]; label = "data")
        Plots.scatter!(plt, tsteps, pred[1, :]; label = "prediction")
        display(Plots.plot(plt))
    end
    return false
end
pinit = ComponentArray(nn_pp)
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
