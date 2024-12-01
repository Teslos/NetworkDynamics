using Pkg
Pkg.activate(".")
using DelimitedFiles
using Graphs
using MLUtils
using Interpolations
using Distributions
using FileIO
using Images
using ScikitLearn
using OneHotArrays
using Flux: shuffle
using Interpolations
using Dierckx
using Random

rng = Xoshiro(1234)
@sk_import datasets: load_digits
include("spikerate.jl")
include("drybean.jl")
include("graph_utils.jl")

# adjust the load path for your system
G = readdlm(joinpath(@__DIR__, "./Norm_G_DTI.txt"),',', Float64, '\n')

using SimpleWeightedGraphs, Graphs

# read the dry bean dataset
db = drybean.read_drybean()
x = Matrix(permutedims(db[shuffle(rng, 1:end), :]))
function normalize_rows(x::AbstractMatrix)
    x = x ./ maximum(x, dims=2)
    return x
end
# target vector
targets = x[17,:]
# normalize only the features
x = normalize_rows(x[1:16,:])


# first we need to create a weighted, directed graph
g_weighted = SimpleWeightedDiGraph(G)



#g_directed, edge_weights = graph_utils.create_watts_strogatz_graph(16*16; k=5, prob=0.15) # the values are taken from Al Beattie
g_directed, edge_weights = graph_utils.create_complete_graph(512)
#g_directed, edge_weights = create_barabasi_albert_graph(512)
#g_directed, edge_weights = create_graph(32,16)
#g_directed, edge_weights = create_erdos_renyi_graph(1024, 0.05)

println("Number of nodes: ", nv(g_directed))
println("Number of edges: ", size(edge_weights))
println("Number of edges: ", ne(g_directed))
using NetworkDynamics


@inline Base.@propagate_inbounds function duffing_vertex!(dv, v, edges, p, t)
    f = p # forcing term
    e_s, e_d = edges
    dv[1] = v[2]
    # Duffing oscillator
    omega = ω - rand()*0.05
    # we are setting the frequency to be constant 
    # omega = ω
    dv[2] = -ω * v[1] - β * v[1]^3 - d*v[2] + f(t)
    for e in e_s
        dv[1] -= e[1]
    end
    for e in e_d
        dv[1] += e[1]
    end
    nothing
end

@inline Base.Base.@propagate_inbounds function duffing_edge!(e, v_s, v_d, p, t)
    e[1] = p*(v_s[1] - v_d[1]) # *σ
    nothing
end

odeelevertex = ODEVertex(; f=duffing_vertex!, dim=2, sym=[:u,:v])
odeeleedge = StaticEdge(; f=duffing_edge!, dim=1, coupling=:directed)

fhn_network! = network_dynamics(odeelevertex, odeeleedge, g_directed)

# Parameter handling
N = nv(g_directed) # Number of nodes in the network

const ϵ = 0.05 # global variables that are accessed several times should be declared as constants
const a = 0.5
const σ = 0.1
const f = 0.1
const β = 1.0

const ω = 1.0
const d = 0.1
# produce the training data
using OrdinaryDiffEq

# using the new plotting package CairoMakie
using GLMakie
function plot_sol(u, t_steps, num_sol)
    fig = Figure()
    ax = GLMakie.Axis(fig[1, 1], xlabel = "Time", ylabel = "u", title = "FitzHugh-Nagumo network")
    for i in 1:num_sol
        lines!(ax, t_steps, u[i,:], label="Oscillator $i")
    end
    fig
    GLMakie.save("Duffing_MNIST.png", fig)
end
using Plots

using Random
using Lux
using Flux
using Optimization
using OptimizationOptimJL
using OptimizationOptimisers
using ComponentArrays
using DiffEqFlux
using Zygote
using OneHotArrays

# network training for the FitzHugh-Nagumo RC last two layers
function load_data(x, yt; shuffling=true, train_ratio = 1.0)
    num_samples = 512
    classes = unique(yt)

    num_train_samples = Int(floor(num_samples * train_ratio))

    if shuffling 
        shuffled_indices = shuffle(rng, 1:num_samples)
    else
        shuffled_indices = 1:num_samples
    end

    # shuffle the data
    train_indices = shuffled_indices[1:num_train_samples]
    test_indices = shuffled_indices[num_train_samples+1:end]

    # convert the data to spike trains
    spike_train = spikerate.rate(x[:,1:num_samples], 32)
    println("spike train before reshape: $(size(spike_train))")
    spike_train_test = spikerate.rate(x[:,num_samples+1:2*num_samples], 32)
    # convert the spike train to a Float32 array and (time, color, 1)
    spike_train = Float32.(reshape(spike_train,:,num_samples))
    spike_train_test = Float32.(reshape(spike_train_test,:,num_samples))
    
    print("spike_train size:",size(spike_train)) 
    tspike = collect(1:size(spike_train,1))
    # tspike = LinRange(0.0, 32.0, size(spike_train,2))
    
    nsamples = size(spike_train,1)
    println("Number of samples: $(nsamples)")
    println("tspike: $(size(tspike)), number vertex: $(nv(g_directed))")
    gs = [Spline1D(tspike, spike_train[:,i], k=2) for i in 1:nv(g_directed)] # do spike train interpolation
    print("gs size:",size(gs))

    # different weights for edges, because the resitivity of the edges are always positive
    w_ij = [pdf(Normal(), x) for x in range(-1, 1, length=ne(g_directed))]
    #w_ij = ones(ne(g_directed))
    nosc = nv(g_directed)
    uall = zeros(Float64, 1, N)

        p = (gs, σ *w_ij)
        #Initial conditions are choosen randomly
        x0 = rand(Float64,2*N)
        # set the problem
        tspan = (0.0, Float64(size(spike_train,1)))
        datasize = 512
        tsteps = range(tspan[1], tspan[2], length=datasize)
        prob = ODEProblem(fhn_network!, x0, tspan, p)
        # solve the Duffing network
        sol = solve(prob, TRBDF2(), saveat=tsteps)
        # if solution converges, then the solution is saved
        if Symbol(sol.retcode) == :Success
            diff_data = Array(sol)
            t = sol.t
            u = sol(sol.t)[1:N,:] # (N, T) nodes, time
            print("u size:",size(u))
            # add the data together
            uall = u

            print("uall size:",size(uall))
        else
            println("Solution did not converge: ", sol.retcode)
            train_x = []
            train_y = []
            test_x = []
            test_y = []
        end

    train_x = uall[train_indices,:]
    train_y = yt[train_indices]
    train_y = onehotbatch(train_y, classes)
    test_x = uall[test_indices,:]
    test_y = yt[test_indices]
    test_y = onehotbatch(test_y, classes)
    return (train_x, train_y), (test_x, test_y)
end

function create_model()
    return Lux.Chain(
        Lux.Dense(640, 512, tanh),
        Lux.Dense(512, 7)
    )
end



# train the network with the output of the FitzHugh-Nagumo network
# the output of the FitzHugh-Nagumo network is the input to the RC network
# loss function is the cross entropy loss
function loss(x, y, model, ps, st)
    #print("x size:",size(x),"y size:",size(y))
    print("x device")
    y_pred, st = model(x, ps, st)
    #println("y_pred size:", size(y_pred))
    
    lossv = -mean(sum(y .* log.(softmax(y_pred)), dims=1))
    println("lossv:",lossv)
    return lossv, st
end


loss_function(x, y, model, ps, states) = loss(x, y, model, ps, states)
function train_model(model, train_data, test_data; epochs=10, batch_size=256, learning_rate=0.001)
    ps, st = Lux.setup(rng, model)
    
    opt = Optimisers.Adam(learning_rate)
    st_opt = Optimisers.setup(opt, ps)

    # train loop
    for epoch in 1:epochs
        for (x, y) in partition(train_data, batch_size)
            #println("x:",size(x),"y:",size(y))
            (loss_value, st), back = pullback(loss_function, x, y, model, ps, st)
        
            grads = back((one(loss_value),nothing))[4]
            st_opt, ps = Optimisers.update(st_opt, ps, grads)
        end
        println("train_labels:", size(train_data[2]))

        train_acc = accuracy(model, ps, st, train_data...)

        test_acc = accuracy(model, ps, st, test_data...)
        println("Epoch $epoch, Train Accuracy: $train_acc, Test Accuracy: $test_acc")
    end
end
# flux model
using FluxOptTools
#using Optim
using Flux, CUDA
# initialize CUDA
Flux.gpu_backend!("CUDA")
#spike_train = spike_train |> Flux.gpu
function loss(x, y)
    y_pred = model_flux(x)
    lossv = sum(abs2, y .- y_pred)
    #println("lossv:",lossv)
    return lossv
end

function loss_ce(x, y)
    y_pred = model_flux(x)
    #print("y_pred:",size(y_pred))
    lossv = Flux.Losses.crossentropy(y_pred, y)
    #println("lossv:",lossv)
    return lossv
end

dim_system = 7
model_flux = Flux.Chain(
    Flux.Dense(512, 512, swish),
    Flux.Dense(512, 256, swish),
    Flux.Dense(256, dim_system),
    Flux.softmax
) |> Flux.gpu
ps = Flux.params(model_flux)
ps = ps |> Flux.gpu
(train_x, train_y), (test_x, test_y) = load_data(x,targets;shuffling=false, train_ratio = 0.9) |> Flux.gpu

# BFGS optimizer for the model
#lossfun, gradfun, fg!, p0 = optfuns(()->loss(model_flux), ps)
#res = Optim.optimize(Optim.only_fg!(fg!), p0, BFGS(), Optim.Options(iterations = 1000, store_trace=true))
# Standard ADAM optimizer for the model
opt = Flux.ADAM(0.001)
epochs = 300
data_loader = Flux.Data.DataLoader((train_x', train_y), batchsize=64, shuffle=true)
for epoch in 1:epochs
    for (x, y) in data_loader
        Flux.train!(loss_ce, ps, [(x, y)], opt)
        println("Epoch $epoch, Loss: $(loss(x, y))")
    end
end

# test accuracy
function accuracy(model, x, y)
    y_pred = model(x)
    pred_class = sum(onecold(y) .== onecold(y_pred))
    acc = pred_class / size(y, 2)
    return acc
end
# define the confusion matrix
using CategoricalArrays, MLJ
function confusion_matrix(model, x, y)
    y_pred = model(x)
    pred_class = onecold(y_pred)
    true_class = onecold(y)
    # get to cpu memory
    pred_class = pred_class |> Flux.cpu  
    predictions = pred_class |> CategoricalArray
    true_class = true_class |> Flux.cpu
    targets = true_class |> CategoricalArray
    cm = MLJ.ConfusionMatrix()(targets, predictions)
    return cm
end
train_acc = accuracy(model_flux, train_x', train_y)
test_acc  = accuracy(model_flux, test_x', test_y)
println("train accuracy $train_acc")
println("test accuracy $test_acc")
conf = confusion_matrix(model_flux, test_x', test_y)
# plot confusion matrix
# Plot confusion matrix for test data using GLMakie
fig = Figure(resolution = (800, 800))
ax = GLMakie.Axis(fig[1, 1], title = "Confusion Matrix", xlabel ="predicted class", ylabel ="true class", aspect = DataAspect())
GLMakie.heatmap!(ax, rotr90(conf.mat), colormap = :viridis)
# Annotate the heatmap with the confusion matrix values
for i in 1:size(conf.mat, 1)
    for j in 1:size(conf.mat, 2)
        GLMakie.text!(ax, j, i, text = string(rotr90(conf.mat)[i,j]), align = (:center, :center), color = :white)
    end
end
fig
GLMakie.save("confusion_matrix.png", fig)
plot_sol(Array(train_x), collect(1:size(train_x,2)), 16)