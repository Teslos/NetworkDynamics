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

# adjust the load path for your system
G = readdlm(joinpath(@__DIR__, "./Norm_G_DTI.txt"),',', Float64, '\n')

using SimpleWeightedGraphs, Graphs
# process digit data from the MNIST dataset
function load_digits_data()
    # read the training data
    directory = joinpath(@__DIR__, "digits")
    lines = readdlm(joinpath(directory, "optdigits.tra"), ',', Int, comment_char='#')
    # read the test data
    lines_test = readdlm(joinpath(directory, "optdigits.tes"), ',', Int, comment_char='#')

    # Print the shape of the data
    println("Training data shape: ", size(lines))
    println("Test data shape: ", size(lines_test))
    print("Training data: ", lines[1:5,:])
    print("Test data: ", lines_test[1:5,:])
    return lines, lines_test
end

using GLMakie
img_train, img_test = load_digits_data()
fig = GLMakie.Figure()


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
#digits = img_train
img_digits = load_digits()
num_digits = 512
# shuffle the data
shuffle_indices = shuffle(rng, 1:num_digits)
#pl_digits = permutedims(reshape(img_digits["data"][shuffle_indices,:],:,8,8),(1,3,2))
#pl_digits = permutedims(reshape(digits[shuffle_indices,1:64], :, 8, 8), (3,2,1))
pl_digits = img_digits["images"][shuffle_indices,:,:]
# target values
targets = img_digits["target"][shuffle_indices]
#targets = digits[shuffle_indices,65]
for i in 1:5
    ax = GLMakie.Axis(fig[1, i], title="Digit $(img_digits["target"][i])", aspect = DataAspect())
    img = rotr90(img_digits["images"][i,:,:])
    GLMakie.heatmap!(ax, img, colormap=:viridis)
end
fig
GLMakie.save("digits.png", fig)
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

g_directed, edge_weights = create_complete_graph(512)
println("Number of nodes: ", nv(g_directed))
println("Number of edges: ", size(edge_weights))
println("Number of edges: ", ne(g_directed))
using NetworkDynamics

x = pl_digits ./ 16.0

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
const σ = 1.0
const f = 0.1
const β = 20.0
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
function load_data(x, y; shuffling=true, train_ratio = 1.0)
    num_samples = num_digits
    classes = 0:9

    num_train_samples = Int(floor(num_samples * train_ratio))

    # convert the data to spike trains
    spike_train = spikerate.rate(x, 8)
    spike_train_test = spikerate.rate(x, 8)
    # convert the spike train to a Float32 array and (time, color, 1)
    spike_train = Float32.(permutedims(spike_train,(2,1,3,4)))
    spike_train_test = Float32.(permutedims(spike_train_test,(2,1,3,4)))
    spike_train = reshape(spike_train, :,8*8*8)
    print("spike_train size:",size(spike_train)) 
    tspike = collect(1:size(spike_train,2))
    nsamples = size(spike_train,1)
    gs = [Spline1D(tspike, spike_train[i,:], k=2) for i in 1:nsamples] # do spike train interpolation
    print("gs size:",size(gs))

    # different weights for edges, because the resitivity of the edges are always positive
    # w_ij = [pdf(Normal(), x) for x in range(-1, 1, length=ne(g_directed))]
    w_ij = ones(ne(g_directed))
    nosc = nv(g_directed)
    uall = zeros(Float64, 1, N)

        p = (gs, w_ij)
        #Initial conditions
        x0 = rand(Float64,2*N)
        # set the problem
        tspan = (0.0, Float64(size(spike_train,2)))
        datasize = size(spike_train,2)
        tsteps = range(tspan[1], tspan[2], length=datasize)
        prob = ODEProblem(fhn_network!, x0, tspan, p)
        # solve the FitzHugh-Nagumo network
        sol = solve(prob, Tsit5(), saveat=tsteps)
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

    train_x = uall[1:num_train_samples,:]
    train_y = y[1:num_train_samples]
    train_y = onehotbatch(train_y, classes)
    test_x = uall[num_train_samples+1:end,:]
    test_y = y[num_train_samples+1:end]
    test_y = onehotbatch(test_y, classes)
    return (train_x, train_y), (test_x, test_y)
end

function create_model()
    return Lux.Chain(
        Lux.Dense(640, 512, tanh),
        Lux.Dense(512, 7)
    )
end

function partition(data, batch_size)
    x, y = data
    return ((x[:, i:min(i+batch_size-1, end)], y[:, i:min(i+batch_size-1, end)]) for i in 1:batch_size:size(x, 2))
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

function accuracy(model, ps, st, x, y)
    total_correct, total = 0, 0
    st = Lux.testmode(st)
    for (x,y) in partition((x,y), 7)
        target_class = onecold(y)
        predicted_class = onecold(Array(first(model(x, ps, st))))
        total_correct += sum(target_class .== predicted_class)
        total += length(y)
    end
    return total_correct / total
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

dim_system = 10
model_flux = Flux.Chain(
    Flux.Dense(512, 512, swish),
    Flux.Dense(512, 256, swish),
    Flux.Dense(256, dim_system),
    Flux.softmax
) |> Flux.gpu
ps = Flux.params(model_flux)
ps = ps |> Flux.gpu
(train_x, train_y), (test_x, test_y) = load_data(x,targets;shuffling=false, train_ratio = 0.8) |> Flux.gpu

# BFGS optimizer for the model
#lossfun, gradfun, fg!, p0 = optfuns(()->loss(model_flux), ps)
#res = Optim.optimize(Optim.only_fg!(fg!), p0, BFGS(), Optim.Options(iterations = 1000, store_trace=true))
# Standard ADAM optimizer for the model
opt = Flux.ADAM(0.001)
epochs = 100
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
accuracy(model_flux, train_x', train_y)
accuracy(model_flux, test_x', test_y)
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