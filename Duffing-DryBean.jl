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
# normalize the data
function standard_scaler(x)
    # Standardise each column to have mean 0 and stddev 1.
    μ = mean(x, dims=2)
    σ = std(x, dims=2)
    return (x .- μ) ./ σ
end
# target vector
targets = x[17,:]
# normalize only the features
# x = normalize_rows(x[1:16,:])
x = standard_scaler(x[1:16,:])


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
    omega = ω - rand()*0.57 # randomize the frequency a bit
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
const σs = 0.1
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
    num_samples = size(x,2)
    classes = unique(yt)

    num_train_samples = Int(floor(num_samples * train_ratio))

    if shuffling 
        shuffled_indices = shuffle(rng, 1:num_samples)
    else
        shuffled_indices = 1:num_samples
    end


    # convert the data to spike trains
    spike_train = spikerate.rate(x, 32)
    spike_train_test = spikerate.rate(x,32)
    # convert the spike train to a Float32 array and (time, color, 1)
    spike_train = Float32.(reshape(spike_train,32*16,:))
    spike_train_test = Float32.(reshape(spike_train_test,32*16,:))
    # reshape the spike train to (num_samples, time)
    spike_train = permutedims(spike_train, (2, 1))
    spike_train_test = permutedims(spike_train_test, (2, 1))
    print("spike_train size:",size(spike_train)) 
    tspike = collect(1:size(spike_train,2))
    # tspike = LinRange(0.0, 32.0, size(spike_train,2))
    
    nsamples = size(spike_train,1)
    println("Number of samples: $(nsamples)")
    println("tspike: $(size(tspike)), number vertex: $(nv(g_directed))")
    gs = [Spline1D(tspike, spike_train[i,:], k=2) for i in 1:nsamples] # do spike train interpolation
    print("gs size:",size(gs))
    N = nv(g_directed) # number of nodes in the network
    num_batches = num_samples ÷ N
    uall = Matrix{Float64}(undef, 0, size(spike_train,2)) # initialize the matrix to store the data
    for i in 1:num_batches
        # different weights for edges, because the resitivity of the edges are always positive
        w_ij = [pdf(Normal(), x) for x in range(-1, 1, length=ne(g_directed))]
        #w_ij = ones(ne(g_directed))
        nosc = nv(g_directed)

        p = (gs[(i-1)*N+1:i*N], σs *w_ij)
        #Initial conditions are choosen randomly
        x0 = rand(Float64,2*N)
        # set the problem
        tspan = (0.0, Float64(size(spike_train,2)))
        datasize = size(spike_train,2)
        tsteps = range(tspan[1], tspan[2], length=datasize)
        prob = ODEProblem(fhn_network!, x0, tspan, p)
        # solve the Duffing network
        sol = solve(prob, Tsit5(), saveat=tsteps)
        # if solution converges, then the solution is saved
        if Symbol(sol.retcode) == :Success
            diff_data = Array(sol)
            t = sol.t
            u = sol(sol.t)[1:2:2*N,:] # (N, T) nodes, time
            print("u size:",size(u))
            # add the data together
            uall = vcat(uall, u)

            print("uall size:",size(uall))
        else
            println("Solution did not converge: ", sol.retcode)
            train_x = []
            train_y = []
            test_x = []
            test_y = []
        end
    end 
    num_train_samples = Int(floor(num_batches * N * train_ratio))
    batch_end = num_batches * N
    train_x = uall[1:num_train_samples ,:]
    train_y = yt[1:num_train_samples]
    train_y = onehotbatch(train_y, classes)
    test_x = uall[num_train_samples+1:batch_end,:]
    test_y = yt[num_train_samples+1:batch_end]
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
opt = Flux.Adam(0.001)
opt_state = Flux.setup(opt, model_flux)
epochs = 300
data_loader = Flux.DataLoader((train_x', train_y), batchsize=64, shuffle=false)
for epoch in 1:epochs
    lossv = 0.0
    for (x,y) in data_loader
        # compute the loss and gradients
        ls, grads = Flux.withgradient(model_flux) do model_flux
            y_pred = model_flux(x)
            Flux.Losses.crossentropy(y_pred, y)
        end
        # update the parameters
        Flux.update!(opt_state, model_flux, grads[1])
        lossv += ls / length(data_loader)
    end
    # Print the loss every 10 epochs
    if epoch % 10 == 0
        println("Epoch $epoch, Loss: $lossv")
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
using CairoMakie
function confusion_matrix_with_labels(model, x, y, class_names)
    y_pred = model(x)
    pred_class = onecold(y_pred)
    true_class = onecold(y)
    
    # Move to CPU memory
    pred_class = pred_class |> Flux.cpu  
    predictions = pred_class |> CategoricalArray
    true_class = true_class |> Flux.cpu
    targets = true_class |> CategoricalArray
    
    # Compute confusion matrix
    cm = MLJ.ConfusionMatrix()(targets, predictions)
    
    # Convert to percentages
    cm_percent = cm.mat ./ sum(cm.mat, dims=2) .* 100
    
    return cm, cm_percent
end

# Define your class names (adjust according to your dry bean classes)
class_names = ["SEKER", "BARBUNYA", "BOMBAY", "CALI", "HOROZ", "SIRA", "DERMASON"]

# Get confusion matrix and percentages
conf, conf_percent = confusion_matrix_with_labels(model_flux, test_x', test_y, class_names)

# Plot confusion matrix with class names and percentages
fig = Figure(resolution = (1000, 800))
ax = CairoMakie.Axis(fig[1, 1], 
    title = "Confusion Matrix (%)", 
    xlabel = "Predicted Class", 
    ylabel = "True Class"
)

# Set custom ticks with class names
ax.xticks = (1:length(class_names), class_names)
ax.yticks = (1:length(class_names), reverse(class_names))
ax.xticklabelrotation = π/4  # Rotate x-axis labels for better readability

# Create heatmap with percentages
hm = CairoMakie.heatmap!(ax, rotr90(conf_percent), colormap = :blues, colorrange = (0, 100))

# Add colorbar
Colorbar(fig[1, 2], hm, label = "Percentage (%)")

# Annotate with percentage values
for i in 1:size(conf_percent, 1)
    for j in 1:size(conf_percent, 2)
        percentage_val = rotr90(conf_percent)[i, j]
        text_color = percentage_val > 50 ? :white : :black  # Use white text for dark cells
        CairoMakie.text!(ax, j, i, 
            text = string(round(percentage_val, digits=1), "%"), 
            align = (:center, :center), 
            color = text_color,
            fontsize = 12
        )
    end
end

# Adjust layout
resize_to_layout!(fig)
# Display and save
display(fig)
save("confusion_matrix_dry_bean_percent_duffing.png", fig)


plot_sol(Array(train_x), collect(1:size(train_x,2)), 16)