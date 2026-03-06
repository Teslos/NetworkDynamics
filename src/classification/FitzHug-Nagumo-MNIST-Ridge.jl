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
using CategoricalArrays

rng = Xoshiro(1234)
@sk_import datasets: load_digits
include("../utils/spikerate.jl")
include("../utils/drybean.jl")

# adjust the load path for your system
G = readdlm(joinpath(@__DIR__, "../../data/Norm_G_DTI.txt"),',', Float64, '\n')

using SimpleWeightedGraphs, Graphs
# process digit data from the MNIST dataset
function load_digits_data()
    # read the training data
    directory = joinpath(@__DIR__, "../../data/digits")
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
using CairoMakie
img_train, img_test = load_digits_data()
fig = CairoMakie.Figure()


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

image_digits = load_image_as_array(joinpath(@__DIR__, "../../data/digits/"))
#digits = img_train
img_digits = load_digits()
num_digits = 1797
# shuffle the data
shuffle_indices = shuffle(rng, 1:num_digits)
#pl_digits = permutedims(reshape(img_digits["data"][shuffle_indices,:],:,8,8),(1,3,2))
#pl_digits = permutedims(reshape(digits[shuffle_indices,1:64], :, 8, 8), (3,2,1))
pl_digits = img_digits["images"][shuffle_indices,:,:]
# target values
targets = img_digits["target"][shuffle_indices]
#targets = digits[shuffle_indices,65]
for i in 1:5
    ax = CairoMakie.Axis(fig[1, i], title="Digit $(img_digits["target"][i])", aspect = DataAspect())
    img = rotr90(img_digits["images"][i,:,:])
    CairoMakie.heatmap!(ax, img, colormap=:viridis)
end
fig
CairoMakie.save("digits.svg", fig)
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

g_directed, edge_weights = create_complete_graph(1797)
println("Number of nodes: ", nv(g_directed))
println("Number of edges: ", size(edge_weights))
println("Number of edges: ", ne(g_directed))

using NetworkDynamics

R0 = 0.5

x = pl_digits ./ 16.0

g0v = [interpolate(i <= 16 ? spike_train[:,i] : g0, BSpline(Quadratic(Line(OnCell())))) for i in 1:nv(g_directed)] # repeat the signal only for some nodes
e0v = [g0 .* R0 for _ in 1:ne(g_directed)] # repeat the signal for all edges

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

Base.@propagate_inbounds function fhn_electrical_vertex!(dv, v, edges, p, t)
    # add external input j0
    g = p

    #println("g size:",size(g))
    #println("g type:",typeof(g))
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
        dv[2] += e[2]
    end
    nothing
end
# set the B rotational matrix with an angle ϕ,
# the default value is ϕ = π/2 - 0.1, but the value causes the numerics to be unstable
ϕ = π/2 - 0.1
B = [cos(ϕ) sin(ϕ); -sin(ϕ) cos(ϕ)]
Base.Base.@propagate_inbounds function electrical_edge!(e, v_s, v_d, p, t)
    #println("p type:",typeof(p))
    #println("v_s size:",size(v_s))
    #println("v_d size:",size(v_d))
    e[1] = p*(B[1,1]*(v_s[1] - v_d[1]) + B[1,2]*(v_s[2] - v_d[2])) # *σ  - edge coupling for current
    e[2] = p*(B[2,1]*(v_s[1] - v_d[1]) + B[2,2]*(v_s[2] - v_d[2])) # *σ  - edge coupling for voltage
    nothing
end


odeelevertex = NetworkDynamics.ODEVertex(; f=fhn_electrical_vertex_simple!, dim=2, sym=[:u, :v])
odeeleedge = StaticEdge(; f=electrical_edge_simple!, dim=2, coupling=:directed)

fhn_network! = network_dynamics(odeelevertex, odeeleedge, g_directed)

# Parameter handling
N = nv(g_directed) # Number of nodes in the network
const ϵ = 0.05 # time scale separation parameter, default value is 0.05
const a = 0.5 # threshold parameter abs(a) < 1 is self-sustained limit cycle, abs(a) = 1 is a Hopf bifurcation
const σ = 0.72



# produce the training data
using OrdinaryDiffEq


# using the new plotting package CairoMakie
function plot_sol(u, t_steps, num_sol)
    fig = Figure()
    ax = CairoMakie.Axis(fig[1, 1], xlabel = "Time", ylabel = "u", title = "FitzHugh-Nagumo network")
    for i in 1:num_sol
        lines!(ax, t_steps, u[2*i-1,:], label="Oscillator $i")
    end
    fig
    CairoMakie.save("FitzHug-Nagumo_MNIST_Ridge.svg", fig)
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
function load_data(x, y; shuffling=true, train_ratio = 1.0)
    num_samples = num_digits
    classes = 0:9

    num_train_samples = Int(floor(num_samples * train_ratio))

    # convert the data to spike trains
    spike_train = spikerate.rate(x, 32)
    spike_train_test = spikerate.rate(x, 32)
    # convert the spike train to a Float32 array and (time, color, 1)
    spike_train = Float32.(permutedims(spike_train,(2,1,3,4)))
    spike_train_test = Float32.(permutedims(spike_train_test,(2,1,3,4)))
    spike_train = reshape(spike_train, :,32*8*8)
    print("spike_train size:",size(spike_train)) 
    tspike = collect(1:size(spike_train,2))
    nsamples = size(spike_train,1)
    gs = [Spline1D(tspike, spike_train[i,:], k=2) for i in 1:nsamples] # do spike train interpolation
    print("gs size:",size(gs))
    # different weights for edges, because the resitivity of the edges are always positive
    w_ij = [pdf(Normal(), x) for x in range(-1, 1, length=ne(g_directed))]
    nosc = nv(g_directed)
    uall = zeros(Float64, 1, N)

        p = (gs, σ * w_ij)
        #Initial conditions
        x0 = rand(Float64, 2*N)
        # set the problem
        tspan = (0.0, Float64(size(spike_train,2)))
        datasize = size(spike_train,2)
        tsteps = range(tspan[1], tspan[2], length=datasize)
        prob = ODEProblem(fhn_network!, x0, tspan, p)
        R0 = 0.5
        # solve the FitzHugh-Nagumo network
        sol = solve(prob, Tsit5(), saveat=tsteps)
        # if solution converges, then the solution is saved
        if Symbol(sol.retcode) == :Success
            diff_data = Array(sol)
            t = sol.t
            u = sol(sol.t)[1:2:2*N,:] # (N, T) nodes, time
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


using MLJ
using DataFrames

# Load RidgeClassifier
ridge_clf = @load LogisticClassifier pkg=MLJLinearModels

# Load and prepare data
(train_x, train_y), (test_x, test_y) = load_data(x, targets; shuffling=false, train_ratio=0.8)

# Convert to DataFrames if needed (MLJ works well with tables)
# Assuming train_x is a matrix where each row is a sample
train_df = DataFrame(train_x, :auto)  # Transpose if needed
test_df = DataFrame(test_x, :auto)

# Ensure target variables are categorical
train_y_cat = categorical(onecold(train_y,0:9))
test_y_cat = categorical(onecold(test_y,0:9))

# Create ridge classifier model
rr = ridge_clf(lambda=0.1)  # You can adjust the regularization parameter

# Create machine and fit
mach = machine(rr, train_df, train_y_cat)
MLJ.fit!(mach)

# Get fitted parameters
params = fitted_params(mach)
println("Ridge classifier coefficients:")
println(params.coefs)

# Make predictions
train_pred = MLJ.predict(mach, train_df)
test_pred = MLJ.predict(mach, test_df)

# Calculate classification accuracy
train_pred_mode = mode.(train_pred)  # Get most likely class
test_pred_mode = mode.(test_pred)

train_accuracy = sum(train_pred_mode .== train_y_cat) / length(train_y_cat)
test_accuracy = sum(test_pred_mode .== test_y_cat) / length(test_y_cat)

println("Training Accuracy: $train_accuracy")
println("Test Accuracy: $test_accuracy")

# If you want crossentropy loss like in the original

train_loss = sum(MLJ.log_loss(train_pred, train_y_cat)) / length(train_y_cat)
test_loss = sum(MLJ.log_loss(test_pred, test_y_cat)) / length(test_y_cat)
println("Training Cross-entropy Loss: $train_loss")
println("Test Cross-entropy Loss: $test_loss")
# Create a confusion matrix function for MLJ
function confusion_matrix_mlj(mach, x_df, y_true)
    # Make predictions using the MLJ machine
    y_pred_dist = MLJ.predict(mach, x_df)
    
    # Get the predicted classes (mode of the distribution)
    pred_class = mode.(y_pred_dist)
    
    # Convert to CategoricalArrays for MLJ's ConfusionMatrix
    predictions = CategoricalArray(pred_class)
    targets = CategoricalArray(y_true)
    
    # Create confusion matrix
    cm = MLJ.ConfusionMatrix()(targets, predictions)
    return cm
end

# Generate confusion matrix for test data
conf = confusion_matrix_mlj(mach, test_df, test_y_cat)

# Plot confusion matrix
using CairoMakie

fig = Figure(resolution = (800, 800))
ax = CairoMakie.Axis(fig[1, 1], 
    title = "Confusion Matrix - Ridge Classifier", 
    xlabel = "Predicted Class", 
    ylabel = "True Class", 
    aspect = DataAspect()
)

# Create heatmap
CairoMakie.heatmap!(ax, rotr90(conf.mat), colormap = :blues)

# Set axis ticks to show digit classes (0-9)
class_labels = string.(0:9)
ax.xticks = (1:10, class_labels)
ax.yticks = (1:10, class_labels)

# Annotate the heatmap with the confusion matrix values
for i in 1:size(conf.mat, 1)
    for j in 1:size(conf.mat, 2)
        value = rotr90(conf.mat)[i, j]
        text_color = value > maximum(conf.mat) / 2 ? :white : :black
        CairoMakie.text!(ax, j, i, 
            text = string(value), 
            align = (:center, :center), 
            color = text_color,
            fontsize = 12
        )
    end
end

# Display and save
display(fig)
CairoMakie.save("confusion_matrix_mnist_ridge.svg", fig)

# Print classification report
println("\nClassification Report:")
println("True classes: ", levels(test_y_cat))
println("Predicted classes: ", levels(mode.(MLJ.predict(mach, test_df))))
println("Confusion Matrix:")
println(conf.mat)

# Calculate per-class accuracy
class_accuracies = diag(conf.mat) ./ sum(conf.mat, dims=2)[:, 1]
for (i, acc) in enumerate(class_accuracies)
    println("Class $(i-1) accuracy: $(round(acc, digits=3))")
end

# Create percentage-based confusion matrix
function plot_confusion_matrix_percent(mach, x_df, y_true, title_str="Confusion Matrix (%)")
    conf = confusion_matrix_mlj(mach, x_df, y_true)
    rot_mat = rotr90(conf.mat)  # Rotate the confusion matrix for better orientation
    # Convert to percentages (row-wise normalization)
    conf_percent = rot_mat ./ sum(rot_mat, dims=2) .* 100
    
    fig = Figure(resolution = (1000, 800))
    ax = CairoMakie.Axis(fig[1, 1], 
        title = title_str, 
        xlabel = "Predicted Class", 
        ylabel = "True Class"
    )
    
    # Set custom ticks with class names
    class_labels = string.(0:9)
    ax.xticks = (1:10, reverse(class_labels))
    ax.yticks = (1:10, class_labels)
    
    # Create heatmap with percentages
    hm = CairoMakie.heatmap!(ax, conf_percent, colormap = :blues, colorrange = (0, 100))
    
    # Add colorbar
    Colorbar(fig[1, 2], hm, label = "Percentage (%)")
    
    # Annotate with percentage values
    for i in 1:size(conf_percent, 1)
        for j in 1:size(conf_percent, 2)
            percentage_val = conf_percent[i, j]
            text_color = percentage_val > 50 ? :white : :black
            CairoMakie.text!(ax, j, i, 
                text = string(round(percentage_val, digits=1), "%"), 
                align = (:center, :center), 
                color = text_color,
                fontsize = 10
            )
        end
    end
    
    return fig
end

# Create percentage confusion matrix
fig_percent = plot_confusion_matrix_percent(mach, test_df, test_y_cat, "MNIST Ridge Classifier - Confusion Matrix (%)")
display(fig_percent)
CairoMakie.save("confusion_matrix_mnist_ridge_percent.svg", fig_percent)

# plot the solution of the FitzHugh-Nagumo network
plot_sol(Array(train_x), collect(1:size(train_x,2)),5)