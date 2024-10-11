using Pkg
Pkg.activate(".")
using DelimitedFiles
using Graphs
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


@sk_import datasets: load_digits
include("spikerate.jl")
include("drybean.jl")

# adjust the load path for your system
G = readdlm(joinpath(@__DIR__, "./Norm_G_DTI.txt"),',', Float64, '\n')

using SimpleWeightedGraphs, Graphs

# read the dry bean dataset
db = drybean.read_drybean()
x = Matrix(permutedims(db[shuffle(1:end), :]))
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

spike_train = spikerate.rate(x[:,1:64], 16)
spike_train_test = spikerate.rate(x[:,65:128], 16)
# convert the spike train to a Float32 array and (time, color, 1)
#spike_train = reshape(Float32.(spike_train), 350, :, 1)
spike_train = Float32.(reshape(spike_train, :,64)) 
spike_train_test = Float32.(reshape(spike_train_test, :,64))
#spike_train = Float32.(spike_train)
tspike = collect(1:size(spike_train,1))

#j0 = zeros(Float64, 350)
g0 =  zeros(Float64, size(spike_train,1))

R0 = 0.5
gs = [Spline1D(tspike, spike_train[:,i], k=2) for i in 1:nv(g_directed)] # repeat the signal for all nodes
gst = [Spline1D(tspike, spike_train_test[:,i], k=2) for i in 1:nv(g_directed)] # repeat the signal for all nodes
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


odeelevertex = ODEVertex(; f=fhn_electrical_vertex_simple!, dim=2, sym=[:u, :v])
odeeleedge = StaticEdge(; f=electrical_edge_simple!, dim=2, coupling=:directed)

fhn_network! = network_dynamics(odeelevertex, odeeleedge, g_directed)

# Parameter handling
N = nv(g_directed) # Number of nodes in the network
const ϵ = 0.05 # time scale separation parameter, default value is 0.05
const a = 0.5 # threshold parameter abs(a) < 1 is self-sustained limit cycle, abs(a) = 1 is a Hopf bifurcation
const σ = 0.006

# different weights for edges, because the resitivity of the edges are always positive
w_ij = [pdf(Normal(), x) for x in range(-1, 1, length=ne(g_directed))]

# Tuple of parameters for nodes and edges
p = (gs,σ * w_ij)
#Initial conditions
rng = Random.default_rng()
x0 = rand(Float64, 2*N)

# produce the training data
using OrdinaryDiffEq

tspan = (0.0, Float64(size(spike_train,1)))
datasize = 256
tsteps = range(tspan[1], tspan[2], length=datasize)
prob = ODEProblem(fhn_network!, x0, tspan, p)
sol = solve(prob, Tsit5(), saveat=tsteps)
diff_data = Array(sol)

# produce the test data
# test parameters
p = (gst,σ * w_ij)
probt = remake(prob, p=p)
solt = solve(probt, Tsit5(), saveat=tsteps)
diff_datat = solt(solt.t)[1:2:128,:]
# using the new plotting package CairoMakie
using GLMakie
fig = Figure()
ax = GLMakie.Axis(fig[1, 1], xlabel = "Time", ylabel = "u", title = "FitzHugh-Nagumo network")
t= sol.t
u = sol(sol.t)[1:2:128,:] # (N, T) nodes, time
# use only the u values
diff_data = u
for i in 1:64
    lines!(ax, t, u[i,:], label="Oscillator $i")
    #text!(ax, t[end], u[i,end]+0.1, text=string("Oscillator ", i), align=(:right, :center))
end
#axislegend(ax, position = :rt)
fig
GLMakie.save("FitzHug-Nagumo_CALI_10.png", fig)
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
    num_samples = size(x, 1)
    classes = unique(y)

    num_train_samples = Int(floor(num_samples * train_ratio))
    if shuffling
        shuffled_indices = shuffle(rng, 1:num_samples)
    else
        shuffled_indices = 1:num_samples
    end

    # shuffle the data
    train_indices = shuffled_indices[1:num_train_samples]
    test_indices = shuffled_indices[num_train_samples+1:end]
    println("train_indices:",train_indices)

    # split the data
    train_x = x[train_indices,:]
    train_y = y[train_indices]
    train_y = onehotbatch(train_y, classes)
    test_x = x[test_indices,:]
    test_y = y[test_indices]
    test_y = onehotbatch(test_y, classes)
    return (train_x, train_y), (test_x, test_y)
end

function create_model()
    return Lux.Chain(
        Lux.Dense(64, 512, tanh),
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
spike_train = spike_train |> Flux.gpu
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
    Flux.Dense(256, 512, swish),
    Flux.Dense(512, 256, swish),
    Flux.Dense(256, dim_system),
    Flux.softmax
) |> Flux.gpu
ps = Flux.params(model_flux)
ps = ps |> Flux.gpu
(train_x, train_y), (test_x, test_y) = load_data(vcat(diff_data,diff_datat),targets;shuffling=false, train_ratio = 0.5) |> Flux.gpu
#test_x, test_y), (_, _) = load_data(diff_datat,targets[65:128];shuffling=false) |> Flux.gpu

# BFGS optimizer for the model
#lossfun, gradfun, fg!, p0 = optfuns(()->loss(model_flux), ps)
#res = Optim.optimize(Optim.only_fg!(fg!), p0, BFGS(), Optim.Options(iterations = 1000, store_trace=true))
# Standard ADAM optimizer for the model
opt = Flux.ADAM(0.001)
epochs = 5000
for epoch in 1:epochs
    Flux.train!(loss_ce, ps, [(train_x', train_y)], opt)
    println("Epoch $epoch, Loss: $(loss(train_x', train_y))")
end

# test accuracy
function accuracy(model, x, y)
    y_pred = model(x)
    pred_class = sum(onecold(y) .== onecold(y_pred))
    acc = pred_class / size(y, 2)
    return acc
end
spike_train_test = spike_train_test |> Flux.gpu
accuracy(model_flux, test_x', test_y)
# Load data
# Create model 
model = create_model()
nn_rc, st_rc = Lux.setup(rng, model)

# train the model
train_model(model,(train_x, train_y), (test_x, test_y); epochs=1000)

