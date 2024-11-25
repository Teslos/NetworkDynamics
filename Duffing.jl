using DelimitedFiles
using Random
using ScikitLearn
using Dierckx
using Interpolations

rng = Xoshiro(1234)
@sk_import datasets: load_digits
include("spikerate.jl")
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

#digits = img_train
img_digits = load_digits()
num_digits = 100
# shuffle the data
shuffle_indices = shuffle(rng, 1:num_digits)
#pl_digits = permutedims(reshape(img_digits["data"][shuffle_indices,:],:,8,8),(1,3,2))
#pl_digits = permutedims(reshape(digits[shuffle_indices,1:64], :, 8, 8), (3,2,1))
pl_digits = img_digits["images"][shuffle_indices,:,:]
# target values
targets = img_digits["target"][shuffle_indices]
#targets = digits[shuffle_indices,65]
using GLMakie
fig = GLMakie.Figure()
for i in 1:5
    ax = GLMakie.Axis(fig[1, i], title="Digit $(img_digits["target"][i])", aspect = DataAspect())
    img = rotr90(img_digits["images"][i,:,:])
    GLMakie.heatmap!(ax, img, colormap=:viridis)
end
fig
GLMakie.save("digits.png", fig)

#g_directed, edge_weights = create_graph()
g_directed = create_barabasi_albert_graph()
edge_weights = ones(length(edges(g_directed)))
x = pl_digits ./ 16.0
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

using NetworkDynamics

Base.@propagate_inbounds function duffing_vertex!(dv, v, edges, p, t)
    g = p
    dv[1] = v[2]
    e_s, e_d  = edges
    # Duffing oscillator
    omega = ω - rand()*0.05
    # we are setting the frequency to be constant 
    # omega = ω
    dv[2] = -v[1] - β * v[1]^3 - d*v[2] + f*cos(omega*t)
    for e in e_s
        dv[1] -= e[1]
    end
    for e in e_d
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
N = nv(g_directed) # Number of nodes
const ϵ = 0.05 # global variables that are accessed several times should be declared as constants
const a = 0.5
const σ = 1.0
const f = 0.1
const β = 20.0
const ω = 1.0
const d = 0.1


# Tuple of parameters for nodes and edges
p = (gs, σ * edge_weights)
#Initial conditions
x0 = randn(2N)
x0[1] = 1.0

# Solving the ODE
using OrdinaryDiffEq

tspan = (0.0, 8.0)
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
save("duffing_brain_graph.png",fig1, px_per_unit = 4)

fig, ax, p = graphplot(g_directed)
hidedecorations!(ax); hidespines!(ax);
save("brain_graph.png",fig)