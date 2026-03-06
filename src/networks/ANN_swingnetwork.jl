## Fit coupling term of swing equation with an ANN
using ComponentArrays
using DiffEqFlux
using NetworkDynamics
using Graphs
using OrdinaryDiffEq
#using GalacticOptim
using Random
using Lux
using Optimization
using OptimizationOptimJL
using OptimizationOptimisers
using Plots

## Defining the graph

N = 20
k = 4
g = barabasi_albert(N, k)

### Defining the network dynamics

@inline function diffusion_vertex!(dv, v, edges, p, t)
    dv[1] = 0.0f0
    for e in edges
        dv[1] += e[1]
    end
    nothing
end

@inline function diffusion_edge!(e, v_s, v_d, p, t)
    e[1] = 1 / 3 * (v_s[1] - v_d[1])
    nothing
end

odevertex = ODEVertex(; f=diffusion_vertex!, dim=1)
staticedge = StaticEdge(; f=diffusion_edge!, dim=1, coupling=:antisymmetric)
diffusion_network! = network_dynamics(odevertex, staticedge, g)

## Simulation 

# generating random values for the parameter value ω_0 of the vertices
v_pars = randn(nv(g))
# coupling stength of edges are set to 1/3
e_pars = 1 / 3 * ones(ne(g))
p = (v_pars, e_pars)

# random initial conditions
x0 = randn(Float32, nv(g))
dx = similar(x0)
datasize = 30 # Number of data points
tspan = (0.0f0, 5.0f0) # Time range
tsteps = range(tspan[1], tspan[2], length=datasize)

diff_prob = ODEProblem(diffusion_network!, x0, tspan, nothing)
diff_sol = solve(diff_prob, Tsit5(); reltol=1e-6, saveat=tsteps)
diff_data = Array(diff_sol)

## Learning the coupling function
rng = Random.default_rng()
ann_diff = Chain(Dense(2, 20, tanh),
    Dense(20, 1))
p, st = Lux.setup(rng, ann_diff)
input_vector = randn(rng, 2)
input_vector = Float32.(input_vector)
out = Lux.apply(ann_diff, input_vector, p, st)
out
@inline function ann_edge!(e, v_s, v_d, p, t)
    in = [v_s[1], v_d[1]]
    e[1] = Lux.apply(ann_diff, in, p, st)[1][1]
    nothing
end

annedge = StaticEdge(; f=ann_edge!, dim=1, coupling=:antisymmetric)
ann_network = network_dynamics(odevertex, annedge, g)

prob_neuralode = ODEProblem(ann_network, x0, tspan, p)

# ## Using MTK to help Enzyme
# using ModelingToolkit
# sys = modelingtoolkitize(prob_neuralode)
# prob_neuralode = ODEProblem(sys, [], tspan)

function predict_neuralode(p)
    tmp_prob = remake(prob_neuralode, p=p)
    Array(solve(tmp_prob, Tsit5(), saveat=tsteps))
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, diff_data .- pred)
    return loss, pred
end

callback = function (p, l, pred; doplot = false)
    println(l)
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
    maxiters = 300)

optprob2 = remake(optprob; u0 = result_neuralode.u)

result_neuralode2 = Optimization.solve(optprob2, Optim.BFGS(; initial_stepnorm = 0.01);
    callback, allow_f_increases = false)

callback(result_neuralode2.u, loss_neuralode(result_neuralode2.u)...;doplot=true)
