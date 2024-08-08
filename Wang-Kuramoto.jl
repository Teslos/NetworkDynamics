# solves classification problem for XOR gate using neural network with oscillators.
using ComponentArrays, DiffEqFlux, NetworkDynamics, Lux, OrdinaryDiffEq, LinearAlgebra
using Graphs
using Optimization # for optimization problem
using OptimizationOptimisers 
using OptimizationOptimJL   
using GLMakie
using Random
using Flux
using Plots

function create_graph(N::Int=5)
    # create all to all graph
    g = Graphs.complete_graph(N)
    edge_weights = ones(length(edges(g)))
    g_weighted = SimpleDiGraph(g)
    g_directed = SimpleDiGraph(g_weighted)
    return g_directed, edge_weights
end

function xor_gate(u::Vector{Int64})
    return u[1] ⊻ u[2]
end

N = 5
g, edge_weights = create_graph(N)

# Functions for edges and vertices
Base.Base.@propagate_inbounds function kiedge!(e, v_s, v_d, p, t)
    pr, pi = p
    if t < forcing_period
        e .= -pr*sin.(v_s .- v_d) # no coupling in the forcing period 
        #e .= 0.0
    else
        e .= -pr*sin.(v_s .- v_d)
    end
    nothing
end

Base.Base.@propagate_inbounds function ki_force_vertex!(dv, v, edges, p, t)
    h, ψ = p
    if t < forcing_period
        dv .= -h*sin.(v .- ψ )
    else
        dv .= -h*sin.(v .- ψ)
    end
    sum_coupling!(dv, edges)
    nothing
end

# generate random values from standard distribution for parameters of the edges
w_ij = randn(ComplexF64, length(edges(g)))
# generate uniform values for the parameters of the vertices
#ξ_0 = [ rand(-0.5:0.001:0.5) + im * rand(-π:0.001:π) for _ in 1:nv(g) ]
ξ_0  = [0.5-π/2*im, 0.5+π/2*im, +π/2*im, 0.5, 0.5+π/2*im]
e_params = [(real(e), imag(e)) for e in w_ij]
v_params = [(real(v), imag(v)) for v in ξ_0]   # parameters for the vertices

# Constructing the network
parameters = (v_params, e_params)
forcing_period = 400.0
forcing_amplitude = 0.1
forcing_phase = 0.0
 
nd_vertex = ODEVertex(; f=ki_force_vertex!, dim=1, sym=[:v])
nd_edge = StaticEdge(; f=kiedge!, dim=1)
nd! = network_dynamics(nd_vertex, nd_edge, g)

# Initial conditions
rng = MersenneTwister(1234)
ϕ0 = randn(rng, nv(g))
tspan = (0.0, 1000.0)
tsteps = range(tspan[1], tspan[2], length=1000)
ode_prob = ODEProblem(nd!, ϕ0, tspan, parameters)
sol = solve(ode_prob, Tsit5(), saveat=tsteps)
xor_odedata = Array(sol)
fig = Figure()
ax = GLMakie.Axis(fig[1, 1]; xlabel="Time", ylabel="u", title="XOR gate")
t = sol.t
u = sol(sol.t)[1:N,:]
for i in 1:N
    lines!(ax, t, u[i,:], linewidth=2)
    text!(ax, t[end], u[i,end]+0.1, text=string("Oscillator ", i), align=(:right, :center))
end
fig

# XOR gate prediction using Optimization 
probflux = ODEProblem(nd!, ϕ0, tspan, parameters; saveat=tspan[1]:0.01:tspan[2])
# Now we are trying to predict the XOR gate using neural network
function predict(p, u0)
    resize!(u0, 4)
    remake(probflux, u0=u0, p=p)
    Array(solve(probflux, Tsit5(), saveat=tspan[1]:0.01:tspan[2], sensealg=ForwardDiffSensitivity()))
end

# in this case minibatch is all possible combinations of XOR gate
# u = [0, 0]->0, [0, 1]->1, [1, 0]->, [1, 1]->0
true_data = [[-π/2*im, -π/2*im], [-π/2*im, +π/2*im], [+π/2*im, -π/2*im], [+π/2*im, +π/2*im]]
xor_data = [-π/2*im, +π/2*im, +π/2*im, -π/2*im]
train_loader = Flux.Data.DataLoader((true_data, xor_data), batchsize=4, shuffle=true)

# define the loss function
function loss(p, batch)
    # do it for all data points in the batch
    output = zeros(ComplexF64, 4)
    for i in batch
        pred = predict(p, i[1])
        # take last value of the prediction
        output = pred[N, end]
    end 
 
    for t in batch
        loss = sum(abs2, 1.0 .- cos.(real(t[2]) .- real(output)))
    end

    #loss = sum(abs2, 1 - cos(batch.data[2] .- output))
    #loss = sqrt(sum(abs2, pred))
    println("Current loss is: ", loss)
    loss, output
end

# callback function for optimization and visualization
cb = function (p, l, pred; doplot=false)
    println("Current loss is: ", l)
    if doplot
        plot(sol, lw=2, label="")
        plot!(pred, lw=2, label="")
    end
    return false
end

# Learning the parameters of the network
rng = Random.default_rng()
ann_wk = Lux.Chain(Lux.Dense(2, 20, tanh),
    Lux.Dense(20, 1))
p, st = Lux.setup(rng, ann_wk)
@inline function wk_edge!(e, v_s, v_d, p, t)
    in = [v_s[1], v_d[1]]
    e[1] = Lux.apply(ann_wk, in, p, st)[1][1]
    nothing 
end

ann_wk_edge = StaticEdge(; f=wk_edge!, dim=1, coupling=:directed)
wk_network! = network_dynamics(nd_vertex, ann_wk_edge, g)

probwk = ODEProblem(wk_network!, ϕ0, tspan, p)

function predict_neuralode(p)
    prob = remake(probwk, p=p)
    Array(solve(prob, Tsit5(), saveat=tsteps, sensealg=ForwardDiffSensitivity()))
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, xor_odedata[5,:] .- pred[5,:])
    println("Current loss is: ", loss)
    return loss, pred
end

callback = function (p, l, pred; doplot = false)
    # plot current prediction against data
    if doplot
        plt = Plots.scatter(tsteps, xor_odedata[5,:]; label = "data")
        Plots.scatter!(plt, tsteps, pred[5,:]; label = "prediction")
        Plots.display(Plots.plot(plt))
    end
    return false
end
pinit = ComponentArray(p)
callback(pinit, loss_neuralode(pinit)...; doplot=true)

# use Optimization.jl to solve the problem
adtype = Optimization.AutoZygote()

optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pinit)
loss_neuralode(pinit)
callback(pinit, loss_neuralode(pinit)...; doplot=true)
result_neuralode = Optimization.solve(optprob, OptimizationOptimisers.AdamW(0.05); callback = callback,
    maxiters=100)

optprob2 = remake(optprob; u0 = result_neuralode.u)

result_neuralode2 = Optimization.solve(optprob2, Optim.BFGS(; initial_stepnorm = 0.01);
        callback, allow_f_increases = false)

callback(result_neuralode2.u, loss_neuralode(result_neuralode2.u)...; doplot=true)

savefig("Wang-Kuramoto-opt.png")
using JLD2

# save the results
@save "Wang-Kuramoto.jld2" result_neuralode2
JLD2.@load "Wang-Kuramoto.jld2" result_neuralode2

# try to predict the XOR gate using the trained network
x0 = randn(ComplexF64, N)
x0[1] = +π/2*im
x0[2] = -π/2*im

probwk = ODEProblem(wk_network!, x0, tspan, result_neuralode2.u)
solwk = solve(probwk, Tsit5(), saveat=tsteps)
xor_pred = Array(solwk)
fig = Figure()
ax = GLMakie.Axis(fig[1, 1]; xlabel="Time", ylabel="u", title="XOR gate")
t = solwk.t
u = real.(solwk(solwk.t)[1:N,:])
for i in 1:N
    lines!(ax, t, u[i,:], linewidth=2)
    text!(ax, t[end], u[i,end]+0.1, text=string("Oscillator ", i), align=(:right, :center))
end
#=
# Initial parameters of the network, that are optimized
pinit = ComponentArray(parameters)
cb(pinit, loss(pinit, train_loader)...; doplot=false)

# We optimize the parameters of the network
adtype = Optimization.AutoEnzyme()
ftest(x,p) = loss(p,x)
optf = Optimization.OptimizationFunction(ftest, adtype)
optprob = Optimization.OptimizationProblem(optf, pinit)
predict(pinit, train_loader.data[1])
loss(pinit, train_loader)
# do optimization to solve the system
res = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.05), maxiters=20)
=#