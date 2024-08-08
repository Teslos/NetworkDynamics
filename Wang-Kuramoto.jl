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
using IterTools
function plot_rect_map(N::Int, M::Int, data::Vector{Float64}, ax)
    #ax = GLMakie.Axis(f[1, 1])
    centers_x = 1:N
    centers_y = 1:M
    data = reshape(abs.(data), N, M)
    GLMakie.heatmap!(ax, centers_x, centers_y, data, colormap = :viridis)
end
function plot_phases(N::Int, M::Int, u::Array{Float64,2}, t::Array{Float64,1}, x0::Array{Float64,1} ,forcing_period::Float64, tspan)
    # Find the index of the value in t that is closest to t0 time
    for t0 in forcing_period:2:tspan[2]
        f = Figure()
        ax = GLMakie.Axis(f[1, 1], xlabel = "Node", ylabel = "u", title = "XOR gate network at time $t0")
        index_closest_to_t = findmin(abs.(t .- t0))[2]
        state_vector_at_t = [mod2pi((u[i,index_closest_to_t]-u[1,index_closest_to_t])) for i in 1:N*M]
        #state_vector_at_t = [mod2pi((u[i,index_closest_to_t]-x0[i]) ) for i in 1:N*M]
        #plot_rect_map(N, M, state_vector_at_t, ax)
        lines!(ax, 1:N, state_vector_at_t, linewidth=2, label="Oscillator")
        # record the frames
        GLMakie.save("./figs/xor_network_diff_phase_t$(t0).png",f, px_per_unit = 4)
    end
end

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
    w = p
    if t < forcing_period
        #e .= -w*sin.(v_s .- v_d) # no coupling in the forcing period 
        e .= 0.0
    else
        e .= -w*sin.(v_s .- v_d)
    end
    nothing
end

Base.Base.@propagate_inbounds function ki_force_vertex!(dv, v, edges, p, t)
    h, ψ = p
    #println("h: ", h, " ψ: ", ψ)
    if t < forcing_period
        dv .= -β * h*sin.(v .- ψ )
    else
        dv .= -h*sin.(v .- ψ)
        #dv .= 0.0
    end
    sum_coupling!(dv, edges)
    nothing
end

# generate random values from standard distribution for parameters of the edges
w_ij = randn(Float64, length(edges(g)))
# generate vertices values Mdata = 4 for all possible combinations of XOR gate
ξ_0 = [[0.5,-π/2], [0.5,-π/2], [0,+π/2], [0,0.5], [0.5,-π/2]]
ξ_1 = [[0.5,-π/2], [0.5,π/2], [0,+π/2], [0,0.5], [0.5,π/2]]
ξ_2 = [[0.5,π/2], [0.5,-π/2], [0,+π/2], [0,0.5], [0.5,π/2]]
ξ_3 = [[0.5,π/2], [0.5,π/2], [0,+π/2], [0,0.5], [0.5,-π/2]]
all_solutions = []
solutions = []
forcing_period = 20.0
β = 10.0
# Initial conditions
rng = MersenneTwister(1254)
ϕ0 = randn(rng, nv(g))
ϕ0[1] = +π/2
ϕ0[2] = +π/2

tspan = (0.0, 100.0)
tsteps = range(tspan[1], tspan[2], length=1000)
nd_vertex = ODEVertex(; f=ki_force_vertex!, dim=1, sym=[:v])
nd_edge = StaticEdge(; f=kiedge!, dim=1)
nd! = network_dynamics(nd_vertex, nd_edge, g)
# all the cases of the XOR gate
pars = [ξ_0, ξ_1, ξ_2, ξ_3]
u0s = zeros(Float64, N, 4)
ode_prob = ODEProblem(nd!, ϕ0, tspan)
# solve ensamble problem for all possible combinations of XOR gate
function prob_func(prob, i, repeat)
    remake(prob; u0 = u0s[:,i], p = (pars[i], w_ij))
end

ens_prob = EnsembleProblem(ode_prob; prob_func=prob_func)
ens_sol = solve(ens_prob, Tsit5(), EnsembleThreads(); trajectories=4, saveat=tsteps)
all_solutions = Array(ens_sol)

fig = Figure()
ax = GLMakie.Axis(fig[1, 1]; xlabel="Time", ylabel="u", title="XOR gate")
sol = ens_sol[2]
t = sol.t
u = real.(sol(sol.t)[1:N,:])
for i in 1:N
    lines!(ax, t, u[i,:], linewidth=2, label="Oscillator $i")
    text!(ax, t[end], u[i,end]+0.1, text=string("Oscillator ", i), align=(:right, :center))
end
axislegend(ax, position = :rt)
fig

plot_phases(N, 1, u, sol.t, ϕ0, forcing_period, tspan)  # plot the phases of the network

# in this case minibatch is all possible combinations of XOR gate
# u = [0, 0]->0, [0, 1]->1, [1, 0]->, [1, 1]->0
using Distributions
uniform = Uniform(-π, π)
# hidden and output layer have uniform distribution
rnd_angle = rand(uniform, 3)
true_data = [vcat([-π/2, -π/2],rnd_angle), vcat([-π/2, +π/2],rnd_angle),
             vcat([+π/2, -π/2],rnd_angle), vcat([+π/2, +π/2],rnd_angle)]
xor_data = [-π/2, +π/2, +π/2, -π/2]

# Learning the parameters of the network
rng = Random.default_rng()
ann_wk = Lux.Chain(Lux.Dense(2, 20, tanh),
    Lux.Dense(20, 1))
nn_pp, st = Lux.setup(rng, ann_wk)
pp = ComponentArray(nn_pp)


@inline function wk_edge!(e, v_s, v_d, p, t)
    nn_p = p
    in = [v_s[1], v_d[1]]
    e[1] = Lux.apply(ann_wk, in, nn_p, st)[1][1]
    nothing 
end

@inline function wk_vertex!(dv, v, edges, p, t)
    h, ψ = p
    #println("h: ", h, " ψ: ", ψ)
    dv .= -h*sin.(v .- ψ)
    sum_coupling!(dv, edges)
    nothing
end

# @note: testing on a single solution (2nd solution)
single_solution = [all_solutions[2]]
single_true = [true_data[2]]
batch_size = 4
train_loader_neural = Flux.Data.DataLoader((all_solutions, true_data), batchsize=batch_size, shuffle=false)
ann_vertex = ODEVertex(; f=wk_vertex!, dim=1, sym=[:v])
ann_wk_edge = StaticEdge(; f=wk_edge!, dim=1, coupling=:directed)
wk_network! = network_dynamics(ann_vertex, ann_wk_edge, g)
probwk = ODEProblem(wk_network!, ϕ0, tspan, pp)

function predict_neuralode(fullp, x0)
    #println("x0: ", x0)
    #println("fullp: ", fullp)
    Array(solve(probwk, Tsit5(), p = fullp, u0=x0, saveat=tsteps, sensealg=ForwardDiffSensitivity()))
end

# batch should contain all possible combinations of XOR gate
function loss_neuralode(p, batch, batch_t)
    # do it for all data points in the batch
    #println("p: ", p, " size of batch_t: ", size(batch_t))
    pred = predict_neuralode(p, batch_t[1])
    loss = sum(abs2, batch[1] .- pred[5,:])
    #pred = predict_neuralode(p)
    #loss = sum(abs2, all_solutions[1][5,:] .- pred[5,:])
    println("Current loss is: ", loss)
    return loss, pred
end
loss_function(data, pred) = sum(abs2, data - pred)
function prob_func_neuralode(p,i,repeat)
    ps = (pars[i], p)
    remake(probwk, p=ps, u0=u0s[:,i], saveat=tsteps)
end

prob_ens = EnsembleProblem(probwk, prob_func = prob_func_neuralode)

# trying the Ensamble problem to solve XOR gate
function loss_multiple_shooting(p; group_size=8)
    # do it for all data points in the batch
    #println("size of batch:", size(batch))
    loss = multiple_shoot(p, all_solutions, tsteps, prob_ens, EnsembleThreads(), loss_function, Tsit5(), group_size;
    continuity_term=300, trajectories = batch_size)
    #println("Current loss is: ", loss)
    return loss
end

callback = function (p, l, pred; doplot = false)
    # plot current prediction against data
    if doplot
        plt = Plots.scatter(tsteps, single_solution; label = "data")
        Plots.scatter!(plt, tsteps, pred[5,:]; label = "prediction")
        Plots.display(Plots.plot(plt))
    end
    return false
end
callback(pp, loss_neuralode(pp, train_loader_neural.data[1], train_loader_neural.data[2])...; doplot=true)

# use Optimization.jl to solve the problem
adtype = Optimization.AutoZygote()

optf = Optimization.OptimizationFunction((x, p) -> loss_multiple_shooting(x), adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentArray(nn_pp))
#result_neuralode = Optimization.solve(optprob, OptimizationOptimisers.AdamW(0.05), ncycle(train_loader_neural, 100); 
#    callback = callback, maxiters=100)

result_neuralode = Optimization.solve(optprob, OptimizationOptimisers.AdamW(0.05); 
    callback = callback, maxiters=100)
optprob2 = remake(optprob; u0 = result_neuralode.u)

result_neuralode2 = Optimization.solve(optprob2, Optim.BFGS(; initial_stepnorm = 0.01);
        callback, allow_f_increases = false)

callback(result_neuralode2.u, loss_neuralode(result_neuralode2.u, train_loader_neural.data[1], train_loader_neural.data[2])...; doplot=true)
using Plots
using PlotlyJS 
plotlyjs()
Plots.savefig("Wang-Kuramoto-opt.png")
using JLD2

# save the results
@save "Wang-Kuramoto.jld2" result_neuralode2
JLD2.@load "Wang-Kuramoto.jld2" result_neuralode_load

# try to predict the XOR gate using the trained network
x0 = randn(Float64, N)
x0[1] = π/2
x0[2] = -π/2

probwk = ODEProblem(wk_network!, x0, tspan, result_neuralode.u)
solwk = solve(probwk, Tsit5(), saveat=tsteps)
xor_pred = Array(solwk)
fig = Figure()
ax = GLMakie.Axis(fig[1, 1]; xlabel="Time", ylabel="u", title="XOR gate")
t = solwk.t
u = real.(solwk(solwk.t)[1:N,:])
for i in [1,2,5]
    lines!(ax, t, u[i,:], linewidth=2, label="Oscillator $i")
    text!(ax, t[end], u[i,end]+0.1, text=string("Oscillator ", i), align=(:right, :center))
end
axislegend(ax, position = :rt)
GLMakie.save("Wang-Kuramoto-opt_true_false.png",fig, px_per_unit = 4)
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
