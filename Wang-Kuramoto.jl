# solves classification problem for XOR gate using neural network with oscillators.
using ComponentArrays, DiffEqFlux, NetworkDynamics, Lux, OrdinaryDiffEq, LinearAlgebra
using Graphs
using Optimization # for optimization problem
using OptimizationOptimisers 
using OptimizationOptimJL   
using GLMakie
using Random
using StableRNGs
using Flux
using Plots
using IterTools
using LaTeXStrings
using Distributions
# default rng
rng = StableRNG(1)

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
        ax = GLMakie.Axis(f[1, 1], xlabel = "Node", ylabel = L"\phi", title = "XOR gate network at time $t0")
        # set y-axis to be in the range of -π to π
        ax.yticks = (0:π/2:2π, ["0", "π/2", "π", "3π/2", "2π"])
        index_closest_to_t = findmin(abs.(t .- t0))[2]
        state_vector_at_t = [mod2pi((u[i,index_closest_to_t]-u[1,index_closest_to_t])) for i in 1:N*M]
        #state_vector_at_t = [mod2pi((u[i,index_closest_to_t]-x0[i]) ) for i in 1:N*M]
        #plot_rect_map(N, M, state_vector_at_t, ax)
        lines!(ax, 1:N, state_vector_at_t, linewidth=2, label="Oscillator")
        # record the frames
        GLMakie.save("./figs/xor_network_sol_phase_t$(t0).png",f, px_per_unit = 4)
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
Base.Base.@propagate_inbounds function kiedge!(e, v_s, v_d, (w,σ), t)
    if t < forcing_period
        #e .= -w*sin.(v_s .- v_d) # no coupling in the forcing period 
        e .= 0.0
    else
        e .= -w*sin.(v_s .- v_d) * σ
    end
    nothing
end

Base.Base.@propagate_inbounds function ki_force_vertex!(dv, v, esum, (h, ψ, β, τ), t)
    #println("h: ", h, " ψ: ", ψ)
    beta = 0.4
    if t < forcing_period
        dv .= -β * h*sin.(v .- ψ ) 
    else
        #dv .= -h*sin.(v .- ψ) - beta * sin.(v .- τ)/(1 .+ cos.(v .- τ)) # forcing term from cost function
        #dv .= 0.0
        dv .= -h*sin.(v .- ψ)
    end
    dv .+= esum[1]
    nothing
end

# generate random values from normal distribution for parameters of the edges
w_ij = randn(rng, length(edges(g)))
# generate random values from uniform distribution for parameters of the vertices :h and :ψ
uniform_ψ = Uniform(-π, π)
uniform_h = Uniform(-0.5, 0.5)
# generate vertices values Mdata = 4 for all possible combinations of XOR gate
# both values are equivalent to the bias the first value is h and the second is ψ
ξ_0 = [[0.5,-π/2], [0.5,-π/2], [0,+π/2], [0,0.5], [0.5,-π/2]]
ξ_1 = [[0.5,-π/2], [0.5,π/2], [0,+π/2], [0,0.5], [0.5,π/2]]
ξ_2 = [[0.5,π/2], [0.5,-π/2], [0,+π/2], [0,0.5], [0.5,π/2]]
ξ_3 = [[0.5,π/2], [0.5,π/2], [0,+π/2], [0,0.5], [0.5,-π/2]]
all_solutions = []
solutions = []
forcing_period = 50.0
# Initial conditions
ϕ0 = randn(rng, nv(g))
ϕ0[1] = +π/2
ϕ0[2] = +π/2

tspan = (0.0, 500.0)
tsteps = range(tspan[1], tspan[2], length=1000)
nd_vertex = VertexModel(; f=ki_force_vertex!, g=StateMask(1:1), dim=1, sym=[:v], psym=[:h, :ψ, :β=>20.0, :τ])
nd_edge = EdgeModel(; g=AntiSymmetric(kiedge!), outdim=1, psym=[:weight,:σ=>1.0])
vertex_list = [nd_vertex for i in vertices(g)]
edge_list = [nd_edge for i in edges(g)]
nd! = Network(g, vertex_list, edge_list)
p_nd = NWParameter(nd!)
p_nd.e[:, :weight] = w_ij
p_nd.v[:, :h] = rand(uniform_h, N)
p_nd.v[:, :ψ] .= -π/2
nothing

# all the cases of the XOR gate
pars = [ξ_0, ξ_1, ξ_2, ξ_3]
#u0s = randn(rng, 4, N)
u0s = zeros(Float64, 4, N)
x0 = randn(rng, N)
#x0 = zeros(Float64, N)
ode_prob = ODEProblem(nd!, ϕ0, tspan)

# solve ensamble problem for all possible combinations of XOR gate
Φ = [-π/2 -π/2 π/2 0 -π/2; 
     -π/2 π/2  π/2 0 π/2; 
      π/2 -π/2 π/2 0 π/2; 
      π/2 π/2  π/2 0 -π/2]
Φr = rand(uniform_ψ, 4, 5)

# solve ensamble problem for all possible combinations of XOR gate
function prob_func(prob, i, repeat)
    new_p = deepcopy(p_nd)
    new_p.v[:,:ψ] = Φ[i,:] # prescribed values for ψ 
    new_p.v[:,:τ] = zeros(Float64, N)
    new_p.v[5,:τ] = Φ[i,5]
    
    prob = remake(prob; u0 = u0s[i,:], p = pflat(new_p))
    return prob
end

ens_prob = EnsembleProblem(ode_prob; prob_func=prob_func)
ens_sol = solve(ens_prob, Tsit5(), EnsembleThreads(); trajectories=4, saveat=tsteps)
all_solutions = Array(ens_sol)

# plot the solutions
fig = Figure(layout=(2,2))
labels_xor = ["FF", "FT", "TF", "TT"]
for (i,sol) in enumerate(ens_sol)
    ax = GLMakie.Axis(fig[i ÷ 3 + 1, i % 2 + (i % 2 == 0 ? 2 : 0)]; xlabel="Time", ylabel="u", title="XOR gate $(labels_xor[i])")
    t = sol.t
    u = sol(sol.t)[1:N,:]
    for i in [1,2,5]
        lines!(ax, t, u[i,:], linewidth=2, label="Oscillator $i")
        text!(ax, t[end], u[i,end]+0.1, text=string("Oscillator ", i), align=(:right, :center))
    end
    axislegend(ax, position = :rt)
end

all_solutions = Array(all_solutions)

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
ann_wk = Lux.Chain(Lux.Dense(2, 20, tanh),
    Lux.Dense(20, 1))
nn_pp, st = Lux.setup(rng, ann_wk)
pp = ComponentArray(nn_pp)


@inline function wk_edge!(e, v_s, v_d, p, t)
    nn_p = p
    in = [v_s[1], v_d[1]]
    e[1] = Lux.apply(ann_wk, in, nn_p, st)[1][1]*sin.(v_s[1] .- v_d[1])
    nothing 
end

@inline function wk_vertex!(dv, v, esum, (h,ψ,β,τ), t)
    println("size of v: ", size(v))
    #h, ψ, β, τ = p
    #if typeof(h) == Float32
    #    println("h: ", h, " ψ: ", ψ)
    #end
    #dv .= -h*sin.(v .- ψ) - β * sin.(v .- ψ)/(1 .+ cos.(v .- ψ))
    dv .= -h*sin.(v .- ψ) + esum[1]
    nothing
end

using MLUtils
# @note: testing on a single solution (2nd solution)
single_solution = [all_solutions[2]]
single_true = [true_data[2]]
batch_size = 4
train_loader_neural = MLUtils.DataLoader((all_solutions, xor_data), batchsize=batch_size, shuffle=false)
ann_vertex = VertexModel(; f=wk_vertex!, g=StateMask(1:1), dim=1, sym=[:v], psym=[:h, :ψ, :β=>20.0, :τ])
ann_wk_edge = EdgeModel(; g=AntiSymmetric(wk_edge!), outdim=1, psym=[:weight])
vertex_list = [ann_vertex for i in vertices(g)]
edge_list = [ann_wk_edge for i in edges(g)]
wk_network! = Network(g, vertex_list, edge_list)

p_wk = NWParameter(wk_network!)
p_wk.e[:, :weight] = [deepcopy(nn_pp) for i in 1:20]

probwk = ODEProblem(wk_network!, ϕ0, tspan, pp)

function distance_func(x, y)
    return sum(1 - cos.(x .- y))
end

function predict_neuralode(fullp, x0)
    #println("x0: ", x0)
    println("fullp: ", fullp)
    println("size of fullp: ", size(fullp))
    Array(solve(probwk, Tsit5(), p = fullp, u0=x0, saveat=tsteps, sensealg=ForwardDiffSensitivity()))
end

# batch should contain all possible combinations of XOR gate
function loss_neuralode(p)
    x0 = randn(Float64, N)  
    # do it for all data points in the batch
    #println("p: ", p, " size of batch_t: ", size(batch_t))
    pred = predict_neuralode(p, x0)
    loss = sum(abs2, all_solutions[1][5,:] .- pred[5,:])
    #pred = predict_neuralode(p)
    #loss = sum(abs2, all_solutions[1][5,:] .- pred[5,:])
    println("Current loss is: ", sum(loss, dims=1))
    return loss, pred
end

output_index = 5
loss_function(data, pred) = begin
    diff = data[output_index,:,:] .- pred[output_index,:,:]
    loss = ones(size(diff)) .- cos.(diff)
    return sum(loss)
end

function prob_func_neuralode(p,i,repeat)
    ps = (p, pars[i])
    println("pars {$i}: ", pars[i])
    remake(probwk, p=p_wk, u0=u0s[:,i], saveat=tsteps)
end

#=
@everywhere using Zygote
@everywhere using SciMLBase
@everywhere using ComponentArrays
@everywhere using NetworkDynamics
@everywhere using DiffEqFlux
=#
prob_ens = EnsembleProblem(probwk, prob_func = prob_func_neuralode)
#addprocs(8)
iter = 0
# trying the Ensamble problem to solve XOR gate
function loss_multiple_shooting(p; group_size=4)
    global iter
    iter = iter + 1
    # do it for all data points in the batch
    #println("Iteration $iter finished:")
    #println("size of p: ",p)
    loss = multiple_shoot(p, all_solutions, tsteps, prob_ens, EnsembleThreads(), loss_function, Tsit5(), group_size;
    continuity_term=300, trajectories = batch_size)
    
    println("Current loss is: ", loss[1])
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
#callback(pp, loss_neuralode(pp, train_loader_neural.data[1], train_loader_neural.data[2])...; doplot=true)

# use Optimization.jl to solve the problem
adtype = Optimization.AutoZygote()

optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentArray(nn_pp))
#result_neuralode = Optimization.solve(optprob, OptimizationOptimisers.AdamW(0.05), ncycle(train_loader_neural, 100); 
#    callback = callback, maxiters=100)

result_neuralode = Optimization.solve(optprob, Optimization.Sophia(); 
    callback = callback, maxiters=100)
optprob2 = remake(optprob; u0 = result_neuralode.minimizer)
using OptimizationPolyalgorithms
#result_neuralode2 = Optimization.solve(optprob2, Optim.BFGS(; initial_stepnorm = 0.01);
#      callback, allow_f_increases = false)
 result_neuralode2 = Optimization.solve(optprob2, Optimization.Sophia(); callback = callback)

callback(result_neuralode2.u, loss_neuralode(result_neuralode2.u, train_loader_neural.data[1], train_loader_neural.data[2])...; doplot=true)
using Plots
using PlotlyJS 
plotlyjs()
Plots.savefig("Wang-Kuramoto-opt.png")
using JLD2

# save the results
@save "Wang-Kuramoto-result-neuralode.jld2" result_neuralode2
JLD2.@load "Wang-Kuramoto-result-neuralode.jld2" result_neuralode

# try to predict the XOR gate using the trained network
x0 = randn(Float64, N)
training_data = (π / 2) * [-1 -1; -1 1; 1 -1; 1 1]
target_data = (π/2) * [-1, 1, 1, -1]
for ind in 1:4
    x0[1:2] .= training_data[ind,:]

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
    GLMakie.save("Wang-Kuramoto-opt_random_init_$(ind).png",fig, px_per_unit = 4)
    println("dist: $(distance_func(u[5,end], xor_data[ind]))")
end

# plot phase difference between all plot_phases
plot_phases(N, 1, u, solwk.t, ϕ0, forcing_period, tspan)
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
