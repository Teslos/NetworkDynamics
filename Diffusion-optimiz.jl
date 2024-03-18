# Learning the diffusion constant with DiffEqFlux.jl
# using the NetworkDynamics.jl package
using ComponentArrays, DiffEqFlux, NetworkDynamics, Lux, OrdinaryDiffEq, LinearAlgebra
using Graphs
using Optimization # for the optimization problem
using OptimizationOptimisers # for the optimization algorithm
#using Plots
### Define the graph
N = 20 # number of nodes
k = 4 # number of edges per node
g = barabasi_albert(N, k)

# Functions for edges and vertices
function diffusionedge!(e, v_s, v_d, p, t)
    D = p[1]
    e .= D .* (v_s .- v_d)
end

function diffusionvertex!(dv, e_s, e_d, p, t)
    dv .= 0.0
    for e in e_s
        dv .-= e
    end
    for e in e_d
        dv .+= e
    end
    nothing
end

# Constructing the network
nd_diffusion_vertex = ODEVertex(; f=diffusionvertex!, dim=1, sym=[:u])
nd_diffusion_edge = StaticEdge(; f=diffusionedge!, dim=1)
nd! = network_dynamics(nd_diffusion_vertex, nd_diffusion_edge, g)

# Simulation
x0 = randn(N) # random initial conditions
Σ = rand(ne(g))
tspan = (0.0, 4.0)
ode_prob = ODEProblem(nd!, x0, tspan, (nothing,Σ))
sol = solve(ode_prob, Tsit5())
#using Plots
# Plotting the solution
#plot(sol, lw=2, label="")
# using the new plotting package GLMakie
using GLMakie
fig = Figure()
ax = GLMakie.Axis(fig[1, 1]; xlabel = "Time", ylabel = "u", title = "Diffusion network")
t= sol.t
u = sol(sol.t)[1:N,:]
for i in 1:N
    lines!(ax, t, u[i,:], color = (:blue, 0.1))
end
fig
GLMakie.save("diffusion_network_optimization.png",fig, px_per_unit = 4)

# Learning the diffusion constant
# using wrapper function to pass tuples as array of parameters
function nd_wrapper!(dx, x, Σ,t)
    nd!(dx, x,(nothing, Σ), t)
end

probflux = ODEProblem(nd_wrapper!, x0, tspan, Σ; saveat=tspan[1]:0.01:tspan[2])

# The prediction function
#
# The function 'predict' integrates the system forward in time.
# Sensitivity analysis refers to computing the sensitivity of the
# output with respect to the parameters.

function predict(p)
    Array(solve(probflux, Tsit5(), p=p, saveat=tspan[1]:0.01:tspan[2], sensealg=ForwardDiffSensitivity()))
    #Array(probflux(p))
end

function loss(p)
    pred = predict(p)
    loss = sqrt(sum(abs2, pred))
    loss, pred
end

# callback
cb = function (p, l, pred; doplot=false)
    println("Current loss is: ", l)
    if doplot
        plot(sol, lw=2, label="")
        plot!(pred, lw=2, label="")
    end
    false
end

pinit = ComponentArray(Σ)
cb(pinit,loss(pinit)...;doplot=false)

# We optimize for optimal local diffusion constants
#res = DiffEqFlux.sciml_train(loss, Σ, ADAM(0.5); cb=cb, maxiters=20)
#DiffEqFlux.trainmode!(loss, Σ, ADAM(0.5); cb=cb, maxiters=20)
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x,p)->loss(x),adtype)
optprob = Optimization.OptimizationProblem(optf,pinit)
predict(pinit)
loss(pinit)
res = Optimization.solve(optprob,OptimizationOptimisers.Adam(0.05), callback=cb, maxiters=20)
res
pinit
optprobr = remake(ode_prob; Σ=res.u)
sol2 = solve(optprobr, Tsit5())

# Plotting the solution
# using the new plotting package GLMakie
fig2 = Figure()
ax = GLMakie.Axis(fig2[1, 1]; xlabel = "Time", ylabel = "u", title = "Fitted network")
t= sol2.t
u = sol2(sol2.t)[1:N,:]
for i in 1:N
    lines!(ax, t, u[i,:], color = (:blue, 0.1))
end
fig2
GLMakie.save("diffusion_network_optimization_fitted.png",fig2, px_per_unit = 4)