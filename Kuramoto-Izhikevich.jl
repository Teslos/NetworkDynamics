# Learning the diffusion constant with DiffEqFlux.jl
# using the NetworkDynamics.jl package
using ComponentArrays, DiffEqFlux, NetworkDynamics, Lux, OrdinaryDiffEq, LinearAlgebra
using Graphs
using Optimization # for the optimization problem
using OptimizationOptimisers
using GLMakie # for the optimization algorithm
function create_graph(N::Int=8, M::Int=10)
    g = Graphs.grid([N, M])
    edge_weights = ones(length(edges(g)))
    return g, edge_weights
end

function plot_rect_map(N::Int, M::Int, data::Vector{Float64}, ax)
    #ax = GLMakie.Axis(f[1, 1])
    centers_x = 1:N
    centers_y = 1:M
    data = reshape(abs.(data), N, M)
    GLMakie.heatmap!(ax, centers_x, centers_y, data, colormap = :viridis)
end

function plot_phases(N::Int, M::Int, u::Array{Float64,2}, t::Array{Float64,1}, x0::Array{Float64,1} ,forcing_period::Float64)
    # Find the index of the value in t that is closest to t0 time
    for t0 in forcing_period:100:2500
        f = Figure()
        ax = GLMakie.Axis(f[1, 1], xlabel = "Node", ylabel = "u", title = "Kuramoto network at time $t0")
        index_closest_to_t = findmin(abs.(t .- t0))[2]
        state_vector_at_t = [mod2pi((u[i,index_closest_to_t]-u[1,index_closest_to_t])-x0[i]) for i in 1:N*M]
        plot_rect_map(N, M, state_vector_at_t, ax)
        # record the frames
        GLMakie.save("./figs/kuramoto_network_diff_phase_t$(t0).png",f, px_per_unit = 4)
    end
end
### Define the graph
# @note: we are defining the diffrent graphs
#N = 20 # number of nodes
#k = 4 # number of edges per node
#g = barabasi_albert(N, k)
N=8; M=10
g, edge_weights = create_graph(N, M)

# create the vector of vectors for the parameter values of the vertices
ξ = [zeros(ComplexF64, nv(g)) for _ in 1:2]
ξ[1][1:2:end] .= 0.1+0*im # activating the first part
ξ[2][1:8] .= 0.1+0*im 
ξ[2][9:8:end] .= 0.1+0*im
ξ[2][8:8:end] .= 0.1+0*im
ξ[2][end-7:end] .= 0.1+0*im
#ξ_0[16:end] .= 0 # activating the end part
f = GLMakie.Figure()
ax = GLMakie.Axis(f[1, 1])
f
ξ[1] = normalize(ξ[1])
ξ[2] = normalize(ξ[2])
plot_rect_map(8,10, abs.(ξ[2]),ax)
f
ω_0 = 1.0 # intrinsic frequency
ϵ = 100 # coupling strength
ϵc = 0.1 # coupling strength for the edges
forcing_period = 400.0 # forcing period

# Functions for edges and vertices
Base.@propagate_inbounds function kiedge!(e, v_s, v_d, p, t)
    # p is matrix of connections, same for ψ
    # e .= p*sin.(v_s .+ ψ .- v_d) # *σ
    pr, pi = p
    if t < forcing_period
        e .= 0.0 # no coupling in the forcing period
    else
        e .= pr*sin.(v_s .+ pi .- v_d) # *σ
    end
    #e .= pr*sin.(v_s .+ pi .- v_d) # *σ
    nothing
end

Base.@propagate_inbounds function ki_force_vertex!(dv, v, edges, p, t)
    # here we are forcing the first node to be active for period of time
    pr, pi = p
    if t < forcing_period
        dv .= ω_0 .+ ϵ * pr * sin.(ω_0 * t + pi)
        #dv .= ϵ * pr * sin.(ω_0 * t + pi)
    else
        dv .= ω_0 
    end
    # sum of all the connection edges
    sum_coupling!(dv, edges)
    nothing
end
# generating the random values for the parameter value ω_0 of the vertices close to 1
#v_pars = [1.0 + 0.01*randn() for v in vertices(g)]
ξ_0 = copy(ξ[1])
ξ_0[1:6] .= 0.0
v_pars = ξ_0
# coupling strength of edges are set to 1/3 @@note: coupling strength of the edges should be given by the
# coupling matrix C_ij = s_ij * exp(ψ_ij*im) where s_ij is the strength of the connection and ψ_ij is the phase difference
# calculate the adjecency matrix from the vector
c_ij = zeros(ComplexF64,ne(g))
i = 1
for e in edges(g)
    u,v = src(e), dst(e)
    #c_ij[i] = ϵc*ξ_0[u]*conj(ξ_0[v])
    # sum contribution of all images memorized in ξ
    c_ij[i] = ϵc/nv(g)*sum([ξ[k][u]*conj(ξ[k][v]) for k in 1:2])

    #c_ij[i] = 1.0/3.0 + 1/3*im
    i += 1
end 


e_pars = c_ij
#e_pars = [1.0 / 3.0 + 1/3*im for e in edges(g)]
v_pars = [(real(v), imag(v)) for v in v_pars]
e_pars = [(real(e), imag(e)) for e in e_pars]
parameters = (v_pars, e_pars)

# Constructing the network
nd_vertex = ODEVertex(; f=ki_force_vertex!, dim=1, sym=[:v])
nd_edge = StaticEdge(; f=kiedge!, dim=1)
nd! = network_dynamics(nd_vertex, nd_edge, g)

# Simulation
using Random
rng = MersenneTwister(1234);
x0 = randn(rng,nv(g)) # random initial conditions
#x0 = ones(nv(g))*10
tspan = (0.0, 2500.0)
ode_prob = ODEProblem(nd!, x0, tspan, parameters)
sol = solve(ode_prob, Tsit5(), saveat=tspan[1]:0.1:tspan[2])
#using Plots
# Plotting the solution
#plot(sol, lw=2, label="")
# using the new plotting package GLMakie
using GLMakie
fig = Figure()
ax = GLMakie.Axis(fig[1, 1]; xlabel = "Time", ylabel = "u", title = "Kuramoto network")
t= sol.t
u = sol(sol.t)[1:N*M,:] # get solution for plotting
for i in 1:N
    lines!(ax, t, u[i,:], color = (:blue, 0.1))
end
fig
GLMakie.save("kuramoto_network_optimization.png",fig, px_per_unit = 4)
# first we get the indices of the vertex variables 'v'
u_idx = idx_containing(nd!, :v)
fig1 = Figure()
ax1 = GLMakie.Axis(fig1[1, 1], xlabel = "Time", ylabel = "u", title = "Kuramoto network")
for i in 2:N
    if i < 4
        cl = (:red, 0.2)
    else
        cl = (:blue, 0.1)
    end
    lines!(ax1, t, (mod2pi.(u[i,:]-u[1,:])), color = cl)
    #heatmap!(ax1, hcat(t, (mod2pi.(u[i,:]-u[1,:]))), color = (:red, 0.1))
end
fig1
GLMakie.save("kuramoto_network_diff_phase.png",fig1, px_per_unit = 4)
using Plots
Plots.plot(sol.t, u[u_idx[2],:], lw=0.2, label="Node 1")
plot_phases(N, M, u, t, x0, forcing_period)

# remake the problem with the new parameters
vpars = zeros(ComplexF64, nv(g))
#vpars = ξ[2]
ξ_0 = copy(ξ[2])
ξ_0[5*8-4] = 0.1 + 0.0im
ξ_0 = normalize(ξ_0)
vpars = [(real(v), imag(v)) for v in ξ_0]
parameters = (vpars, e_pars)
u0 = real(ξ_0)
optprobr = remake(ode_prob; u0=u0, p=parameters)
forcing_period = 0.0
sol2 = solve(optprobr, Tsit5(), saveat=tspan[1]:0.1:tspan[2])
u = sol2(sol2.t)[1:N*M,:]
plot_phases(N, M, abs.(u), sol2.t, zeros(nv(g)), forcing_period)

# Learning the diffusion constant
# using wrapper function to pass tuples as array of parameters
function nd_wrapper!(dx, x, Σ,t)
    nd!(dx, x, Σ, t)
end
# The prediction function
#
# The function 'predict' integrates the system forward in time.
# Sensitivity analysis refers to computing the sensitivity of the
# output with respect to the parameters.

probflux = ODEProblem(nd!, x0, tspan, parameters; saveat=tspan[1]:0.01:tspan[2])

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
pinit = ComponentArray(parameters)
cb(pinit,loss(pinit)...;doplot=false)

# We optimize for optimal local diffusion constants
#res = DiffEqFlux.sciml_train(loss, Σ, ADAM(0.5); cb=cb, maxiters=20)
#DiffEqFlux.trainmode!(loss, Σ, ADAM(0.5); cb=cb, maxiters=20)
adtype = Optimization.AutoZygote()
ftest(x,p) = loss(p)
optf = Optimization.OptimizationFunction(ftest,adtype)
optprob = Optimization.OptimizationProblem(optf,pinit)
predict(pinit)
loss(pinit)
res = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.05),maxiters=20)
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