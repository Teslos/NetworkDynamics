function diffusion_edge!(e, v_s, v_d, p, t)
    # usually e, v_s, v_d are arrays, hence we use the brodcasting operator.
    e .= v_s - v_d
    nothing
end

function diffusion_vertex!(dv, v, edges, p, t)
    # usually v, e are arrays, hence we use the brodcasting operator.
    dv .= 0 
    for e in edges
        dv .+= e
    end

    nothing
end

using Graphs
N = 20 # Number of vertices
k = 4 # Number of edges per vertex
g = barabasi_albert(N, k) # a little more exciting than a regular graph

using NetworkDynamics
nd_diffusion_vertex = ODEVertex(; f=diffusion_vertex!, dim=1)
nd_diffusion_edge = StaticEdge(; f=diffusion_edge!, dim=1)

nd = network_dynamics(nd_diffusion_vertex, nd_diffusion_edge, g)

using OrdinaryDiffEq

x0 = randn(N) # random initial condition
ode_prob = ODEProblem(nd, x0, (0.0, 4.0))
sol = solve(ode_prob, Tsit5())

using Plots
plot(sol, vars=syms_containing(nd, "v"), fmt=:png)

# multi-dimensional diffusion
N = 10
k = 4
g = barabasi_albert(N, k)

# We will have two independent diffusion processes on the graph, dim=2
nd_diffusion_vertex_2 = ODEVertex(; f=diffusion_vertex!, dim=2, sym=[:x,:ϕ])
nd_diffusion_edge_2 = StaticEdge(; f=diffusion_edge!, dim=2)
nd_2 = network_dynamics(nd_diffusion_vertex_2, nd_diffusion_edge_2, g)

x0_2 = vec(transpose([randn(N) .^2 randn(N)]))
ode_prob_2 = ODEProblem(nd_2, x0_2, (0.0, 3.0))
sol_2 = solve(ode_prob_2, Tsit5());

# plot the solution
plot(sol_2; vars=syms_containing(nd_2, "x"), fmt=:png)