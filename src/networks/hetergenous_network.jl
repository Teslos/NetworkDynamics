## Heterogenous network
using NetworkDynamics, OrdinaryDiffEq, Plots, Graphs

N = 9
g = watts_strogatz(N, 3, 0.0) # ring network

function kuramoto_edge!(e, θ_s, θ_d, K, t)
    e[1] = K *sin(θ_s[1] - θ_d[1])
    nothing
end

function kuramoto_vertex!(dθ, θ, edges, ω, t)
    dθ[1] = ω
    sum_coupling!(dθ, edges)
    nothing
end

vertex! = ODEVertex(; f=kuramoto_vertex!, dim=1, sym=[:θ])
edge! = StaticEdge(; f=kuramoto_edge!, dim=1)
nd! = network_dynamics(vertex!, edge!, g)

# Introduction of Heterogenous parameters is as easy as defining a array.
# Here the vertex parameters are hetereogenous, while the edge parameters are homogenous
# parameter K.

ω = (collect(1:N) .- sum(1:N)/N) /N
K = 3.0
p = (ω, K)

# Integrate and plot
x0 = (collect(1:N) .- sum(1:N)/N) /N
tspan = (0.0, 10.0)
prob = ODEProblem(nd!, x0, tspan, p)
sol = solve(prob, Tsit5())
plot(sol; ylabel="θ")


# Two paradigmatic modifications of the node model above are static nodes and nodes with 
# inertia. A static node has no internal dynamic and instead fixes the variable at a constant value.
# A Kuramoto model with inertia consits of two internal variables leading to 
# more complicated (and for many applications more realistic) dynamics.

static! = StaticVertex(; f=(θ, edges, c, t)-> θ .= c, dim=1, sym=[:θ])

function kuramoto_inertia!(dv, v, edges, P, t)
    dv[1] = v[2]
    dv[2] = P - 1.0 * v[2]
    for e in edges
        dv[2] += e[1]
    end
    nothing
end

inertia! = ODEVertex(; f=kuramoto_inertia!, dim=2, sym=[:θ, :ω])

# Since now we model a system with hetereogenous node dynamics, we can no longer
# straightforwardly pass a single VertexFunction to 'network_dynamics'. but instead have
# to hand over an Array.

vertex_array = Array{VertexFunction}([vertex! for i in 1:N])
vertex_array[1] = static!
vertex_array[5] = inertia! # index should correspond to the node's index in the graph
nd_hetero! = network_dynamics(vertex_array, edge!, g)

# Now we have to take a bit more care with defining initial conditions and parameters.
# For the first dimension we keep the initial condition from above and insert! another OrdinaryDiffEq
# into 'x0' at the correct index.
x0[1] = ω[1]

# The node with inertia is two-dimensional, hence we need to specify two initial conditions,
# For the first dimension we keep the initial condition from above and insert! another OrdinaryDiffEq
# into 'x0' at the correct index.

inertia_ic_2 = 5
insert!(x0, 6, inertia_ic_2)

# 'x0[1:4]` holds ic for nodes 1 to 4, 'x0[5:6]' holds the two
# initial conditions for node 5, and 'x0[7:9]' holds ic for nodes 6 to 8.

prob_hetero = ODEProblem(nd_hetero!, x0, tspan, p)
sol_hetero = solve(prob_hetero, Rodas4())

# For clarity, we plot only the variables reffering to the oscillators angle and color
# them according to their type.

membership = ones(Int64, N)
membership[1] = 2
membership[5] = 3
nodecolor = [colorant"lightseagreen", colorant"orange", colorant"darkred"]
nodefillc = reshape(nodecolor[membership], 1, N)

vars = syms_containing(nd_hetero!, :θ)
plot(sol_hetero, vars=vars, lc=nodefillc, ylabel="θ")

# Components with algebraic constraints

function edgeA!(de, e, v_s, v_d, p, t)
    de[1] = f(e, v_s, v_d, p, t) # dynamic variable
    e[2] = g(e, v_s, v_d, p, t) # static variable
    nothing
end

M = zeros(2,2)
M[1,1] = 1.0

nd_edgeA! = ODEEdge(; f=edgeA!, dim=2, sym=[:e, :s], coupling=:undirected, mass_matrix=M)
