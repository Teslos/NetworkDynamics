function swig_equation!(dv, v, edges, P, t)
    dv[1] = v[2]
    dv[2] = P - 0.1 * v[2]
    for e in edges
        dv[2] += e[1]
    end
    nothing
end

function powerflow!(e, v_s, v_d, K, t)
    e[1] = K * sin(v_s[1] - v_d[1])
end

using Graphs
g = watts_strogatz(4, 2, 0.0)

using NetworkDynamics
swig_vertex = ODEVertex(; f=swig_equation!, dim=2, sym=[:θ, :ω])
powerflow_edge = StaticEdge(; f=powerflow!, dim=1)

nd = network_dynamics(swig_vertex, powerflow_edge, g)

K = 6.0
P = [1.0, -1.0, 1.0, -1.0]
p = (P, K)

u0 = find_fixpoint(nd, p, zeros(8))

using OrdinaryDiffEq, StochasticDiffEq
ode_prob = ODEProblem(nd, u0, (0.0, 500.0), p)
ode_sol = solve(ode_prob, Tsit5())

using Plots, LaTeXStrings
plot(ode_sol; vars = syms_containing(nd, "ω"), ylims=(-1.0,1.0), fmt=:png, legend=false, xlabel=L"t", ylabel=L"\theta")

h = SimpleGraph(4,0)

function fluctuation!(dx, x, edges, p, t)
    dx[1] = 0.0
    dx[2] = 0.05
end

sde_prob = SDEProblem(nd, fluctuation!, u0, (0.0, 500.0), p)
sde_sol = solve(sde_prob, SOSRA(), dt=0.01)
plot(sde_sol; vars = syms_containing(nd, "ω"), ylims=(-1.0,1.0), fmt=:png, legend=false, xlabel=L"t", ylabel=L"\theta")