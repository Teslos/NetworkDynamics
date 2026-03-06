using Test
using NetworkDynamics
using Graphs
using OrdinaryDiffEq
using LinearAlgebra

# ── Shared parameters ─────────────────────────────────────────────────────────
const σ_test = 0.7
const a_test = 0.5
const ϵ_test = 0.05

# ── FitzHugh-Nagumo network (small, no external forcing) ──────────────────────
@inline Base.@propagate_inbounds function fhn_vertex_test!(dv, v, edges, p, t)
    dv[1] = v[1] - v[1]^3 / 3 - v[2]
    dv[2] = ϵ_test * (v[1] - a_test)
    for e in edges[1]; dv[1] -= e[1]; end
    for e in edges[2]; dv[1] += e[1]; end
    nothing
end

@inline Base.@propagate_inbounds function fhn_edge_test!(e, v_s, v_d, p, t)
    e[1] = σ_test * (v_s[1] - v_d[1])  # use global coupling constant
    nothing
end

@testset "FitzHugh-Nagumo ODE network" begin

    @testset "completes without error on complete graph (N=4)" begin
        N = 4
        g = SimpleDiGraph(complete_graph(N))
        vertex = ODEVertex(f=fhn_vertex_test!, dim=2, sym=[:u, :v])
        edge   = StaticEdge(f=fhn_edge_test!, dim=1, coupling=:directed)
        nd     = network_dynamics(vertex, edge, g)
        x0 = randn(2 * N)
        prob = ODEProblem(nd, x0, (0.0, 10.0), nothing)
        sol  = solve(prob, Tsit5())
        @test sol.retcode == ReturnCode.Success
        @test size(sol[end]) == (2 * N,)
        @test !any(isnan, sol[end])
        @test !any(isinf, sol[end])
    end

    @testset "solution grows monotonically in time (more time steps with saveat)" begin
        N = 3
        g = SimpleDiGraph(complete_graph(N))
        vertex = ODEVertex(f=fhn_vertex_test!, dim=2, sym=[:u, :v])
        edge   = StaticEdge(f=fhn_edge_test!, dim=1, coupling=:directed)
        nd     = network_dynamics(vertex, edge, g)
        x0 = randn(2 * N)
        prob = ODEProblem(nd, x0, (0.0, 5.0), nothing)
        sol  = solve(prob, Tsit5(); saveat=0.5)
        @test length(sol.t) == 11   # t = 0.0 : 0.5 : 5.0
        @test sol.t[1] ≈ 0.0
        @test sol.t[end] ≈ 5.0
    end

    @testset "state dimension equals 2N" begin
        for N in [3, 5, 8]
            g = SimpleDiGraph(complete_graph(N))
            vertex = ODEVertex(f=fhn_vertex_test!, dim=2, sym=[:u, :v])
            edge   = StaticEdge(f=fhn_edge_test!, dim=1, coupling=:directed)
            nd = network_dynamics(vertex, edge, g)
            x0 = randn(2 * N)
            sol = solve(ODEProblem(nd, x0, (0.0, 1.0), nothing), Tsit5())
            @test size(sol[end], 1) == 2 * N
        end
    end

end

# ── Kuramoto oscillator network ───────────────────────────────────────────────
@inline Base.@propagate_inbounds function kuramoto_edge_test!(e, v_s, v_d, p, t)
    e[1] = p * sin(v_s[1] - v_d[1])
    nothing
end

@inline Base.@propagate_inbounds function kuramoto_vertex_test!(dv, v, edges, p, t)
    dv[1] = p   # natural frequency
    for e in edges[1]; dv[1] -= e[1]; end
    for e in edges[2]; dv[1] += e[1]; end
    nothing
end

@testset "Kuramoto ODE network" begin

    @testset "completes on complete graph (N=5)" begin
        N = 5
        g = SimpleDiGraph(complete_graph(N))
        vertex = ODEVertex(f=kuramoto_vertex_test!, dim=1, sym=[:θ])
        edge   = StaticEdge(f=kuramoto_edge_test!, dim=1, coupling=:directed)
        nd     = network_dynamics(vertex, edge, g)
        ω = randn(N)       # natural frequencies
        K = 0.5 * ones(ne(g))   # coupling
        p = (ω, K)
        # flatten to vector expected by NetworkDynamics (vertex params)
        x0 = 2π * rand(N)
        prob = ODEProblem(nd, x0, (0.0, 10.0), (ω, K))
        sol  = solve(prob, Tsit5())
        @test sol.retcode == ReturnCode.Success
        @test !any(isnan, sol[end])
    end

end

# ── Duffing oscillator (scalar, no network) ────────────────────────────────────
@testset "Duffing oscillator (standalone ODE)" begin

    function duffing!(du, u, p, t)
        α, β, δ, γ, ω = p
        du[1] = u[2]
        du[2] = -δ * u[2] - α * u[1] - β * u[1]^3 + γ * cos(ω * t)
        nothing
    end

    @testset "solves without error" begin
        p = (1.0, -1.0, 0.3, 0.5, 1.2)   # α, β, δ, γ, ω
        x0 = [0.0, 0.0]
        prob = ODEProblem(duffing!, x0, (0.0, 50.0), p)
        sol  = solve(prob, Tsit5(); saveat=0.1)
        @test sol.retcode == ReturnCode.Success
        @test !any(isnan, reduce(vcat, sol.u))
        @test length(sol.t) == 501
    end

    @testset "changes initial condition produces different trajectory" begin
        p = (1.0, -1.0, 0.3, 0.5, 1.2)
        sol1 = solve(ODEProblem(duffing!, [0.0, 0.0], (0.0, 10.0), p), Tsit5())
        sol2 = solve(ODEProblem(duffing!, [1.0, 0.0], (0.0, 10.0), p), Tsit5())
        @test sol1[end] != sol2[end]
    end

end
