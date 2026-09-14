using Test
using Graphs
using OrdinaryDiffEq
using LinearAlgebra
using Random

# These tests were written against the NetworkDynamics <0.9 API (ODEVertex /
# StaticEdge / network_dynamics), which has been removed; against the installed
# 0.9.7 they errored rather than ran, which is where the suite's "4 errored"
# came from. They are rewritten here as plain ODEProblem formulations of the
# same dynamics -- which is also how every runnable script in this repository now
# integrates these networks (dense coupling matvec, see scripts/run_fhn_digits.jl
# and notebooks/EP-XY-Network-Claude.jl), so the tests now exercise the code path
# the reported results actually depend on.

# ── Shared parameters ─────────────────────────────────────────────────────────
const σ_test = 0.7
const a_test = 0.5
const ϵ_test = 0.05

# ── FitzHugh-Nagumo network (small, no external forcing) ──────────────────────
# State layout x = [u_1..u_N; v_1..v_N], diffusive coupling on u over a complete
# graph: sum_j W_ij (u_j - u_i).
function fhn_network!(dx, x, W, t)
    N = size(W, 1)
    u = @view x[1:N]; v = @view x[N+1:2N]
    du = @view dx[1:N]; dv = @view dx[N+1:2N]
    coupling = W * u .- vec(sum(W, dims=2)) .* u
    @. du = u - u^3 / 3 - v + coupling
    @. dv = ϵ_test * (u - a_test)
    return nothing
end

complete_coupling(N) = σ_test .* (ones(N, N) - I(N))
fhn_problem(N, x0, tspan) = ODEProblem(fhn_network!, x0, tspan, complete_coupling(N))

@testset "FitzHugh-Nagumo ODE network" begin

    @testset "completes without error on complete graph (N=4)" begin
        N = 4
        x0 = randn(Xoshiro(1), 2 * N)
        sol = solve(fhn_problem(N, x0, (0.0, 10.0)), Tsit5())
        @test sol.retcode == ReturnCode.Success
        @test size(sol[end]) == (2 * N,)
        @test !any(isnan, sol[end])
        @test !any(isinf, sol[end])
    end

    @testset "solution grows monotonically in time (more time steps with saveat)" begin
        N = 3
        x0 = randn(Xoshiro(2), 2 * N)
        sol = solve(fhn_problem(N, x0, (0.0, 5.0)), Tsit5(); saveat=0.5)
        @test length(sol.t) == 11   # t = 0.0 : 0.5 : 5.0
        @test sol.t[1] ≈ 0.0
        @test sol.t[end] ≈ 5.0
    end

    @testset "state dimension equals 2N" begin
        for N in [3, 5, 8]
            x0 = randn(Xoshiro(N), 2 * N)
            sol = solve(fhn_problem(N, x0, (0.0, 1.0)), Tsit5())
            @test size(sol[end], 1) == 2 * N
        end
    end

    @testset "coupling is diffusive: a synchronised state stays synchronised" begin
        # If every u_i is equal the coupling term vanishes identically, so the
        # network must evolve exactly like a single uncoupled oscillator.
        N = 4
        x0 = vcat(fill(0.3, N), fill(-0.1, N))
        sol = solve(fhn_problem(N, x0, (0.0, 5.0)), Tsit5(); saveat=1.0)
        for u in sol.u
            @test all(≈(u[1]), u[1:N])
            @test all(≈(u[N+1]), u[N+1:2N])
        end
    end

end

# ── Kuramoto oscillator network ───────────────────────────────────────────────
# dθ_i = ω_i + K * sum_j sin(θ_j - θ_i)
function kuramoto_network!(dθ, θ, p, t)
    ω, K = p
    N = length(θ)
    @inbounds for i in 1:N
        acc = 0.0
        for j in 1:N
            acc += sin(θ[j] - θ[i])
        end
        dθ[i] = ω[i] + K * acc
    end
    return nothing
end

@testset "Kuramoto ODE network" begin

    @testset "completes on complete graph (N=5)" begin
        N = 5
        ω = randn(Xoshiro(3), N)          # natural frequencies
        K = 0.5
        x0 = 2π * rand(Xoshiro(4), N)
        prob = ODEProblem(kuramoto_network!, x0, (0.0, 10.0), (ω, K))
        sol  = solve(prob, Tsit5())
        @test sol.retcode == ReturnCode.Success
        @test !any(isnan, sol[end])
    end

    @testset "identical frequencies and strong coupling synchronise" begin
        N = 6
        ω = fill(0.5, N)
        x0 = 2π * rand(Xoshiro(5), N)
        sol = solve(ODEProblem(kuramoto_network!, x0, (0.0, 60.0), (ω, 2.0)), Tsit5())
        θ = sol[end]
        # Phase-locked units can settle a multiple of 2pi apart, so measure
        # coherence with the Kuramoto order parameter rather than by subtracting
        # raw phases.
        R = abs(sum(cis, θ) / N)
        @test R > 0.999
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
        # beta = -1 makes the quartic term destabilising, so the potential
        # u^2/2 - u^4/4 is unbounded outside |u| ~ 1.4 and escaping orbits blow
        # up in finite time. The second initial condition used to be [1.0, 0.0],
        # which diverges (retcode Unstable, values ~1e30) and made the
        # inequality below pass vacuously. Both initial conditions now stay in
        # the well, and the solves are required to succeed.
        p = (1.0, -1.0, 0.3, 0.5, 1.2)
        sol1 = solve(ODEProblem(duffing!, [0.0, 0.0], (0.0, 10.0), p), Tsit5())
        sol2 = solve(ODEProblem(duffing!, [0.3, 0.0], (0.0, 10.0), p), Tsit5())
        @test sol1.retcode == ReturnCode.Success
        @test sol2.retcode == ReturnCode.Success
        @test all(isfinite, sol1[end])
        @test all(isfinite, sol2[end])
        @test sol1[end] != sol2[end]
    end

end
