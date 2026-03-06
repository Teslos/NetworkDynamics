"""
Entry point for the NetworkDynamics test suite.
Run with: julia tests/runtests.jl
Or from the Julia REPL: include("tests/runtests.jl")
"""

using Test

@testset "NetworkDynamics" begin

    @testset "Graph utilities" begin
        include("test_graph_utils.jl")
    end

    @testset "Spike rate encoding" begin
        include("test_spikerate.jl")
    end

    @testset "ODE models" begin
        include("test_ode_models.jl")
    end

end
