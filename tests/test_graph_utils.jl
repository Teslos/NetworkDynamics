using Test
using Graphs
using SimpleWeightedGraphs

include("../src/networks/graph_utils.jl")
using .graph_utils

@testset "graph_utils" begin

    @testset "create_complete_graph" begin
        N = 5
        g, w = create_complete_graph(N)
        @test nv(g) == N
        @test ne(g) == N * (N - 1)   # directed complete graph has N*(N-1) edges
        # weights are built from the undirected base graph (half the directed edges)
        @test length(w) == ne(g) ÷ 2
        @test all(w .== 1.0)
    end

    @testset "create_barabasi_albert_graph" begin
        N = 20
        g, w = create_barabasi_albert_graph(N)
        @test nv(g) == N
        @test ne(g) > 0
        @test length(w) == ne(g) ÷ 2
        @test all(w .== 1.0)
    end

    @testset "create_erdos_renyi_graph" begin
        N = 15
        g, w = create_erdos_renyi_graph(N, 0.6)
        @test nv(g) == N
        @test length(w) == ne(g) ÷ 2
        @test all(w .== 1.0)
    end

    @testset "create_watts_strogatz_graph" begin
        N = 12
        g, w = create_watts_strogatz_graph(N; k=4, prob=0.2)
        @test nv(g) == N
        @test length(w) == ne(g) ÷ 2
        @test all(w .== 1.0)
    end

    @testset "create_graph (grid N×M)" begin
        g, w = create_graph(4, 5)
        @test nv(g) == 4 * 5          # 4×5 grid = 20 nodes
        @test length(w) == ne(g) ÷ 2
        @test all(w .== 1.0)
    end

    @testset "return types are SimpleDiGraph and Vector" begin
        g, w = create_complete_graph(4)
        @test g isa SimpleDiGraph
        @test w isa Vector
    end

end
