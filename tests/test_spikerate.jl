using Test

include("../src/utils/spikerate.jl")
using .spikerate

@testset "spikerate" begin

    @testset "rate_conv: output shape and binary values" begin
        data = rand(Float64, 3, 4)
        spikes = rate_conv(data)
        @test size(spikes) == (3, 4)
        @test all(x -> x == 0 || x == 1, spikes)
    end

    @testset "rate_conv: clamps values outside [0, 1]" begin
        data = [-2.0 0.5; 3.0 0.0]
        spikes = rate_conv(data)
        @test all(x -> x == 0 || x == 1, spikes)
    end

    @testset "rate_conv: all-zeros input produces all-zero spikes" begin
        data = zeros(Float64, 5, 5)
        spikes = rate_conv(data)
        @test all(spikes .== 0)
    end

    @testset "rate_conv: all-ones input produces all-one spikes" begin
        data = ones(Float64, 5, 5)
        spikes = rate_conv(data)
        @test all(spikes .== 1)
    end

    @testset "rate: output shape [num_steps × batch × features]" begin
        data = rand(Float64, 3, 4)
        spikes = rate(data, 10)
        @test size(spikes) == (10, 3, 4)
        @test all(x -> x == 0 || x == 1, spikes)
    end

    @testset "rate: first_spike_time zeros early timesteps" begin
        fst = 3
        data = ones(Float64, 2, 4)   # all-ones → spikes at every step after fst
        spikes = rate(data, 10; first_spike_time=fst)
        @test all(spikes[1:fst, :, :] .== 0)
    end

    @testset "rate: time_var_input path" begin
        data = rand(Float64, 8, 3, 4)   # time × batch × features
        spikes = rate(data, 0; time_var_input=true)   # num_steps=0 ≡ false here
        @test size(spikes) == size(data)
        @test all(x -> x == 0 || x == 1, spikes)
    end

    @testset "rate: raises on negative num_steps" begin
        @test_throws ArgumentError rate(rand(2, 3), -1)
    end

    @testset "rate: raises when first_spike_time > num_steps-1" begin
        @test_throws ArgumentError rate(rand(2, 3), 5; first_spike_time=10)
    end

end
