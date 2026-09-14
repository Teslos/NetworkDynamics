# Regression tests for the production FHN digit classifier.
#
# These tests deliberately call the same fit/feature/predict API used by
# scripts/run_fhn_digits.jl. In particular, they exercise the production solver
# tolerances; a test-only copy with tighter tolerances previously hid numerical
# cross-talk between training and test nodes in a joint adaptive solve.

using Test
using Random
using LinearAlgebra

include("../src/baselines/fhn_digit_classifier.jl")
using .FHNDigitClassifier

@testset "FHN digit classifier" begin
    @testset "sample encoding is reproducible and batch-independent" begin
        X = Float64.(rand(Xoshiro(1), 0:16, 8, 6))
        S = encode_digit_sequences(X; nsteps=3, seed=42)
        @test S == encode_digit_sequences(X; nsteps=3, seed=42)
        @test S != encode_digit_sequences(X; nsteps=3, seed=43)

        permutation = [4, 1, 6, 2, 5, 3]
        @test encode_digit_sequences(X[:, permutation]; nsteps=3, seed=42) ==
              S[permutation, :]
        @test encode_digit_sequences(X[:, 3:3]; nsteps=3, seed=42)[1, :] ==
              S[3, :]
    end

    @testset "coupling strength has a size-independent meaning" begin
        sigma = 0.72
        for n in (5, 13)
            W = build_fhn_coupling(
                n, sigma; rng=Xoshiro(10), normalization=:row_total)
            @test diag(W) == zeros(n)
            @test all(isapprox.(vec(sum(W, dims=2)), sigma; atol=1e-12))
        end

        W_legacy = build_fhn_coupling(
            13, sigma; rng=Xoshiro(10), normalization=:per_edge)
        @test !all(isapprox.(vec(sum(W_legacy, dims=2)), sigma; atol=1e-12))
        @test_throws ArgumentError build_fhn_coupling(
            5, sigma; normalization=:unknown)
    end

    @testset "fit is seeded and prediction is genuinely inductive" begin
        rng = Xoshiro(11)
        X_train = Float64.(rand(rng, 0:16, 8, 12))
        y_train = repeat(0:2, inner=4)
        fit_args = (
            nsteps=2, seed=7, sigma=0.72, normalization=:row_total,
            epochs=8,
        )

        model = fit_fhn_digit_classifier(X_train, y_train; fit_args...)
        repeated = fit_fhn_digit_classifier(X_train, y_train; fit_args...)
        @test model.train_states == repeated.train_states
        @test model.query_weights == repeated.query_weights
        @test model.query_initial_state == repeated.query_initial_state
        @test model.training_features == repeated.training_features
        @test model.readout.weight == repeated.readout.weight
        @test size(model.train_states) == (length(y_train), 16)
        @test model.training_features == fhn_digit_features(model, X_train)

        X_query = Float64.(rand(rng, 0:16, 8, 4))
        F_batch = fhn_digit_features(model, X_query)

        # Every query runs in its own two-state ODE against an immutable cached
        # train reservoir. Its trajectory is therefore exactly unchanged by
        # batching, order, or unrelated query values.
        for j in axes(X_query, 2)
            @test F_batch[:, j] ==
                  fhn_digit_features(model, X_query[:, j:j])[:, 1]
        end
        permutation = [3, 1, 4, 2]
        @test fhn_digit_features(model, X_query[:, permutation]) ==
              F_batch[:, permutation]

        unrelated = Float64.(rand(rng, 0:16, 8, 1))
        F_extended = fhn_digit_features(model, hcat(X_query, unrelated))
        @test F_extended[:, 1:4] == F_batch

        prediction = predict_fhn_digits(model, X_query)
        @test length(prediction) == 4
        @test all(in(model.classes), prediction)
        @test length(predict_training_digits(model)) == length(y_train)
    end
end
