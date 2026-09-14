# Regression tests for the FHN digit reservoir (scripts/run_fhn_digits.jl).
#
# These cover the three properties the inductive redesign rests on, each of
# which was broken or unverified at some point:
#
#   1. Reproducibility -- a seed must reproduce a run. Two unseeded sources were
#      found here: the spike encoder drew from the global RNG, and Flux's `Dense`
#      initialised the readout from the global RNG even when an `rng` was passed.
#   2. Test-input isolation -- in `fixed_reservoir` mode, the training nodes must
#      be unaffected by the test inputs. This is what makes the classifier
#      inductive rather than transductive.
#   3. Batched == individual -- a test node's trajectory must not depend on which
#      other test samples happen to be in the batch, which is what licenses the
#      claim that the mode is equivalent to inserting each query on its own.
#
# The dynamics below mirror scripts/run_fhn_digits.jl; properties 2 and 3 are
# structural facts about the masked coupling, so they are checked directly on the
# integrator rather than through the script.

using Test
using OrdinaryDiffEq, LinearAlgebra, Statistics, Random, Distributions

include("../src/utils/spikerate.jl")
include("../src/baselines/baseline_utils.jl")
include("../src/baselines/baseline_models.jl")
using .spikerate, .BaselineUtils, .BaselineModels

const RES_EPS = 0.05
const RES_A = 0.5
const RES_R0 = 0.5
const RES_SIGMA = 0.72

"Complete-graph diffusive coupling, optionally masking the columns of `masked`."
function build_coupling(n, rng; masked=Int[])
    W = [pdf(Normal(), r) for r in (2 .* rand(rng, n, n) .- 1)]
    W = RES_SIGMA .* (W .+ W') ./ 2
    W[diagind(W)] .= 0
    isempty(masked) || (W[:, masked] .= 0.0)
    return W
end

"Relax the reservoir driven by `S` (n, T). Returns u-trajectories (n, T)."
function relax(S, W, z0)
    n, T = size(S)
    rowsum = vec(sum(W, dims=2))
    function rhs!(dz, z, _, t)
        u = @view z[1:n]; v = @view z[n+1:2n]
        du = @view dz[1:n]; dv = @view dz[n+1:2n]
        i = t <= 1 ? 1 : min(floor(Int, t), T - 1)
        f = t <= 1 ? 0.0 : t - i
        g = (1 - f) .* @view(S[:, i]) .+ f .* @view(S[:, min(i + 1, T)])
        c = W * u .- rowsum .* u
        @. du = g + u - u^3 / 3 - v + c
        @. dv = (g * RES_R0 + u - RES_A) * RES_EPS
        return nothing
    end
    sol = solve(ODEProblem(rhs!, z0, (0.0, Float64(T))), Tsit5();
                saveat=1.0:1.0:T, save_idxs=1:n, reltol=1e-8, abstol=1e-10)
    return Array(sol)
end

@testset "FHN reservoir" begin

    @testset "spike encoder is reproducible from an explicit rng" begin
        data = rand(Xoshiro(1), 6, 8)
        a = spikerate.rate(data, 4; rng=Xoshiro(42))
        b = spikerate.rate(data, 4; rng=Xoshiro(42))
        c = spikerate.rate(data, 4; rng=Xoshiro(43))
        @test a == b                      # same seed, same spikes
        @test a != c                      # different seed, different spikes
        @test all(x -> x in (false, true, 0, 1), a)
    end

    @testset "readout fit is reproducible from an explicit rng" begin
        X = randn(Xoshiro(5), 6, 40)
        Y = zeros(3, 40); for j in 1:40; Y[rand(Xoshiro(j), 1:3), j] = 1.0; end
        m1 = train_logreg(X, Y; epochs=20, rng=Xoshiro(7))
        m2 = train_logreg(X, Y; epochs=20, rng=Xoshiro(7))
        m3 = train_logreg(X, Y; epochs=20, rng=Xoshiro(8))
        @test predict_nn(m1, X) == predict_nn(m2, X)
        @test m1.weight == m2.weight
        @test m1.weight != m3.weight
    end

    # ---- structural properties of the fixed_reservoir masking ----------------
    n, T = 12, 24
    train_idx, test_idx = collect(1:8), collect(9:12)
    rng = Xoshiro(11)
    S = Float64.(rand(rng, n, T) .< 0.3)
    W_masked = build_coupling(n, Xoshiro(3); masked=test_idx)
    W_full = build_coupling(n, Xoshiro(3))
    z0 = rand(Xoshiro(4), 2n)

    @testset "test inputs cannot reach the training nodes" begin
        U_ref = relax(S, W_masked, z0)
        S_perturbed = copy(S)
        S_perturbed[test_idx, :] .= 1.0 .- S_perturbed[test_idx, :]   # flip every test spike
        U_perturbed = relax(S_perturbed, W_masked, z0)
        train_shift = maximum(abs, U_ref[train_idx, :] .- U_perturbed[train_idx, :])
        test_shift = maximum(abs, U_ref[test_idx, :] .- U_perturbed[test_idx, :])

        # The masked rows make the training nodes mathematically independent of
        # the test inputs, but this is NOT bit-identical, and deliberately tested
        # as such: the adaptive solver chooses its steps from an error norm over
        # ALL states, so perturbing the test nodes shifts the step sequence and
        # moves the training values at solver-tolerance level (~1e-8 here, with
        # reltol 1e-8). The residual channel is numerical, not dynamical, and it
        # disappears once the training reservoir is integrated in its own solve
        # and cached -- at which point this assertion should tighten to `==`.
        # Measured at reltol 1e-8: train_shift ~ 3e-6, test_shift ~ 1. The bound
        # is on the SEPARATION rather than an absolute value, because the
        # residual scales with the solver tolerance, so a looser tolerance (as
        # in the script, which uses the defaults) will show a larger one.
        @test train_shift < 1e-4
        @test train_shift < test_shift / 1e3
        @test test_shift > 1e-2          # the test nodes really did move

        # control: without the mask the same perturbation moves the training
        # nodes by orders of magnitude more, i.e. the assertion above can fail.
        V_ref = relax(S, W_full, z0)
        V_perturbed = relax(S_perturbed, W_full, z0)
        @test maximum(abs, V_ref[train_idx, :] .- V_perturbed[train_idx, :]) > 1e-2
    end

    @testset "a test node does not depend on the rest of the test batch" begin
        U_all = relax(S, W_masked, z0)
        keep = vcat(train_idx, [test_idx[1]])          # train set + ONE query
        S_one = S[keep, :]
        W_one = W_masked[keep, keep]
        z0_one = vcat(z0[keep], z0[n .+ keep])
        U_one = relax(S_one, W_one, z0_one)
        # the query is the last row of the reduced system
        @test isapprox(U_all[test_idx[1], :], U_one[end, :]; rtol=1e-5, atol=1e-6)
        @test isapprox(U_all[train_idx, :], U_one[1:length(train_idx), :]; rtol=1e-5, atol=1e-6)
    end
end
