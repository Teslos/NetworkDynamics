module FHNDigitClassifier

using OrdinaryDiffEq
using SciMLBase
using LinearAlgebra
using Random
using Distributions
using Statistics

include(joinpath(@__DIR__, "..", "utils", "spikerate.jl"))
include(joinpath(@__DIR__, "baseline_utils.jl"))
include(joinpath(@__DIR__, "baseline_models.jl"))
using .spikerate
using .BaselineUtils
using .BaselineModels

export FHNDigitModel,
       load_digit_data,
       encode_digit_sequences,
       build_fhn_coupling,
       fit_fhn_digit_classifier,
       fhn_digit_features,
       predict_fhn_digits,
       predict_training_digits

const DEFAULT_EPS = 0.05
const DEFAULT_A = 0.5
const DEFAULT_R0 = 0.5

"""
Fitted inductive FHN digit classifier.

`train_states` contains only the training oscillators. Test queries never enter
the training ODE: each query is subsequently driven by its own spike sequence
and by a fixed weighted projection of the cached training trajectory.
"""
struct FHNDigitModel{M,C}
    train_states::Matrix{Float64}  # training nodes x encoded time
    query_weights::Vector{Float64}
    query_initial_state::Vector{Float64}
    training_features::Matrix{Float64}  # encoded time x training samples
    feature_mean::Vector{Float64}
    feature_scale::Vector{Float64}
    readout::M
    classes::Vector{C}
    nsteps::Int
    sigma::Float64
    normalization::Symbol
    encoding_seed::Int
    eps::Float64
    a::Float64
    r0::Float64
    reltol::Float64
    abstol::Float64
end

load_digit_data(; path=joinpath(@__DIR__, "..", "..", "data", "digits", "optdigits.tes")) =
    BaselineUtils.load_digits(; path)

function _check_normalization(normalization::Symbol)
    normalization in (:row_total, :per_edge) ||
        throw(ArgumentError("normalization must be :row_total or :per_edge"))
    return normalization
end

"Stable content-derived seed: independent of process, batch order, and batch size."
function _sample_seed(base_seed::Integer, x::AbstractVector, domain::UInt64)
    h = UInt64(0xcbf29ce484222325) ⊻ UInt64(base_seed) ⊻ domain
    @inbounds for value in x
        h ⊻= reinterpret(UInt64, Float64(value))
        h *= UInt64(0x00000100000001b3)
    end
    return h
end

"""
    encode_digit_sequences(X; nsteps=32, seed=1234, input_max=16.0)

Rate-encode columns of `X` independently. The random stream for a sample is
derived from its pixel values and `seed`, so the same query receives the same
encoding whether evaluated alone, reordered, or inside a different batch.
Returns a `(samples, nsteps * features)` drive matrix.
"""
function encode_digit_sequences(X::AbstractMatrix; nsteps::Int=32,
                                seed::Integer=1234, input_max::Real=16.0)
    nsteps > 0 || throw(ArgumentError("nsteps must be positive"))
    input_max > 0 || throw(ArgumentError("input_max must be positive"))
    d, nsamples = size(X)
    S = Matrix{Float64}(undef, nsamples, nsteps * d)
    for sample in 1:nsamples
        x = Float64.(@view X[:, sample])
        rng = Xoshiro(_sample_seed(seed, x, UInt64(0x656e636f64696e67)))
        probs = reshape(x ./ input_max, 1, d)
        spikes = spikerate.rate(probs, nsteps; rng)
        S[sample, :] .= vec(@view spikes[:, 1, :])
    end
    return S
end

"""
    build_fhn_coupling(n, sigma; rng, normalization=:row_total)

Construct the positive random training coupling. With `:row_total` (default),
every row is normalized to sum to `sigma`, making the dynamical regime
comparable across reservoir sizes. `:per_edge` preserves the historical rule in
which every edge is multiplied by `sigma` and aggregate coupling grows with `n`.
"""
function build_fhn_coupling(n::Int, sigma::Real;
                            rng::AbstractRNG=Random.default_rng(),
                            normalization::Symbol=:row_total)
    n >= 2 || throw(ArgumentError("the training reservoir needs at least two nodes"))
    sigma >= 0 || throw(ArgumentError("sigma must be non-negative"))
    _check_normalization(normalization)
    raw = 2 .* rand(rng, n, n) .- 1
    W = pdf.(Normal(), raw)
    W = Float64(sigma) .* (W .+ W') ./ 2
    W[diagind(W)] .= 0
    if normalization === :row_total && sigma > 0
        for i in axes(W, 1)
            total = sum(@view W[i, :])
            total > 0 || error("training coupling row $i has zero weight")
            @views W[i, :] .*= sigma / total
        end
    end
    return W
end

@inline function _lerp_row(S::AbstractMatrix, row::Int, t::Real)
    steps = size(S, 2)
    t <= 1 && return S[row, 1]
    t >= steps && return S[row, steps]
    i = floor(Int, t)
    f = t - i
    return (1 - f) * S[row, i] + f * S[row, i + 1]
end

@inline function _lerp_vector(x::AbstractVector, t::Real)
    steps = length(x)
    t <= 1 && return x[1]
    t >= steps && return x[steps]
    i = floor(Int, t)
    f = t - i
    return (1 - f) * x[i] + f * x[i + 1]
end

function _require_success(sol, expected_steps::Int, context::AbstractString)
    SciMLBase.successful_retcode(sol.retcode) ||
        error("$context failed with retcode $(sol.retcode)")
    length(sol.t) == expected_steps ||
        error("$context returned $(length(sol.t)) saved steps; expected $expected_steps")
    return sol
end

"Solve only the training population; no test state exists in this ODE."
function _training_states(S::AbstractMatrix, W::AbstractMatrix, z0::AbstractVector;
                          eps::Real, a::Real, r0::Real,
                          reltol::Real, abstol::Real)
    n, steps = size(S)
    size(W) == (n, n) || throw(DimensionMismatch("W must be $n x $n"))
    length(z0) == 2n || throw(DimensionMismatch("z0 must contain 2n states"))
    rowsum = vec(sum(W, dims=2))
    gbuf = zeros(n)
    function rhs!(dz, z, _, t)
        u = @view z[1:n]
        v = @view z[(n + 1):(2n)]
        du = @view dz[1:n]
        dv = @view dz[(n + 1):(2n)]
        if t <= 1
            gbuf .= @view S[:, 1]
        elseif t >= steps
            gbuf .= @view S[:, steps]
        else
            i = floor(Int, t)
            f = t - i
            @views @. gbuf = (1 - f) * S[:, i] + f * S[:, i + 1]
        end
        mul!(du, W, u)
        @. du = gbuf + u - u^3 / 3 - v + du - rowsum * u
        @. dv = (gbuf * r0 + u - a) * eps
        return nothing
    end
    prob = ODEProblem(rhs!, copy(z0), (0.0, Float64(steps)))
    sol = solve(prob, Tsit5(); saveat=1.0:1.0:steps, save_idxs=1:n,
                reltol=Float64(reltol), abstol=Float64(abstol))
    _require_success(sol, steps, "training-reservoir solve")
    return Matrix{Float64}(Array(sol))
end

function _query_weights(ntrain::Int, sigma::Real, normalization::Symbol,
                        rng::AbstractRNG)
    _check_normalization(normalization)
    w = Float64(sigma) .* pdf.(Normal(), 2 .* rand(rng, ntrain) .- 1)
    if normalization === :row_total && sigma > 0
        total = sum(w)
        total > 0 || error("query coupling has zero weight")
        w .*= sigma / total
    end
    return w
end

"Integrate one query against the cached training trajectory."
function _query_trajectory(S::AbstractMatrix, row::Int,
                           train_states::AbstractMatrix, w::AbstractVector,
                           z0::AbstractVector; eps::Real, a::Real, r0::Real,
                           reltol::Real, abstol::Real)
    steps = size(S, 2)
    size(train_states, 2) == steps ||
        throw(DimensionMismatch("query and cached training trajectories need equal lengths"))
    length(w) == size(train_states, 1) ||
        throw(DimensionMismatch("one query weight is required per training node"))
    length(z0) == 2 || throw(DimensionMismatch("query z0 must have length two"))
    incoming = vec(transpose(train_states) * w)
    total = sum(w)
    function rhs!(dz, z, _, t)
        g = _lerp_row(S, row, t)
        h = _lerp_vector(incoming, t)
        u, v = z
        dz[1] = g + u - u^3 / 3 - v + h - total * u
        dz[2] = (g * r0 + u - a) * eps
        return nothing
    end
    prob = ODEProblem(rhs!, Float64.(z0), (0.0, Float64(steps)))
    sol = solve(prob, Tsit5(); saveat=1.0:1.0:steps,
                reltol=Float64(reltol), abstol=Float64(abstol))
    _require_success(sol, steps, "query solve")
    return [state[1] for state in sol.u]
end

"Standardized cached features for the samples that formed the training reservoir."
function _training_features(model::FHNDigitModel)
    return BaselineUtils.standardize_apply(
        model.training_features, (model.feature_mean, model.feature_scale))
end

function _query_feature_matrix(
        X_query::AbstractMatrix, train_states::AbstractMatrix,
        query_weights::AbstractVector, query_initial_state::AbstractVector;
        nsteps::Int, encoding_seed::Int, eps::Real, a::Real, r0::Real,
        reltol::Real, abstol::Real)
    S_query = encode_digit_sequences(
        X_query; nsteps=nsteps, seed=encoding_seed)
    nquery = size(X_query, 2)
    features = Matrix{Float64}(undef, size(S_query, 2), nquery)
    for query in 1:nquery
        features[:, query] .= _query_trajectory(
            S_query, query, train_states, query_weights, query_initial_state;
            eps, a, r0, reltol, abstol)
    end
    return features
end

"""
Fit an inductive FHN classifier using only `(X_train, y_train)`.

The fitted object is independent of the number, values, ordering, and labels of
future queries. `X_train` uses features x samples orientation.
"""
function fit_fhn_digit_classifier(X_train::AbstractMatrix, y_train::AbstractVector;
                                  nsteps::Int=32, seed::Int=1234,
                                  sigma::Real=0.72,
                                  normalization::Symbol=:row_total,
                                  eps::Real=DEFAULT_EPS, a::Real=DEFAULT_A,
                                  r0::Real=DEFAULT_R0, epochs::Int=500,
                                  reltol::Real=1e-6, abstol::Real=1e-8)
    size(X_train, 2) == length(y_train) ||
        throw(DimensionMismatch("X_train columns must match y_train"))
    length(y_train) >= 2 || throw(ArgumentError("at least two training samples are required"))
    classes = sort(unique(y_train))
    length(classes) >= 2 || throw(ArgumentError("at least two classes are required"))
    _check_normalization(normalization)

    S_train = encode_digit_sequences(X_train; nsteps, seed=seed + 100_000)
    W_train = build_fhn_coupling(length(y_train), sigma;
                                 rng=Xoshiro(seed + 200_000), normalization)
    z0_train = rand(Xoshiro(seed + 300_000), 2length(y_train))
    U_train = _training_states(S_train, W_train, z0_train;
                               eps, a, r0, reltol, abstol)

    # Train the readout on exactly the same virtual-query transformation used
    # at inference time. Using internal reservoir-node states here would create
    # a train/predict distribution mismatch because those nodes have different
    # coupling rows and participate reciprocally in the reservoir.
    query_rng = Xoshiro(seed + 500_000)
    query_weights = _query_weights(
        length(y_train), sigma, normalization, query_rng)
    query_initial_state = rand(query_rng, 2)
    X_features = _query_feature_matrix(
        X_train, U_train, query_weights, query_initial_state;
        nsteps, encoding_seed=seed + 100_000, eps, a, r0, reltol, abstol)
    feature_mean, feature_scale = BaselineUtils.standardize_fit(X_features)
    X_standard = BaselineUtils.standardize_apply(
        X_features, (feature_mean, feature_scale))
    Y = BaselineUtils.onehot(y_train, classes)
    readout = BaselineModels.train_logreg(
        X_standard, Y; epochs, rng=Xoshiro(seed + 400_000))

    return FHNDigitModel(
        U_train, query_weights, query_initial_state,
        X_features, feature_mean, feature_scale, readout, collect(classes), nsteps,
        Float64(sigma), normalization, seed + 100_000,
        Float64(eps), Float64(a), Float64(r0), Float64(reltol), Float64(abstol))
end

"""
Generate unstandardized trajectory features for new digit columns.

Each query is solved as its own two-state ODE. Consequently the result for a
query is independent of every other query in the batch, including the adaptive
solver's step selection.
"""
function fhn_digit_features(model::FHNDigitModel, X_query::AbstractMatrix)
    return _query_feature_matrix(
        X_query, model.train_states,
        model.query_weights, model.query_initial_state;
        nsteps=model.nsteps, encoding_seed=model.encoding_seed,
        eps=model.eps, a=model.a, r0=model.r0,
        reltol=model.reltol, abstol=model.abstol)
end

function predict_fhn_digits(model::FHNDigitModel, X_query::AbstractMatrix)
    X_features = fhn_digit_features(model, X_query)
    X_standard = BaselineUtils.standardize_apply(
        X_features, (model.feature_mean, model.feature_scale))
    return model.classes[BaselineModels.predict_nn(model.readout, X_standard)]
end

function predict_training_digits(model::FHNDigitModel)
    return model.classes[BaselineModels.predict_nn(model.readout, _training_features(model))]
end

end # module FHNDigitClassifier
