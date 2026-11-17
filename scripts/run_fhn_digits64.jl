# Standard spike-encoded FHN reservoir classifier for the sklearn/UCI 8x8 digits
# dataset (1797 samples x 64 pixels, 10 classes, 80/20 stratified split).
#
# Unlike the sample-per-oscillator convention in `run_fhn_digits.jl` (where
# N = 1438 oscillators represent training samples), this script follows the
# standard Reservoir Computing convention (matching `run_fhn_drybean64.jl`):
#   - The reservoir has N_res = 64 coupled FHN oscillators (one per pixel).
#   - For each sample independently, the 64 pixels (in [0, 16], scaled to [0, 1])
#     are rate-encoded into 64 parallel spike trains of length T = 32 steps.
#   - Oscillator k (k = 1..64) is driven by pixel k's 32-step spike train, while
#     diffusive coupling W mixes the 64 oscillators spatially.
#   - Each sample is an independent 64-oscillator ODE solve over t in (0, 32).
#
# Usage:
#   julia --project=. scripts/run_fhn_digits64.jl
#   julia --project=. scripts/run_fhn_digits64.jl --seeds 10

include(joinpath(@__DIR__, "..", "src", "baselines", "baseline_utils.jl"))
include(joinpath(@__DIR__, "..", "src", "baselines", "baseline_models.jl"))
include(joinpath(@__DIR__, "..", "src", "utils", "spikerate.jl"))
using .BaselineUtils, .BaselineModels, .spikerate
using OrdinaryDiffEq, LinearAlgebra, Statistics, Random, Distributions, Printf

function option(name::String, default, parse_value)
    index = findfirst(==(name), ARGS)
    index === nothing && return default
    index < length(ARGS) || error("$name requires a value")
    return parse_value(ARGS[index + 1])
end

const NSEEDS      = option("--seeds", 1, x -> parse(Int, x))
const SEED        = option("--seed", 1234, x -> parse(Int, x))
const NRES        = option("--nres", 64, x -> parse(Int, x))
const NSTEPS      = option("--nsteps", 32, x -> parse(Int, x))
const SIGMA       = option("--sigma", 0.1, x -> parse(Float64, x))
const NORM        = Symbol(option("--normalization", "row_total", identity))
const TRAIN_RATIO = option("--train-ratio", 0.8, x -> parse(Float64, x))
const EPOCHS      = option("--epochs", 500, x -> parse(Int, x))

const NPIX = 64
const EPS  = 0.05
const AFHN = 0.5
const R0   = 0.5

NRES >= NPIX || error("--nres must be at least $NPIX")
NORM in (:row_total, :none) ||
    error("--normalization must be row_total or none (got $NORM)")

# Ridge strengths tried for the closed-form readout, and the share of the
# training set held out to choose among them. The test set is never used.
const RIDGE_LAMBDAS      = [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]
const RIDGE_VAL_FRACTION = 0.2

println("Loading sklearn/UCI 8x8 digits (optdigits.tes)...")
X_all, y_all = BaselineUtils.load_digits()
classes = sort(unique(y_all))
nsamples = size(X_all, 2)
println("Dataset: $nsamples samples, $(size(X_all, 1)) pixels, $(length(classes)) classes")

function build_coupling(n::Int, sigma::Float64, rng::AbstractRNG, normalization::Symbol)
    raw = 2 .* rand(rng, n, n) .- 1
    W = [pdf(Normal(), r) for r in raw]
    W = sigma .* (W .+ W') ./ 2
    W[diagind(W)] .= 0.0
    if normalization === :row_total && sigma > 0
        for i in 1:n
            s = sum(@view W[i, :])
            s > 0 && (@views W[i, :] .*= sigma / s)
        end
    end
    return W
end

function simulate_reservoir(X::AbstractMatrix, W::Matrix{Float64};
                            nsteps::Int=NSTEPS, seed::Int=SEED)
    d, nuse = size(X)
    nres = size(W, 1)
    rowsum = vec(sum(W, dims=2))

    # Scale pixel intensities [0, 16] -> [0, 1] and rate-encode into (nsteps, nuse, 64)
    probs = permutedims(X ./ 16.0) # (nuse, 64)
    spikes = spikerate.rate(probs, nsteps; rng=Xoshiro(seed + 100_000))
    Sp = Float64.(permutedims(spikes, (2, 3, 1))) # (nuse, 64, nsteps)

    Feat_mean = Matrix{Float64}(undef, nres, nuse)
    Feat_stat = Matrix{Float64}(undef, 4 * nres, nuse)
    Feat_traj = Matrix{Float64}(undef, nres * nsteps, nuse)

    gvec = zeros(nres)
    coupling = zeros(nres)
    z0 = zeros(2 * nres)

    function rhs!(dz, z, drive, t)
        u = @view z[1:nres]
        v = @view z[(nres + 1):(2 * nres)]
        du = @view dz[1:nres]
        dv = @view dz[(nres + 1):(2 * nres)]
        idx = clamp(floor(Int, t) + 1, 1, nsteps)
        @inbounds for k in 1:nres
            gvec[k] = k <= d ? drive[k, idx] : 0.0
        end
        mul!(coupling, W, u)
        @. du = gvec + u - u^3 / 3 - v + coupling - rowsum * u
        @. dv = (gvec * R0 + u - AFHN) * EPS
        return nothing
    end

    for s in 1:nuse
        drive = @view Sp[s, :, :]
        prob = ODEProblem(rhs!, z0, (0.0, Float64(nsteps)), drive)
        sol = solve(prob, Tsit5(); saveat=1.0:1.0:nsteps, save_idxs=1:nres,
                    abstol=1e-6, reltol=1e-6)
        U = Array(sol) # (nres, nsteps)
        m_u = vec(mean(U, dims=2))
        s_u = vec(std(U, dims=2))
        end_u = @view U[:, end]
        max_u = vec(maximum(U, dims=2))

        Feat_mean[:, s] .= m_u
        Feat_stat[1:nres, s] .= m_u
        Feat_stat[(nres + 1):(2 * nres), s] .= end_u
        Feat_stat[(2 * nres + 1):(3 * nres), s] .= s_u
        Feat_stat[(3 * nres + 1):(4 * nres), s] .= max_u
        Feat_traj[:, s] .= vec(U)
    end

    return Feat_mean, Feat_stat, Feat_traj
end

# Closed-form ridge readout on features F (features, samples) with a bias row.
function ridge_fit(F::AbstractMatrix, Y::AbstractMatrix, lambda::Float64)
    Fb = vcat(F, ones(1, size(F, 2)))
    return (Float64.(Y) * Fb') / (Fb * Fb' + lambda * I)
end

ridge_predict(Wout, F) = vec(map(argmax, eachcol(Wout * vcat(F, ones(1, size(F, 2))))))

# Choose the ridge strength on a stratified validation split carved from the
# TRAINING indices. A fixed value does not transfer across feature sets: with
# 2048 trajectory features and 1438 training samples, lambda = 1 nearly
# interpolates the one-hot targets (test ~0.51) where lambda = 1e4 gives ~0.93.
function select_ridge_lambda(Feat::AbstractMatrix, y::AbstractVector,
                             train_idx::Vector{Int}; seed::Int=SEED)
    fit_in, val_in = BaselineUtils.stratified_split(
        y[train_idx], 1 - RIDGE_VAL_FRACTION; rng=Xoshiro(seed + 300_000))
    fit_idx, val_idx = train_idx[fit_in], train_idx[val_in]
    sc = BaselineUtils.standardize_fit(Feat[:, fit_idx])
    Xfit = BaselineUtils.standardize_apply(Feat[:, fit_idx], sc)
    Xval = BaselineUtils.standardize_apply(Feat[:, val_idx], sc)
    Yfit = BaselineUtils.onehot(y[fit_idx], classes)
    best_lambda, best_acc = first(RIDGE_LAMBDAS), -Inf
    for lambda in RIDGE_LAMBDAS
        pred = classes[ridge_predict(ridge_fit(Xfit, Yfit, lambda), Xval)]
        acc = BaselineUtils.accuracy(pred, y[val_idx])
        if acc > best_acc
            best_lambda, best_acc = lambda, acc
        end
    end
    return best_lambda
end

function evaluate_features(Feat::AbstractMatrix, y::AbstractVector,
                           train_idx::Vector{Int}, test_idx::Vector{Int};
                           seed::Int=SEED)
    sc = BaselineUtils.standardize_fit(Feat[:, train_idx])
    Xtr = BaselineUtils.standardize_apply(Feat[:, train_idx], sc)
    Xte = BaselineUtils.standardize_apply(Feat[:, test_idx], sc)
    Ytr = BaselineUtils.onehot(y[train_idx], classes)

    # 1. Standard RC linear readout (multinomial logistic regression)
    logreg = BaselineModels.train_logreg(Xtr, Ytr; epochs=EPOCHS, rng=Xoshiro(seed + 400_000))
    pred_tr_lin = classes[BaselineModels.predict_nn(logreg, Xtr)]
    pred_te_lin = classes[BaselineModels.predict_nn(logreg, Xte)]
    rep_lin = BaselineUtils.classification_report(pred_te_lin, y[test_idx], classes)

    # 2. Closed-form Ridge readout: lambda chosen on a validation split of the
    #    training set, then refit on all training samples.
    ridge_lambda = select_ridge_lambda(Feat, y, train_idx; seed=seed)
    Wout = ridge_fit(Xtr, Ytr, ridge_lambda)
    pred_te_ridge = classes[ridge_predict(Wout, Xte)]
    rep_ridge = BaselineUtils.classification_report(pred_te_ridge, y[test_idx], classes)

    # 3. 1-hidden-layer MLP readout (run_fhn_drybean64.jl uses the same model
    #    with 512 hidden units)
    mlp = BaselineModels.train_mlp(Xtr, Ytr; hidden=256, epochs=EPOCHS, rng=Xoshiro(seed + 500_000))
    pred_tr_mlp = classes[BaselineModels.predict_nn(mlp, Xtr)]
    pred_te_mlp = classes[BaselineModels.predict_nn(mlp, Xte)]
    rep_mlp = BaselineUtils.classification_report(pred_te_mlp, y[test_idx], classes)

    return (
        lin_train = BaselineUtils.accuracy(pred_tr_lin, y[train_idx]),
        lin_test  = rep_lin.accuracy,
        lin_f1    = rep_lin.macro_f1,
        ridge_test = rep_ridge.accuracy,
        ridge_f1   = rep_ridge.macro_f1,
        ridge_lambda = ridge_lambda,
        mlp_train = BaselineUtils.accuracy(pred_tr_mlp, y[train_idx]),
        mlp_test  = rep_mlp.accuracy,
        mlp_f1    = rep_mlp.macro_f1,
    )
end

seeds_to_run = NSEEDS == 1 ? [SEED] : collect(1:NSEEDS)

acc_mean_lin = Float64[]; acc_mean_mlp = Float64[]
acc_stat_lin = Float64[]; acc_stat_mlp = Float64[]
acc_traj_lin = Float64[]; acc_traj_ridge = Float64[]; acc_traj_mlp = Float64[]

for s in seeds_to_run
    t0 = time()
    rng_w = Xoshiro(s + 200_000)
    W = build_coupling(NRES, SIGMA, rng_w, NORM)
    Feat_mean, Feat_stat, Feat_traj = simulate_reservoir(X_all, W; nsteps=NSTEPS, seed=s)

    train_idx, test_idx = BaselineUtils.stratified_split(
        y_all, TRAIN_RATIO; rng=Xoshiro(s + 1))

    r_mean = evaluate_features(Feat_mean, y_all, train_idx, test_idx; seed=s)
    r_stat = evaluate_features(Feat_stat, y_all, train_idx, test_idx; seed=s)
    r_traj = evaluate_features(Feat_traj, y_all, train_idx, test_idx; seed=s)

    push!(acc_mean_lin, r_mean.lin_test)
    push!(acc_mean_mlp, r_mean.mlp_test)
    push!(acc_stat_lin, r_stat.lin_test)
    push!(acc_stat_mlp, r_stat.mlp_test)
    push!(acc_traj_lin, r_traj.lin_test)
    push!(acc_traj_ridge, r_traj.ridge_test)
    push!(acc_traj_mlp, r_traj.mlp_test)

    @printf("[seed=%d, %.1fs] Mean(64d): lin=%.4f mlp=%.4f | Stats(256d): lin=%.4f mlp=%.4f | Traj(2048d): lin=%.4f ridge=%.4f (lambda=%g) mlp=%.4f\n",
            s, time() - t0,
            r_mean.lin_test, r_mean.mlp_test,
            r_stat.lin_test, r_stat.mlp_test,
            r_traj.lin_test, r_traj.ridge_test, r_traj.ridge_lambda, r_traj.mlp_test)
end

if length(seeds_to_run) > 1
    println("\n========== Summary over $(length(seeds_to_run)) seeds (mean ± std) ==========")
    @printf("Time-averaged node state (64 features)   - Linear: %.4f ± %.4f | MLP: %.4f ± %.4f\n",
            mean(acc_mean_lin), std(acc_mean_lin), mean(acc_mean_mlp), std(acc_mean_mlp))
    @printf("Node summary statistics  (256 features)  - Linear: %.4f ± %.4f | MLP: %.4f ± %.4f\n",
            mean(acc_stat_lin), std(acc_stat_lin), mean(acc_stat_mlp), std(acc_stat_mlp))
    @printf("Full node trajectories   (2048 features) - Linear: %.4f ± %.4f | Ridge: %.4f ± %.4f | MLP: %.4f ± %.4f\n",
            mean(acc_traj_lin), std(acc_traj_lin), mean(acc_traj_ridge), std(acc_traj_ridge), mean(acc_traj_mlp), std(acc_traj_mlp))
end
