# Inductive FHN-reservoir classifier for the sklearn/UCI 8x8 digits dataset.
#
# Unlike the historical implementation, test samples are never nodes in the
# training ODE. The training population is solved once and cached. Each query is
# then integrated as an independent two-state FHN oscillator driven by:
#   1. its own deterministic rate encoding; and
#   2. a fixed weighted projection of the cached training trajectories.
#
# This gives an actual fit/predict boundary: the fitted model is independent of
# the number, ordering, and values of test samples, and predicting a sample does
# not require re-solving the training reservoir.
#
# Usage:
#   julia --project=. scripts/run_fhn_digits.jl
#   julia --project=. scripts/run_fhn_digits.jl --quick
#   julia --project=. scripts/run_fhn_digits.jl --n 300 --nsteps 4
#   julia --project=. scripts/run_fhn_digits.jl --seed 7 --sigma 0.72
#   julia --project=. scripts/run_fhn_digits.jl --normalization per_edge
#
# `row_total` is the paper-safe default: every oscillator receives total
# coupling `sigma`, independent of reservoir size. `per_edge` is retained only
# as an explicit ablation matching the historical size-dependent convention.

include(joinpath(@__DIR__, "..", "src", "baselines", "fhn_digit_classifier.jl"))
using .FHNDigitClassifier
using .FHNDigitClassifier.BaselineUtils
using Random
using Statistics
using Printf

function option(name::String, default, parse_value)
    index = findfirst(==(name), ARGS)
    index === nothing && return default
    index < length(ARGS) || error("$name requires a value")
    return parse_value(ARGS[index + 1])
end

const QUICK = "--quick" in ARGS
const NSAMP = option("--n", QUICK ? 300 : 1797, x -> parse(Int, x))
const NSTEPS = option("--nsteps", QUICK ? 4 : 32, x -> parse(Int, x))
const SEED = option("--seed", 1234, x -> parse(Int, x))
const SIGMA = option("--sigma", 0.72, x -> parse(Float64, x))
const TRAIN_RATIO = option("--train-ratio", 0.8, x -> parse(Float64, x))
const EPOCHS = option("--epochs", QUICK ? 100 : 500, x -> parse(Int, x))
const NORMALIZATION = Symbol(option("--normalization", "row_total", identity))

NSAMP > 1 || error("--n must be at least 2")
NSTEPS > 0 || error("--nsteps must be positive")
0 < TRAIN_RATIO < 1 || error("--train-ratio must lie strictly between 0 and 1")
NORMALIZATION in (:row_total, :per_edge) ||
    error("--normalization must be row_total or per_edge")

println("Loading digits...")
X_all, y_all = load_digit_data()
NSAMP <= size(X_all, 2) || error("--n=$NSAMP exceeds the $(size(X_all, 2))-sample dataset")

# Subset selection and splitting are deterministic. Only training columns are
# passed to fit_fhn_digit_classifier; the model cannot inspect the test set.
selected = shuffle(Xoshiro(SEED), 1:size(X_all, 2))[1:NSAMP]
X = X_all[:, selected]
y = y_all[selected]
train_idx, test_idx = BaselineUtils.stratified_split(
    y, TRAIN_RATIO; rng=Xoshiro(SEED + 1))
X_train, y_train = X[:, train_idx], y[train_idx]
X_test, y_test = X[:, test_idx], y[test_idx]

println("FHN digit classifier: $(length(train_idx)) train / $(length(test_idx)) test, ",
        "nsteps=$NSTEPS, sigma=$SIGMA, normalization=$NORMALIZATION, seed=$SEED")
println("Fitting and caching the train-only reservoir...")
fit_seconds = @elapsed model = fit_fhn_digit_classifier(
    X_train, y_train; nsteps=NSTEPS, seed=SEED, sigma=SIGMA,
    normalization=NORMALIZATION, epochs=EPOCHS)

println("Predicting independent test queries from the cached reservoir...")
query_seconds = @elapsed pred_test = predict_fhn_digits(model, X_test)
pred_train = predict_training_digits(model)
classes = sort(unique(y_train))
report = BaselineUtils.classification_report(pred_test, y_test, classes)
train_accuracy = BaselineUtils.accuracy(pred_train, y_train)

println("\n========== Inductive FHN digit classification ==========")
println(@sprintf("Training reservoir: %d nodes   feature length: %d",
                 length(train_idx), NSTEPS * size(X, 1)))
println(@sprintf("Train-only fit/cache time: %.1f s", fit_seconds))
println(@sprintf("Independent query time: %.1f s total (%.3f s/sample)",
                 query_seconds, query_seconds / length(test_idx)))
println(@sprintf("Train accuracy: %.4f", train_accuracy))
println(@sprintf("Test  accuracy: %.4f", report.accuracy))
println(@sprintf("Test  macro-F1: %.4f", report.macro_f1))
println(@sprintf(
    "RESULT mode=inductive quick=%s seed=%d N=%d n_train=%d n_test=%d nsteps=%d sigma=%.6g normalization=%s test_acc=%.4f macro_f1=%.4f fit_s=%.1f query_s=%.1f",
    string(QUICK), SEED, NSAMP, length(train_idx), length(test_idx), NSTEPS, SIGMA,
    string(NORMALIZATION), report.accuracy, report.macro_f1,
    fit_seconds, query_seconds))
