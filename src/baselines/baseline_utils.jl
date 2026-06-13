# Shared utilities for the baseline experiments requested by the manuscript
# critique (docs/critique_chaotic_oscillator_networks.md):
#   - data loading for the two datasets used in the paper
#   - standardization, stratified split, stratified k-fold
#   - classification metrics: accuracy, per-class precision/recall/F1,
#     macro-F1, weighted-F1, confusion matrix
#   - paired Wilcoxon signed-rank test for seed-to-seed model comparison
#
# Pure Julia (no Python/sklearn). NOTE: scikit-learn's load_digits() is exactly
# the UCI optdigits test partition shipped in data/digits/optdigits.tes
# (1797 samples x 64 features, 10 classes), so we reproduce the paper's dataset
# without PyCall.

module BaselineUtils

using DelimitedFiles
using CSV
using DataFrames
using Statistics
using Random
using LinearAlgebra

export load_digits, load_drybean, standardize_fit, standardize_apply,
       stratified_split, stratified_kfold, onehot, accuracy, confusion_matrix,
       precision_recall_f1, macro_f1, weighted_f1, classification_report,
       wilcoxon_signed_rank, summarize, silhouette_score, fisher_ratio

# --------------------------------------------------------------- data loading

"Load the digits dataset (== sklearn load_digits): X is (64, 1797), y in 0:9."
function load_digits(; path=joinpath(@__DIR__, "..", "..", "data", "digits", "optdigits.tes"))
    raw = readdlm(path, ',', Int)
    X = Float64.(permutedims(raw[:, 1:end-1]))   # (features, samples)
    y = raw[:, end]
    return X, y
end

"Load the Dry Bean dataset: X is (16, 13611), y is a vector of class strings."
function load_drybean(; path=joinpath(@__DIR__, "..", "..", "data", "DryBeanDataset.csv"))
    df = CSV.read(path, DataFrame)
    X = permutedims(Matrix{Float64}(df[:, 1:end-1]))
    y = String.(df[:, end])
    return X, y
end

# ------------------------------------------------------------- preprocessing

"Fit per-feature mean/std on columns-as-samples matrix X (features, samples)."
function standardize_fit(X::AbstractMatrix)
    mu = vec(mean(X, dims=2))
    sigma = vec(std(X, dims=2))
    sigma[sigma .== 0] .= 1.0
    return (mu, sigma)
end

standardize_apply(X, (mu, sigma)) = (X .- mu) ./ sigma

# ------------------------------------------------------------------ encoding

"One-hot encode labels y given the ordered `classes`; returns (K, N) matrix."
function onehot(y, classes)
    idx = Dict(c => i for (i, c) in enumerate(classes))
    Y = zeros(Float32, length(classes), length(y))
    for (j, c) in enumerate(y)
        Y[idx[c], j] = 1
    end
    return Y
end

# -------------------------------------------------------------------- splits

"Stratified train/test split. Returns (train_idx, test_idx)."
function stratified_split(y, train_ratio; rng=Random.default_rng())
    train_idx = Int[]
    test_idx = Int[]
    for c in unique(y)
        idx = shuffle(rng, findall(==(c), y))
        ntr = round(Int, train_ratio * length(idx))
        append!(train_idx, idx[1:ntr])
        append!(test_idx, idx[ntr+1:end])
    end
    return shuffle(rng, train_idx), shuffle(rng, test_idx)
end

"Stratified k-fold. Returns a vector of (train_idx, test_idx) tuples."
function stratified_kfold(y, k; rng=Random.default_rng())
    folds = [Int[] for _ in 1:k]
    for c in unique(y)
        idx = shuffle(rng, findall(==(c), y))
        for (i, s) in enumerate(idx)
            push!(folds[mod1(i, k)], s)
        end
    end
    return [(reduce(vcat, folds[setdiff(1:k, f)]), folds[f]) for f in 1:k]
end

# ------------------------------------------------------------------- metrics

accuracy(ypred, ytrue) = mean(ypred .== ytrue)

"Confusion matrix C[i,j] = # true class i predicted as class j, over `classes`."
function confusion_matrix(ypred, ytrue, classes)
    idx = Dict(c => i for (i, c) in enumerate(classes))
    K = length(classes)
    C = zeros(Int, K, K)
    for (p, t) in zip(ypred, ytrue)
        C[idx[t], idx[p]] += 1
    end
    return C
end

"Per-class precision, recall, F1 and support from a confusion matrix."
function precision_recall_f1(C::AbstractMatrix)
    K = size(C, 1)
    prec = zeros(K); rec = zeros(K); f1 = zeros(K); support = zeros(Int, K)
    for i in 1:K
        tp = C[i, i]
        fp = sum(C[:, i]) - tp
        fn = sum(C[i, :]) - tp
        prec[i] = tp + fp == 0 ? 0.0 : tp / (tp + fp)
        rec[i]  = tp + fn == 0 ? 0.0 : tp / (tp + fn)
        f1[i]   = prec[i] + rec[i] == 0 ? 0.0 : 2prec[i] * rec[i] / (prec[i] + rec[i])
        support[i] = sum(C[i, :])
    end
    return (precision=prec, recall=rec, f1=f1, support=support)
end

macro_f1(C) = mean(precision_recall_f1(C).f1)

function weighted_f1(C)
    m = precision_recall_f1(C)
    return sum(m.f1 .* m.support) / sum(m.support)
end

"Full report: accuracy, macro-F1, weighted-F1, plus per-class table."
function classification_report(ypred, ytrue, classes)
    C = confusion_matrix(ypred, ytrue, classes)
    m = precision_recall_f1(C)
    return (accuracy=accuracy(ypred, ytrue), macro_f1=mean(m.f1),
            weighted_f1=sum(m.f1 .* m.support) / sum(m.support),
            per_class=m, confusion=C, classes=classes)
end

# --------------------------------------------------------- separability (B7)

# Euclidean pairwise distance matrix for columns-as-samples X (features, N).
function _pairwise_dist(X)
    G = X' * X
    sq = diag(G)
    D2 = sq .+ sq' .- 2 .* G
    return sqrt.(max.(D2, 0.0))
end

"""
Mean silhouette score of samples X (features, N) under `labels`, in [-1, 1].
Higher = classes are more separated in this feature space. Compare reservoir
states vs raw inputs to quantify whether the reservoir improves separability.
"""
function silhouette_score(X, labels)
    N = size(X, 2)
    D = _pairwise_dist(X)
    classes = unique(labels)
    s = zeros(N)
    for i in 1:N
        same = filter(!=(i), findall(==(labels[i]), labels))
        a = isempty(same) ? 0.0 : mean(D[i, same])
        b = Inf
        for c in classes
            c == labels[i] && continue
            b = min(b, mean(D[i, findall(==(c), labels)]))
        end
        denom = max(a, b)
        s[i] = denom == 0 ? 0.0 : (b - a) / denom
    end
    return mean(s)
end

"""
Fisher discriminant ratio tr(S_between)/tr(S_within) for X (features, N).
Higher = larger between-class spread relative to within-class spread, i.e. more
linearly separable.
"""
function fisher_ratio(X, labels)
    mu = vec(mean(X, dims=2))
    Sw = 0.0; Sb = 0.0
    for c in unique(labels)
        idx = findall(==(c), labels)
        Xc = @view X[:, idx]
        muc = vec(mean(Xc, dims=2))
        Sw += sum(abs2, Xc .- muc)
        Sb += length(idx) * sum(abs2, muc .- mu)
    end
    return Sw == 0 ? Inf : Sb / Sw
end

# ------------------------------------------------------------------ stats

"mean and sample std of a vector, returned as a NamedTuple."
summarize(v) = (mean=mean(v), std=length(v) > 1 ? std(v) : 0.0, n=length(v))

"""
Paired Wilcoxon signed-rank test (two-sided), normal approximation with
continuity + tie correction. `x`, `y` are paired measurements (e.g. per-seed
accuracies of two models). Returns (W, z, p). Use to test whether one model
beats another across seeds, as the critique requires before topology claims.
"""
function wilcoxon_signed_rank(x::AbstractVector, y::AbstractVector)
    @assert length(x) == length(y)
    d = x .- y
    keep = d .!= 0
    d = d[keep]
    n = length(d)
    n == 0 && return (W=0.0, z=0.0, p=1.0)
    # rank |d| with average ranks for ties
    order = sortperm(abs.(d))
    ad = abs.(d)[order]
    ranks = zeros(Float64, n)
    i = 1
    while i <= n
        j = i
        while j < n && ad[j+1] == ad[i]
            j += 1
        end
        ranks[i:j] .= (i + j) / 2
        i = j + 1
    end
    signs = sign.(d[order])
    Wpos = sum(ranks[signs .> 0])
    Wneg = sum(ranks[signs .< 0])
    W = min(Wpos, Wneg)
    muW = n * (n + 1) / 4
    # tie correction
    tie_term = 0.0
    i = 1
    while i <= n
        j = i
        while j < n && ad[j+1] == ad[i]
            j += 1
        end
        t = j - i + 1
        tie_term += t^3 - t
        i = j + 1
    end
    sigmaW = sqrt(n * (n + 1) * (2n + 1) / 24 - tie_term / 48)
    z = sigmaW == 0 ? 0.0 : (W - muW + 0.5 * sign(muW - W)) / sigmaW
    p = 2 * (1 - normal_cdf(abs(z)))
    return (W=W, z=z, p=clamp(p, 0.0, 1.0))
end

normal_cdf(z) = 0.5 * (1 + erf_approx(z / sqrt(2)))

# Abramowitz & Stegun 7.1.26 rational approximation of erf.
function erf_approx(x)
    s = sign(x); x = abs(x)
    t = 1 / (1 + 0.3275911x)
    y = 1 - (((((1.061405429t - 1.453152027)t) + 1.421413741)t - 0.284496736)t + 0.254829592)t * exp(-x^2)
    return s * y
end

end # module BaselineUtils
