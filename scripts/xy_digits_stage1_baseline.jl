# Softmax logistic-regression baseline for EP-XY digits Stage 1, on the SAME
# train/test split the XY run used (identical SEED and selection order), so the
# numbers pair directly with the XY accuracies in scripts/xy_digits_stage1.jl.
# Fast (no ODEs) -- used to recompute the baseline after fixing the argmax bug
# without re-running the ~2 h XY training.
#
# Run: julia --project=. scripts/xy_digits_stage1_baseline.jl

using LinearAlgebra, Statistics, Random, Printf, DelimitedFiles

const SEED       = 1
const TASKS      = [[3, 5, 8], [0, 1, 2, 3, 4]]
const N_TRAIN_PC = 60
const N_TEST_PC  = 30

raw = readdlm(joinpath(@__DIR__, "..", "data", "digits", "optdigits.tes"), ',', Int)
const X_ALL = Float64.(raw[:, 1:64]); const Y_ALL = raw[:, 65]

function logreg_accuracy(Xtr, ytr, Xte, yte, n_class; iters=600, lr=0.5, l2=1e-3)
    n, d = size(Xtr)
    W = zeros(d, n_class); b = zeros(n_class)
    Y = zeros(n, n_class); for i in 1:n; Y[i, ytr[i]] = 1.0; end
    for _ in 1:iters
        logits = Xtr * W .+ b'
        e = exp.(logits .- maximum(logits, dims=2)); P = e ./ sum(e, dims=2)
        G = (P .- Y) ./ n
        W .-= lr .* (Xtr' * G .+ l2 .* W)
        b .-= lr .* vec(sum(G, dims=1))
    end
    logits = Xte * W .+ b'
    pred = [argmax(@view logits[i, :]) for i in 1:size(Xte, 1)]
    return mean(pred .== yte)
end

# Replicates the split in xy_digits_stage1.jl::run_task exactly.
function split_for(classes)
    rng = MersenneTwister(SEED)
    classcol = Dict(c => j for (j, c) in enumerate(classes))
    tr = Int[]; te = Int[]
    for c in classes
        ci = shuffle(rng, findall(==(c), Y_ALL))
        append!(tr, ci[1:N_TRAIN_PC]); append!(te, ci[N_TRAIN_PC+1:N_TRAIN_PC+N_TEST_PC])
    end
    shuffle!(rng, tr); shuffle!(rng, te)
    Xtr = X_ALL[tr, :] ./ 16.0; Xte = X_ALL[te, :] ./ 16.0
    ytr = [classcol[c] for c in Y_ALL[tr]]; yte = [classcol[c] for c in Y_ALL[te]]
    return Xtr, ytr, Xte, yte
end

println("Logreg baseline (corrected), same split as the XY run:\n")
@printf("%-18s | %-7s %-9s\n", "task", "chance", "logreg test")
println("-"^40)
for classes in TASKS
    Xtr, ytr, Xte, yte = split_for(classes)
    acc = logreg_accuracy(Xtr, ytr, Xte, yte, length(classes))
    @printf("%-18s | %-7.3f %.3f\n", string(classes), 1/length(classes), acc)
end
