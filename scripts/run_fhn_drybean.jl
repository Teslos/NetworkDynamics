# Runnable reproduction of the FHN reservoir Dry-Bean classifier, and the source of
# the paper's Dry-Bean confusion matrix.
#
# The original (src/classification/FitzHug-Nagumo-DryBean.jl) cannot run here: it
# uses `@sk_import` (PyCall is not built), the removed NetworkDynamics old API
# (network_dynamics/ODEVertex/StaticEdge), and GLMakie. This script reproduces the
# *method* with the same architecture the digits port uses
# (scripts/run_fhn_digits.jl), which is the same architecture the original used:
#   - Dry-Bean CSV, 16 features standardised per feature, 7 classes
#   - rate spike encoding of each sample's features -> one drive train per sample
#   - one FHN oscillator per sample on a complete graph, diffusive coupling
#   - readout = logistic regression on each node's u-trajectory
#   - stratified 80/20 split
#
# It additionally writes the confusion matrix, which the original produced only as
# a PNG with no stored numbers -- so the paper's figure could not be re-plotted.
#
# Usage:
#   julia --project=. scripts/run_fhn_drybean.jl                # default N=2046
#   julia --project=. scripts/run_fhn_drybean.jl --n 600        # quick subset
#   julia --project=. scripts/run_fhn_drybean.jl --seed 3

include(joinpath(@__DIR__, "..", "src", "baselines", "baseline_utils.jl"))
include(joinpath(@__DIR__, "..", "src", "baselines", "baseline_models.jl"))
include(joinpath(@__DIR__, "..", "src", "utils", "spikerate.jl"))
include(joinpath(@__DIR__, "..", "src", "utils", "drybean.jl"))
using .BaselineUtils, .BaselineModels, .spikerate, .drybean
using OrdinaryDiffEq, LinearAlgebra, Statistics, Random, Distributions, Printf, DelimitedFiles

# ----- args
n_arg = findfirst(==("--n"), ARGS)
const NSAMP = n_arg === nothing ? 2046 : parse(Int, ARGS[n_arg + 1])
seed_arg = findfirst(==("--seed"), ARGS)
const SEED = seed_arg === nothing ? 1234 : parse(Int, ARGS[seed_arg + 1])

# ----- FHN reservoir params (from the original script)
const EPS = 0.05
const A = 0.5
const R0 = 0.5
const SIGMA = 0.006          # sigma_s in FitzHug-Nagumo-DryBean.jl
const NSTEPS = 32            # spike-encoding time steps

println("Loading Dry-Bean...")
db = drybean.read_drybean()
raw = Matrix(permutedims(db))                 # (17, n_samples): 16 features + label row
y_all = string.(raw[17, :])
X_all = Float64.(raw[1:16, :])

# The original drives one node per sample and processes the dataset in batches of
# N (= reservoir size), concatenating the per-batch node trajectories:
#     num_batches = nsamples / N;  uall = vcat(uall, u)
# so the readout is trained on ALL usable samples, not on one reservoir's worth.
# Subsampling to N total was the main reason an earlier version of this port
# overfitted (512 trajectory features against ~1.6k training rows).
rng = Xoshiro(SEED)
const N = NSAMP                                   # reservoir size = samples per batch
perm = shuffle(rng, 1:size(X_all, 2))
const NBATCH = length(perm) ÷ N
use = perm[1:(N * NBATCH)]
Xs = X_all[:, use]
y  = y_all[use]
classes = sort(unique(y))
println("Reservoir N=$N nodes, $NBATCH batches -> $(N*NBATCH) of $(length(perm)) samples")
println("$(length(classes)) classes: ", join(classes, ", "))

# ----- standardise each feature, then map to [0,1] for rate encoding
mu = mean(Xs, dims = 2); sd = std(Xs, dims = 2) .+ 1e-9
# The original feeds the STANDARDISED values straight to spikerate.rate, which
# clips outside [0,1] and treats the value as a Bernoulli probability. So a feature
# is effectively encoded by how far above +0 sigma it sits, saturating at +1 sigma.
# Remapping +-3 sigma onto [0,1] instead (a graded code) changes the representation
# and is not what produced the published numbers, so we match the original.
Z  = (Xs .- mu) ./ sd

# ----- spike encode: (16, N) rates -> drive matrix S (N, 16*NSTEPS)
S = spikerate.rate(permutedims(Z), NSTEPS)    # (NSTEPS, N, 16)
S = permutedims(S, (2, 1, 3))                 # (N, NSTEPS, 16)
S = Float64.(reshape(S, N * NBATCH, NSTEPS * 16))   # (all used samples, 512)
const T = size(S, 2)
println("Drive: $(size(S)) (each node gets a $(T)-length spike train)")

# coupling and drive are rebuilt per batch below
Wc = zeros(N, N); rowsum = zeros(N); Sb = zeros(N, 1)

@inline function drive!(out, t)
    if t <= 1
        @inbounds out .= @view Sb[:, 1]
    else
        i = min(floor(Int, t), T - 1); f = t - i
        @inbounds out .= (1 - f) .* @view(Sb[:, i]) .+ f .* @view(Sb[:, i + 1])
    end
    return out
end

function fhn_rhs!(dz, z, gbuf, t)
    u = @view z[1:N]; v = @view z[N+1:2N]
    du = @view dz[1:N]; dv = @view dz[N+1:2N]
    g = drive!(gbuf, t)
    coupling = Wc * u .- rowsum .* u
    @. du = g + u - u^3 / 3 - v + coupling
    @. dv = (g * R0 + u - A) * EPS
    return nothing
end

println("Solving $NBATCH FHN reservoir batches ($(2N) ODE states each)...")
Xfeat = Matrix{Float64}(undef, T, N * NBATCH)   # filled by index: no rebinding,
                                                # so no soft-scope capture, and no O(n^2) hcat
for b in 1:NBATCH
    global Sb = S[((b - 1) * N + 1):(b * N), :]   # this batch's drive trains
    # fresh coupling weights and initial condition per batch, as in the original
    Wb = [pdf(Normal(), r) for r in (2 .* rand(rng, N, N) .- 1)]
    Wb = SIGMA .* (Wb .+ Wb') ./ 2
    Wb[diagind(Wb)] .= 0
    global Wc = Wb
    global rowsum = vec(sum(Wb, dims = 2))
    z0 = rand(rng, 2N)
    prob = ODEProblem(fhn_rhs!, z0, (0.0, Float64(T)), zeros(N))
    t_solve = @elapsed sol = solve(prob, Tsit5(); saveat = 1.0:1.0:T, save_idxs = 1:N)
    @printf("  batch %d/%d done in %.1f s, retcode=%s
", b, NBATCH, t_solve, sol.retcode)
    Xfeat[:, ((b - 1) * N + 1):(b * N)] = permutedims(Array(sol))
end

# ----- readout: logistic regression on the trajectories, stratified 80/20
tr, te = stratified_split(y, 0.8; rng = Xoshiro(SEED))
sc  = standardize_fit(Xfeat[:, tr])
Xtr = standardize_apply(Xfeat[:, tr], sc)
Xte = standardize_apply(Xfeat[:, te], sc)
Ytr = onehot(y[tr], classes)

# The original trains "the FitzHugh-Nagumo RC last two layers" with Zygote, i.e. a
# small neural readout rather than a linear one. `--logreg` keeps the linear
# readout for comparison.
use_mlp = !("--logreg" in ARGS)
model   = use_mlp ? train_mlp(Xtr, Ytr; hidden = 128, epochs = 500, rng = Xoshiro(SEED)) :
                    train_logreg(Xtr, Ytr; epochs = 500, rng = Xoshiro(SEED))
println("Readout: ", use_mlp ? "MLP (128 hidden)" : "logistic regression")
pred_tr = classes[predict_nn(model, Xtr)]
pred_te = classes[predict_nn(model, Xte)]
rep     = classification_report(pred_te, y[te], classes)
C       = confusion_matrix(pred_te, y[te], classes)

println("\n========== FHN reservoir Dry-Bean classification ==========")
println(@sprintf("Nodes/samples: %d   features (trajectory length): %d", N, T))
println(@sprintf("Train accuracy: %.4f", accuracy(pred_tr, y[tr])))
println(@sprintf("Test  accuracy: %.4f", rep.accuracy))
println(@sprintf("Test  macro-F1: %.4f", rep.macro_f1))
println(@sprintf("RESULT seed=%d N=%d test_acc=%.4f macro_f1=%.4f",
                 SEED, N, rep.accuracy, rep.macro_f1))

# ----- persist the confusion matrix as NUMBERS so the figure can be re-plotted
outdir = joinpath(@__DIR__, "..", "results", "confusion_matrices")
isdir(outdir) || mkpath(outdir)
open(joinpath(outdir, "fhn_drybean_confusion_matrix.txt"), "w") do io
    println(io, "# FHN reservoir, Dry-Bean, seed=$SEED N=$N")
    println(io, "# rows = true class, cols = predicted class")
    println(io, "# classes: ", join(classes, ","))
    writedlm(io, C)
end
println("Confusion matrix written to results/confusion_matrices/fhn_drybean_confusion_matrix.txt")
