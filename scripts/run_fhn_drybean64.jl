# FHN reservoir Dry-Bean classifier, 64-NODE variant.
#
# The committed original (src/classification/FitzHug-Nagumo-DryBean.jl) is
# internally inconsistent: it builds a 2046-node reservoir and 512-long trajectory
# features, but its readout is `Lux.Dense(64, 512, tanh)`, i.e. 64 inputs. It also
# cannot run (PyCall, removed NetworkDynamics API). The manuscript says "our
# reservoir is made up of 64 FitzHugh-Nagumo oscillators", and its exploratory
# section drives only the first 16 nodes:
#     g0v = [i <= 16 ? spike_train[:, i] : g0 for i in 1:nv(g)]
#
# That configuration is self-consistent and matches both the readout width and the
# paper: a 64-node reservoir, 16 of whose nodes receive the sample's features, read
# out by ONE scalar per node -> 64 features. This script implements it, so the
# published ~92% can be tested against a configuration that could actually produce
# it. Each sample is a separate (small) ODE solve.
#
# Usage:
#   julia --project=. scripts/run_fhn_drybean64.jl --n 3000     # subset, quick
#   julia --project=. scripts/run_fhn_drybean64.jl              # all samples

include(joinpath(@__DIR__, "..", "src", "baselines", "baseline_utils.jl"))
include(joinpath(@__DIR__, "..", "src", "baselines", "baseline_models.jl"))
include(joinpath(@__DIR__, "..", "src", "utils", "spikerate.jl"))
include(joinpath(@__DIR__, "..", "src", "utils", "drybean.jl"))
using .BaselineUtils, .BaselineModels, .spikerate, .drybean
using OrdinaryDiffEq, LinearAlgebra, Statistics, Random, Distributions, Printf, DelimitedFiles

n_arg    = findfirst(==("--n"), ARGS)
seed_arg = findfirst(==("--seed"), ARGS)
const NSAMP = n_arg    === nothing ? 0    : parse(Int, ARGS[n_arg + 1])     # 0 = all
const SEED  = seed_arg === nothing ? 1234 : parse(Int, ARGS[seed_arg + 1])

const NRES   = 64      # reservoir nodes, per the manuscript
const NIN    = 16      # driven nodes = number of Dry-Bean features
const EPS    = 0.05
const AFHN   = 0.5
const R0     = 0.5
# sigma_s = 0.006 in the original was set for a 2046-node graph, where total
# in-coupling per node is ~sigma*(N-1)*E[w] ~ 3.7. At N=64 the same value gives
# ~0.11, i.e. 33x weaker: the undriven nodes barely respond and their features
# carry little. Matching total in-coupling suggests sigma ~ 0.006*2045/63 ~ 0.19.
sig_arg = findfirst(a -> startswith(a, "--sigma="), ARGS)
const SIGMA  = sig_arg === nothing ? 0.006 : parse(Float64, ARGS[sig_arg][9:end])
const NSTEPS = 32

println("Loading Dry-Bean...")
db = drybean.read_drybean()
raw = Matrix(permutedims(db))
y_all = string.(raw[17, :])
X_all = Float64.(raw[1:16, :])

rng = Xoshiro(SEED)
perm = shuffle(rng, 1:size(X_all, 2))
nuse = NSAMP == 0 ? length(perm) : min(NSAMP, length(perm))
use  = perm[1:nuse]
Xs, y = X_all[:, use], y_all[use]
classes = sort(unique(y))
println("$nuse samples, $(length(classes)) classes; reservoir $NRES nodes, $NIN driven, sigma=$SIGMA")

mu = mean(Xs, dims = 2); sd = std(Xs, dims = 2) .+ 1e-9
Z  = (Xs .- mu) ./ sd                      # standardised, as the original feeds it

# spike-encode: (16, nuse) -> (NSTEPS, nuse, 16)
Sp = spikerate.rate(permutedims(Z), NSTEPS)
Sp = Float64.(permutedims(Sp, (2, 3, 1)))  # (nuse, 16, NSTEPS)

# reservoir coupling: complete graph over NRES nodes, diffusive, positive weights
Wc = [pdf(Normal(), r) for r in (2 .* rand(rng, NRES, NRES) .- 1)]
Wc = SIGMA .* (Wc .+ Wc') ./ 2
Wc[diagind(Wc)] .= 0
const rowsum = vec(sum(Wc, dims = 2))

gvec = zeros(NRES)
function rhs!(dz, z, drive, t)
    u = @view z[1:NRES]; v = @view z[NRES+1:2NRES]
    du = @view dz[1:NRES]; dv = @view dz[NRES+1:2NRES]
    i = clamp(floor(Int, t) + 1, 1, NSTEPS)
    @inbounds for k in 1:NRES
        gvec[k] = k <= NIN ? drive[k, i] : 0.0      # only the first NIN nodes are driven
    end
    coupling = Wc * u .- rowsum .* u
    @. du = gvec + u - u^3 / 3 - v + coupling
    @. dv = (gvec * R0 + u - AFHN) * EPS
    return nothing
end

println("Solving $nuse per-sample reservoir runs ($(2NRES) states each)...")
Feat = Matrix{Float64}(undef, NRES, nuse)          # one feature per node
z0 = zeros(2NRES)
t_all = @elapsed for s in 1:nuse
    drive = @view Sp[s, :, :]                       # (16, NSTEPS)
    prob = ODEProblem(rhs!, z0, (0.0, Float64(NSTEPS)), drive)
    sol = solve(prob, Tsit5(); saveat = 1.0:1.0:NSTEPS, save_idxs = 1:NRES,
                abstol = 1e-6, reltol = 1e-6)
    U = Array(sol)                                  # (NRES, NSTEPS)
    Feat[:, s] = vec(mean(U, dims = 2))             # time-averaged activity per node
    s % 2000 == 0 && @printf("  %d/%d\n", s, nuse)
end
@printf("  done in %.1f s\n", t_all)

tr, te = stratified_split(y, 0.9; rng = Xoshiro(SEED))   # original used train_ratio = 0.9
sc  = standardize_fit(Feat[:, tr])
Xtr = standardize_apply(Feat[:, tr], sc)
Xte = standardize_apply(Feat[:, te], sc)
Ytr = onehot(y[tr], classes)

model   = train_mlp(Xtr, Ytr; hidden = 512, epochs = 500, rng = Xoshiro(SEED))
pred_tr = classes[predict_nn(model, Xtr)]
pred_te = classes[predict_nn(model, Xte)]
rep     = classification_report(pred_te, y[te], classes)
C       = confusion_matrix(pred_te, y[te], classes)

println("\n===== FHN reservoir (64 nodes, 1 feature/node) Dry-Bean =====")
@printf("Train accuracy: %.4f\n", accuracy(pred_tr, y[tr]))
@printf("Test  accuracy: %.4f\n", rep.accuracy)
@printf("Test  macro-F1: %.4f\n", rep.macro_f1)
@printf("RESULT seed=%d nodes=%d nsamples=%d test_acc=%.4f macro_f1=%.4f\n",
        SEED, NRES, nuse, rep.accuracy, rep.macro_f1)

outdir = joinpath(@__DIR__, "..", "results", "confusion_matrices")
isdir(outdir) || mkpath(outdir)
open(joinpath(outdir, "fhn_drybean64_confusion_matrix.txt"), "w") do io
    println(io, "# FHN reservoir 64 nodes / 16 driven, Dry-Bean, seed=$SEED n=$nuse")
    println(io, "# rows = true class, cols = predicted class")
    println(io, "# classes: ", join(classes, ","))
    writedlm(io, C)
end
println("Confusion matrix -> results/confusion_matrices/fhn_drybean64_confusion_matrix.txt")
