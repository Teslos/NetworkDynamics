# Faithful, runnable reproduction of the FHN reservoir digit classifier from
# src/classification/FitzHug-Nagumo-MNIST-Ridge.jl, to check the paper's 88%.
#
# The original cannot run on this machine: it uses the removed NetworkDynamics
# old API (network_dynamics/ODEVertex/StaticEdge; installed version is 0.9.7),
# `@sk_import load_digits` (PyCall not built), and references undefined globals.
# This script reproduces the *method* exactly:
#   - digits from optdigits.tes (== sklearn load_digits), pixels/16
#   - rate spike encoding, 32 steps -> 32*8*8 = 2048-length drive per digit
#   - a complete-graph FHN reservoir with N = N_samples nodes, each node driven
#     by one digit, diffusive coupling of strength sigma=0.72, eps=0.05, a=0.5
#   - readout = logistic regression on each node's u-trajectory (2048 features)
#   - 80/20 stratified split (1437/360, matching the paper)
#
# The all-to-all diffusive coupling is written as a dense matvec W*u, so the
# 1797-node / 3.2M-edge solve is cheap.
#
# Usage:
#   julia --project=. scripts/run_fhn_digits.jl           # full N=1797
#   julia --project=. scripts/run_fhn_digits.jl --n 300   # quick subset

include(joinpath(@__DIR__, "..", "src", "baselines", "baseline_utils.jl"))
include(joinpath(@__DIR__, "..", "src", "baselines", "baseline_models.jl"))
include(joinpath(@__DIR__, "..", "src", "utils", "spikerate.jl"))
using .BaselineUtils, .BaselineModels, .spikerate
using OrdinaryDiffEq, LinearAlgebra, Statistics, Random, Distributions, Printf, DelimitedFiles

# ----- args
n_arg = findfirst(==("--n"), ARGS)
const NSAMP = n_arg === nothing ? 1797 : parse(Int, ARGS[n_arg + 1])
const SEED = 1234

# ----- FHN reservoir params (from the original script)
const EPS = 0.05
const A = 0.5
const R0 = 0.5
const SIGMA = 0.72
const NSTEPS = 32           # spike-encoding time steps

# ----- data: optdigits.tes == sklearn load_digits, as (samples, 8, 8)
function load_digit_images(; path=joinpath(@__DIR__, "..", "data", "digits", "optdigits.tes"))
    raw = readdlm(path, ',', Int)
    imgs = reshape(permutedims(Float64.(raw[:, 1:64])), 8, 8, :)   # (8,8,N)
    imgs = permutedims(imgs, (3, 1, 2))                            # (N,8,8)
    return imgs, raw[:, 65]
end

println("Loading digits...")
imgs_all, y_all = load_digit_images()
rng = Xoshiro(SEED)
idx = shuffle(rng, 1:size(imgs_all, 1))[1:NSAMP]
imgs = imgs_all[idx, :, :]
y = y_all[idx]
N = NSAMP
println("Using N=$N digit-nodes (classes: $(sort(unique(y))))")

# ----- spike encode: (N,8,8) pixels/16 -> drive matrix S (N, 2048)
x = imgs ./ 16.0
S = spikerate.rate(x, NSTEPS)                       # (32, N, 8, 8)
S = permutedims(S, (2, 1, 3, 4))                    # (N, 32, 8, 8)
S = Float64.(reshape(S, N, NSTEPS * 64))            # (N, 2048)
const T = size(S, 2)
println("Drive: $(size(S)) (each node gets a $(T)-length spike train)")

# ----- coupling: complete graph, diffusive, weights ~ sigma * Normal-pdf(U[-1,1])
Wc = [pdf(Normal(), r) for r in (2 .* rand(rng, N, N) .- 1)]
Wc = SIGMA .* (Wc .+ Wc') ./ 2
Wc[diagind(Wc)] .= 0
const rowsum = vec(sum(Wc, dims=2))

# ----- vectorized input drive g_i(t) by linear interpolation across columns
@inline function drive!(out, t)
    if t <= 1
        @inbounds out .= @view S[:, 1]
    else
        i = min(floor(Int, t), T - 1); f = t - i
        @inbounds out .= (1 - f) .* @view(S[:, i]) .+ f .* @view(S[:, i + 1])
    end
    return out
end

# ----- FHN reservoir RHS (u = z[1:N], v = z[N+1:2N])
function fhn_rhs!(dz, z, gbuf, t)
    u = @view z[1:N]; v = @view z[N+1:2N]
    du = @view dz[1:N]; dv = @view dz[N+1:2N]
    g = drive!(gbuf, t)
    coupling = Wc * u .- rowsum .* u                 # diffusive, dense matvec
    @. du = g + u - u^3 / 3 - v + coupling
    @. dv = (g * R0 + u - A) * EPS
    return nothing
end

println("Solving FHN reservoir ($(2N) ODE states)...")
z0 = rand(rng, 2N)
gbuf = zeros(N)
prob = ODEProblem(fhn_rhs!, z0, (0.0, Float64(T)), gbuf)
t_solve = @elapsed sol = solve(prob, Tsit5(); saveat=1.0:1.0:T, save_idxs=1:N)
println(@sprintf("  solve done in %.1f s, retcode=%s", t_solve, sol.retcode))

U = Array(sol)                                       # (N, T) node u-trajectories
Xfeat = permutedims(U)                               # (T, N): col j = sample j features (T-length trajectory)

# ----- readout: logistic regression on the reservoir trajectories, 80/20 split
classes = sort(unique(y))
tr, te = stratified_split(y, 0.8; rng=Xoshiro(SEED))
sc = standardize_fit(Xfeat[:, tr])
Xtr = standardize_apply(Xfeat[:, tr], sc)
Xte = standardize_apply(Xfeat[:, te], sc)
Ytr = onehot(y[tr], classes)

model = train_logreg(Xtr, Ytr; epochs=500, rng=Xoshiro(SEED))
pred_tr = classes[predict_nn(model, Xtr)]
pred_te = classes[predict_nn(model, Xte)]
rep = classification_report(pred_te, y[te], classes)

println("\n========== FHN reservoir digit classification ==========")
println(@sprintf("Nodes/samples: %d   features (trajectory length): %d", N, T))
println(@sprintf("Train accuracy: %.4f", accuracy(pred_tr, y[tr])))
println(@sprintf("Test  accuracy: %.4f   (paper claims 0.88)", rep.accuracy))
println(@sprintf("Test  macro-F1: %.4f", rep.macro_f1))
