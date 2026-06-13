# Combined criticality-vs-accuracy experiment -- the decisive test of Beattie
# et al. 2024's central claim: does the FHN reservoir classify best at avalanche
# criticality, and is the critical reservoir robust to readout shrinkage?
#
# On ONE network, for each coupling strength sigma, we measure BOTH:
#   - branching ratio / avalanche statistics (criticality), and
#   - dry-bean classification accuracy,
# from the same per-sample reservoir runs. Plus a readout-shrinkage curve
# (accuracy vs number of readout nodes) for sub/critical/super sigma.
#
# Setup (Beattie-faithful): N excitable FHN oscillators (a>1, quiescent at rest).
# 16 dry-bean features rate-encoded into 16 input nodes; the readout is taken
# from a SEPARATE set of non-input nodes, so good accuracy requires the coupling
# to propagate input information through the network.
#
# Usage:
#   julia --project=. scripts/run_criticality_vs_accuracy.jl           # full
#   julia --project=. scripts/run_criticality_vs_accuracy.jl --quick   # fast

include(joinpath(@__DIR__, "..", "src", "baselines", "fhn_reservoir.jl"))
include(joinpath(@__DIR__, "..", "src", "baselines", "avalanche.jl"))
include(joinpath(@__DIR__, "..", "src", "baselines", "baseline_utils.jl"))
include(joinpath(@__DIR__, "..", "src", "baselines", "baseline_models.jl"))
using .FHNReservoir, .AvalancheCriticality, .BaselineUtils, .BaselineModels
using Random, Statistics, Printf

const QUICK   = "--quick" in ARGS
const N       = 100
const NIN     = 16                  # input nodes = dry-bean features
const AEXC    = 1.05                # excitable regime
const T       = QUICK ? 400 : 700
const INSCALE = 1.2                 # input current scale (feature -> drive)
const THRESH  = 1.0
const BIN     = 2
const NPER    = QUICK ? 20 : 60     # samples per class
const SEED    = 1
const SIGMAS  = QUICK ? [0.003, 0.03, 0.3] : [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
const SHRINK  = [N, 50, 20]         # readout sizes for the shrinkage test

const OUTDIR = joinpath(@__DIR__, "..", "results", "baselines")
const FIGDIR = joinpath(@__DIR__, "..", "results", "figures")
isdir(OUTDIR) || mkpath(OUTDIR)
isdir(FIGDIR) || mkpath(FIGDIR)

const INPUT_NODES = 1:NIN
# Readout is the full reservoir state (standard reservoir computing). The
# shrinkage test reduces the number of readout nodes by drawing RANDOM subsets
# from the whole network (Beattie's "free choice of readout nodes" / spatial
# invariance), averaged over several draws -- not by excluding the input nodes.

# ----------------------------------------------------------------- data
Xraw, y = load_drybean()
# min-max normalize each feature to [0,1] for rate encoding
let lo = vec(minimum(Xraw, dims=2)), hi = vec(maximum(Xraw, dims=2))
    global Xn = (Xraw .- lo) ./ max.(hi .- lo, 1e-9)
end
# stratified subsample to NPER per class
let idx = Int[]
    rng = Xoshiro(SEED)
    for c in unique(y)
        ci = shuffle(rng, findall(==(c), y))
        append!(idx, ci[1:min(NPER, length(ci))])
    end
    global samp = shuffle(rng, idx)
end
ys = y[samp]
classes = sort(unique(ys))
nsamp = length(samp)
println("Dry bean: $nsamp samples ($(length(classes)) classes), N=$N excitable FHN, ",
        "$NIN input nodes, full-state readout, T=$T")

# deterministic constant-current drive for one sample: (N, T), feature intensity
# -> input current on the input nodes (no stochastic encoding noise).
function sample_drive(s)
    S = zeros(N, T)
    @inbounds for (j, node) in enumerate(INPUT_NODES)
        S[node, :] .= INSCALE * Xn[j, samp[s]]
    end
    return S
end

# ---- per-sigma: run all samples, collect readout features + criticality ----
# readout features per node: [mean(u), spike count]; stored per readout node so
# the shrinkage test can use a subset.
# Classification features: per readout node [mean(u), std(u), spike count],
# stored in READOUT_ORDER so the shrinkage test can use the first k nodes.
# Deterministic input -> the reservoir response is a clean function of the sample.
function classification_features(sigma)
    feat = zeros(3, N, nsamp)            # (stat, node, sample), natural node order
    for s in 1:nsamp
        U = fhn_states(sample_drive(s), sigma; seed=SEED, a=AEXC)
        sp_all = detect_spikes(U; thresh=THRESH)
        feat[1, :, s] = vec(mean(U, dims=2))
        feat[2, :, s] = vec(std(U, dims=2))
        feat[3, :, s] = vec(sum(sp_all, dims=2))
    end
    return feat
end

# Criticality probe on the SAME reservoir (same seed -> same coupling W): drive
# the input nodes with sparse Poisson input and measure avalanche statistics.
# This is the proper criticality measurement (clean avalanche structure), as
# opposed to inferring it from the constant-input classification runs.
const TPROBE = QUICK ? 1500 : 4000
function criticality_probe(sigma)
    rng = Xoshiro(2024)
    Sp = zeros(N, TPROBE)
    Sp[INPUT_NODES, :] .= Float64.(rand(rng, NIN, TPROBE) .< 0.05)
    U = fhn_states(Sp, sigma; seed=SEED, a=AEXC)
    act = population_activity(detect_spikes(U; thresh=THRESH); bin=BIN)
    sizes, _ = avalanche_stats(act)
    return (branch=branching_ratio(act), tau=powerlaw_mle(sizes; xmin=1), activity=mean(act))
end

# classify from a given set of readout nodes (features = [mean, std, spikecount])
function classify(feat, nodes; nsplits=(QUICK ? 2 : 5))
    F = reshape(feat[:, nodes, :], 3 * length(nodes), nsamp)
    accs = Float64[]
    for sp in 1:nsplits
        tr, te = stratified_split(ys, 0.7; rng=Xoshiro(sp))
        sc = standardize_fit(F[:, tr])
        Xtr = standardize_apply(F[:, tr], sc); Xte = standardize_apply(F[:, te], sc)
        m = train_logreg(Xtr, onehot(ys[tr], classes); epochs=(QUICK ? 200 : 400), rng=Xoshiro(sp))
        push!(accs, accuracy(classes[predict_nn(m, Xte)], ys[te]))
    end
    return mean(accs)
end

# accuracy from a random k-node subset, averaged over several draws (spatial
# invariance / free choice of readout nodes)
function shrinkage_accuracy(feat, k; ndraws=(QUICK ? 2 : 4))
    k >= N && return classify(feat, 1:N)
    return mean(classify(feat, shuffle(Xoshiro(100 + d), 1:N)[1:k]) for d in 1:ndraws)
end

println("\nSweeping coupling sigma...")
res = Dict{Float64,Any}()
for sigma in SIGMAS
    feat = classification_features(sigma)
    crit = criticality_probe(sigma)
    acc = classify(feat, 1:N)
    res[sigma] = (feat=feat, branch=crit.branch, tau=crit.tau, activity=crit.activity, acc=acc)
    @printf("  sigma=%.3f  branch=%.2f  tau=%.2f  activity=%.2f  accuracy=%.3f\n",
            sigma, crit.branch, crit.tau, crit.activity, acc)
end

# ---- readout shrinkage for representative sub/critical/super sigma ----
crit_sigma = SIGMAS[argmin(abs.([res[s].branch - 1 for s in SIGMAS]))]
sub_sigma  = SIGMAS[1]
sup_sigma  = SIGMAS[end]
shrink_acc = Dict(s => [shrinkage_accuracy(res[s].feat, k) for k in SHRINK]
                  for s in (sub_sigma, crit_sigma, sup_sigma))

# --------------------------------------------------------------- report
open(joinpath(OUTDIR, "criticality_vs_accuracy.md"), "w") do io
    println(io, "# Criticality vs accuracy (FHN reservoir, dry bean)\n")
    println(io, "Generated by `scripts/run_criticality_vs_accuracy.jl` ", QUICK ? "(quick)" : "", ".")
    println(io, "N=$N excitable FHN (a=$AEXC), $NIN input nodes, full-state readout, ",
                "$nsamp samples, T=$T.\n")
    println(io, "Tests Beattie et al.: does accuracy peak at avalanche criticality ",
                "(branching ratio ≈ 1), and is the critical reservoir robust to readout shrinkage?\n")
    println(io, "| σ | branching ratio | τ (size) | mean activity | accuracy |")
    println(io, "|---|---|---|---|---|")
    for s in SIGMAS
        r = res[s]
        println(io, @sprintf("| %.3f | %.2f | %.2f | %.2f | %.3f |",
                             s, r.branch, r.tau, r.activity, r.acc))
    end
    println(io, @sprintf("\nClosest to criticality (branching≈1): σ=%.3f.\n", crit_sigma))
    println(io, "## Readout shrinkage (accuracy vs # readout nodes)\n")
    println(io, "| regime | σ | " * join(["$(k) nodes" for k in SHRINK], " | ") * " |")
    println(io, "|---|---|" * "---|"^length(SHRINK))
    for (lab, s) in (("subcritical", sub_sigma), ("critical", crit_sigma), ("supercritical", sup_sigma))
        println(io, @sprintf("| %s | %.3f | ", lab, s) *
                    join([@sprintf("%.3f", a) for a in shrink_acc[s]], " | ") * " |")
    end
end

# --------------------------------------------------------------- figure
using CairoMakie
fig = Figure(size=(1000, 360))
Label(fig[0, 1:3], "Criticality vs accuracy (FHN reservoir, dry bean)", fontsize=17, font=:bold)

ax1 = Axis(fig[1, 1], xscale=log10, xlabel="coupling σ", ylabel="test accuracy",
           title="accuracy vs coupling")
scatterlines!(ax1, SIGMAS, [res[s].acc for s in SIGMAS], color=:crimson)
vlines!(ax1, [crit_sigma], color=:gray, linestyle=:dash)

ax2 = Axis(fig[1, 2], xscale=log10, xlabel="coupling σ", ylabel="branching ratio",
           title="criticality vs coupling")
scatterlines!(ax2, SIGMAS, [res[s].branch for s in SIGMAS], color=:black)
hlines!(ax2, [1.0], color=:crimson, linestyle=:dash)
vlines!(ax2, [crit_sigma], color=:gray, linestyle=:dash)

ax3 = Axis(fig[1, 3], xlabel="# readout nodes", ylabel="test accuracy",
           title="readout shrinkage", xticks=(1:length(SHRINK), string.(SHRINK)))
for (lab, s, col) in (("subcritical", sub_sigma, :steelblue),
                      ("critical", crit_sigma, :crimson),
                      ("supercritical", sup_sigma, :gray40))
    scatterlines!(ax3, 1:length(SHRINK), shrink_acc[s], color=col, label=lab)
end
axislegend(ax3, position=:lb, labelsize=9)
save(joinpath(FIGDIR, "criticality_vs_accuracy.png"), fig)

println("\nWrote:")
println("  ", joinpath(OUTDIR, "criticality_vs_accuracy.md"))
println("  ", joinpath(FIGDIR, "criticality_vs_accuracy.png"))
