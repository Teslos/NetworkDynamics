# Reservoir diagnostics for the manuscript critique:
#   B9  edge-of-chaos: accuracy + echo-state-property (ESP) divergence vs the
#       reservoir control knob -- coupling strength sigma for the FHN reservoir,
#       spectral radius rho for the tanh-ESN -- over seeds, with error bars.
#   B7  separability: silhouette score, Fisher ratio, and linear-probe accuracy
#       on raw pixels vs FHN reservoir states vs ESN states, quantifying whether
#       the reservoir actually makes the classes more separable.
#
# Usage:
#   julia --project=. scripts/run_reservoir_diagnostics.jl           # full
#   julia --project=. scripts/run_reservoir_diagnostics.jl --quick   # fast

include(joinpath(@__DIR__, "..", "src", "baselines", "baseline_utils.jl"))
include(joinpath(@__DIR__, "..", "src", "baselines", "baseline_models.jl"))
include(joinpath(@__DIR__, "..", "src", "baselines", "fhn_reservoir.jl"))
include(joinpath(@__DIR__, "..", "src", "utils", "spikerate.jl"))
using .BaselineUtils, .BaselineModels, .FHNReservoir, .spikerate
using Random, Statistics, LinearAlgebra, Printf, DelimitedFiles

const QUICK = "--quick" in ARGS
const SEEDS = QUICK ? 2 : 4
const NFHN  = QUICK ? 120 : 300         # FHN reservoir nodes (= samples)
const NRESN = QUICK ? 200 : 500         # ESN reservoir units
const SIGMAS = QUICK ? [0.0, 0.4, 0.72, 1.5] : [0.0, 0.1, 0.2, 0.4, 0.72, 1.0, 1.5, 2.5]
const RHOS   = QUICK ? [0.3, 0.9, 1.5] : [0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.2, 2.6]
const NSTEPS = 32

const OUTDIR = joinpath(@__DIR__, "..", "results", "baselines")
const FIGDIR = joinpath(@__DIR__, "..", "results", "figures")
isdir(OUTDIR) || mkpath(OUTDIR)
isdir(FIGDIR) || mkpath(FIGDIR)

# --------------------------------------------------------------------- data
function load_digit_images(; path=joinpath(@__DIR__, "..", "data", "digits", "optdigits.tes"))
    raw = readdlm(path, ',', Int)
    imgs = permutedims(reshape(permutedims(Float64.(raw[:, 1:64])), 8, 8, :), (3, 1, 2))
    return imgs, raw[:, 65]
end

# rate-encode (M,8,8) images into a drive matrix (M, 32*64)
function encode(imgs)
    M = size(imgs, 1)
    S = spikerate.rate(imgs ./ 16.0, NSTEPS)        # (32, M, 8, 8)
    S = permutedims(S, (2, 1, 3, 4))
    return Float64.(reshape(S, M, NSTEPS * 64))     # (M, 2048)
end

imgs_all, y_all = load_digit_images()
classes = sort(unique(y_all))

# Xpix: standardized raw pixels (64, N) for the ESN and the raw-pixel baseline
Xpix_all = permutedims(reshape(permutedims(imgs_all, (1, 3, 2)), :, 64))  # (64, N)

# Local Lyapunov-style edge-of-chaos probe: under a weak common drive, perturb
# the reservoir state and measure the mean per-step log growth of the
# perturbation. < 0 => contracting (echo-state property holds); > 0 => chaotic;
# the zero crossing marks the edge of chaos.
function esn_lyap(esn; T=600, washout=150, eps=1e-8, rng=Random.default_rng())
    din = size(esn.Win, 2) - 1
    Nr = size(esn.Wr, 1)
    U = 0.05 .* randn(rng, din, T)          # weak drive so recurrence dominates
    step(x, t) = (1 - esn.leak) .* x .+ esn.leak .* tanh.(esn.Wr * x .+ esn.Win * vcat(@view(U[:, t]), 1.0))
    x = zeros(Nr)
    for t in 1:washout
        x = step(x, t)
    end
    xb = x .+ eps .* randn(rng, Nr)
    d0 = norm(xb - x)
    rate = 0.0
    for t in (washout+1):T
        x = step(x, t); xb = step(xb, t)
        d = norm(xb - x)
        rate += log(d / d0)
        xb = x .+ (d0 / (d + 1e-300)) .* (xb .- x)   # renormalize (Benettin)
    end
    return rate / (T - washout)
end

# readout accuracy on a feature matrix (features, samples)
function probe_accuracy(F, y; seed=1)
    tr, te = stratified_split(y, 0.8; rng=Xoshiro(seed))
    sc = standardize_fit(F[:, tr])
    Xtr = standardize_apply(F[:, tr], sc); Xte = standardize_apply(F[:, te], sc)
    m = train_logreg(Xtr, onehot(y[tr], classes); epochs=(QUICK ? 200 : 400), rng=Xoshiro(seed))
    return accuracy(classes[predict_nn(m, Xte)], y[te])
end

# ============================================================== B9: FHN sweep
println("B9: FHN reservoir sigma-sweep (N=$NFHN, $(SEEDS) seeds)...")
rng0 = Xoshiro(0)
fhn_idx = shuffle(rng0, 1:size(imgs_all, 1))[1:NFHN]
S_fhn = encode(imgs_all[fhn_idx, :, :])
y_fhn = y_all[fhn_idx]

fhn_acc = Dict(s => Float64[] for s in SIGMAS)
fhn_esp = Dict(s => Float64[] for s in SIGMAS)
for s in SIGMAS, seed in 1:SEEDS
    U = fhn_states(S_fhn, s; seed=seed)
    push!(fhn_acc[s], probe_accuracy(permutedims(U), y_fhn; seed=seed))
    push!(fhn_esp[s], fhn_esp_divergence(S_fhn, s; seed=seed))
    @printf("  sigma=%.2f seed=%d  acc=%.3f esp=%.3f\n", s, seed, fhn_acc[s][end], fhn_esp[s][end])
end

# ============================================================== B9: ESN sweep
println("B9: tanh-ESN spectral-radius sweep (Nr=$NRESN, $(SEEDS) seeds)...")
esn_acc = Dict(r => Float64[] for r in RHOS)
esn_lyap_d = Dict(r => Float64[] for r in RHOS)
let Xs = standardize_apply(Xpix_all, standardize_fit(Xpix_all))
    for r in RHOS, seed in 1:SEEDS
        tr, te = stratified_split(y_all, 0.8; rng=Xoshiro(seed))
        esn = build_esn(NRESN, 1; spectral_radius=r, rng=Xoshiro(seed))
        pred = classes[esn_classify(esn, Xs[:, tr], onehot(y_all[tr], classes), Xs[:, te])]
        push!(esn_acc[r], accuracy(pred, y_all[te]))
        push!(esn_lyap_d[r], esn_lyap(esn; rng=Xoshiro(seed)))
        @printf("  rho=%.2f seed=%d  acc=%.3f lyap=%+.3f\n", r, seed, esn_acc[r][end], esn_lyap_d[r][end])
    end
end

# ============================================================ B7: separability
println("B7: separability (silhouette / Fisher / linear probe)...")
# subsample for the O(N^2) silhouette
sil_n = QUICK ? 200 : 500
sidx = shuffle(Xoshiro(7), 1:size(imgs_all, 1))[1:sil_n]
ysil = y_all[sidx]

# raw pixels
Xpix_s = standardize_apply(Xpix_all[:, sidx], standardize_fit(Xpix_all[:, sidx]))
# ESN states (operating point rho=0.9)
esn_op = build_esn(NRESN, 1; spectral_radius=0.9, rng=Xoshiro(1))
Fesn = esn_features(esn_op, standardize_apply(Xpix_all[:, sidx], standardize_fit(Xpix_all[:, sidx])))
Fesn_s = standardize_apply(Fesn, standardize_fit(Fesn))
# FHN states at operating sigma=0.72 (use the NFHN subset that was solved)
Ufhn = fhn_states(S_fhn, 0.72; seed=1)
Ffhn = permutedims(Ufhn)
Ffhn_s = standardize_apply(Ffhn, standardize_fit(Ffhn))

sep = Dict{String,NamedTuple}()
sep["raw pixels"]  = (sil=silhouette_score(Xpix_s, ysil), fish=fisher_ratio(Xpix_s, ysil),
                      probe=mean(probe_accuracy(Xpix_all, y_all; seed=s) for s in 1:SEEDS))
sep["ESN states"]  = (sil=silhouette_score(Fesn_s, ysil), fish=fisher_ratio(Fesn_s, ysil),
                      probe=mean(esn_acc[0.9]))
sep["FHN states"]  = (sil=silhouette_score(Ffhn_s, y_fhn), fish=fisher_ratio(Ffhn_s, y_fhn),
                      probe=mean(fhn_acc[0.72]))

# --------------------------------------------------------------- write report
fmt(v) = (s = summarize(v); @sprintf("%.3f ± %.3f", s.mean, s.std))
open(joinpath(OUTDIR, "reservoir_diagnostics.md"), "w") do io
    println(io, "# Reservoir diagnostics (B9 edge-of-chaos, B7 separability)\n")
    println(io, "Generated by `scripts/run_reservoir_diagnostics.jl` ", QUICK ? "(quick)" : "", ".\n")

    println(io, "## B9a — FHN reservoir vs coupling strength σ (N=$NFHN, $(SEEDS) seeds)\n")
    println(io, "| σ | Test accuracy | ESP divergence |")
    println(io, "|---|---|---|")
    for s in SIGMAS
        println(io, @sprintf("| %.2f | %s | %s |", s, fmt(fhn_acc[s]), fmt(fhn_esp[s])))
    end

    println(io, "\n## B9b — tanh-ESN vs spectral radius ρ (Nr=$NRESN, $(SEEDS) seeds)\n")
    println(io, "Lyapunov exponent < 0 ⇒ echo-state property holds; > 0 ⇒ chaos; ",
                "the zero crossing is the edge of chaos.\n")
    println(io, "| ρ | Test accuracy | Lyapunov exponent |")
    println(io, "|---|---|---|")
    for r in RHOS
        println(io, @sprintf("| %.2f | %s | %s |", r, fmt(esn_acc[r]), fmt(esn_lyap_d[r])))
    end

    println(io, "\n## B7 — separability of feature spaces\n")
    println(io, "| Feature space | Silhouette | Fisher ratio | Linear-probe acc |")
    println(io, "|---|---|---|---|")
    for k in ("raw pixels", "ESN states", "FHN states")
        v = sep[k]
        println(io, @sprintf("| %s | %.3f | %.3f | %.3f |", k, v.sil, v.fish, v.probe))
    end
    println(io, "\nHigher silhouette / Fisher ratio = more linearly separable. ",
                "If the reservoir spaces don't exceed raw pixels, the ",
                "\"resonance improves separability\" claim is not supported.")
end

# --------------------------------------------------------------------- figures
using CairoMakie
m(v) = mean(v); e(v) = (length(v) > 1 ? std(v) : 0.0)

fig = Figure(size=(960, 720))
Label(fig[0, 1:2], "B9 — Edge of chaos: accuracy and echo-state property", fontsize=17, font=:bold)
# FHN column
axA = Axis(fig[1, 1], xlabel="coupling σ", ylabel="test accuracy", title="FHN reservoir")
errorbars!(axA, SIGMAS, [m(fhn_acc[s]) for s in SIGMAS], [e(fhn_acc[s]) for s in SIGMAS])
scatterlines!(axA, SIGMAS, [m(fhn_acc[s]) for s in SIGMAS], color=:crimson)
axB = Axis(fig[2, 1], xlabel="coupling σ", ylabel="ESP divergence")
scatterlines!(axB, SIGMAS, [m(fhn_esp[s]) for s in SIGMAS], color=:black)
# ESN column
axC = Axis(fig[1, 2], xlabel="spectral radius ρ", ylabel="test accuracy", title="tanh-ESN")
errorbars!(axC, RHOS, [m(esn_acc[r]) for r in RHOS], [e(esn_acc[r]) for r in RHOS])
scatterlines!(axC, RHOS, [m(esn_acc[r]) for r in RHOS], color=:crimson)
vlines!(axC, [1.0], color=:gray, linestyle=:dash)
axD = Axis(fig[2, 2], xlabel="spectral radius ρ", ylabel="Lyapunov exponent")
scatterlines!(axD, RHOS, [m(esn_lyap_d[r]) for r in RHOS], color=:black)
hlines!(axD, [0.0], color=:gray, linestyle=:dot)
vlines!(axD, [1.0], color=:gray, linestyle=:dash)
save(joinpath(FIGDIR, "edge_of_chaos.png"), fig)

# separability bar chart (normalize each metric to raw=1 for a fair visual)
fig2 = Figure(size=(820, 360))
Label(fig2[0, 1:3], "B7 — Separability relative to raw pixels", fontsize=16, font=:bold)
spaces = ["raw pixels", "ESN states", "FHN states"]
for (col, (mname, key)) in enumerate((("silhouette", :sil), ("Fisher ratio", :fish), ("linear-probe acc", :probe)))
    ax = Axis(fig2[1, col], xticks=(1:3, spaces), ylabel=mname, xticklabelrotation=π/6)
    vals = [getfield(sep[s], key) for s in spaces]
    barplot!(ax, 1:3, vals, color=[:gray70, :steelblue, :crimson])
end
save(joinpath(FIGDIR, "separability.png"), fig2)

println("\nWrote:")
println("  ", joinpath(OUTDIR, "reservoir_diagnostics.md"))
println("  ", joinpath(FIGDIR, "edge_of_chaos.png"))
println("  ", joinpath(FIGDIR, "separability.png"))
