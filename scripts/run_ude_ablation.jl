# B8 -- learned-coupling (UDE) vs fixed-reservoir ablation (critique §1.2).
#
# The manuscript's signature contribution is *learning the coupling terms* of
# the oscillator network (UDE formulation), but it only demonstrates this on
# XOR; the headline digit/dry-bean results use a fixed/random reservoir with a
# trained readout. This experiment runs both on the SAME task with an identical
# architecture, so the only difference is whether the coupling matrix W is
# trainable:
#   - fixed reservoir : W frozen (random), train readout only  (reservoir computing)
#   - learned coupling: W trained end-to-end with the readout  (UDE-style)
#
# The reservoir is a discrete-time (fixed-step Euler) FHN-style oscillator
# network -- a fast, fully-differentiable stand-in for the continuous UDE, so we
# can backprop through the dynamics with Flux/Zygote.
#
# Usage:
#   julia --project=. scripts/run_ude_ablation.jl           # full
#   julia --project=. scripts/run_ude_ablation.jl --quick   # fast

include(joinpath(@__DIR__, "..", "src", "baselines", "baseline_utils.jl"))
include(joinpath(@__DIR__, "..", "src", "baselines", "baseline_models.jl"))
using .BaselineUtils, .BaselineModels
using Flux, Zygote
using Random, Statistics, Printf, LinearAlgebra
const Opt = Flux.Optimisers   # Optimisers is a Flux dependency, not a direct one

const QUICK  = "--quick" in ARGS
const N      = 30                   # oscillators
const DIN    = 16                   # dry-bean features
const K      = 7                    # classes
const TSTEP  = 25                   # Euler steps
const DT     = 0.2f0
const EPSf   = 0.08f0
const AF     = 0.7f0
const BDEC   = 0.8f0                # recovery decay
const NPER   = QUICK ? 25 : 60      # samples per class
const SEEDS  = QUICK ? 3 : 8
const EPOCHS = QUICK ? 150 : 400
const LR     = 0.01f0

const OUTDIR = joinpath(@__DIR__, "..", "results", "baselines")
isdir(OUTDIR) || mkpath(OUTDIR)

# ----------------------------------------------------------------- data
Xraw, y = load_drybean()
let lo = vec(minimum(Xraw, dims=2)), hi = vec(maximum(Xraw, dims=2))
    global Xn = Float32.((Xraw .- lo) ./ max.(hi .- lo, 1e-9))
end
function subsample(seed)
    rng = Xoshiro(seed); idx = Int[]
    for c in unique(y)
        ci = shuffle(rng, findall(==(c), y)); append!(idx, ci[1:min(NPER, length(ci))])
    end
    return shuffle(rng, idx)
end
const CLASSES = sort(unique(y))

# ---- differentiable reservoir: input X (DIN, batch) -> features (2N, batch) ----
# discrete FHN: u += dt(drive + u - u^3/3 - v + W u); v += dt eps (u + a - b v)
function features(W, Win, X)
    drive = Win * X
    u = zero(drive); v = zero(drive); su = zero(drive)
    for _ in 1:TSTEP
        u = u .+ DT .* (drive .+ u .- u .^ 3 ./ 3f0 .- v .+ W * u)
        v = v .+ DT .* (EPSf .* (u .+ AF .- BDEC .* v))
        su = su .+ u
    end
    return vcat(u, su ./ TSTEP)
end

struct Resv{A,B,C}; W::A; Wout::B; bout::C; end
Flux.@layer Resv

# spectral-radius-scaled random coupling
function init_W(rng; rho=0.9f0)
    W = randn(rng, Float32, N, N) ./ sqrt(Float32(N))
    W[diagind(W)] .= 0
    r = maximum(abs, eigvals(Float64.(W)))
    return Float32.(W .* (rho / max(r, 1e-6)))
end

function train(Winit, Win, Xtr, Ytr; freeze_W)
    m = Resv(copy(Winit), 0.01f0 .* randn(Float32, K, 2N), zeros(Float32, K))
    opt = Opt.setup(Opt.OptimiserChain(Opt.ClipGrad(1f0), Opt.Adam(LR)), m)
    for _ in 1:EPOCHS
        g = Zygote.gradient(mm -> Flux.logitcrossentropy(mm.Wout * features(mm.W, Win, Xtr) .+ mm.bout, Ytr), m)[1]
        freeze_W && (g = (W=nothing, Wout=g.Wout, bout=g.bout))
        opt, m = Opt.update(opt, m, g)
    end
    return m
end

predict(m, Win, X) = CLASSES[vec(map(argmax, eachcol(m.Wout * features(m.W, Win, X) .+ m.bout)))]

# ----------------------------------------------------------------- run
println("B8 ablation: N=$N FHN reservoir, dry bean ($K classes, $(NPER)/class), $(SEEDS) seeds")
acc_fixed = Float64[]; acc_learned = Float64[]; acc_raw = Float64[]
for seed in 1:SEEDS
    rng = Xoshiro(seed)
    idx = subsample(seed)
    Xs = Xn[:, idx]; ys = y[idx]
    tr, te = stratified_split(ys, 0.7; rng=rng)
    Xtr, Xte = Xs[:, tr], Xs[:, te]
    Ytr = onehot(ys[tr], CLASSES)

    Win = randn(rng, Float32, N, DIN) .* 0.5f0
    Winit = init_W(rng)

    mA = train(Winit, Win, Xtr, Ytr; freeze_W=true)    # fixed reservoir
    mB = train(Winit, Win, Xtr, Ytr; freeze_W=false)   # learned coupling
    aA = accuracy(predict(mA, Win, Xte), ys[te])
    aB = accuracy(predict(mB, Win, Xte), ys[te])

    # raw-feature logistic-regression reference
    mR = train_logreg(Xtr, Ytr; epochs=400, rng=rng)
    aR = accuracy(CLASSES[predict_nn(mR, Xte)], ys[te])

    push!(acc_fixed, aA); push!(acc_learned, aB); push!(acc_raw, aR)
    @printf("  seed %d: fixed=%.3f  learned=%.3f  raw-logreg=%.3f\n", seed, aA, aB, aR)
end

w = wilcoxon_signed_rank(acc_learned, acc_fixed)
sf = summarize(acc_fixed); sl = summarize(acc_learned); sr = summarize(acc_raw)

open(joinpath(OUTDIR, "ude_ablation.md"), "w") do io
    println(io, "# B8 — learned-coupling (UDE) vs fixed-reservoir ablation\n")
    println(io, "Generated by `scripts/run_ude_ablation.jl` ", QUICK ? "(quick)" : "", ".")
    println(io, "Discrete-time FHN reservoir (N=$N), dry bean ($K classes, $(NPER)/class), ",
                "$(SEEDS) seeds. Identical architecture/training; only W trainability differs.\n")
    println(io, "| Model | Test accuracy |")
    println(io, "|---|---|")
    println(io, @sprintf("| Fixed reservoir (W frozen) | %.3f ± %.3f |", sf.mean, sf.std))
    println(io, @sprintf("| Learned coupling (W trained, UDE) | %.3f ± %.3f |", sl.mean, sl.std))
    println(io, @sprintf("| Raw-feature logistic regression | %.3f ± %.3f |", sr.mean, sr.std))
    println(io, @sprintf("\nPaired Wilcoxon (learned vs fixed): W=%.1f, z=%.2f, p=%.4f", w.W, w.z, w.p))
    verdict = w.p < 0.05 ? (sl.mean > sf.mean ? "learning the coupling significantly helps" :
                                                "learning the coupling significantly hurts") :
                           "no significant difference between learned and fixed coupling"
    println(io, "\nVerdict: $verdict.")
end

println()
@printf("Fixed reservoir : %.3f ± %.3f\n", sf.mean, sf.std)
@printf("Learned coupling: %.3f ± %.3f\n", sl.mean, sl.std)
@printf("Raw logreg      : %.3f ± %.3f\n", sr.mean, sr.std)
@printf("Paired Wilcoxon (learned vs fixed): p=%.4f\n", w.p)
println("\nWrote ", joinpath(OUTDIR, "ude_ablation.md"))
