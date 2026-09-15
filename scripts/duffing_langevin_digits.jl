# Multi-class DIGITS classification with finite-temperature (Langevin-sampled)
# thermodynamic Equilibrium Propagation on a monostable Duffing network.
#
# Uses the MONOSTABLE regime, which is (a) required for graded multi-class readout
# (the bistable readout fails at chance -- see the paper) and (b) exactly the
# regime where the finite-T EP gradient is faithful and Langevin sampling mixes
# cleanly (unimodal; see results/ep_duffing_langevin.md, Result 2). So the Duffing
# digits pipeline of scripts/duffing_digits_mono_v2.jl is reused verbatim -- 16
# pooled inputs -> monostable-hidden -> 10 linear output cells, layered coupling,
# softmax-CE readout -- with only the relaxer swapped for the overdamped Langevin
# sampler and the gradient replaced by the thermal-average contrast
#   dL/dW_ij = (<x_i x_j>_{-beta} - <x_i x_j>_{+beta}) / 2beta,   <.> time averages,
# with common random numbers across the free/+-beta phases.
#
# Efficiency: the batch-summed correlation the gradient needs is just
# sum_t X_t' X_t (one gemm per sampled step), so no per-sample NxN tensors.
#
# Run: julia -t auto --project=. scripts/duffing_langevin_digits.jl

using Random, Statistics, LinearAlgebra, DelimitedFiles, Printf
EP_DUFFING_SKIP_RUN = true
include(joinpath(@__DIR__, "..", "notebooks", "EP-Duffing-Network.jl"))   # adam_update

# ---------------- hyperparameters ----------------
const SEEDS   = 1:parse(Int, get(ENV, "DUF_LGV_SEEDS", "5"))
const VAL_FRAC = 0.2
const FULLRES = ("fullres" in ARGS)   # full 64-px inputs vs 4x4-pooled 16 (default)
const NHID    = FULLRES ? 64 : 40     # monostable hidden units
const A_H     = 1.0          # hidden quadratic coeff > 0  -> single well (monostable)
const T_SAMP  = 0.30         # sampling temperature (Langevin)
const BETA    = 0.1
const LR      = 0.01
const N_ITER  = parse(Int, get(ENV, "DUF_LGV_ITER", FULLRES ? "450" : "300"))
const BATCH   = 64
const N_BURN  = 200
const N_SAMPLE= 350
const N_TRAIN_PC = FULLRES ? 120 : 40
const N_TEST_PC  = 40
const EVAL_EVERY = 20
const OUTFILE = joinpath(@__DIR__, "..", "results",
    FULLRES ? "ep_duffing_langevin_digits_fullres_seeds.json" : "ep_duffing_langevin_digits_seeds.json")

include(joinpath(@__DIR__, "..", "src", "utils", "eval_protocol.jl"))
using .EvalProtocol

# Evaluation protocol (revised 2026-09-12): the checkpoint is selected on a
# stratified 20% validation split carved out of the training partition, the test
# partition is evaluated once per seed on checkpoints fixed in advance, and the
# result is a mean over seeds. See src/utils/eval_protocol.jl. The earlier
# single-seed, test-selected number for this script was 0.80 (pooled 16).

# ---------------- data ----------------
raw  = readdlm(joinpath(@__DIR__, "..", "data", "digits", "optdigits.tes"), ',', Int)
const XALL = Float64.(raw[:, 1:64]); const YALL = raw[:, 65]
pool4x4(v) = (img = reshape(v, 8, 8);
    [(img[bi,bj]+img[bi+1,bj]+img[bi,bj+1]+img[bi+1,bj+1])/4 for bi in 1:2:8 for bj in 1:2:8])
poolall(X) = permutedims(reduce(hcat, [pool4x4(X[i, :]) for i in 1:size(X, 1)]))
# feature matrix in [0,1]: full 64 px or 4x4-pooled 16
featmat(idx) = FULLRES ? (XALL[idx, :] ./ 16.0) : (poolall(XALL[idx, :]) ./ 16.0)

const NIN = FULLRES ? 64 : 16; const NCLS = 10; const NN = NIN + NHID + NCLS
const INP = collect(1:NIN); const HID = collect(NIN+1:NIN+NHID)
const OUTc = collect(NN-NCLS+1:NN); const VARc = vcat(HID, OUTc)
const MSK = let M = zeros(NN, NN)
    for i in INP, j in HID;  M[i,j]=1.0; M[j,i]=1.0; end
    for i in HID, j in OUTc; M[i,j]=1.0; M[j,i]=1.0; end; M end
# per-column on-site coeffs: hidden -> monostable quartic (c=1,a=A_H); output -> linear (-x)
const C3 = [i in HID ? 1.0 : 0.0 for i in 1:NN]
const A1 = [i in HID ? A_H : (i in OUTc ? 1.0 : 0.0) for i in 1:NN]

# ---------------- Langevin sampler (monostable) ----------------
# Returns M1 (sum_t sum_d x_i), M2 (sum_t X'X), meanx (per-sample time-mean).
function lang_relax(W, h, x0, Y, beta, T; dt=0.02, n_burn=N_BURN, n_sample=N_SAMPLE,
                    rng=Random.default_rng())
    nb, N = size(x0)
    X = copy(x0); Xin = x0[:, INP]; @views X[:, INP] .= Xin
    hrow = reshape(h, 1, N); sq = sqrt(2 * T * dt)
    F = zeros(nb, N); noise = zeros(nb, N)
    M1 = zeros(N); M2 = zeros(N, N); meanx = zeros(nb, N); P = zeros(nb, NCLS)
    doact = beta != 0.0
    step! = function ()
        @. F = -(C3' * X^3 + A1' * X)
        mul!(F, X, W, 1.0, 1.0); F .+= hrow
        if doact
            O = @view X[:, OUTc]; m = maximum(O, dims=2); @. P = exp(O - m); P ./= sum(P, dims=2)
            @views F[:, OUTc] .-= beta .* (P .- Y)
        end
        randn!(rng, noise); @. X += F * dt + sq * noise; @views X[:, INP] .= Xin
    end
    for _ in 1:n_burn; step!(); end
    for _ in 1:n_sample
        step!(); meanx .+= X; M1 .+= vec(sum(X, dims=1)); mul!(M2, X', X, 1.0, 1.0)
    end
    meanx ./= n_sample
    return M1, M2, meanx
end

# Symmetric finite-T EP gradient (softmax-CE nudge), common random numbers.
function lang_grad(W, h, x0, Y, beta, T; dt=0.02, n_burn=N_BURN, n_sample=N_SAMPLE)
    nb, N = size(x0); base = rand(UInt32)
    _, M2f_unused, mxf = lang_relax(W, h, x0, Y, 0.0, T; dt=dt, n_burn=n_burn, n_sample=n_sample, rng=MersenneTwister(base))
    xs = copy(x0); @views xs[:, VARc] .= mxf[:, VARc]
    M1p, M2p, _ = lang_relax(W, h, xs, Y, beta, T; dt=dt, n_burn=n_burn, n_sample=n_sample, rng=MersenneTwister(base))
    M1m, M2m, _ = lang_relax(W, h, xs, Y, -beta, T; dt=dt, n_burn=n_burn, n_sample=n_sample, rng=MersenneTwister(base))
    f = 1.0 / (nb * n_sample * 2beta)
    gW = (M2m .- M2p) .* f; gW .*= MSK; gW[diagind(gW)] .= 0
    gh = (M1m .- M1p) .* f
    O = mxf[:, OUTc]; ce = 0.0
    for d in 1:nb
        o = O[d, :]; mo = maximum(o); pe = exp.(o .- mo); pe ./= sum(pe)
        ce += -sum(Y[d, :] .* log.(pe .+ 1e-12))
    end
    return gW, gh, ce / nb
end

# ---------------- baselines (softmax logreg, 1-hidden MLP) ----------------
logreg_acc(Xtr, ytr, Xte, yte, nc; iters=800, lr=0.5, l2=1e-3) = begin
    n, d = size(Xtr); W = zeros(d, nc); b = zeros(nc); Y = zeros(n, nc); for i in 1:n; Y[i, ytr[i]] = 1.0; end
    for _ in 1:iters
        L = Xtr*W .+ b'; e = exp.(L .- maximum(L, dims=2)); Pp = e ./ sum(e, dims=2)
        G = (Pp .- Y) ./ n; W .-= lr .* (Xtr'*G .+ l2 .* W); b .-= lr .* vec(sum(G, dims=1))
    end
    L = Xte*W .+ b'; mean([argmax(@view L[i, :]) for i in 1:size(Xte, 1)] .== yte)
end
mlp_acc(Xtr, ytr, Xte, yte, nc, seed; hh=64, iters=3000, lr=0.2, l2=1e-4) = begin
    rng = MersenneTwister(seed); n, d = size(Xtr)
    W1 = 0.1*randn(rng, d, hh); b1 = zeros(hh); W2 = 0.1*randn(rng, hh, nc); b2 = zeros(nc)
    Y = zeros(n, nc); for i in 1:n; Y[i, ytr[i]] = 1.0; end
    for _ in 1:iters
        A1h = tanh.(Xtr*W1 .+ b1'); Lg = A1h*W2 .+ b2'; e = exp.(Lg .- maximum(Lg, dims=2)); Pp = e ./ sum(e, dims=2)
        dL = (Pp .- Y) ./ n; gW2 = A1h'*dL .+ l2 .* W2; gb2 = vec(sum(dL, dims=1))
        dZ1 = (dL*W2') .* (1 .- A1h.^2); gW1 = Xtr'*dZ1 .+ l2 .* W1; gb1 = vec(sum(dZ1, dims=1))
        W1 .-= lr.*gW1; b1 .-= lr.*gb1; W2 .-= lr.*gW2; b2 .-= lr.*gb2
    end
    A1h = tanh.(Xte*W1 .+ b1'); Lg = A1h*W2 .+ b2'; mean([argmax(@view Lg[i, :]) for i in 1:size(Xte, 1)] .== yte)
end

# ---------------- split ----------------
const CC = Dict(c => j for (j, c) in enumerate(0:9))
function make_split(seed)
    tr, va, te = class_split(YALL, collect(0:9), N_TRAIN_PC, N_TEST_PC; val_frac=VAL_FRAC, seed=seed)
    lab(idx) = [CC[c] for c in YALL[idx]]
    ftr, fva, fte = featmat(tr), featmat(va), featmat(te)
    return (Xtrp=ftr, Xvap=fva, Xtep=fte,
            Xtr=2 .* ftr .- 1, Xva=2 .* fva .- 1, Xte=2 .* fte .- 1,
            ytr=lab(tr), yva=lab(va), yte=lab(te))
end

# `var_init` is a frozen draw and the sampler RNG is fixed, so the score depends
# only on (W, h).
function digit_acc(W, h, X, y, T, var_init)
    n = size(X, 1); x0 = zeros(n, NN); x0[:, INP] .= X; x0[:, VARc] .= var_init
    _, _, mx = lang_relax(W, h, x0, zeros(n, NCLS), 0.0, T; n_burn=250, n_sample=400, rng=MersenneTwister(77))
    o = mx[:, OUTc]; mean([argmax(@view o[i, :]) for i in 1:n] .== y)
end

# ---------------- train ----------------
println("threads=$(Threads.nthreads())  N=$NN ($NIN in, $NHID mono-hidden, $NCLS out)  " *
        "T=$T_SAMP  seeds=$(collect(SEEDS))  (checkpoint selected on validation, test once per seed)")

function run_seed(seed)
    s = make_split(seed); Nd = length(s.ytr)
    Ytr = [s.ytr[i] == j ? 1.0 : 0.0 for i in eachindex(s.ytr), j in 1:NCLS]
    rng = MersenneTwister(seed)
    W = 0.1*randn(rng, NN, NN); W = (W+W')/2; W .*= MSK; h = zeros(NN)
    nvar = length(VARc)
    init_tr = frozen_init(7000+seed, Nd, nvar; scale=0.1)
    init_va = frozen_init(8000+seed, length(s.yva), nvar; scale=0.1)
    init_te = frozen_init(9000+seed, length(s.yte), nvar; scale=0.1)
    sW = zeros(NN, NN); rW = zeros(NN, NN); sh = zeros(NN); rh = zeros(NN)
    best_va = -1.0; bW = copy(W); bh = copy(h); best_it = 0
    @printf("=== seed %d: train=%d val=%d test=%d ===\n", seed, Nd, length(s.yva), length(s.yte))
    t0 = time()
    for it in 1:N_ITER
        bi = rand(rng, 1:Nd, BATCH)
        x0 = zeros(BATCH, NN); x0[:, INP] .= s.Xtr[bi, :]; x0[:, VARc] .= 0.1*randn(rng, BATCH, nvar)
        gW, gh, ce = lang_grad(W, h, x0, Ytr[bi, :], BETA, T_SAMP)
        W, sW, rW = adam_update(W, gW, LR, it, sW, rW); W = (W+W')/2; W .*= MSK
        h, sh, rh = adam_update(h, gh, LR, it, sh, rh)
        if it == 1 || it % EVAL_EVERY == 0
            va = digit_acc(W, h, s.Xva, s.yva, T_SAMP, init_va)
            if va > best_va; best_va = va; bW = copy(W); bh = copy(h); best_it = it; end
            @printf("  it %d: CE %.3f  val %.3f (best %.3f @ %d) [%.0fs]\n", it, ce, va, best_va, best_it, time()-t0)
        end
    end
    secs = time() - t0
    # The test partition is evaluated here only, on checkpoints fixed in advance.
    te_sel = digit_acc(bW, bh, s.Xte, s.yte, T_SAMP, init_te)
    te_fin = digit_acc(W, h, s.Xte, s.yte, T_SAMP, init_te)
    tr_acc = digit_acc(bW, bh, s.Xtr, s.ytr, T_SAMP, init_tr)
    lr_te = logreg_acc(s.Xtrp, s.ytr, s.Xtep, s.yte, NCLS)
    ml_te = mlp_acc(s.Xtrp, s.ytr, s.Xtep, s.yte, NCLS, seed)
    @printf("  seed %d done in %.0fs: train %.3f | val %.3f @ it %d | test %.3f (final iterate %.3f) | logreg %.3f | MLP %.3f\n\n",
            seed, secs, tr_acc, best_va, best_it, te_sel, te_fin, lr_te, ml_te)
    return (seed=seed, train=tr_acc, val=best_va, selected_iter=best_it, test=te_sel,
            test_final=te_fin, logreg=lr_te, mlp=ml_te, seconds=secs)
end

results = [run_seed(seed) for seed in SEEDS]
du = [r.test for r in results]; duf = [r.test_final for r in results]
const FEATLBL = FULLRES ? "64 px" : "pooled 16"
println("\n", "="^68)
@printf("%d seeds, %s, chance %.2f. Test evaluated once per seed.\n", length(results), FEATLBL, 1/NCLS)
println("-"^68)
@printf("%-30s | %-16s %-16s\n", "model", "train", "test")
@printf("%-30s | %-16s %-16s\n", "Langevin monostable EP (val-sel)", msfmt([r.train for r in results]), msfmt(du))
@printf("%-30s | %-16s %-16s\n", "  (final iterate)", "-", msfmt(duf))
@printf("%-30s | %-16s %-16s\n", "logreg ($FEATLBL)", "-", msfmt([r.logreg for r in results]))
@printf("%-30s | %-16s %-16s\n", "MLP ($FEATLBL)", "-", msfmt([r.mlp for r in results]))
println("-"^68)
@printf("per-seed test: %s\n", join((@sprintf("%.3f", a) for a in du), ", "))
write_seed_record(OUTFILE, Dict("seeds"=>collect(SEEDS), "iterations"=>N_ITER, "hidden"=>NHID,
    "validation_fraction"=>VAL_FRAC, "train_per_class"=>N_TRAIN_PC, "test_per_class"=>N_TEST_PC,
    "beta"=>BETA, "temperature"=>T_SAMP, "inputs"=>FEATLBL), results)
println("\nwrote ", OUTFILE)
