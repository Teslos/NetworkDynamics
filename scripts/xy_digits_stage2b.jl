# EP-XY digits scale-up, STAGE 2b: is Stage 2's 10-class shortfall under-training
# or a real ceiling?
#
# Stage 2 (results/xy_digits_stage2.md) got XY 0.767 vs logreg 0.843 / MLP 0.883
# on full 10-class, but XY was UNDERFITTING (train 0.792 ~ test, below logreg's
# train) and the cost was unconverged/bouncing -- the signature of under-training
# from the compute cuts. Stage 2 ran in 165 s, so we can re-spend the budget.
#
# Stage 2b removes the cheap-but-lossy knobs while keeping the rest controlled
# (same 4x4 downsampling and same baselines, so numbers compare to Stage 2):
#   * symmetric (+-beta) gradient instead of one-sided  -> cleaner gradient
#   * 150 epochs instead of 80                           -> let the cost converge
#   * 40 hidden instead of 20                            -> more capacity
#   * 70 train / 35 test per class instead of 50/30      -> more data
#
# Read: if XY now FITS the train set and closes on logreg/MLP, Stage 2's gap was
# under-training. If it still plateaus below despite fitting train, that is a
# genuine capacity/conditioning ceiling for EP-XY at 10 classes.
#
# Run: julia -t auto --project=. scripts/xy_digits_stage2b.jl

using LinearAlgebra, Statistics, Random, Printf, DelimitedFiles
using OrdinaryDiffEq
using SciMLBase: get_du

EP_XY_SKIP_RUN = true
include(joinpath(@__DIR__, "..", "notebooks", "EP-XY-Network-Claude.jl"))
include(joinpath(@__DIR__, "..", "src", "utils", "eval_protocol.jl"))
using .EvalProtocol

# Evaluation protocol (revised 2026-09-12): this script already selected its
# checkpoint on the TRAINING cost rather than on the test set, so the test
# partition was never used for selection. What it lacked was seed averaging: the
# reported 0.797 came from one seed. It now runs SEEDS (resampling the split, the
# initialisation and the batch order), carves a stratified 20% validation split
# out of the training partition so a validation-selected checkpoint can be
# reported alongside the training-cost one, and evaluates the test partition once
# per seed on checkpoints fixed in advance. See src/utils/eval_protocol.jl.

const STEADY_TOL_S2B = 1e-3
steady_state_callback() = DiscreteCallback(
    (u, t, integrator) -> maximum(abs, get_du(integrator)) < STEADY_TOL_S2B,
    terminate!; save_positions=(false, false))

# ---------------------------------------------------------------- config (re-spent budget)
const SEEDS      = 1:parse(Int, get(ENV, "XY_S2B_SEEDS", "5"))
const VAL_FRAC   = 0.2
const SYMMETRIC  = true
const EVAL_EVERY = 10
const CLASSES    = collect(0:9)
const N_TRAIN_PC = 70
const N_TEST_PC  = 35
const N_HIDDEN   = 40
const N_EV       = 600            # T = 60 (same horizon as Stage 2, controlled)
const DT         = 0.1
const BETA       = 0.01
const STUDY_RATE = 0.05
const N_EPOCH    = parse(Int, get(ENV, "XY_S2B_EPOCH", "150"))
const BATCH      = 50
const W_SCALE    = 0.1
const L2         = 1e-4
const OUTFILE    = joinpath(@__DIR__, "..", "results", "xy_digits_stage2b_seeds.json")
const ON, OFF    = π/2, -π/2

println("threads = ", Threads.nthreads(), ", tol = ", STEADY_TOL_S2B,
        ", N_ev = ", N_EV, " (T=", N_EV*DT, "), SYMMETRIC grad, L2 = ", L2,
        ", hidden = ", N_HIDDEN, ", ", N_TRAIN_PC, "tr/cls, ", N_EPOCH, " epochs\n")

raw = readdlm(joinpath(@__DIR__, "..", "data", "digits", "optdigits.tes"), ',', Int)
const X_ALL = Float64.(raw[:, 1:64]); const Y_ALL = raw[:, 65]

function pool4x4(v)
    img = reshape(v, 8, 8); out = Vector{Float64}(undef, 16); k = 1
    for bi in 1:2:8, bj in 1:2:8
        out[k] = (img[bi,bj]+img[bi+1,bj]+img[bi,bj+1]+img[bi+1,bj+1])/4; k += 1
    end
    return out
end
pool_all(X) = permutedims(reduce(hcat, [pool4x4(X[i, :]) for i in 1:size(X, 1)]))

function logreg_accuracy(Xtr, ytr, Xte, yte, n_class; iters=800, lr=0.5, l2=1e-3)
    n, d = size(Xtr); W = zeros(d, n_class); b = zeros(n_class)
    Y = zeros(n, n_class); for i in 1:n; Y[i, ytr[i]] = 1.0; end
    for _ in 1:iters
        e = exp.((Xtr*W .+ b') .- maximum(Xtr*W .+ b', dims=2)); P = e ./ sum(e, dims=2)
        G = (P .- Y) ./ n
        W .-= lr .* (Xtr' * G .+ l2 .* W); b .-= lr .* vec(sum(G, dims=1))
    end
    L = Xte*W .+ b'
    return mean([argmax(@view L[i, :]) for i in 1:size(Xte,1)] .== yte)
end

function mlp_accuracy(Xtr, ytr, Xte, yte, n_class, seed; h=64, iters=3000, lr=0.2, l2=1e-4)
    rng = MersenneTwister(seed); n, d = size(Xtr)
    W1 = 0.1*randn(rng,d,h); b1 = zeros(h); W2 = 0.1*randn(rng,h,n_class); b2 = zeros(n_class)
    Y = zeros(n, n_class); for i in 1:n; Y[i, ytr[i]] = 1.0; end
    for _ in 1:iters
        Z1 = Xtr*W1 .+ b1'; A1 = tanh.(Z1)
        Lg = A1*W2 .+ b2'; e = exp.(Lg .- maximum(Lg, dims=2)); P = e ./ sum(e, dims=2)
        dL = (P .- Y) ./ n
        gW2 = A1'*dL .+ l2.*W2; gb2 = vec(sum(dL, dims=1))
        dZ1 = (dL*W2') .* (1 .- A1.^2)
        gW1 = Xtr'*dZ1 .+ l2.*W1; gb1 = vec(sum(dZ1, dims=1))
        W1 .-= lr.*gW1; b1 .-= lr.*gb1; W2 .-= lr.*gW2; b2 .-= lr.*gb2
    end
    A1 = tanh.(Xte*W1 .+ b1'); Lg = A1*W2 .+ b2'
    return mean([argmax(@view Lg[i, :]) for i in 1:size(Xte,1)] .== yte)
end

function train_xy_l2(W0, bias0, Xtr, Ttr, Xva, yva, input_index, variable_index, output_index;
                     rng, accuracy, init_va)
    N = size(W0, 1); N_data = size(Xtr, 1)
    W = copy(W0); bias = copy(bias0)
    sW = zeros(size(W)); rW = zeros(size(W)); sB = zeros(size(bias)); rB = zeros(size(bias))
    best_cost = Inf; bestW = copy(W); bestB = copy(bias); ch = zeros(N_EPOCH)
    best_va = -1.0; vaW = copy(W); vaB = copy(bias); best_va_epoch = 0
    for epoch in 1:N_EPOCH
        perm = shuffle(rng, 1:N_data)
        ec = 0.0
        for s in 1:BATCH:N_data
            bidx = perm[s:min(s+BATCH-1, N_data)]
            phase0 = zeros(length(bidx), N)
            phase0[:, input_index] .= Xtr[bidx, :]
            phase0[:, variable_index] .= 0.1 * randn(rng, length(bidx), length(variable_index))
            gW, gB, cost, _ = EP_param_gradient(W, bias, phase0, Ttr[bidx, :], BETA,
                                                N_EV, DT, input_index, variable_index,
                                                output_index; symmetric=SYMMETRIC)
            gW = gW .+ L2 .* W                          # L2 weight decay
            W, sW, rW = Adam_update(W, gW, STUDY_RATE, epoch, sW, rW)
            bias, sB, rB = Adam_update(bias, gB, STUDY_RATE, epoch, sB, rB)
            ec += cost * length(bidx)
        end
        ch[epoch] = ec / N_data
        if ch[epoch] < best_cost; best_cost = ch[epoch]; bestW = copy(W); bestB = copy(bias); end
        if epoch == 1 || epoch % EVAL_EVERY == 0
            va = accuracy(W, bias, Xva, yva, init_va)
            if va > best_va; best_va = va; vaW = copy(W); vaB = copy(bias); best_va_epoch = epoch; end
            @printf("  epoch %d: cost %.4f  val %.3f (best %.3f @ %d)\n",
                    epoch, ch[epoch], va, best_va, best_va_epoch)
        end
    end
    return (cost_W=bestW, cost_b=bestB, val_W=vaW, val_b=vaB,
            best_va=best_va, best_va_epoch=best_va_epoch, history=ch)
end

# ---------------------------------------------------------------- run
const N_CLS = length(CLASSES)
const N = 16 + N_HIDDEN + N_CLS
const INPUT_INDEX = collect(1:16)
const OUTPUT_INDEX = collect(N-N_CLS+1:N)
const VARIABLE_INDEX = setdiff(1:N, INPUT_INDEX)
const CC = Dict(c => j for (j, c) in enumerate(CLASSES))

function make_split(seed)
    tr, va, te = class_split(Y_ALL, CLASSES, N_TRAIN_PC, N_TEST_PC; val_frac=VAL_FRAC, seed=seed)
    feat(idx) = pool_all(X_ALL[idx, :]) ./ 16.0
    lab(idx) = [CC[c] for c in Y_ALL[idx]]
    ftr, fva, fte = feat(tr), feat(va), feat(te)
    phase(P) = (P .- 0.5) .* pi
    return (Xtr_p=ftr, Xva_p=fva, Xte_p=fte,
            Xtr=phase(ftr), Xva=phase(fva), Xte=phase(fte),
            ytr=lab(tr), yva=lab(va), yte=lab(te))
end

# `var_init` is a frozen draw, so the score depends only on (W, bias).
function xy_accuracy(W, bias, X, y, var_init)
    n = size(X, 1)
    phase0 = zeros(n, N); phase0[:, INPUT_INDEX] .= X
    phase0[:, VARIABLE_INDEX] .= var_init
    eq = run_network_batch(phase0, N_EV*DT, W, bias, fill(OFF, n, N_CLS), 0.0,
                           INPUT_INDEX, OUTPUT_INDEX)
    out = eq[:, OUTPUT_INDEX]
    return mean([argmax(@view out[i, :]) for i in 1:n] .== y)
end

function run_seed(seed)
    s = make_split(seed)
    Ttr = [s.ytr[i] == j ? ON : OFF for i in eachindex(s.ytr), j in 1:N_CLS]
    rng = MersenneTwister(seed)
    W0 = W_SCALE * randn(rng, N, N); W0 = (W0 + W0') / 2; W0[diagind(W0)] .= 0
    bias0 = zeros(2, N); bias0[1, :] .= 0.1*rand(rng, N); bias0[2, :] .= 2pi .* (rand(rng, N) .- 0.5)
    nvar = length(VARIABLE_INDEX)
    init_tr = frozen_init(7000+seed, length(s.ytr), nvar; scale=0.1)
    init_va = frozen_init(8000+seed, length(s.yva), nvar; scale=0.1)
    init_te = frozen_init(9000+seed, length(s.yte), nvar; scale=0.1)

    @printf("=== seed %d: N=%d (16 inputs, %d hidden, %d outputs), train=%d val=%d test=%d ===\n",
            seed, N, N_HIDDEN, N_CLS, length(s.ytr), length(s.yva), length(s.yte))
    t0 = time()
    fit = train_xy_l2(W0, bias0, s.Xtr, Ttr, s.Xva, s.yva,
                      INPUT_INDEX, VARIABLE_INDEX, OUTPUT_INDEX;
                      rng=rng, accuracy=xy_accuracy, init_va=init_va)
    secs = time() - t0

    # The test partition is evaluated here only, on checkpoints fixed in advance.
    te_cost = xy_accuracy(fit.cost_W, fit.cost_b, s.Xte, s.yte, init_te)
    te_val  = xy_accuracy(fit.val_W,  fit.val_b,  s.Xte, s.yte, init_te)
    tr_acc  = xy_accuracy(fit.cost_W, fit.cost_b, s.Xtr, s.ytr, init_tr)
    lr_te = logreg_accuracy(s.Xtr_p, s.ytr, s.Xte_p, s.yte, N_CLS)
    ml_te = mlp_accuracy(s.Xtr_p, s.ytr, s.Xte_p, s.yte, N_CLS, seed)
    @printf("  seed %d done in %.0fs: train %.3f | test %.3f (cost-selected) %.3f (val-selected) | logreg %.3f | MLP %.3f\n\n",
            seed, secs, tr_acc, te_cost, te_val, lr_te, ml_te)
    return (seed=seed, train=tr_acc, test=te_cost, test_val_selected=te_val,
            val=fit.best_va, selected_epoch=fit.best_va_epoch,
            cost_first=fit.history[1], cost_last=fit.history[end],
            logreg=lr_te, mlp=ml_te, seconds=secs)
end

results = [run_seed(seed) for seed in SEEDS]
xy = [r.test for r in results]; xyv = [r.test_val_selected for r in results]
println("="^68)
@printf("%d seeds, 4x4 inputs, chance %.3f. Test evaluated once per seed.\n", length(results), 1/N_CLS)
println("-"^68)
@printf("%-28s | %-16s %-16s\n", "model", "train acc", "test acc")
@printf("%-28s | %-16s %-16s\n", "XY (EP, symmetric), cost-sel", msfmt([r.train for r in results]), msfmt(xy))
@printf("%-28s | %-16s %-16s\n", "  val-selected checkpoint", "-", msfmt(xyv))
@printf("%-28s | %-16s %-16s\n", "logreg", "-", msfmt([r.logreg for r in results]))
@printf("%-28s | %-16s %-16s\n", "MLP", "-", msfmt([r.mlp for r in results]))
println("-"^68)
@printf("per-seed test: %s\n", join((@sprintf("%.3f", a) for a in xy), ", "))
write_seed_record(OUTFILE, Dict("seeds"=>collect(SEEDS), "epochs"=>N_EPOCH, "hidden"=>N_HIDDEN,
    "validation_fraction"=>VAL_FRAC, "train_per_class"=>N_TRAIN_PC, "test_per_class"=>N_TEST_PC,
    "beta"=>BETA, "symmetric_gradient"=>SYMMETRIC, "inputs"=>"4x4 pooled"), results)
println("\nwrote ", OUTFILE)
