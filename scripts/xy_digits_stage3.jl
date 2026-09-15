# EP-XY digits scale-up, STAGE 3: adopt Wang et al. (2024)'s exact protocol to
# test whether our Stage 2b "10-class ceiling" was an implementation artifact.
#
# Analysis of docs/Wang_2024_Neuromorph._Comput._Eng._4_034014.pdf (which our
# EP-XY-Network-Claude.jl implements) showed Wang reaches 93.3% all-to-all with
# only 11 hidden units on the SAME sklearn 8x8 digits -- beating his linear
# classifier -- while our Stage 2/2b hit ~0.80 and could not fit the train set.
# The differences, in priority order:
#   1. MULTISTABILITY: Wang initializes hidden+output UNIFORMLY over [-pi, pi)
#      every step and averages the EP gradient over basins via a large batch
#      ("trains all fixed points simultaneously"). We used 0.1*randn near phi=0 ->
#      one basin, no averaging. This is the primary fix.
#   2. RESOLUTION: Wang uses full 64 pixels; we downsampled to 4x4.
#   3. beta = 0.1 (Wang) vs our 0.01 (+ loose tol) -> cleaner EP gradient.
#   4. weight init N(0, 1/N) (Xavier-like) + bias strength h = 0.
#   5. large batch (~300) and many iterations.
#
# EVALUATION PROTOCOL (revised). The first version of this script selected the
# best checkpoint by repeatedly scoring the TEST set and then reported that same
# maximum -- a selection-biased number, and from a single seed, while the
# manuscript states every accuracy is a mean over seeds. Both are fixed here,
# following src/hybrid/readout_ablation.py:
#   * a stratified 20% VALIDATION split is carved out of Wang's 100-image/class
#     training partition (so training sees 80/class); the checkpoint is selected
#     on validation accuracy;
#   * the test partition (70/class) is touched exactly once per seed, at the end,
#     for two checkpoints fixed in advance -- the validation-selected one and the
#     final iterate -- so nothing is selected on test;
#   * everything runs over SEEDS, which resample the split, the weight init and
#     the batch order; results are reported as mean +/- std;
#   * the logreg/MLP baselines are refit per seed on the SAME reduced training
#     split, so the comparison stays like-for-like.
# Evaluation uses a frozen uniform [-pi,pi) draw for the free cells (one draw per
# split, reused at every checkpoint), so the validation curve tracks the weights
# rather than the initialization noise.
#
# Run: julia -t auto --project=. scripts/xy_digits_stage3.jl   (heavy: full 64px)
#      XY_S3_SEEDS=1 XY_S3_ITER=50 julia -t auto --project=. scripts/xy_digits_stage3.jl

using LinearAlgebra, Statistics, Random, Printf, DelimitedFiles, JSON
using OrdinaryDiffEq
using SciMLBase: get_du

EP_XY_SKIP_RUN = true
include(joinpath(@__DIR__, "..", "notebooks", "EP-XY-Network-Claude.jl"))

# Integrate to equilibrium (Wang's protocol relaxes fully). Keep a modest tol so
# uniform-init trajectories, which start far from equilibrium, still terminate.
const STEADY_TOL_S3 = 5e-4
steady_state_callback() = DiscreteCallback(
    (u, t, integrator) -> maximum(abs, get_du(integrator)) < STEADY_TOL_S3,
    terminate!; save_positions=(false, false))

# ---------------------------------------------------------------- config (Wang protocol)
const SEEDS      = 1:parse(Int, get(ENV, "XY_S3_SEEDS", "5"))
const CLASSES    = collect(0:9)
const N_TRAIN_PC = 100           # Wang: first 100 images / digit (train + validation)
const N_TEST_PC  = 70            # Wang: next 70 images / digit
const VAL_FRAC   = 0.2           # readout_ablation.py's VALIDATION_FRACTION
const N_HIDDEN   = 11            # Wang's best all-to-all (N = 85)
const N_EV       = 800           # T = 80
const DT         = 0.1
const BETA       = 0.1           # Wang's conventional choice (was 0.01)
const STUDY_RATE = 0.1           # Wang: eta = 0.1
const N_ITER     = parse(Int, get(ENV, "XY_S3_ITER", "400"))  # Wang uses 1000
const BATCH      = 100           # random images / iteration (basin averaging)
const EVAL_EVERY = 25
const ON, OFF    = π/2, -π/2
const OUTFILE    = joinpath(@__DIR__, "..", "results", "xy_digits_stage3_seeds.json")

println("threads = ", Threads.nthreads(), ", WANG PROTOCOL: uniform [-pi,pi) init, ",
        "beta = ", BETA, ", full 64px, N(0,1/N) weights, one-sided grad")
println("N_ev = ", N_EV, " (T=", N_EV*DT, "), tol = ", STEADY_TOL_S3, ", ",
        N_ITER, " iters, batch ", BATCH, ", ", N_HIDDEN, " hidden")
println("seeds = ", collect(SEEDS), ", validation fraction = ", VAL_FRAC,
        " (checkpoint selected on validation, test evaluated once per seed)\n")

raw = readdlm(joinpath(@__DIR__, "..", "data", "digits", "optdigits.tes"), ',', Int)
const X_ALL = Float64.(raw[:, 1:64]); const Y_ALL = raw[:, 65]

# ---------------------------------------------------------------- baselines (full 64px)
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

function mlp_accuracy(Xtr, ytr, Xte, yte, n_class, seed; h=64, iters=4000, lr=0.2, l2=1e-4)
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

# ---------------------------------------------------------------- layout (seed-independent)
const N_CLS = length(CLASSES)
const N = 64 + N_HIDDEN + N_CLS
const INPUT_INDEX    = collect(1:64)
const OUTPUT_INDEX   = collect(N-N_CLS+1:N)
const VARIABLE_INDEX = setdiff(1:N, INPUT_INDEX)
const CLASSCOL = Dict(c => j for (j, c) in enumerate(CLASSES))

# Wang's 100/70-per-class split, with a stratified validation hold-out carved
# out of the training partition (readout_ablation.stratified_split). The seed
# moves the split as well as the initialization.
function make_split(seed)
    rng = MersenneTwister(1000 + seed)
    n_val = max(1, round(Int, VAL_FRAC * N_TRAIN_PC))
    tr = Int[]; va = Int[]; te = Int[]
    for c in CLASSES
        ci = shuffle(rng, findall(==(c), Y_ALL))
        append!(va, ci[1:n_val])
        append!(tr, ci[n_val+1:N_TRAIN_PC])
        append!(te, ci[N_TRAIN_PC+1:N_TRAIN_PC+N_TEST_PC])
    end
    pixels(idx) = X_ALL[idx, :] ./ 16.0
    labels(idx) = [CLASSCOL[c] for c in Y_ALL[idx]]
    phase(P) = (P .- 0.5) .* π                       # phase in [-pi/2, pi/2]
    return (Xtr_raw=pixels(tr), Xva_raw=pixels(va), Xte_raw=pixels(te),
            Xtr=phase(pixels(tr)), Xva=phase(pixels(va)), Xte=phase(pixels(te)),
            ytr=labels(tr), yva=labels(va), yte=labels(te))
end

uniform_init(rng, n) = 2π .* rand(rng, n, length(VARIABLE_INDEX)) .- π

# Wang readout: p_i ~ 1 + sin(phi_i), predicted = argmax over output cells.
# `var_init` is a frozen uniform [-pi,pi) draw, so the score depends only on W.
function xy_accuracy(W, bias, X, y, var_init)
    n = size(X, 1)
    phase0 = zeros(n, N)
    phase0[:, INPUT_INDEX] .= X
    phase0[:, VARIABLE_INDEX] .= var_init
    eq = run_network_batch(phase0, N_EV*DT, W, bias, fill(OFF, n, N_CLS), 0.0,
                           INPUT_INDEX, OUTPUT_INDEX)
    out = sin.(eq[:, OUTPUT_INDEX])
    return mean([argmax(@view out[i, :]) for i in 1:n] .== y)
end

# ---------------------------------------------------------------- one seed
function run_seed(seed)
    s = make_split(seed)
    Ttr = [s.ytr[i] == j ? ON : OFF for i in eachindex(s.ytr), j in 1:N_CLS]
    Nd = length(s.ytr)

    rng = MersenneTwister(seed)
    # Wang init: weights N(0, 1/N), bias strength h = 0, bias direction uniform.
    W = randn(rng, N, N) ./ sqrt(N); W = (W + W') / 2; W[diagind(W)] .= 0
    bias = zeros(2, N); bias[2, :] .= 2π .* (rand(rng, N) .- 0.5)

    # Frozen evaluation inits (one draw per split, reused at every checkpoint).
    init_tr = uniform_init(MersenneTwister(7000 + seed), Nd)
    init_va = uniform_init(MersenneTwister(8000 + seed), length(s.yva))
    init_te = uniform_init(MersenneTwister(9000 + seed), length(s.yte))

    sW = zeros(size(W)); rW = zeros(size(W)); sB = zeros(size(bias)); rB = zeros(size(bias))
    best_va = -1.0; best_W = copy(W); best_b = copy(bias); best_it = 0
    history = Any[]

    @printf("=== seed %d: train=%d val=%d test=%d ===\n",
            seed, Nd, length(s.yva), length(s.yte))
    t0 = time()
    for it in 1:N_ITER
        bidx = rand(rng, 1:Nd, BATCH)                              # random batch
        phase0 = zeros(BATCH, N)
        phase0[:, INPUT_INDEX] .= s.Xtr[bidx, :]
        phase0[:, VARIABLE_INDEX] .= uniform_init(rng, BATCH)      # UNIFORM [-pi,pi)
        gW, gB, cost, _ = EP_param_gradient(W, bias, phase0, Ttr[bidx, :], BETA,
                                            N_EV, DT, INPUT_INDEX, VARIABLE_INDEX,
                                            OUTPUT_INDEX; symmetric=false)  # one-sided (Wang)
        W, sW, rW = Adam_update(W, gW, STUDY_RATE, it, sW, rW)
        W = (W + W') / 2; W[diagind(W)] .= 0
        bias, sB, rB = Adam_update(bias, gB, STUDY_RATE, it, sB, rB)
        if it == 1 || it % EVAL_EVERY == 0
            va = xy_accuracy(W, bias, s.Xva, s.yva, init_va)
            if va > best_va
                best_va = va; best_W = copy(W); best_b = copy(bias); best_it = it
            end
            push!(history, Dict("iter" => it, "cost" => cost, "val" => va))
            @printf("  iter %d: cost %.4f  val acc %.3f  (best %.3f @ %d)  [%.0fs]\n",
                    it, cost, va, best_va, best_it, time()-t0)
        end
    end
    secs = time() - t0

    # The test partition is touched here only, for checkpoints fixed in advance.
    xy_te_sel = xy_accuracy(best_W, best_b, s.Xte, s.yte, init_te)
    xy_te_fin = xy_accuracy(W, bias, s.Xte, s.yte, init_te)
    xy_tr = xy_accuracy(best_W, best_b, s.Xtr, s.ytr, init_tr)
    lr_te = logreg_accuracy(s.Xtr_raw, s.ytr, s.Xte_raw, s.yte, N_CLS)
    ml_te = mlp_accuracy(s.Xtr_raw, s.ytr, s.Xte_raw, s.yte, N_CLS, seed)

    @printf("  seed %d done in %.0fs: train %.3f | val %.3f @ iter %d | test %.3f (final iterate %.3f) | logreg %.3f | MLP %.3f\n\n",
            seed, secs, xy_tr, best_va, best_it, xy_te_sel, xy_te_fin, lr_te, ml_te)

    return (seed=seed, train=xy_tr, val=best_va, selected_iter=best_it,
            test=xy_te_sel, test_final=xy_te_fin, logreg=lr_te, mlp=ml_te,
            seconds=secs, history=history)
end

# ---------------------------------------------------------------- all seeds
results = [run_seed(seed) for seed in SEEDS]

ms(v) = length(v) > 1 ? @sprintf("%.3f +/- %.3f", mean(v), std(v)) : @sprintf("%.3f", only(v))
xy   = [r.test for r in results]
xyf  = [r.test_final for r in results]
xytr = [r.train for r in results]
lrb  = [r.logreg for r in results]
mlb  = [r.mlp for r in results]
gap  = 100 .* (xy .- lrb)

println("="^72)
@printf("%d seeds, full 64px, chance %.3f. Test evaluated once per seed.\n",
        length(results), 1/N_CLS)
println("-"^72)
@printf("%-26s | %-18s %-18s\n", "model", "train acc", "test acc")
@printf("%-26s | %-18s %-18s\n", "XY (EP, Wang, val-sel)", ms(xytr), ms(xy))
@printf("%-26s | %-18s %-18s\n", "XY (final iterate)", "-", ms(xyf))
@printf("%-26s | %-18s %-18s\n", "logreg", "-", ms(lrb))
@printf("%-26s | %-18s %-18s\n", "MLP", "-", ms(mlb))
println("-"^72)
@printf("per-seed test: %s\n", join((@sprintf("%.3f", a) for a in xy), ", "))
@printf("XY - logreg, paired: %s pp (XY ahead on %d/%d seeds)\n",
        ms(gap), count(>(0), gap), length(gap))
println("\nWang paper (full 64px, all-to-all 11 hidden): XY 93.3%, linear 90.4%, ANN 94.3%")
println("Our Stage 2b (4x4, 40 hidden):                 XY 0.797, logreg 0.837, MLP 0.900")

open(OUTFILE, "w") do io
    JSON.print(io, Dict(
        "seeds" => collect(SEEDS), "iterations" => N_ITER, "hidden" => N_HIDDEN,
        "validation_fraction" => VAL_FRAC, "train_per_class" => N_TRAIN_PC,
        "test_per_class" => N_TEST_PC, "beta" => BETA,
        "per_seed" => [Dict(string(k) => getfield(r, k) for k in keys(r)) for r in results],
        "summary" => Dict(
            "xy_test_mean" => mean(xy),
            "xy_test_std" => length(xy) > 1 ? std(xy) : NaN,
            "xy_test_final_mean" => mean(xyf),
            "xy_train_mean" => mean(xytr),
            "logreg_test_mean" => mean(lrb),
            "mlp_test_mean" => mean(mlb),
            "xy_minus_logreg_pp_mean" => mean(gap)),
    ), 2)
end
println("\nwrote ", OUTFILE)
