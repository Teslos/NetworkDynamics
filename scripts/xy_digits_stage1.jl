# EP-XY digits scale-up, STAGE 1: harder/confusable classes vs a logreg baseline.
#
# Stage 0 (results/xy_digits_stage0.md) proved the pipeline trains and the N~100
# XY net relaxes (slow gradient flow, no frustration), but used an easy 3-class
# subset (0,1,2) where even a linear model hits 100% -- uninformative. Stage 1
# raises difficulty and adds an interpretable baseline:
#   * Task A: confusable {3, 5, 8}
#   * Task B: 5-class {0, 1, 2, 3, 4}
# For each, EP-train the XY net and train a softmax logistic-regression baseline
# on the SAME train/test split (raw pixel features), so XY accuracy is read
# relative to a known linear bar rather than in the abstract.
#
# Operating point from Stage 0: N_ev=1000 (T=100) with the steady-state tolerance
# relaxed to 1e-3 (the net is at a fixed point to ~1e-4 by T=100, so 1e-3 lets
# most solves terminate early instead of paying the long 1e-5 tail).
#
# Run: julia -t auto --project=. scripts/xy_digits_stage1.jl

using LinearAlgebra, Statistics, Random, Printf, DelimitedFiles
using OrdinaryDiffEq
using SciMLBase: get_du

EP_XY_SKIP_RUN = true
include(joinpath(@__DIR__, "..", "notebooks", "EP-XY-Network-Claude.jl"))

# --- Relax the steady-state tolerance to 1e-3 (Stage 0 operating point). This
# overrides the 1e-5 callback in the included file; run_network_batch resolves
# steady_state_callback by name at call time, so it picks up this definition.
const STEADY_TOL_S1 = 1e-3
steady_state_callback() = DiscreteCallback(
    (u, t, integrator) -> maximum(abs, get_du(integrator)) < STEADY_TOL_S1,
    terminate!; save_positions=(false, false))

# ---------------------------------------------------------------- config
const SEED       = 1
const TASKS      = [[3, 5, 8], [0, 1, 2, 3, 4]]
const N_TRAIN_PC = 60
const N_TEST_PC  = 30
const N_HIDDEN   = 30
const N_EV       = 1000           # T = 100
const DT         = 0.1
const BETA       = 0.01
const STUDY_RATE = 0.05
const N_EPOCH    = 120
const BATCH      = 30
const W_SCALE    = 0.1
const ON, OFF    = π/2, -π/2

Random.seed!(SEED)
println("threads = ", Threads.nthreads(), ", steady tol = ", STEADY_TOL_S1,
        ", N_ev = ", N_EV, " (T=", N_EV*DT, ")\n")

raw = readdlm(joinpath(@__DIR__, "..", "data", "digits", "optdigits.tes"), ',', Int)
const X_ALL = Float64.(raw[:, 1:64]); const Y_ALL = raw[:, 65]

# ---------------------------------------------------------------- baseline
# Multinomial (softmax) logistic regression, full-batch GD on raw pixels/16.
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
    logits = Xte * W .+ b'                                   # (n_test, n_class)
    pred = [argmax(@view logits[i, :]) for i in 1:size(Xte, 1)]
    return mean(pred .== yte)
end

# ---------------------------------------------------------------- one task
function run_task(classes)
    rng = MersenneTwister(SEED)
    n_cls = length(classes)
    classcol = Dict(c => j for (j, c) in enumerate(classes))

    tr = Int[]; te = Int[]
    for c in classes
        ci = shuffle(rng, findall(==(c), Y_ALL))
        append!(tr, ci[1:N_TRAIN_PC]); append!(te, ci[N_TRAIN_PC+1:N_TRAIN_PC+N_TEST_PC])
    end
    shuffle!(rng, tr); shuffle!(rng, te)

    Xtr_raw = X_ALL[tr, :] ./ 16.0;  Xte_raw = X_ALL[te, :] ./ 16.0
    ytr = [classcol[c] for c in Y_ALL[tr]];  yte = [classcol[c] for c in Y_ALL[te]]
    Xtr = (Xtr_raw .- 0.5) .* π;  Xte = (Xte_raw .- 0.5) .* π     # phase encoding

    N = 64 + N_HIDDEN + n_cls
    input_index  = collect(1:64)
    output_index = collect(N-n_cls+1:N)
    variable_index = setdiff(1:N, input_index)
    Ttr = [ytr[i] == j ? ON : OFF for i in eachindex(ytr), j in 1:n_cls]

    W0 = W_SCALE * randn(rng, N, N); W0 = (W0 + W0') / 2; W0[diagind(W0)] .= 0
    bias0 = zeros(2, N)
    bias0[1, :] .= 0.1 * rand(rng, N)
    bias0[2, :] .= 2π .* (rand(rng, N) .- 0.5)

    function accuracy(W, bias, X, y)
        n = size(X, 1)
        phase0 = zeros(n, N)
        phase0[:, input_index] .= X
        phase0[:, variable_index] .= 0.1 * randn(rng, n, length(variable_index))
        dummy = fill(OFF, n, n_cls)
        eq = run_network_batch(phase0, N_EV*DT, W, bias, dummy, 0.0, input_index, output_index)
        out = eq[:, output_index]
        pred = [argmax(@view out[i, :]) for i in 1:n]
        return mean(pred .== y)
    end

    t0 = time()
    Wf, biasf, ch = train_network(W0, bias0, Xtr, Ttr, BETA, STUDY_RATE,
                                  N_EPOCH, BATCH, N_EV, DT,
                                  input_index, variable_index, output_index;
                                  use_adam=true, symmetric=true, print_every=20)
    dt_train = time() - t0

    xy_tr = accuracy(Wf, biasf, Xtr, ytr)
    xy_te = accuracy(Wf, biasf, Xte, yte)
    lr_te = logreg_accuracy(Xtr_raw, ytr, Xte_raw, yte, n_cls)

    return (classes=classes, n_cls=n_cls, N=N, chance=1/n_cls,
            cost1=ch[1], costN=ch[end], xy_tr=xy_tr, xy_te=xy_te,
            lr_te=lr_te, secs=dt_train, ntr=length(ytr), nte=length(yte))
end

results = NamedTuple[]
for classes in TASKS
    println("=== Task: classes $classes ===")
    r = run_task(classes)
    push!(results, r)
    @printf("  N=%d  train=%d test=%d  cost %.3f->%.3f  %.0fs\n",
            r.N, r.ntr, r.nte, r.cost1, r.costN, r.secs)
    @printf("  XY  train %.3f  test %.3f   |  logreg test %.3f   (chance %.3f)\n\n",
            r.xy_tr, r.xy_te, r.lr_te, r.chance)
end

println("="^64)
@printf("%-18s | %-7s %-9s %-9s %-9s\n", "task", "chance", "XY test", "logreg", "XY train")
println("-"^64)
for r in results
    @printf("%-18s | %-7.3f %-9.3f %-9.3f %-9.3f\n",
            string(r.classes), r.chance, r.xy_te, r.lr_te, r.xy_tr)
end
