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

const STEADY_TOL_S2B = 1e-3
steady_state_callback() = DiscreteCallback(
    (u, t, integrator) -> maximum(abs, get_du(integrator)) < STEADY_TOL_S2B,
    terminate!; save_positions=(false, false))

# ---------------------------------------------------------------- config (re-spent budget)
const SEED       = 1
const CLASSES    = collect(0:9)
const N_TRAIN_PC = 70
const N_TEST_PC  = 35
const N_HIDDEN   = 40
const N_EV       = 600            # T = 60 (same horizon as Stage 2, controlled)
const DT         = 0.1
const BETA       = 0.01
const STUDY_RATE = 0.05
const N_EPOCH    = 150
const BATCH      = 50
const W_SCALE    = 0.1
const L2         = 1e-4
const ON, OFF    = π/2, -π/2

Random.seed!(SEED)
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

function mlp_accuracy(Xtr, ytr, Xte, yte, n_class; h=64, iters=3000, lr=0.2, l2=1e-4)
    rng = MersenneTwister(SEED); n, d = size(Xtr)
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

function train_xy_l2(W0, bias0, Xtr, Ttr, input_index, variable_index, output_index; rng)
    N = size(W0, 1); N_data = size(Xtr, 1)
    W = copy(W0); bias = copy(bias0)
    sW = zeros(size(W)); rW = zeros(size(W)); sB = zeros(size(bias)); rB = zeros(size(bias))
    best_cost = Inf; bestW = copy(W); bestB = copy(bias); ch = zeros(N_EPOCH)
    for epoch in 1:N_EPOCH
        perm = shuffle(rng, 1:N_data); ec = 0.0
        for s in 1:BATCH:N_data
            bidx = perm[s:min(s+BATCH-1, N_data)]
            phase0 = zeros(length(bidx), N)
            phase0[:, input_index] .= Xtr[bidx, :]
            phase0[:, variable_index] .= 0.1 * randn(rng, length(bidx), length(variable_index))
            gW, gB, cost, _ = EP_param_gradient(W, bias, phase0, Ttr[bidx, :], BETA,
                                                N_EV, DT, input_index, variable_index,
                                                output_index; symmetric=true)
            gW = gW .+ L2 .* W
            W, sW, rW = Adam_update(W, gW, STUDY_RATE, epoch, sW, rW)
            bias, sB, rB = Adam_update(bias, gB, STUDY_RATE, epoch, sB, rB)
            ec += cost * length(bidx)
        end
        ch[epoch] = ec / N_data
        if ch[epoch] < best_cost; best_cost = ch[epoch]; bestW = copy(W); bestB = copy(bias); end
        if epoch == 1 || epoch % 15 == 0; @printf("  epoch %d: cost %.4f\n", epoch, ch[epoch]); end
    end
    return bestW, bestB, ch
end

# ---------------------------------------------------------------- run
rng = MersenneTwister(SEED)
classcol = Dict(c => j for (j, c) in enumerate(CLASSES))
tr = Int[]; te = Int[]
for c in CLASSES
    ci = shuffle(rng, findall(==(c), Y_ALL))
    append!(tr, ci[1:N_TRAIN_PC]); append!(te, ci[N_TRAIN_PC+1:N_TRAIN_PC+N_TEST_PC])
end
shuffle!(rng, tr); shuffle!(rng, te)

Xtr_p = pool_all(X_ALL[tr, :]) ./ 16.0;  Xte_p = pool_all(X_ALL[te, :]) ./ 16.0
ytr = [classcol[c] for c in Y_ALL[tr]];  yte = [classcol[c] for c in Y_ALL[te]]
Xtr = (Xtr_p .- 0.5) .* π;  Xte = (Xte_p .- 0.5) .* π

const N_CLS = length(CLASSES)
const N = 16 + N_HIDDEN + N_CLS
input_index  = collect(1:16)
output_index = collect(N-N_CLS+1:N)
variable_index = setdiff(1:N, input_index)
Ttr = [ytr[i] == j ? ON : OFF for i in eachindex(ytr), j in 1:N_CLS]

println("Full 10-class: N=$N (16 inputs, $N_HIDDEN hidden, $N_CLS outputs), ",
        "train=$(length(ytr)), test=$(length(yte))\n")

W0 = W_SCALE * randn(rng, N, N); W0 = (W0 + W0') / 2; W0[diagind(W0)] .= 0
bias0 = zeros(2, N); bias0[1, :] .= 0.1*rand(rng, N); bias0[2, :] .= 2π .* (rand(rng, N) .- 0.5)

function xy_accuracy(W, bias, X, y)
    n = size(X, 1)
    phase0 = zeros(n, N); phase0[:, input_index] .= X
    phase0[:, variable_index] .= 0.1 * randn(rng, n, length(variable_index))
    eq = run_network_batch(phase0, N_EV*DT, W, bias, fill(OFF, n, N_CLS), 0.0, input_index, output_index)
    out = eq[:, output_index]
    return mean([argmax(@view out[i, :]) for i in 1:n] .== y)
end

println("Training XY net (symmetric EP, L2)...")
t0 = time()
Wf, biasf, ch = train_xy_l2(W0, bias0, Xtr, Ttr, input_index, variable_index, output_index; rng=rng)
secs = time() - t0

xy_tr = xy_accuracy(Wf, biasf, Xtr, ytr); xy_te = xy_accuracy(Wf, biasf, Xte, yte)
lr_tr = logreg_accuracy(Xtr_p, ytr, Xtr_p, ytr, N_CLS); lr_te = logreg_accuracy(Xtr_p, ytr, Xte_p, yte, N_CLS)
ml_tr = mlp_accuracy(Xtr_p, ytr, Xtr_p, ytr, N_CLS);    ml_te = mlp_accuracy(Xtr_p, ytr, Xte_p, yte, N_CLS)

@printf("\ntrained %d epochs in %.0fs  (cost %.3f -> %.3f, min %.3f)\n",
        N_EPOCH, secs, ch[1], ch[end], minimum(ch))
println("="^58)
@printf("%-10s | %-10s %-10s   (chance %.3f, 4x4 inputs)\n", "model", "train acc", "test acc", 1/N_CLS)
println("-"^58)
@printf("%-10s | %-10.3f %-10.3f\n", "XY (EP)", xy_tr, xy_te)
@printf("%-10s | %-10.3f %-10.3f\n", "logreg",  lr_tr, lr_te)
@printf("%-10s | %-10.3f %-10.3f\n", "MLP",     ml_tr, ml_te)
println("\nStage 2 was: XY 0.792/0.767, logreg 0.852/0.843, MLP 0.962/0.883 (train/test)")
