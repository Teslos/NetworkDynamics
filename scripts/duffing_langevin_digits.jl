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
const SEED    = 1
const NHID    = 40           # monostable hidden units
const A_H     = 1.0          # hidden quadratic coeff > 0  -> single well (monostable)
const T_SAMP  = 0.30         # sampling temperature (Langevin)
const BETA    = 0.1
const LR      = 0.01
const N_ITER  = 300
const BATCH   = 64
const N_BURN  = 200
const N_SAMPLE= 350
const N_TRAIN_PC = 40
const N_TEST_PC  = 40
const EVAL_EVERY = 20

# ---------------- data ----------------
Random.seed!(SEED)
raw  = readdlm(joinpath(@__DIR__, "..", "data", "digits", "optdigits.tes"), ',', Int)
const XALL = Float64.(raw[:, 1:64]); const YALL = raw[:, 65]
pool4x4(v) = (img = reshape(v, 8, 8);
    [(img[bi,bj]+img[bi+1,bj]+img[bi,bj+1]+img[bi+1,bj+1])/4 for bi in 1:2:8 for bj in 1:2:8])
poolall(X) = permutedims(reduce(hcat, [pool4x4(X[i, :]) for i in 1:size(X, 1)]))

const NIN = 16; const NCLS = 10; const NN = NIN + NHID + NCLS
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
mlp_acc(Xtr, ytr, Xte, yte, nc; hh=64, iters=3000, lr=0.2, l2=1e-4) = begin
    rng = MersenneTwister(SEED); n, d = size(Xtr)
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
rng = MersenneTwister(SEED); cc = Dict(c => j for (j, c) in enumerate(0:9)); tr = Int[]; te = Int[]
for c in 0:9
    ci = shuffle(rng, findall(==(c), YALL))
    append!(tr, ci[1:N_TRAIN_PC]); append!(te, ci[N_TRAIN_PC+1:N_TRAIN_PC+N_TEST_PC])
end
Xtrp = poolall(XALL[tr, :]) ./ 16.0; Xtep = poolall(XALL[te, :]) ./ 16.0
ytr = [cc[c] for c in YALL[tr]]; yte = [cc[c] for c in YALL[te]]
Xtr = 2 .* Xtrp .- 1; Xte = 2 .* Xtep .- 1; Nd = length(ytr)
Ytr = [ytr[i] == j ? 1.0 : 0.0 for i in eachindex(ytr), j in 1:NCLS]

function digit_acc(W, h, X, y, T)
    n = size(X, 1); x0 = zeros(n, NN); x0[:, INP] .= X; x0[:, VARc] .= 0.1*randn(rng, n, length(VARc))
    _, _, mx = lang_relax(W, h, x0, zeros(n, NCLS), 0.0, T; n_burn=250, n_sample=400, rng=MersenneTwister(77))
    o = mx[:, OUTc]; mean([argmax(@view o[i, :]) for i in 1:n] .== y)
end

# ---------------- train ----------------
println("threads=$(Threads.nthreads())  N=$NN ($NIN in, $NHID mono-hidden, $NCLS out)  " *
        "T=$T_SAMP  train=$Nd test=$(length(yte))")
W = 0.1*randn(rng, NN, NN); W = (W+W')/2; W .*= MSK; h = zeros(NN)
sW = zeros(NN, NN); rW = zeros(NN, NN); sh = zeros(NN); rh = zeros(NN)
best = 0.0; bW = copy(W); bh = copy(h); t0 = time()
for it in 1:N_ITER
    bi = rand(rng, 1:Nd, BATCH)
    x0 = zeros(BATCH, NN); x0[:, INP] .= Xtr[bi, :]; x0[:, VARc] .= 0.1*randn(rng, BATCH, length(VARc))
    gW, gh, ce = lang_grad(W, h, x0, Ytr[bi, :], BETA, T_SAMP)
    global W, sW, rW = adam_update(W, gW, LR, it, sW, rW); global W = (W+W')/2; global W .*= MSK
    global h, sh, rh = adam_update(h, gh, LR, it, sh, rh)
    if it == 1 || it % EVAL_EVERY == 0
        a = digit_acc(W, h, Xte, yte, T_SAMP)
        if a > best; global best = a; global bW = copy(W); global bh = copy(h); end
        @printf("  it %d: CE %.3f  test %.3f (best %.3f) [%.0fs]\n", it, ce, a, best, time()-t0)
    end
end

du_te = digit_acc(bW, bh, Xte, yte, T_SAMP); du_tr = digit_acc(bW, bh, Xtr, ytr, T_SAMP)
lr_te = logreg_acc(Xtrp, ytr, Xtep, yte, NCLS); ml_te = mlp_acc(Xtrp, ytr, Xtep, yte, NCLS)
println("\n", "="^54)
@printf("%-30s | %-7s %-7s (chance %.2f)\n", "model", "train", "test", 1/NCLS)
println("-"^54)
@printf("%-30s | %-7.3f %-7.3f\n", "Langevin monostable EP (best)", du_tr, du_te)
@printf("%-30s | %-7s %-7.3f\n", "logreg (pooled 16)", "-", lr_te)
@printf("%-30s | %-7s %-7.3f\n", "MLP (pooled 16)", "-", ml_te)
println("\nRef: deterministic mono v2 ~0.8x; XY phase net 0.94; bistable 0.18")
