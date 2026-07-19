# UDE-SubReservoir.jl
#
# Hybrid reservoir computing: UDE-trained Kuramoto oscillator sub-group
# provides class-discriminative phase context that biases a random ESN reservoir.
#
# Architecture:
#   input u (64-dim optdigits)
#     ├─→ W_in_ude → [UDE sub-group, N_OSC oscillators] → φ_ude (phase features)
#     └─→ [random ESN, N_RES nodes] biased by W_CTX * [sin(φ),cos(φ)]
#   readout: ridge regression on [ESN+bias features; sin(φ_ude); cos(φ_ude)]
#
# UDE training: cross-entropy loss with jointly-trained linear head.
#   Coupling NN and W_cls trained end-to-end so phases become linearly separable.
#
# Four ablations:
#   1. Plain ESN (no UDE)
#   2. UDE phases only (no ESN)
#   3. ESN biased by UDE context + raw UDE phases
#   4. Plain ESN + UDE phases concatenated (no context injection)
#
# Run: julia --project=. src/models/UDE-SubReservoir.jl [--quick]

using ComponentArrays, LinearAlgebra
using CairoMakie
using Random, StableRNGs, Statistics
using Printf, JLD2
import Enzyme   # reverse-mode AD: param-count-independent, lifts the old ForwardDiff batch cap

include(joinpath(@__DIR__, "..", "baselines", "baseline_utils.jl"))
include(joinpath(@__DIR__, "..", "baselines", "baseline_models.jl"))
using .BaselineUtils, .BaselineModels

rng = StableRNG(42)
QUICK = ("--quick" in ARGS) || haskey(ENV, "UDE_QUICK")

# ── Hyperparameters ───────────────────────────────────────────────────────────
const N_IN       = 64       # optdigits features (8×8 flattened)
const N_ROWS     = 8        # image rows — ESN timesteps
const N_COLS     = N_IN ÷ N_ROWS   # pixels per row = 8 (ESN input dim)
const N_CLS      = 10       # digit classes 0-9
const N_OSC      = 10       # UDE oscillators
const N_RES      = 500      # random ESN reservoir nodes
const H_BIAS     = 0.5      # Kuramoto bias strength
const T_SETTLE   = 1.0      # UDE integration time
const DT_UDE     = 0.2      # RK4 step → NSTEPS = 5
const NSTEPS     = round(Int, T_SETTLE / DT_UDE)
const N_PER_CLS  = QUICK ? 5  : 16   # samples per class per mini-batch (Enzyme lifts the old 80-sample ForwardDiff cap)
const N_ITER_1   = QUICK ? 150 : 1500 # Adam iterations
const N_ITER_2   = QUICK ? 50  : 500  # AdamW refinement

# ── Data loading ──────────────────────────────────────────────────────────────
X_all, y_all = load_digits()
tr, te  = stratified_split(y_all, 0.8; rng=rng)
sc      = standardize_fit(X_all[:, tr])
X_train = standardize_apply(X_all[:, tr], sc)
X_test  = standardize_apply(X_all[:, te], sc)
y_train = y_all[tr]
y_test  = y_all[te]
N_train = length(tr)
N_test  = length(te)
println("Dataset: $(size(X_all,2)) samples — train $N_train / test $N_test")

# Pre-group training indices by class for fast balanced sampling
class_pools = [findall(==(c), y_train) for c in 0:9]

# ── Random input projection: N_IN → N_OSC ────────────────────────────────────
const W_IN_UDE = randn(rng, N_OSC, N_IN) ./ sqrt(Float64(N_IN))

# ── Neural network for Kuramoto coupling ──────────────────────────────────────
# 4 → 16 (tanh) → 1 (linear), shared across all directed edges
const NN_INIT = ComponentArray(
    W1    = randn(rng, Float64, 16 * 4)        .* 0.1,
    b1    = zeros(Float64, 16),
    W2    = randn(rng, Float64, 16)            .* 0.1,
    b2    = zeros(Float64, 1),
    W_cls = zeros(Float64, N_CLS * 2 * N_OSC)       # linear head: (N_CLS, 2*N_OSC)
)

# ── Kuramoto RHS (scalar, non-allocating — Enzyme/ForwardDiff friendly) ───────
# coupling weight w_ij = W2·tanh(W1·[sin φj, cos φj, sin φi, cos φi] + b1) + b2,
# shared across all directed edges. A per-edge apply_nn with a Dual matmul is a
# severe AD compile trap; the flat scalar loop compiles fast and Enzyme is happy.
function kuramoto_rhs(φ, p, h_vec, ψ_vec)
    Tp  = promote_type(eltype(φ), eltype(p))
    W1  = reshape(p.W1, 16, 4)
    b1  = p.b1; W2 = p.W2; b2 = p.b2[1]
    out = Vector{Tp}(undef, N_OSC)
    @inbounds for i in 1:N_OSC
        si = sin(φ[i]); ci = cos(φ[i])
        # i-side pre-activation (W1[:,3]·sinφi + W1[:,4]·cosφi + b1) depends only on i;
        # hoist it out of the edge loop (~1.17× faster Enzyme gradient, math unchanged).
        bi = Vector{Tp}(undef, 16)
        for h in 1:16
            bi[h] = W1[h,3]*si + W1[h,4]*ci + b1[h]
        end
        s  = -h_vec[i] * sin(φ[i] - ψ_vec[i])
        for j in 1:N_OSC
            j == i && continue
            sj = sin(φ[j]); cj = cos(φ[j])
            w = b2
            for h in 1:16
                z = W1[h,1]*sj + W1[h,2]*cj + bi[h]
                w += W2[h] * tanh(z)
            end
            s += w * sin(φ[j] - φ[i])
        end
        out[i] = s
    end
    out
end

function settle_phases(ψ_vec, p)
    h = fill(H_BIAS, N_OSC)
    φ = zeros(eltype(p), N_OSC)
    for _ in 1:NSTEPS
        k1 = kuramoto_rhs(φ,                    p, h, ψ_vec)
        k2 = kuramoto_rhs(φ .+ 0.5*DT_UDE.*k1, p, h, ψ_vec)
        k3 = kuramoto_rhs(φ .+ 0.5*DT_UDE.*k2, p, h, ψ_vec)
        k4 = kuramoto_rhs(φ .+    DT_UDE.*k3,  p, h, ψ_vec)
        φ  = φ .+ (DT_UDE/6.0) .* (k1 .+ 2.0.*k2 .+ 2.0.*k3 .+ k4)
    end
    return φ
end

# ── Cross-entropy loss with jointly-trained linear head ───────────────────────
# Settle phases for each sample, form [sin(φ); cos(φ)], apply linear W_cls,
# then compute softmax cross-entropy. Coupling NN and W_cls trained end-to-end.
# Per-sample softmax cross-entropy (no hcat/matrix build — cleaner for Enzyme).
function loss_crossentropy(p, X_batch, y_batch)
    B     = size(X_batch, 2)
    W_cls = reshape(p.W_cls, N_CLS, 2 * N_OSC)
    loss  = zero(eltype(p))
    for n in 1:B
        φ    = settle_phases(W_IN_UDE * X_batch[:, n], p)
        feat = vcat(sin.(φ), cos.(φ))               # (2*N_OSC,)
        lg   = W_cls * feat                         # (N_CLS,)
        m    = maximum(lg)
        lse  = m + log(sum(exp.(lg .- m)))
        loss += lse - lg[y_batch[n] + 1]
    end
    return loss / B
end

# ── Balanced mini-batch sampler ───────────────────────────────────────────────
function sample_batch()
    idx = Int[]
    for pool in class_pools
        n = min(N_PER_CLS, length(pool))
        append!(idx, pool[randperm(rng, length(pool))[1:n]])
    end
    return idx
end

# ── UDE training (Enzyme reverse-mode + manual Adam, resampled mini-batches) ──
# Enzyme first autodiff call per process compiles the pipeline (~5 min, one-time).
enz_grad(p, X, y) = Enzyme.gradient(Enzyme.Reverse, loss_crossentropy,
                                    p, Enzyme.Const(X), Enzyme.Const(y))[1]

function train_ude(p0; iters1=N_ITER_1, iters2=N_ITER_2, lr1=0.005, lr2=0.001, clip=10.0)
    p = copy(p0)
    β1=0.9; β2=0.999; ϵ=1e-8; m = zero(p); v = zero(p); t = 0
    stage(lr, iters, tag) = for it in 1:iters
        bi = sample_batch(); Xb = X_train[:, bi]; yb = y_train[bi]
        g  = enz_grad(p, Xb, yb)
        gn = norm(vec(g)); gn > clip && (g .*= clip/gn)
        t += 1
        @. m = β1*m + (1-β1)*g
        @. v = β2*v + (1-β2)*g*g
        @. p = p - lr * (m/(1-β1^t)) / (sqrt(v/(1-β2^t)) + ϵ)
        it % 50 == 0 && @printf("[%s] iter %4d  xent = %.4f  (min=0, init≈%.4f)\n",
                                tag, t, loss_crossentropy(p, Xb, yb), log(Float64(N_CLS)))
    end
    stage(lr1, iters1, "S1"); stage(lr2, iters2, "S2")
    p
end

bsz = N_CLS * N_PER_CLS
println("\nMini-batch size: $bsz  ($(N_PER_CLS) per class)  — cross-entropy loss (Enzyme AD)")
println("── Stage 1: Adam(0.005) → Stage 2: Adam(0.001) ───────────────────────")
trained_p = train_ude(NN_INIT)
# Evaluate on large batch to report final xent + accuracy
eval_idx  = vcat([pool[1:min(20, length(pool))] for pool in class_pools]...)
eval_X    = X_train[:, eval_idx]
eval_y    = y_train[eval_idx]
final_loss = loss_crossentropy(trained_p, eval_X, eval_y)
# Quick accuracy from W_cls head alone (sanity check on UDE training)
phi_eval  = hcat([settle_phases(W_IN_UDE * eval_X[:, n], trained_p) for n in 1:length(eval_idx)]...)
feat_eval = vcat(sin.(phi_eval), cos.(phi_eval))
W_cls_mat = reshape(trained_p.W_cls, N_CLS, 2*N_OSC)
preds_eval = [argmax(W_cls_mat * feat_eval[:, n]) - 1 for n in 1:length(eval_idx)]
acc_eval   = mean(preds_eval .== eval_y)
@printf("Final xent (eval batch): %.4f  |  UDE head train-subset acc: %.1f%%\n",
        final_loss, 100acc_eval)

# ── Extract UDE phase features for all samples ────────────────────────────────
function ude_phase_features(p, X)
    N = size(X, 2)
    sin_feats = zeros(N_OSC, N)
    cos_feats = zeros(N_OSC, N)
    for n in 1:N
        φ = settle_phases(W_IN_UDE * X[:, n], p)
        sin_feats[:, n] = sin.(φ)
        cos_feats[:, n] = cos.(φ)
    end
    return vcat(sin_feats, cos_feats)   # (2*N_OSC, N)
end

println("\nExtracting UDE phase features (train)...")
F_ude_train = ude_phase_features(trained_p, X_train)
println("Extracting UDE phase features (test)...")
F_ude_test  = ude_phase_features(trained_p, X_test)

# ── ESN — row-step encoding (8 rows × 8 pixels, not 64 scalar steps) ─────────
# Each 8×8 image is presented as 8 timesteps of dim-8 input (one row per tick).
# This preserves horizontal structure and cuts sequence length 8×, eliminating
# the forgetting problem that pixel-by-pixel scanning causes at step 64.
esn = build_esn(N_RES, N_COLS; spectral_radius=0.9, density=0.1,
                input_scale=1.0, leak=0.1, rng=rng)

# Row-aware feature extraction: reshape (64,) → (8, 8), iterate over rows.
# Julia is column-major: reshape(v, 8, 8)[:,t] = t-th group of 8 = t-th image row.
function esn_features_rows(esn, X)
    d, N  = size(X)
    Nr    = size(esn.Wr, 1)
    F     = zeros(2Nr + 1, N)
    for j in 1:N
        U = reshape(X[:, j], N_COLS, N_ROWS)   # (8 pixels, 8 rows)
        r = zeros(Nr)
        S = zeros(Nr, N_ROWS)
        for t in 1:N_ROWS
            pre  = esn.Wr * r .+ esn.Win * vcat(U[:, t], 1.0)
            r    = (1 - esn.leak) .* r .+ esn.leak .* tanh.(pre)
            S[:, t] = r
        end
        F[1:Nr, j]     = S[:, end]
        F[Nr+1:2Nr, j] = vec(mean(S, dims=2))
        F[end, j]      = 1.0
    end
    return F
end

println("Computing plain ESN features...")
F_esn_plain_train = esn_features_rows(esn, X_train)
F_esn_plain_test  = esn_features_rows(esn, X_test)

const W_CTX = randn(rng, N_RES, 2*N_OSC) .* 0.05

function esn_ude_features(esn, X, F_ude, W_ctx)
    Nr = size(esn.Wr, 1)
    N  = size(X, 2)
    F  = zeros(2*Nr + 1, N)
    for n in 1:N
        ctx = W_ctx * F_ude[:, n]
        U   = reshape(X[:, n], N_COLS, N_ROWS)   # (8 pixels, 8 rows)
        r   = zeros(Nr)
        S   = zeros(Nr, N_ROWS)
        for t in 1:N_ROWS
            pre = esn.Wr * r .+ esn.Win * vcat(U[:, t], 1.0) .+ ctx
            r   = (1 - esn.leak) .* r .+ esn.leak .* tanh.(pre)
            S[:, t] = r
        end
        F[1:Nr, n]     = S[:, end]
        F[Nr+1:2Nr, n] = vec(mean(S, dims=2))
        F[end, n]      = 1.0
    end
    return F
end

println("Computing UDE-biased ESN features (train)...")
F_esn_ude_train = esn_ude_features(esn, X_train, F_ude_train, W_CTX)
println("Computing UDE-biased ESN features (test)...")
F_esn_ude_test  = esn_ude_features(esn, X_test,  F_ude_test,  W_CTX)

# ── Ridge regression readout ──────────────────────────────────────────────────
function ridge_fit(F, y; λ=1e-3)
    K = N_CLS
    Y = zeros(K, length(y))
    for (n, c) in enumerate(y)
        Y[c+1, n] = 1.0
    end
    return (Y * F') / (F * F' + λ * I)
end

ridge_predict_labels(W, F) = [argmax(W * F[:, n]) - 1 for n in 1:size(F, 2)]
acc(ŷ, y) = mean(ŷ .== y)

# 1. Plain ESN
W1 = ridge_fit(F_esn_plain_train, y_train)
acc_esn_tr = acc(ridge_predict_labels(W1, F_esn_plain_train), y_train)
acc_esn_te = acc(ridge_predict_labels(W1, F_esn_plain_test),  y_test)

# 2. UDE phases only
F_ude_b_tr = vcat(F_ude_train, ones(1, N_train))
F_ude_b_te = vcat(F_ude_test,  ones(1, N_test))
W2 = ridge_fit(F_ude_b_tr, y_train)
acc_ude_tr = acc(ridge_predict_labels(W2, F_ude_b_tr), y_train)
acc_ude_te = acc(ridge_predict_labels(W2, F_ude_b_te), y_test)

# 3. ESN biased by UDE context + raw UDE phase features
F_comb_tr = vcat(F_esn_ude_train, F_ude_train)
F_comb_te = vcat(F_esn_ude_test,  F_ude_test)
W3 = ridge_fit(F_comb_tr, y_train)
acc_comb_tr = acc(ridge_predict_labels(W3, F_comb_tr), y_train)
acc_comb_te = acc(ridge_predict_labels(W3, F_comb_te), y_test)

# 4. Plain ESN + UDE phases (no context injection — ridge selects what to use)
F_cat_tr = vcat(F_esn_plain_train, F_ude_train)
F_cat_te = vcat(F_esn_plain_test,  F_ude_test)
W4 = ridge_fit(F_cat_tr, y_train)
acc_cat_tr = acc(ridge_predict_labels(W4, F_cat_tr), y_train)
acc_cat_te = acc(ridge_predict_labels(W4, F_cat_te), y_test)

println("\n── Results ──────────────────────────────────────────────────────────")
println("                              Train      Test")
@printf("1. Plain ESN:                 %.2f%%    %.2f%%\n", 100acc_esn_tr, 100acc_esn_te)
@printf("2. UDE only:                  %.2f%%    %.2f%%\n", 100acc_ude_tr, 100acc_ude_te)
@printf("3. ESN (UDE-biased) + UDE:    %.2f%%    %.2f%%  (%+.2fpp)\n",
        100acc_comb_tr, 100acc_comb_te, (acc_comb_te-acc_esn_te)*100)
@printf("4. Plain ESN + UDE (concat):  %.2f%%    %.2f%%  (%+.2fpp)\n",
        100acc_cat_tr, 100acc_cat_te, (acc_cat_te-acc_esn_te)*100)

mkpath(joinpath(@__DIR__, "../../results/models"))
jldsave(joinpath(@__DIR__, "../../results/models/ude_subreservoir.jld2");
        trained_p, W_IN_UDE, W_CTX, acc_esn_te, acc_ude_te, acc_comb_te, acc_cat_te)

# ── Figure ────────────────────────────────────────────────────────────────────
fig = Figure(size=(700, 420))
ax  = CairoMakie.Axis(fig[1, 1];
    title  = "UDE Sub-Reservoir (contrastive) — Test Accuracy (optdigits)",
    ylabel = "Test accuracy (%)",
    xticks = (1:3, ["Plain ESN", "UDE only", "ESN + UDE"]))
accs   = [100acc_esn_te, 100acc_ude_te, 100acc_comb_te]
barplot!(ax, 1:3, accs; color=[:steelblue, :coral, :seagreen], width=0.55)
ylims!(ax, max(0.0, minimum(accs) - 10), 100)
for (i, a) in enumerate(accs)
    text!(ax, i, a + 0.5; text=@sprintf("%.1f%%", a),
          align=(:center, :bottom), fontsize=13)
end
figdir = joinpath(@__DIR__, "..", "..", "results", "figures")
isdir(figdir) || mkpath(figdir)
save(joinpath(figdir, "ude_subreservoir_accuracy.png"), fig)
println("Saved: results/figures/ude_subreservoir_accuracy.png")
