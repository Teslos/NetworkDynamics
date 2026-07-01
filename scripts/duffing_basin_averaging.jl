# Basin-averaging for the bistable EP-Duffing network: port Wang et al.'s
# multistability remedy (the fix that took our EP-XY 10-class run from 0.80 to
# 0.94) to the substrate where we first hit the bistable-forward-inference wall.
#
# Diagnosis recap (results/ep_duffing_*.md): EP gradients on the Duffing net are
# faithful, but the double-well's FREE equilibrium picks its output basin from the
# hidden/output INITIALIZATION, not the input -> output is not a deterministic
# function of the input -> robust XOR ~ chance. Every prior Duffing run initialized
# the free cells with tiny noise near x=0 (one basin, no averaging) -- exactly the
# mistake we later fixed for XY.
#
# Wang's remedy: initialize the free (hidden AND output) cells over the FULL range
# spanning both wells every step, and AVERAGE the EP gradient over many random
# initializations (Minit). This trains all stable fixed points simultaneously,
# forcing the output to become a deterministic function of the input regardless of
# basin. XOR has only 4 patterns, so we set Minit explicitly (Wang's large-batch
# argument doesn't apply). We build one big batch of Minit x 4 rows -- each pattern
# repeated with independent full-range inits -- so EP_param_gradient averages the
# gradient over (patterns x inits) in a single call.
#
# HONEST TEST: robust accuracy is measured with free cells initialized over the
# FULL range [-1.5, 1.5] (both wells), i.e. the actual test of basin-invariance --
# stricter than the near-0 test-noise (0.1) used in earlier Duffing scripts.
# Evaluated at the deep well (s=1) and at the s=0.5 sweet spot.
#
# Usage: julia --project=. scripts/duffing_basin_averaging.jl

using Random, Printf, Statistics

EP_DUFFING_SKIP_RUN = true
include(joinpath(@__DIR__, "..", "notebooks", "EP-Duffing-Network.jl"))

const SEEDS       = 1:5
const N_EPOCH     = 1000
const N_HIDDEN    = 2               # N = 2 inputs + 2 hidden + 1 output = 5
const BETA        = 0.1
const LR          = 0.02
const TEST_RANGE  = 1.5             # full-range test init (wells at +-1)
const N_TEST_INIT = 20
const S_EVAL      = [1.0, 0.5]      # deep well, and the sweet-spot operating depth

# (label, init-mode, Minit).  :near0 = 0.1*randn (old protocol); :full = uniform.
const CONFIGS = [
    ("near0  Minit=1  (control)", :near0, 1),
    ("full   Minit=1",            :full,  1),
    ("full   Minit=10",           :full,  10),
    ("full   Minit=20",           :full,  20),
]

data   = Float64[-1 -1; -1 1; 1 -1; 1 1]
target = reshape(Float64[-1, 1, 1, -1], 4, 1)

# Draw free-cell initial phases for one row.
draw_init(mode, n, rng) = mode === :near0 ? 0.1 .* randn(rng, n) :
                                            TEST_RANGE .* (2 .* rand(rng, n) .- 1)

# Basin-averaging trainer: each step builds Minit*4 rows (each XOR pattern repeated
# Minit times with independent full-range free-cell inits); EP_param_gradient then
# averages the gradient over patterns x inits. One-sided gradient (Wang-style).
function train_basin!(net, input_idx, var_idx, mode, Minit; rng)
    N = net.N
    s_W = zeros(N, N); r_W = zeros(N, N); s_h = zeros(N); r_h = zeros(N)
    best_cost = Inf; best_W = copy(net.W); best_h = copy(net.h); ch = zeros(N_EPOCH)
    nrow = 4 * Minit
    for epoch in 1:N_EPOCH
        x0  = zeros(nrow, N); tgt = zeros(nrow, 1); row = 1
        for p in 1:4, _ in 1:Minit
            x0[row, input_idx] .= data[p, :]
            x0[row, var_idx]   .= draw_init(mode, length(var_idx), rng)
            tgt[row, 1] = target[p, 1]; row += 1
        end
        gW, gh, cost, _ = EP_param_gradient(net, x0, tgt, BETA; symmetric=false)
        net.W, s_W, r_W = adam_update(net.W, gW, LR, epoch, s_W, r_W)
        net.W = (net.W + net.W') / 2; net.W[diagind(net.W)] .= 0
        net.h, s_h, r_h = adam_update(net.h, gh, LR, epoch, s_h, r_h)
        ch[epoch] = cost
        if cost < best_cost; best_cost = cost; best_W = copy(net.W); best_h = copy(net.h); end
    end
    net.W = best_W; net.h = best_h
    return ch, best_cost
end

# Robust XOR accuracy at well depth s_eval, with FULL-RANGE test init (both wells).
function robust_acc(net, input_idx, var_idx, out_idx, s_eval, rng)
    net.a = -s_eval; net.c = s_eval
    correct = 0
    for i in 1:4, _ in 1:N_TEST_INIT
        x0 = zeros(1, net.N)
        x0[1, input_idx] .= data[i, :]
        x0[1, var_idx]   .= TEST_RANGE .* (2 .* rand(rng, length(var_idx)) .- 1)
        eq = relax_batch(net, x0, reshape(target[i, :], 1, :), 0.0)
        sign(eq[1, out_idx[1]]) == sign(target[i, 1]) && (correct += 1)
    end
    return correct / (4 * N_TEST_INIT)
end

Nn = 2 + N_HIDDEN + 1
input_idx = [1, 2]; out_idx = [Nn]; var_idx = setdiff(1:Nn, input_idx)

println("EP-Duffing basin-averaging -- robust XOR accuracy under FULL-RANGE test init")
println(length(SEEDS), " seeds x ", N_TEST_INIT, " draws (uniform [-", TEST_RANGE,
        ",", TEST_RANGE, "]), ", N_EPOCH, " epochs, one-sided grad, deep-well training\n")
@printf("%-26s | %-18s | %-18s | %s\n",
        "config", "robust@deep s=1", "robust@sweet s=0.5", "med final cost")
@printf("%-26s | %-6s %-5s %-5s | %-6s %-5s %-5s |\n",
        "", "mean", "max", "min", "mean", "max", "min")
println("-"^82)

for (label, mode, Minit) in CONFIGS
    accD = Float64[]; accS = Float64[]; fc = Float64[]
    for seed in SEEDS
        rng = MersenneTwister(seed)
        net = DuffingNetwork(Nn, input_idx, out_idx; a=-1.0, c=1.0, delta=1.0)
        random_init!(net; rng=rng)
        ch, _ = train_basin!(net, input_idx, var_idx, mode, Minit; rng=rng)
        push!(accD, robust_acc(net, input_idx, var_idx, out_idx, 1.0, MersenneTwister(9000+seed)))
        push!(accS, robust_acc(net, input_idx, var_idx, out_idx, 0.5, MersenneTwister(9000+seed)))
        push!(fc, ch[end])
    end
    @printf("%-26s | %-6.0f %-5.0f %-5.0f | %-6.0f %-5.0f %-5.0f | %.3f\n",
            label, 100mean(accD), 100maximum(accD), 100minimum(accD),
            100mean(accS), 100maximum(accS), 100minimum(accS), median(fc))
end
