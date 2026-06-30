# Training at the sweet-spot depth: can we turn the s=0.5 operating-point gain
# (basin tracking: 86% vs 71% at s=1.0) into a trained, checkpointed result?
#
# Basin tracking (results/ep_duffing_basin_tracking.md) showed the round-1
# champion -- annealed to s=1.0 -- is most accurate when EVALUATED at s=0.5
# (mean 86% vs 71% at s=1.0). But the static sweep showed FIXED training at
# s=0.5 reaches only 57%. So the good weights come from the curriculum, and the
# good operating point is s=0.5. This script crosses the two factors directly:
#
#   train mode           eval s   prior reference
#   ------------------   ------   ---------------------------------------------
#   fixed   s=0.5        0.5      static sweep ~57%
#   anneal  ->0.5        0.5      NEW -- curriculum that STOPS at the sweet spot
#   anneal  ->1.0        1.0      round-1 champion ~73%
#   anneal  ->1.0        0.5      basin tracking ~86% (good weights, good op pt)
#
# a = -s, c = s keeps minima at +-1 for every depth; only the barrier (s/4)
# changes, so eval depth is a free operating lever. hidden=2, 6 seeds, 2000
# epochs, 40 init draws (matches basin tracking for comparability).
#
# Usage: julia --project=. scripts/duffing_sweetspot.jl

using Random, Printf, Statistics

EP_DUFFING_SKIP_RUN = true
include(joinpath(@__DIR__, "..", "notebooks", "EP-Duffing-Network.jl"))

const SEEDS       = 1:6
const N_EPOCH     = 2000
const ANNEAL_FRAC = 0.5
const N_HIDDEN    = 2
const N_TEST_INIT = 40
const TEST_NOISE  = 0.1
const S_START     = 0.05

data   = Float64[-1 -1; -1 1; 1 -1; 1 1]
target = reshape(Float64[-1, 1, 1, -1], 4, 1)

# Linear ramp S_START -> s_target over the first frac of training, then hold.
# frac = 0 => fixed at s_target the whole run.
function s_at(epoch, n_epoch, frac, s_target)
    frac <= 0.0 && return s_target
    n_ramp = max(1, round(Int, frac * n_epoch))
    epoch >= n_ramp && return s_target
    return S_START + (s_target - S_START) * (epoch - 1) / (n_ramp == 1 ? 1 : n_ramp - 1)
end

# Train to a target depth (annealed or fixed); checkpoint best free cost once s
# has reached s_target so restored weights are valid at that depth.
function train_to!(net, beta, lr, n_epoch, frac, s_target; noise=0.1, rng=Random.default_rng())
    N = net.N; N_data = size(data, 1)
    s_W = zeros(N, N); r_W = zeros(N, N); s_h = zeros(N); r_h = zeros(N)
    best_cost = Inf; best_W = copy(net.W); best_h = copy(net.h)
    for epoch in 1:n_epoch
        s = s_at(epoch, n_epoch, frac, s_target); net.a = -s; net.c = s
        x0 = zeros(N_data, N)
        x0[:, net.input_index] .= data
        x0[:, net.variable_index] .= noise * randn(rng, N_data, length(net.variable_index))
        gW, gh, cost, _ = EP_param_gradient(net, x0, target, beta; symmetric=true)
        net.W, s_W, r_W = adam_update(net.W, gW, lr, epoch, s_W, r_W)
        net.W = (net.W + net.W') / 2; net.W[diagind(net.W)] .= 0
        net.h, s_h, r_h = adam_update(net.h, gh, lr, epoch, s_h, r_h)
        if s == s_target && cost < best_cost
            best_cost = cost; best_W = copy(net.W); best_h = copy(net.h)
        end
    end
    net.W = best_W; net.h = best_h
    return best_cost
end

function robust_accuracy(net, input_idx, var_idx, out_idx, s_eval, rng)
    net.a = -s_eval; net.c = s_eval
    correct = 0
    for i in 1:4, _ in 1:N_TEST_INIT
        x0 = zeros(1, net.N)
        x0[1, input_idx] .= data[i, :]
        x0[1, var_idx]   .= TEST_NOISE * randn(rng, length(var_idx))
        eq = relax_batch(net, x0, reshape(target[i, :], 1, :), 0.0)
        sign(eq[1, out_idx[1]]) == sign(target[i, 1]) && (correct += 1)
    end
    return correct / (4 * N_TEST_INIT)
end

# (label, anneal_frac, s_target, s_eval)
CONFIGS = [
    ("fixed  s=0.5 -> eval 0.5", 0.0,         0.5, 0.5),
    ("anneal->0.5  -> eval 0.5", ANNEAL_FRAC, 0.5, 0.5),
    ("anneal->1.0  -> eval 1.0", ANNEAL_FRAC, 1.0, 1.0),
    ("anneal->1.0  -> eval 0.5", ANNEAL_FRAC, 1.0, 0.5),
]

Nn = 2 + N_HIDDEN + 1
input_idx = [1, 2]; out_idx = [Nn]; var_idx = setdiff(1:Nn, input_idx)

println("EP-Duffing sweet-spot training -- robust XOR accuracy (", length(SEEDS),
        " seeds x ", N_TEST_INIT, " init draws, test noise ", TEST_NOISE, ")")
println("hidden=", N_HIDDEN, ", ", N_EPOCH, " epochs, anneal frac=", ANNEAL_FRAC, "\n")
@printf("%-26s | %-10s %-10s %-10s | %s\n",
        "config", "mean acc", "max acc", "min acc", "median best-cost")
println("-"^74)

for (label, frac, s_target, s_eval) in CONFIGS
    accs = Float64[]; bestc = Float64[]
    for seed in SEEDS
        rng = MersenneTwister(seed)
        net = DuffingNetwork(Nn, input_idx, out_idx; a=-s_target, c=s_target, delta=1.0)
        random_init!(net; rng=rng)
        bc = train_to!(net, 0.1, 0.02, N_EPOCH, frac, s_target; rng=rng)
        push!(accs, robust_accuracy(net, input_idx, var_idx, out_idx, s_eval,
                                    MersenneTwister(9000 + seed)))
        push!(bestc, bc)
    end
    @printf("%-26s | %-10.0f %-10.0f %-10.0f | %.4f\n",
            label, 100*mean(accs), 100*maximum(accs), 100*minimum(accs), median(bestc))
end
