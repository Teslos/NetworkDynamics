# Per-pattern basin tracking: WHICH XOR patterns flip into the wrong basin as the
# Duffing wells deepen?
#
# The branch's headline is that bistable FORWARD INFERENCE -- not the EP gradient
# -- is what blocks robust Duffing XOR. The anneal experiments support this
# indirectly (start smooth -> partial rescue). This script observes the mechanism
# DIRECTLY: train the round-1 champion (hidden=2, linear anneal frac=0.5, 2000
# epochs), then freeze the weights and sweep the EVALUATION well depth s_eval from
# near-flat to deep, measuring per-pattern sign-correctness over many init draws.
#
# Prediction if the diagnosis is right:
#   * shallow s_eval (input-determined, smooth inference): all 4 patterns ~100%
#     sign-correct -- the learned map is intact.
#   * deep s_eval (bistable): some pattern(s) collapse toward ~50% as the
#     hidden-cell init, not the input, starts selecting the output basin.
# The pattern(s) that collapse are the ones whose correct output basin is the
# harder-to-reach one given the input clamp.
#
# Same potential as everywhere: a = -s, c = s -> minima fixed at +-1, barrier s/4.
#
# Usage: julia --project=. scripts/duffing_basin_tracking.jl

using Random, Printf, Statistics

EP_DUFFING_SKIP_RUN = true
include(joinpath(@__DIR__, "..", "notebooks", "EP-Duffing-Network.jl"))

const SEEDS       = 1:6
const N_EPOCH     = 2000        # round-1 champion config
const ANNEAL_FRAC = 0.5
const N_HIDDEN    = 2
const N_TEST_INIT = 40          # more draws -> tighter per-pattern fractions
const TEST_NOISE  = 0.1
const S_START     = 0.05
const S_FINAL     = 1.0
const S_EVAL      = [0.05, 0.1, 0.2, 0.5, 1.0]   # evaluation well depths

data   = Float64[-1 -1; -1 1; 1 -1; 1 1]
target = reshape(Float64[-1, 1, 1, -1], 4, 1)
const PAT_LABELS = ["(-1,-1)->-1", "(-1,+1)->+1", "(+1,-1)->+1", "(+1,+1)->-1"]

s_at(epoch, n_epoch, frac) =
    frac <= 0.0 ? S_FINAL :
    (n_ramp = max(1, round(Int, frac * n_epoch));
     epoch >= n_ramp ? S_FINAL :
     S_START + (S_FINAL - S_START) * (epoch - 1) / (n_ramp == 1 ? 1 : n_ramp - 1))

# Round-1 annealing trainer (checkpoint only at the deep well).
function train_anneal!(net, data, target, beta, lr, n_epoch, frac;
                       noise=0.1, rng=Random.default_rng())
    N = net.N; N_data = size(data, 1)
    s_W = zeros(N, N); r_W = zeros(N, N); s_h = zeros(N); r_h = zeros(N)
    best_cost = Inf; best_W = copy(net.W); best_h = copy(net.h)
    for epoch in 1:n_epoch
        s = s_at(epoch, n_epoch, frac); net.a = -s; net.c = s
        x0 = zeros(N_data, N)
        x0[:, net.input_index] .= data
        x0[:, net.variable_index] .= noise * randn(rng, N_data, length(net.variable_index))
        gW, gh, cost, _ = EP_param_gradient(net, x0, target, beta; symmetric=true)
        net.W, s_W, r_W = adam_update(net.W, gW, lr, epoch, s_W, r_W)
        net.W = (net.W + net.W') / 2; net.W[diagind(net.W)] .= 0
        net.h, s_h, r_h = adam_update(net.h, gh, lr, epoch, s_h, r_h)
        if s == S_FINAL && cost < best_cost
            best_cost = cost; best_W = copy(net.W); best_h = copy(net.h)
        end
    end
    net.a = -S_FINAL; net.c = S_FINAL; net.W = best_W; net.h = best_h
    return net
end

# Per-pattern sign-correct fraction over init draws at the net's current depth.
function per_pattern_acc(net, input_idx, var_idx, out_idx, rng)
    acc = zeros(4)
    for i in 1:4
        c = 0
        for _ in 1:N_TEST_INIT
            x0 = zeros(1, net.N)
            x0[1, input_idx] .= data[i, :]
            x0[1, var_idx]   .= TEST_NOISE * randn(rng, length(var_idx))
            eq = relax_batch(net, x0, reshape(target[i, :], 1, :), 0.0)
            sign(eq[1, out_idx[1]]) == sign(target[i, 1]) && (c += 1)
        end
        acc[i] = c / N_TEST_INIT
    end
    return acc
end

Nn = 2 + N_HIDDEN + 1
input_idx = [1, 2]; out_idx = [Nn]; var_idx = setdiff(1:Nn, input_idx)

# acc_by_s[k] = 4-vector of per-pattern accuracy at S_EVAL[k], averaged over seeds.
acc_by_s = [zeros(4) for _ in S_EVAL]
for seed in SEEDS
    rng = MersenneTwister(seed)
    net = DuffingNetwork(Nn, input_idx, out_idx; a=-S_FINAL, c=S_FINAL, delta=1.0)
    random_init!(net; rng=rng)
    train_anneal!(net, data, target, 0.1, 0.02, N_EPOCH, ANNEAL_FRAC; rng=rng)
    for (k, s) in enumerate(S_EVAL)
        net.a = -s; net.c = s
        acc_by_s[k] .+= per_pattern_acc(net, input_idx, var_idx, out_idx,
                                        MersenneTwister(9000 + seed)) ./ length(SEEDS)
    end
end

println("EP-Duffing per-pattern basin tracking (champion config: hidden=", N_HIDDEN,
        ", linear anneal frac=", ANNEAL_FRAC, ", ", N_EPOCH, " epochs)")
println(length(SEEDS), " seeds x ", N_TEST_INIT, " init draws, test noise ",
        TEST_NOISE, "; sign-correct fraction per XOR pattern vs eval well depth\n")
@printf("%-8s | %-12s %-12s %-12s %-12s | %s\n",
        "s_eval", PAT_LABELS[1], PAT_LABELS[2], PAT_LABELS[3], PAT_LABELS[4], "mean")
println("-"^78)
for (k, s) in enumerate(S_EVAL)
    a = acc_by_s[k]
    @printf("%-8.2f | %-12.0f %-12.0f %-12.0f %-12.0f | %.0f\n",
            s, 100a[1], 100a[2], 100a[3], 100a[4], 100 * mean(a))
end
