# Barrier annealing for the EP-Duffing network: does growing the double-well
# barrier DURING training rescue XOR?
#
# Motivation. The static design sweep (scripts/duffing_design_sweep.jl) showed
# that reshaping the wells at a FIXED depth leaves robust XOR accuracy at chance:
# the forward equilibrium is already bistable from epoch 1, so the output basin
# is picked by the hidden-cell init, not the input, before training can steer it.
#
# Annealing attacks exactly that. With a = -s, c = s the minima stay pinned at
# x = +-1 (so the +-1 encoding is unchanged) while the barrier depth = s/4.
#   * Start near-flat (s ~ s_start): the potential barely constrains the cells,
#     so the free equilibrium is INPUT-DETERMINED and smooth -- the regime where
#     the XY/Kuramoto net solves XOR robustly.
#   * Ramp s up to s_final = 1.0: the wells deepen and LOCK IN the +-1 levels,
#     ideally around an XOR map already learned in the easy regime.
#
# Falsifiable prediction: if bistable forward inference is THE bottleneck, a slow
# enough anneal recovers robust XOR; the no-anneal control stays at chance.
#
# We checkpoint the best free-phase cost only once s has reached s_final, so the
# restored weights are valid at the target (deep-well) substrate, and we score
# ROBUST accuracy (sign-correct over many init draws) at s_final.
#
# Usage: julia --project=. scripts/duffing_anneal.jl

using Random, Printf, Statistics

EP_DUFFING_SKIP_RUN = true
include(joinpath(@__DIR__, "..", "notebooks", "EP-Duffing-Network.jl"))

const SEEDS       = 1:6
const N_EPOCH     = 2000
const N_TEST_INIT = 15
const TEST_NOISE  = 0.1
const S_START     = 0.05      # near-flat potential at the start of an anneal
const S_FINAL     = 1.0       # target deep well (barrier depth 0.25, minima +-1)
const N_HIDDEN    = 2         # N = 2 inputs + N_HIDDEN + 1 output

# anneal_frac: fraction of epochs spent ramping s from S_START to S_FINAL
# (linearly). 0.0 = no anneal (control, s = S_FINAL throughout). The remaining
# epochs hold at S_FINAL.
const SCHEDULES = [0.0, 0.5, 0.9]

data   = Float64[-1 -1; -1 1; 1 -1; 1 1]
target = reshape(Float64[-1, 1, 1, -1], 4, 1)

# s for a given epoch under a linear ramp over the first anneal_frac of training.
function s_at(epoch, n_epoch, anneal_frac)
    anneal_frac <= 0.0 && return S_FINAL
    n_ramp = max(1, round(Int, anneal_frac * n_epoch))
    epoch >= n_ramp && return S_FINAL
    return S_START + (S_FINAL - S_START) * (epoch - 1) / (n_ramp - 1 == 0 ? 1 : n_ramp - 1)
end

# Annealing trainer: mirrors train! in EP-Duffing-Network.jl but (1) sets the
# well depth per epoch via net.a / net.c, and (2) only checkpoints the best free
# cost once s has reached S_FINAL, so restored weights are valid at the deep well.
function train_anneal!(net, data, target, beta, lr, n_epoch, anneal_frac;
                       noise=0.1, rng=Random.default_rng())
    N = net.N
    N_data = size(data, 1)
    cost_history = zeros(n_epoch)
    s_W = zeros(N, N); r_W = zeros(N, N)
    s_h = zeros(N);    r_h = zeros(N)

    best_cost = Inf
    best_W = copy(net.W); best_h = copy(net.h)

    for epoch in 1:n_epoch
        s = s_at(epoch, n_epoch, anneal_frac)
        net.a = -s; net.c = s

        x0 = zeros(N_data, N)
        x0[:, net.input_index] .= data
        x0[:, net.variable_index] .= noise * randn(rng, N_data, length(net.variable_index))

        gW, gh, cost, _ = EP_param_gradient(net, x0, target, beta; symmetric=true)

        net.W, s_W, r_W = adam_update(net.W, gW, lr, epoch, s_W, r_W)
        net.W = (net.W + net.W') / 2
        net.W[diagind(net.W)] .= 0
        net.h, s_h, r_h = adam_update(net.h, gh, lr, epoch, s_h, r_h)

        cost_history[epoch] = cost
        if s == S_FINAL && cost < best_cost     # only checkpoint at the deep well
            best_cost = cost
            best_W = copy(net.W); best_h = copy(net.h)
        end
    end

    net.a = -S_FINAL; net.c = S_FINAL           # leave net at the target substrate
    net.W = best_W; net.h = best_h
    return cost_history, best_cost
end

# Robust XOR accuracy at the deep well: fraction of (pattern x init-draw) that
# are sign-correct under fresh free-cell noise.
function robust_accuracy(net, input_idx, var_idx, out_idx, rng)
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

Nn = 2 + N_HIDDEN + 1
input_idx = [1, 2]
out_idx   = [Nn]
var_idx   = setdiff(1:Nn, input_idx)

println("EP-Duffing barrier annealing -- robust XOR accuracy (", length(SEEDS),
        " seeds x ", N_TEST_INIT, " init draws, test noise ", TEST_NOISE, ")")
println("N = ", Nn, " (", N_HIDDEN, " hidden), s: ", S_START, " -> ", S_FINAL,
        ", ", N_EPOCH, " epochs\n")
@printf("%-12s | %-10s %-10s %-10s | %s\n",
        "anneal frac", "mean acc", "max acc", "min acc", "median best/final cost")
println("-"^74)

for frac in SCHEDULES
    accs = Float64[]; bestc = Float64[]; finalc = Float64[]
    for seed in SEEDS
        rng = MersenneTwister(seed)
        net = DuffingNetwork(Nn, input_idx, out_idx; a=-S_FINAL, c=S_FINAL, delta=1.0)
        random_init!(net; rng=rng)
        ch, bc = train_anneal!(net, data, target, 0.1, 0.02, N_EPOCH, frac; rng=rng)
        push!(accs, robust_accuracy(net, input_idx, var_idx, out_idx, MersenneTwister(9000 + seed)))
        push!(bestc, bc); push!(finalc, ch[end])
    end
    tag = frac == 0.0 ? "0.0 (ctrl)" : @sprintf("%.2f", frac)
    @printf("%-12s | %-10.0f %-10.0f %-10.0f | %.4f / %.4f\n",
            tag, 100 * mean(accs), 100 * maximum(accs), 100 * minimum(accs),
            median(bestc), median(finalc))
end
