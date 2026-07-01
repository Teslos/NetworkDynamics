# EP-Duffing basin-averaging, STRONGER config: does more Minit + more epochs
# (+ optional annealing) close the floor to a robust ALL-SEED XOR solve?
#
# results/ep_duffing_basin_averaging.md showed basin-averaging (Wang's
# multistability remedy) lifts robust XOR monotonically with Minit -- 51% (near-0
# control) -> 76% mean at Minit=20, best seed 100%, under the honest full-range
# test -- but with high seed variance (min 64%). This pushes the two knobs the
# trend pointed at:
#   * Minit 30 and 50 (more basin samples averaged per gradient step),
#   * 1500 epochs (was 1000),
#   * a variant that STACKS basin-averaging with barrier annealing (s: 0.3 -> 1.0
#     over the first half of training), to see if shaping the landscape over time
#     helps the basins agree faster.
#
# Reports, per config: robust XOR under FULL-RANGE test init (uniform [-1.5,1.5],
# both wells) at the deep well and the s=0.5 sweet spot, plus the number of seeds
# that SOLVE (robust acc >= 0.90 at the deep well) -- the "floor closing" metric.
#
# Usage: julia --project=. scripts/duffing_basin_averaging_strong.jl

using Random, Printf, Statistics

EP_DUFFING_SKIP_RUN = true
include(joinpath(@__DIR__, "..", "notebooks", "EP-Duffing-Network.jl"))

const SEEDS        = 1:6
const N_EPOCH      = 1500
const N_HIDDEN     = 2
const BETA         = 0.1
const LR           = 0.02
const TEST_RANGE   = 1.5
const N_TEST_INIT  = 25
const SOLVE_THRESH = 0.90
const S_START      = 0.3            # anneal start depth (barrier s/4)

# (label, Minit, anneal_frac).  anneal_frac = 0 -> deep well (s=1) throughout.
const CONFIGS = [
    ("Minit=30",          30, 0.0),
    ("Minit=50",          50, 0.0),
    ("Minit=50 +anneal",  50, 0.5),
]

data   = Float64[-1 -1; -1 1; 1 -1; 1 1]
target = reshape(Float64[-1, 1, 1, -1], 4, 1)

s_at(epoch, frac) = frac <= 0.0 ? 1.0 :
    (n = max(1, round(Int, frac * N_EPOCH));
     epoch >= n ? 1.0 : S_START + (1.0 - S_START) * (epoch - 1) / (n == 1 ? 1 : n - 1))

# Basin-averaging trainer (full-range init, gradient averaged over Minit x 4 rows,
# one-sided). Optional barrier annealing; checkpoint best cost only at the deep well.
function train_strong!(net, input_idx, var_idx, Minit, frac; rng)
    N = net.N
    s_W = zeros(N, N); r_W = zeros(N, N); s_h = zeros(N); r_h = zeros(N)
    best_cost = Inf; best_W = copy(net.W); best_h = copy(net.h); ch = zeros(N_EPOCH)
    nrow = 4 * Minit
    for epoch in 1:N_EPOCH
        s = s_at(epoch, frac); net.a = -s; net.c = s
        x0 = zeros(nrow, N); tgt = zeros(nrow, 1); row = 1
        for p in 1:4, _ in 1:Minit
            x0[row, input_idx] .= data[p, :]
            x0[row, var_idx]   .= TEST_RANGE .* (2 .* rand(rng, length(var_idx)) .- 1)
            tgt[row, 1] = target[p, 1]; row += 1
        end
        gW, gh, cost, _ = EP_param_gradient(net, x0, tgt, BETA; symmetric=false)
        net.W, s_W, r_W = adam_update(net.W, gW, LR, epoch, s_W, r_W)
        net.W = (net.W + net.W') / 2; net.W[diagind(net.W)] .= 0
        net.h, s_h, r_h = adam_update(net.h, gh, LR, epoch, s_h, r_h)
        ch[epoch] = cost
        if s == 1.0 && cost < best_cost; best_cost = cost; best_W = copy(net.W); best_h = copy(net.h); end
    end
    net.a = -1.0; net.c = 1.0; net.W = best_W; net.h = best_h
    return ch, best_cost
end

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

println("EP-Duffing basin-averaging (STRONG) -- robust XOR under FULL-RANGE test init")
println(length(SEEDS), " seeds x ", N_TEST_INIT, " draws, ", N_EPOCH,
        " epochs, one-sided grad; solve = robust >= ", Int(100SOLVE_THRESH), "% @ deep\n")
@printf("%-20s | %-6s %-5s %-5s %-8s | %-6s | %s\n",
        "config", "mean", "max", "min", "solved", "sweet", "med cost")
println("-"^68)

for (label, Minit, frac) in CONFIGS
    accD = Float64[]; accS = Float64[]; fc = Float64[]
    for seed in SEEDS
        rng = MersenneTwister(seed)
        net = DuffingNetwork(Nn, input_idx, out_idx; a=-1.0, c=1.0, delta=1.0)
        random_init!(net; rng=rng)
        ch, _ = train_strong!(net, input_idx, var_idx, Minit, frac; rng=rng)
        push!(accD, robust_acc(net, input_idx, var_idx, out_idx, 1.0, MersenneTwister(9000+seed)))
        push!(accS, robust_acc(net, input_idx, var_idx, out_idx, 0.5, MersenneTwister(9000+seed)))
        push!(fc, ch[end])
    end
    solved = count(>=(SOLVE_THRESH), accD)
    @printf("%-20s | %-6.0f %-5.0f %-5.0f %d/%-6d | %-6.0f | %.3f\n",
            label, 100mean(accD), 100maximum(accD), 100minimum(accD),
            solved, length(SEEDS), 100mean(accS), median(fc))
end
