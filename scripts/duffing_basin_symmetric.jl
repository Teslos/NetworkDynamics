# EP-Duffing basin-averaging: symmetric (+-beta) gradient vs one-sided, at the
# best config so far (Minit=50 + annealing). Aims to close the floor to an
# all-seed XOR solve.
#
# results/ep_duffing_basin_averaging.md: Minit=50 + anneal with a ONE-SIDED EP
# gradient reached 81% mean / 3-of-6 seeds solved (robust >= 90% under full-range
# test), the best Duffing result of the arc, but not a clean all-seed solve.
# Unlike Wang's log cost (which blows up when nudged the wrong way, forcing
# one-sided), our Duffing cost is quadratic C = 1/2 sum (x - target)^2, so the
# +-beta nudge is well-behaved both directions -- the symmetric (central-difference)
# estimator is safe and less biased. This runs the SAME config with symmetric=true
# and compares to the recorded one-sided baseline.
#
# Everything else identical to duffing_basin_averaging_strong.jl's best row
# (Minit=50, anneal s:0.3->1.0 over first half, 1500 epochs, 6 seeds, full-range
# test), so the only change is the gradient estimator.
#
# Usage: julia --project=. scripts/duffing_basin_symmetric.jl

using Random, Printf, Statistics

EP_DUFFING_SKIP_RUN = true
include(joinpath(@__DIR__, "..", "notebooks", "EP-Duffing-Network.jl"))

const SEEDS        = 1:6
const N_EPOCH      = 1500
const N_HIDDEN     = 2
const MINIT        = 50
const ANNEAL_FRAC  = 0.5
const S_START      = 0.3
const BETA         = 0.1
const LR           = 0.02
const TEST_RANGE   = 1.5
const N_TEST_INIT  = 25
const SOLVE_THRESH = 0.90

# (label, symmetric?)
const CONFIGS = [("one-sided (baseline)", false), ("symmetric +-beta", true)]

data   = Float64[-1 -1; -1 1; 1 -1; 1 1]
target = reshape(Float64[-1, 1, 1, -1], 4, 1)

s_at(epoch) = ANNEAL_FRAC <= 0.0 ? 1.0 :
    (n = max(1, round(Int, ANNEAL_FRAC * N_EPOCH));
     epoch >= n ? 1.0 : S_START + (1.0 - S_START) * (epoch - 1) / (n == 1 ? 1 : n - 1))

function train!(net, input_idx, var_idx, symmetric; rng)
    N = net.N
    s_W = zeros(N, N); r_W = zeros(N, N); s_h = zeros(N); r_h = zeros(N)
    best_cost = Inf; best_W = copy(net.W); best_h = copy(net.h); ch = zeros(N_EPOCH)
    nrow = 4 * MINIT
    for epoch in 1:N_EPOCH
        s = s_at(epoch); net.a = -s; net.c = s
        x0 = zeros(nrow, N); tgt = zeros(nrow, 1); row = 1
        for p in 1:4, _ in 1:MINIT
            x0[row, input_idx] .= data[p, :]
            x0[row, var_idx]   .= TEST_RANGE .* (2 .* rand(rng, length(var_idx)) .- 1)
            tgt[row, 1] = target[p, 1]; row += 1
        end
        gW, gh, cost, _ = EP_param_gradient(net, x0, tgt, BETA; symmetric=symmetric)
        net.W, s_W, r_W = adam_update(net.W, gW, LR, epoch, s_W, r_W)
        net.W = (net.W + net.W') / 2; net.W[diagind(net.W)] .= 0
        net.h, s_h, r_h = adam_update(net.h, gh, LR, epoch, s_h, r_h)
        ch[epoch] = cost
        if s == 1.0 && cost < best_cost; best_cost = cost; best_W = copy(net.W); best_h = copy(net.h); end
    end
    net.a = -1.0; net.c = 1.0; net.W = best_W; net.h = best_h
    return ch
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

println("EP-Duffing basin-averaging: symmetric vs one-sided gradient")
println("Minit=", MINIT, ", anneal s:", S_START, "->1.0 (frac ", ANNEAL_FRAC, "), ",
        N_EPOCH, " epochs, ", length(SEEDS), " seeds; full-range test, solve>=",
        Int(100SOLVE_THRESH), "% @ deep\n")
@printf("%-22s | %-6s %-5s %-5s %-8s | %-6s | %s\n",
        "gradient", "mean", "max", "min", "solved", "sweet", "med cost")
println("-"^68)

for (label, symmetric) in CONFIGS
    accD = Float64[]; accS = Float64[]; fc = Float64[]
    for seed in SEEDS
        rng = MersenneTwister(seed)
        net = DuffingNetwork(Nn, input_idx, out_idx; a=-1.0, c=1.0, delta=1.0)
        random_init!(net; rng=rng)
        ch = train!(net, input_idx, var_idx, symmetric; rng=rng)
        push!(accD, robust_acc(net, input_idx, var_idx, out_idx, 1.0, MersenneTwister(9000+seed)))
        push!(accS, robust_acc(net, input_idx, var_idx, out_idx, 0.5, MersenneTwister(9000+seed)))
        push!(fc, ch[end])
    end
    solved = count(>=(SOLVE_THRESH), accD)
    @printf("%-22s | %-6.0f %-5.0f %-5.0f %d/%-6d | %-6.0f | %.3f\n",
            label, 100mean(accD), 100maximum(accD), 100minimum(accD),
            solved, length(SEEDS), 100mean(accS), median(fc))
end
