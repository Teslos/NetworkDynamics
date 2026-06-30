# EP-Duffing barrier annealing, round 2: does a SLOWER ramp combined with MORE
# hidden units push past the partial rescue (mean ~73%, best 93%) from
# scripts/duffing_anneal.jl?
#
# Two levers, on top of the smooth->deep curriculum:
#   * Slower ramp -- more epochs (3000 vs 2000) AND a cosine ease-in schedule
#     that keeps the wells shallow (input-determined inference) for longer before
#     deepening, in addition to the plain linear ramp.
#   * More hidden units -- N_HIDDEN in {4, 6} (the static sweep's other lever,
#     which did nothing at fixed depth; the question is whether it helps once the
#     curriculum has put cells in input-determined basins).
#
# As before: a = -s, c = s keeps minima at +-1; barrier depth = s/4; start near
# flat (s=0.05), ramp to s=1.0 over the first anneal_frac of epochs, then hold;
# checkpoint best free cost ONLY at the deep well; score ROBUST accuracy there.
#
# Usage: julia --project=. scripts/duffing_anneal_slow.jl

using Random, Printf, Statistics

EP_DUFFING_SKIP_RUN = true
include(joinpath(@__DIR__, "..", "notebooks", "EP-Duffing-Network.jl"))

const SEEDS       = 1:6
const N_EPOCH     = 3000      # slower than the 2000-epoch round 1
const N_TEST_INIT = 15
const TEST_NOISE  = 0.1
const S_START     = 0.05
const S_FINAL     = 1.0

# (schedule shape, anneal_frac). :linear ramps s uniformly; :cosine is an ease-in
# (slow start) that keeps s near S_START longer -> a genuinely slower deepening.
const SCHEDULES = [(:linear, 0.5), (:linear, 0.9), (:cosine, 0.9)]
const HIDDEN    = [2, 4, 6]   # 2 included as the round-1 reference

data   = Float64[-1 -1; -1 1; 1 -1; 1 1]
target = reshape(Float64[-1, 1, 1, -1], 4, 1)

function s_at(epoch, n_epoch, shape, anneal_frac)
    anneal_frac <= 0.0 && return S_FINAL
    n_ramp = max(1, round(Int, anneal_frac * n_epoch))
    epoch >= n_ramp && return S_FINAL
    u = n_ramp == 1 ? 1.0 : (epoch - 1) / (n_ramp - 1)        # progress in [0,1]
    f = shape === :cosine ? (1 - cos(u * pi / 2)) : u          # ease-in vs linear
    return S_START + (S_FINAL - S_START) * f
end

function train_anneal!(net, data, target, beta, lr, n_epoch, shape, anneal_frac;
                       noise=0.1, rng=Random.default_rng())
    N = net.N
    N_data = size(data, 1)
    cost_history = zeros(n_epoch)
    s_W = zeros(N, N); r_W = zeros(N, N)
    s_h = zeros(N);    r_h = zeros(N)
    best_cost = Inf
    best_W = copy(net.W); best_h = copy(net.h)

    for epoch in 1:n_epoch
        s = s_at(epoch, n_epoch, shape, anneal_frac)
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
        if s == S_FINAL && cost < best_cost
            best_cost = cost
            best_W = copy(net.W); best_h = copy(net.h)
        end
    end

    net.a = -S_FINAL; net.c = S_FINAL
    net.W = best_W; net.h = best_h
    return cost_history, best_cost
end

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

println("EP-Duffing slower-ramp x more-hidden -- robust XOR accuracy (",
        length(SEEDS), " seeds x ", N_TEST_INIT, " init draws, test noise ",
        TEST_NOISE, ")")
println("s: ", S_START, " -> ", S_FINAL, ", ", N_EPOCH, " epochs\n")
@printf("%-8s %-14s | %-10s %-10s %-10s | %s\n",
        "hidden", "schedule", "mean acc", "max acc", "min acc", "median best/final cost")
println("-"^84)

for nh in HIDDEN, (shape, frac) in SCHEDULES
    Nn = 2 + nh + 1
    input_idx = [1, 2]
    out_idx   = [Nn]
    var_idx   = setdiff(1:Nn, input_idx)

    accs = Float64[]; bestc = Float64[]; finalc = Float64[]
    for seed in SEEDS
        rng = MersenneTwister(seed)
        net = DuffingNetwork(Nn, input_idx, out_idx; a=-S_FINAL, c=S_FINAL, delta=1.0)
        random_init!(net; rng=rng)
        ch, bc = train_anneal!(net, data, target, 0.1, 0.02, N_EPOCH, shape, frac; rng=rng)
        push!(accs, robust_accuracy(net, input_idx, var_idx, out_idx, MersenneTwister(9000 + seed)))
        push!(bestc, bc); push!(finalc, ch[end])
    end
    @printf("%-8d %-14s | %-10.0f %-10.0f %-10.0f | %.4f / %.4f\n",
            nh, string(shape, " ", frac), 100 * mean(accs), 100 * maximum(accs),
            100 * minimum(accs), median(bestc), median(finalc))
end
