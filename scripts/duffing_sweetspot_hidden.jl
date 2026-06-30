# Does "train deep, operate shallow" generalize across width?
#
# Sweet-spot study (results/ep_duffing_sweetspot.md) found that for hidden=2 the
# best EP-Duffing XOR config is TRAIN DEEP (anneal->1.0, sharp weights) + OPERATE
# SHALLOW (eval at s=0.5, input-determined inference): 86% / min 82% vs the
# champion's 71% / min 61%. Meanwhile anneal round 2 showed that EVALUATED AT
# s=1.0, more hidden units (4, 6) collapse back to chance (~47-56%).
#
# Open question: was that collapse a *training* failure, or just the wrong
# operating point? This script trains each width deep (anneal->1.0) once per seed,
# then sweeps the OPERATING depth s_eval and reports robust XOR accuracy. If
# "operate shallow" is general, the 4/6-hidden nets should recover at s~0.5 the
# way hidden=2 does.
#
# a = -s, c = s keeps minima at +-1 for every depth. hidden in {2,4,6}, 6 seeds,
# 2000 epochs, 40 init draws. Cells show mean (min) over seeds.
#
# Usage: julia --project=. scripts/duffing_sweetspot_hidden.jl

using Random, Printf, Statistics

EP_DUFFING_SKIP_RUN = true
include(joinpath(@__DIR__, "..", "notebooks", "EP-Duffing-Network.jl"))

const SEEDS       = 1:6
const N_EPOCH     = 2000
const ANNEAL_FRAC = 0.5
const HIDDEN      = [2, 4, 6]
const S_EVAL      = [0.3, 0.5, 0.7, 1.0]
const N_TEST_INIT = 40
const TEST_NOISE  = 0.1
const S_START     = 0.05
const S_FINAL     = 1.0

data   = Float64[-1 -1; -1 1; 1 -1; 1 1]
target = reshape(Float64[-1, 1, 1, -1], 4, 1)

s_at(epoch, n_epoch, frac) =
    frac <= 0.0 ? S_FINAL :
    (n_ramp = max(1, round(Int, frac * n_epoch));
     epoch >= n_ramp ? S_FINAL :
     S_START + (S_FINAL - S_START) * (epoch - 1) / (n_ramp == 1 ? 1 : n_ramp - 1))

# Train deep via anneal->S_FINAL; checkpoint best free cost at the deep well.
function train_deep!(net, beta, lr, n_epoch, frac; noise=0.1, rng=Random.default_rng())
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
    net.W = best_W; net.h = best_h
    return net
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

println("EP-Duffing 'train deep, operate shallow' across width -- robust XOR accuracy")
println(length(SEEDS), " seeds x ", N_TEST_INIT, " draws, test noise ", TEST_NOISE,
        "; train: anneal->", S_FINAL, ", frac=", ANNEAL_FRAC, ", ", N_EPOCH, " epochs")
println("cells = mean (min) over seeds\n")
@printf("%-8s | %s\n", "hidden",
        join([@sprintf("eval %-9s", string(s)) for s in S_EVAL], " "))
println("-"^(10 + 15 * length(S_EVAL)))

for nh in HIDDEN
    Nn = 2 + nh + 1
    input_idx = [1, 2]; out_idx = [Nn]; var_idx = setdiff(1:Nn, input_idx)

    # Train once per seed (deep), reuse across all eval depths.
    nets = DuffingNetwork[]
    for seed in SEEDS
        rng = MersenneTwister(seed)
        net = DuffingNetwork(Nn, input_idx, out_idx; a=-S_FINAL, c=S_FINAL, delta=1.0)
        random_init!(net; rng=rng)
        train_deep!(net, 0.1, 0.02, N_EPOCH, ANNEAL_FRAC; rng=rng)
        push!(nets, net)
    end

    cells = String[]
    for s in S_EVAL
        accs = [robust_accuracy(nets[k], input_idx, var_idx, out_idx, s,
                                MersenneTwister(9000 + SEEDS[k])) for k in eachindex(SEEDS)]
        push!(cells, @sprintf("%3.0f (%3.0f)", 100*mean(accs), 100*minimum(accs)))
    end
    @printf("%-8d | %s\n", nh, join([@sprintf("%-13s", c) for c in cells], " "))
end
