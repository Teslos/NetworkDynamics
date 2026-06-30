# Can a shallower double-well and more hidden units make the EP-Duffing network
# robustly solve XOR? The deep-well / 2-hidden baseline sits at chance
# (scripts/multiseed_convergence.jl) because the bistable forward inference lets
# the hidden-cell init, not the input, pick the output basin.
#
# Two levers, swept as a grid and scored by ROBUST accuracy (sign-correct over
# many init draws, multi-seed):
#   * Well depth: a = -s, c = s keeps the minima fixed at x = +-1 (so the +-1
#     encoding is unchanged) while the barrier depth = s/4. Smaller s => shallower
#     wells => the input coupling/bias can more easily drive a cell into the
#     input-determined basin (closer to the smooth XY inference).
#   * Hidden units: N = 2 inputs + n_hidden + 1 output.
#
# Usage: julia --project=. scripts/duffing_design_sweep.jl

using Random, Printf, Statistics

EP_DUFFING_SKIP_RUN = true
include(joinpath(@__DIR__, "..", "notebooks", "EP-Duffing-Network.jl"))

const SEEDS = 1:6
const N_EPOCH = 2000
const N_TEST_INIT = 15
const TEST_NOISE = 0.1     # matches the training init noise; acc is flat in this
const DEPTHS = [1.0, 0.5, 0.3]      # s: a=-s, c=s  -> barrier depth s/4
const HIDDEN = [2, 6]

data   = Float64[-1 -1; -1 1; 1 -1; 1 1]
target = reshape(Float64[-1, 1, 1, -1], 4, 1)

# Robust XOR accuracy of a trained net: fraction of (pattern x init-draw) that
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

println("EP-Duffing design sweep -- robust XOR accuracy (", length(SEEDS),
        " seeds x ", N_TEST_INIT, " init draws, test noise ", TEST_NOISE, ")\n")
@printf("%-8s %-8s | %-10s %-10s %-10s | %s\n",
        "depth s", "hidden", "mean acc", "max acc", "min acc", "median best/final cost")
println("-"^78)

for s in DEPTHS, nh in HIDDEN
    Nn = 2 + nh + 1
    input_idx = [1, 2]
    out_idx = [Nn]
    var_idx = setdiff(1:Nn, input_idx)

    accs = Float64[]; bestc = Float64[]; finalc = Float64[]
    for seed in SEEDS
        rng = MersenneTwister(seed)
        net = DuffingNetwork(Nn, input_idx, out_idx; a=-s, c=s, delta=1.0)
        random_init!(net; rng=rng)
        ch = train!(net, data, target, 0.1, 0.02, N_EPOCH; print_every=10^9, rng=rng)
        push!(accs, robust_accuracy(net, input_idx, var_idx, out_idx, MersenneTwister(9000 + seed)))
        push!(bestc, minimum(ch)); push!(finalc, ch[end])
    end
    @printf("%-8.2f %-8d | %-10.0f %-10.0f %-10.0f | %.4f / %.4f\n",
            s, nh, 100 * mean(accs), 100 * maximum(accs), 100 * minimum(accs),
            median(bestc), median(finalc))
end
