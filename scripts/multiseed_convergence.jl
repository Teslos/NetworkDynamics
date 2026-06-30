# Multi-seed convergence test for EP on oscillator networks (XOR).
#
# Trains the network from scratch on many random seeds and reports how RELIABLY
# EP converges, not just whether one lucky seed solves XOR. Two distinct rates:
#
#   * solve rate (best checkpoint) -- did training ever reach a solving config?
#     This is what the shipped trainers deliver (they restore the best weights).
#   * stability rate (final epoch) -- was the LAST epoch also solving? The gap
#     between the two quantifies the double-well / phase-locking basin-hopping
#     that makes the free cost bounce, and that the checkpoint papers over.
#
# Usage (run per substrate; the two model files share global names):
#   julia --project=. scripts/multiseed_convergence.jl duffing
#   julia --project=. scripts/multiseed_convergence.jl xy

using Random, Printf, Statistics

const SUBSTRATE = length(ARGS) >= 1 ? ARGS[1] : "duffing"
const N = 5
const INPUT_IDX = [1, 2]
const OUTPUT_IDX = [5]
const VAR_IDX = setdiff(1:N, INPUT_IDX)
const SOLVE_THRESH = 0.1   # free-relaxation cost below this == "solved" by cost
const N_TEST_INIT = 20     # init draws per pattern for the robust XOR accuracy
const ROBUST_THRESH = 0.95 # robust accuracy above this == genuinely solved

# Robust XOR accuracy: fraction of (pattern x init-draw) that are sign-correct.
# A genuine XOR solver should be insensitive to the free-cell initialization;
# a basin-lucky one is not. `relax_out(pattern_i, rng) -> output value`.
function robust_accuracy(relax_out, rng)
    correct = 0
    for i in 1:4, _ in 1:N_TEST_INIT
        out, tsign = relax_out(i, rng)
        sign(out) == tsign && (correct += 1)
    end
    return correct / (4 * N_TEST_INIT)
end

function summarize(robust_acc, stable, best_cost, final_cost)
    n = length(robust_acc)
    solved = robust_acc .>= ROBUST_THRESH
    println("\n", "="^64)
    @printf("Seeds: %d   (robust accuracy = sign-correct over 4 patterns x %d inits)\n",
            n, N_TEST_INIT)
    @printf("Robust solve rate (acc >= %.0f%%):  %d/%d = %.0f%%\n",
            100 * ROBUST_THRESH, sum(solved), n, 100 * mean(solved))
    @printf("Robust XOR accuracy: median %.0f%%, mean %.0f%%, min %.0f%%, max %.0f%%\n",
            100 * median(robust_acc), 100 * mean(robust_acc),
            100 * minimum(robust_acc), 100 * maximum(robust_acc))
    @printf("Cost-based solve rate (best<%.2f): %d/%d  -- selection-biased, see note\n",
            SOLVE_THRESH, sum(best_cost .< SOLVE_THRESH), n)
    @printf("Stable rate (final epoch <%.2f):   %d/%d = %.0f%%\n",
            SOLVE_THRESH, sum(final_cost .< SOLVE_THRESH), n, 100 * mean(final_cost .< SOLVE_THRESH))
    @printf("Best cost  median %.4f | Final cost median %.4f (bounce)\n",
            median(best_cost), median(final_cost))
    println("Note: best-checkpoint cost is the min over many noisy-init epochs, so it")
    println("is optimistically selected; robust accuracy is the honest convergence metric.")
    println("="^64)
end

# ============================================================================
if SUBSTRATE == "duffing"
    EP_DUFFING_SKIP_RUN = true
    include(joinpath(@__DIR__, "..", "notebooks", "EP-Duffing-Network.jl"))

    const N_SEEDS = 15
    const N_EPOCH = 2000
    data   = Float64[-1 -1; -1 1; 1 -1; 1 1]
    target = reshape(Float64[-1, 1, 1, -1], 4, 1)

    # Sweep the test-init noise: separates "output is input-independent" (acc ~
    # chance at every noise level) from "input-determined wells with thin margins"
    # (acc high at low noise, degrading as noise grows).
    NOISE_LEVELS = [0.0, 0.01, 0.05, 0.1]
    acc_by_noise = [Float64[] for _ in NOISE_LEVELS]
    best_cost = Float64[]; final_cost = Float64[]
    println("Multi-seed EP convergence -- damped Duffing (XOR), $N_SEEDS seeds x $N_EPOCH epochs")
    for seed in 1:N_SEEDS
        rng = MersenneTwister(seed)
        net = DuffingNetwork(N, INPUT_IDX, OUTPUT_IDX); random_init!(net; rng=rng)
        ch = train!(net, data, target, 0.1, 0.02, N_EPOCH; print_every=10^9, rng=rng)
        push!(best_cost, minimum(ch)); push!(final_cost, ch[end])

        accs = map(enumerate(NOISE_LEVELS)) do (k, nz)
            relax_out = function (i, r)
                x0 = zeros(1, N)
                x0[1, INPUT_IDX] .= data[i, :]
                x0[1, VAR_IDX]   .= nz * randn(r, length(VAR_IDX))
                eq = relax_batch(net, x0, reshape(target[i, :], 1, :), 0.0)
                return eq[1, OUTPUT_IDX[1]], sign(target[i, 1])
            end
            a = robust_accuracy(relax_out, MersenneTwister(1000 + seed + 100k))
            push!(acc_by_noise[k], a); a
        end
        @printf("  seed %2d: acc@noise[%s] = [%s]  | best %.4f final %.4f\n", seed,
                join(NOISE_LEVELS, ","), join((@sprintf("%.0f%%", 100a) for a in accs), " "),
                minimum(ch), ch[end])
    end
    println("\n", "="^64)
    @printf("Seeds: %d   (robust accuracy over 4 patterns x %d inits)\n", N_SEEDS, N_TEST_INIT)
    for (k, nz) in enumerate(NOISE_LEVELS)
        @printf("  test-init noise %.2f:  mean robust acc %.0f%%  (min %.0f%%, max %.0f%%)\n",
                nz, 100 * mean(acc_by_noise[k]), 100 * minimum(acc_by_noise[k]),
                100 * maximum(acc_by_noise[k]))
    end
    @printf("Best cost median %.4f | Final cost median %.4f (bounce)\n",
            median(best_cost), median(final_cost))
    println("="^64)

# ============================================================================
elseif SUBSTRATE == "xy"
    EP_XY_SKIP_RUN = true
    include(joinpath(@__DIR__, "..", "notebooks", "EP-XY-Network-Claude.jl"))

    const N_SEEDS = 10
    const N_EPOCH = 3000
    const N_EV = 200
    const DT = 0.1
    training_data   = (π / 2) * Float64[-1 -1; -1 1; 1 -1; 1 1]
    training_target = (π / 2) * Float64[-1, 1, 1, -1]

    robust_acc = Float64[]; best_cost = Float64[]; final_cost = Float64[]
    println("Multi-seed EP convergence -- XY/Kuramoto (XOR), $N_SEEDS seeds x $N_EPOCH epochs")
    for seed in 1:N_SEEDS
        Random.seed!(seed)
        net = SP_XY_Network(N, N_EV, DT, INPUT_IDX, OUTPUT_IDX); get_beta!(net, 0.01)
        random_state_initiation!(net); net.weights_0 *= 0.1; net.bias_0 *= 0.1
        Wf, bf, ch = train_network(net.weights_0, net.bias_0, training_data, training_target,
                                   0.01, 0.05, N_EPOCH, 4, N_EV, DT,
                                   INPUT_IDX, net.variable_index, OUTPUT_IDX; print_every=10^9)
        net.weights = Wf; net.bias = bf

        relax_out = function (i, rng)
            phase0 = zeros(N)
            phase0[INPUT_IDX] .= training_data[i, :]
            phase0[net.variable_index] .= 0.1 * randn(rng, length(net.variable_index))
            final_phase = run_network(net, phase0, training_target[i]; beta=0.0)
            return final_phase[OUTPUT_IDX[1]], sign(training_target[i])
        end
        acc = robust_accuracy(relax_out, MersenneTwister(1000 + seed))
        push!(robust_acc, acc); push!(best_cost, minimum(ch)); push!(final_cost, ch[end])
        @printf("  seed %2d: robust acc %5.1f%%  | best %.4f  final %.4f\n",
                seed, 100 * acc, minimum(ch), ch[end])
    end
    summarize(robust_acc, nothing, best_cost, final_cost)

else
    error("unknown substrate $SUBSTRATE (use 'duffing' or 'xy')")
end
