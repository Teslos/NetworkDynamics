# LAYERED (feedforward-symmetric) + temperature-annealed Langevin EP, Duffing XOR.
#
# Restricts the symmetric coupling to input<->hidden and hidden<->output edges
# (layered_mask in EP-Duffing-Langevin.jl). The energy stays a symmetric gradient
# system, but the output field comes solely from the input-driven hidden layer,
# so the output well is input-determined -- the remedy that makes the phase
# network's XOR robust in the paper.
#
# Finding: layering only helps with ENOUGH hidden capacity (the mask removes
# edges). Capacity scan (annealed, 8 seeds, full-range test inits):
#   layered 2-2-1 : ~81% mean, 0/8
#   layered 2-3-1 : ~86% mean, 0/8
#   layered 2-4-1 : ~96% mean, median ~96%, every seed >=92%   <-- no stalls
# vs all-to-all annealed ~90% mean but with a 58% stall seed. The 2-4-1 layered
# net removes the catastrophic stalls and matches the paper's layered phase
# network (~95%, all seeds solved).
#
# Usage:  julia --project=. scripts/duffing_langevin_layered_xor.jl
#         julia --project=. scripts/duffing_langevin_layered_xor.jl 8   # N_seed

using Random
using Statistics
using Printf

EP_DUFFING_LANGEVIN_SKIP_RUN = true
include(joinpath(@__DIR__, "..", "notebooks", "EP-Duffing-Langevin.jl"))

const DATA   = Float64[-1 -1; -1 1; 1 -1; 1 1]
const TARGET = reshape(Float64[-1, 1, 1, -1], 4, 1)

eval_annealed(net; n_draw=6) = mean(begin
    acc, _ = langevin_anneal_xor_accuracy(net, DATA, TARGET; T_hi=0.15, T_lo=0.05,
        n_ramp=4000, n_read=2000, init_range=1.0, rng=MersenneTwister(2000 + r))
    acc
end for r in 1:n_draw)

# Train one seed with annealing; `hidden` = hidden-unit indices (nothing -> all-to-all).
function run_seed(seed; N, hidden, beta=0.1, lr=0.03, N_epoch=300, n_burn=1200, n_sample=2500)
    rng = MersenneTwister(seed)
    net = DuffingNetwork(N, [1, 2], [N]); random_init!(net; rng=rng)
    mask = hidden === nothing ? nothing : layered_mask(N, [1, 2], hidden, [N])
    train_langevin_anneal!(net, DATA, TARGET; beta=beta, T_hi=0.20, T_lo=0.06, lr=lr,
        N_epoch=N_epoch, dt=0.02, n_burn=n_burn, n_sample=n_sample, init_noise=0.5,
        mask=mask, rng=rng)
    return eval_annealed(net)
end

function main(seeds)
    println("Layered + annealed Langevin EP -- robust XOR (a=-1, barrier 0.25)")
    println("  all-to-all annealed ~90% mean but with stall seeds; layered fixes stalls\n")
    @printf("  %-22s  %-7s  %-7s  %-6s  %-9s  %s\n",
            "config", "mean", "median", "min", "solved", "per-seed %")
    configs = (("all-to-all (N=5)", 5, nothing),
               ("layered 2-2-1 (N=5)", 5, [3, 4]),
               ("layered 2-4-1 (N=7)", 7, [3, 4, 5, 6]))
    for (label, N, hidden) in configs
        accs = [run_seed(s; N=N, hidden=hidden) for s in seeds]
        solved = count(a -> a >= 0.99, accs)
        @printf("  %-22s  %5.1f%%  %5.1f%%  %4.0f%%  %-9s  %s\n",
                label, 100mean(accs), 100median(accs), 100minimum(accs),
                "$solved/$(length(seeds))", string(round.(Int, 100 .* accs)))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    nseed = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 8
    main(1:nseed)
end
