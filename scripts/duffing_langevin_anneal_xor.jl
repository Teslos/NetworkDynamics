# Temperature-ANNEALED finite-T thermodynamic EP on the Duffing network, XOR.
#
# Extends scripts/duffing_langevin_xor.jl with a hot->cold cooling schedule:
#   * annealed TRAINING: T geometrically cooled T_hi -> T_lo over epochs. Hot
#     epochs mix / escape wells and give lower-variance gradients; cold epochs
#     sharpen the landscape.
#   * annealed READOUT: at test the free relaxation ramps T_hi -> T_lo within a
#     single run, so the trained field pulls each output to the input-selected
#     well (warm) and then commits it (cold). A fixed cold readout would instead
#     re-pin in whatever well the init landed in.
#
# Result (a=-1, barrier 0.25, N=5, 8 seeds, full-range test inits):
#   fixed-T (best)                        ~80% mean, 1/8 solved
#   annealed train + annealed readout     ~90% mean, median ~94%, 3/8 solved
# i.e. annealing lifts the fixed-T thermodynamic result by ~+10 points and gets
# several seeds to a full XOR solve. The annealed TRAINING is the main driver.
#
# Usage:  julia --project=. scripts/duffing_langevin_anneal_xor.jl
#         julia --project=. scripts/duffing_langevin_anneal_xor.jl 8   # N_seed

using Random
using Statistics
using Printf

EP_DUFFING_LANGEVIN_SKIP_RUN = true
include(joinpath(@__DIR__, "..", "notebooks", "EP-Duffing-Langevin.jl"))

const DATA   = Float64[-1 -1; -1 1; 1 -1; 1 1]
const TARGET = reshape(Float64[-1, 1, 1, -1], 4, 1)

# Robust accuracy with the annealed readout, averaged over full-range test-init draws.
function eval_annealed(net; T_hi=0.15, T_lo=0.05, n_ramp=4000, n_read=2000, n_draw=6)
    accs = Float64[]
    for r in 1:n_draw
        acc, _ = langevin_anneal_xor_accuracy(net, DATA, TARGET; T_hi=T_hi, T_lo=T_lo,
            n_ramp=n_ramp, n_read=n_read, init_range=1.0, rng=MersenneTwister(2000 + r))
        push!(accs, acc)
    end
    return mean(accs)
end

# Train one seed (annealed or fixed-T), evaluate with the annealed readout.
function run_seed(seed; annealed::Bool, T_hi=0.20, T_lo=0.06, T_fixed=0.15,
                  N=5, beta=0.1, lr=0.03, N_epoch=300, n_burn=1200, n_sample=2500)
    rng = MersenneTwister(seed)
    net = DuffingNetwork(N, [1, 2], [N]); random_init!(net; rng=rng)
    if annealed
        train_langevin_anneal!(net, DATA, TARGET; beta=beta, T_hi=T_hi, T_lo=T_lo, lr=lr,
            N_epoch=N_epoch, dt=0.02, n_burn=n_burn, n_sample=n_sample, init_noise=0.5, rng=rng)
    else
        train_langevin!(net, DATA, TARGET; beta=beta, T=T_fixed, lr=lr, N_epoch=N_epoch,
            dt=0.02, n_burn=n_burn, n_sample=n_sample, init_noise=0.5, print_every=10^9, rng=rng)
    end
    return eval_annealed(net)
end

function main(seeds)
    println("Temperature-annealed Langevin EP -- robust XOR (a=-1, barrier 0.25), N=5")
    println("  deterministic baseline ~57% (chance); fixed-T thermodynamic best ~80%\n")
    @printf("  %-34s  %-7s  %-7s  %-9s  %s\n", "config", "mean", "median", "solved", "per-seed %")
    for (label, annealed) in (("fixed-T(0.15) + annealed readout", false),
                              ("annealed train + annealed readout", true))
        accs = [run_seed(s; annealed=annealed) for s in seeds]
        solved = count(a -> a >= 0.99, accs)
        @printf("  %-34s  %5.1f%%  %5.1f%%  %-9s  %s\n",
                label, 100mean(accs), 100median(accs), "$solved/$(length(seeds))",
                string(round.(Int, 100 .* accs)))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    nseed = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 8
    main(1:nseed)
end
