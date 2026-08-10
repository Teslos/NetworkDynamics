# Finite-temperature thermodynamic EP on a Duffing (double-well) network, XOR.
#
# Trains the all-to-all N=5 bistable Duffing network with the overdamped Langevin
# sampler (EP-Duffing-Langevin.jl) and reports ROBUST XOR accuracy over a
# temperature sweep and multiple training seeds, evaluated with full-range random
# free-cell inits (the honest test that the deterministic relaxer fails at ~57%).
#
# Headline finding: there is a temperature window (a=-1,c=1 -> barrier 0.25;
# window ~ T in [0.13, 0.17], i.e. dV/T ~ 1.5-2) in which thermal sampling lifts
# accuracy from chance to ~80% mean (best seeds 96-100%) -- comparable to the
# basin-averaging remedy, but from a single physical parameter (temperature).
# Too cold (T<=0.06) does not mix (reproduces the ~57% deterministic failure);
# too hot (T>=0.2) washes out the wells.
#
# Usage:  julia --project=. scripts/duffing_langevin_xor.jl
#         julia --project=. scripts/duffing_langevin_xor.jl 0.15 6   # single T, N_seed

using Random
using Statistics
using Printf

EP_DUFFING_LANGEVIN_SKIP_RUN = true
include(joinpath(@__DIR__, "..", "notebooks", "EP-Duffing-Langevin.jl"))

const DATA   = Float64[-1 -1; -1 1; 1 -1; 1 1]
const TARGET = reshape(Float64[-1, 1, 1, -1], 4, 1)

# Train one seed, then report robust accuracy averaged over several full-range
# test-init draws (the output basin must be set by the input, not the init).
function train_and_eval(; T, seed, N=5, a=-1.0, c=1.0, beta=0.1, lr=0.03,
                        N_epoch=300, n_burn=1200, n_sample=2500, init_noise=0.5,
                        n_test_draw=6)
    rng = MersenneTwister(seed)
    net = DuffingNetwork(N, [1, 2], [N]; a=a, c=c)
    random_init!(net; rng=rng)
    train_langevin!(net, DATA, TARGET; beta=beta, T=T, lr=lr, N_epoch=N_epoch,
        dt=0.02, n_burn=n_burn, n_sample=n_sample, init_noise=init_noise,
        print_every=10^9, rng=rng)
    accs = Float64[]
    for r in 1:n_test_draw
        acc, _ = langevin_xor_accuracy(net, DATA, TARGET; T=T, dt=0.02,
            n_burn=n_burn, n_sample=n_sample, init_range=1.0, rng=MersenneTwister(1000 + r))
        push!(accs, acc)
    end
    return mean(accs)
end

function sweep(Ts, seeds)
    println("Finite-T Langevin EP -- robust XOR (a=-1, barrier 0.25), N=5")
    println("  deterministic baseline (EP-Duffing-Network.jl): ~57% (chance)\n")
    @printf("  %-6s  %-7s  %-7s  %-14s  %s\n", "T", "mean", "median", "solved(>=99%)", "per-seed %")
    for T in Ts
        accs = [train_and_eval(T=T, seed=s) for s in seeds]
        solved = count(a -> a >= 0.99, accs)
        @printf("  %-6.2f  %5.1f%%  %5.1f%%  %-14s  %s\n",
                T, 100mean(accs), 100median(accs), "$solved/$(length(seeds))",
                string(round.(Int, 100 .* accs)))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) >= 1
        T = parse(Float64, ARGS[1])
        nseed = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 8
        sweep([T], 1:nseed)
    else
        sweep([0.06, 0.10, 0.13, 0.15, 0.17, 0.20], 1:8)
    end
end
