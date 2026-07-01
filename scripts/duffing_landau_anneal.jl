# Landau / deterministic-annealing EP-Duffing: cool through the phase transition.
#
# Phase-field / Landau free energy: F(x) = 1/2 a(T) x^2 + 1/4 b x^4, with the
# quadratic coefficient set by temperature, a(T) = a0 (T - Tc), b > 0 fixed.
#   * T > Tc  (a > 0): SINGLE well at x=0 -- monostable "disordered" phase. The
#     equilibrium is UNIQUE and set by the field (bias + input coupling) -> forward
#     inference is purely input-determined, no basin ambiguity (the XY-like regime).
#   * T < Tc  (a < 0): DOUBLE well, minima at +-sqrt(-a/b) -- bistable "ordered"
#     phase. At a=-1, b=1 the wells sit at the +-1 logic levels.
#   * Cooling through Tc (a: + -> 0 -> -) is a supercritical pitchfork; a nonzero
#     symmetry-breaking field (the input) unfolds it (cusp), so the state slides
#     continuously into the input-selected well and is frozen in as the barrier
#     grows. The answer is fixed while monostable, then locked by cooling
#     (= deterministic annealing / graduated non-convexity).
#
# This tests the key refinement over our earlier annealing: START ABOVE Tc (a>0,
# genuinely monostable) vs our earlier start at a shallow double well (a<0, already
# bistable). We anneal ONLY a (temperature); the quartic b=c is fixed, so the
# minima EMERGE from 0 and grow to +-1 -- a true order-parameter phase transition.
# Both runs use basin-averaging (full-range init + gradient over Minit basins) and
# the honest full-range test. Reuses the Duffing dynamics (anneal net.a, fix net.c).
#
# Usage: julia --project=. scripts/duffing_landau_anneal.jl

using Random, Printf, Statistics, LinearAlgebra

EP_DUFFING_SKIP_RUN = true
include(joinpath(@__DIR__, "..", "notebooks", "EP-Duffing-Network.jl"))

const SEEDS        = 1:6
const N_EPOCH      = 1200
const N_HIDDEN     = 2
const MINIT        = 40
const ANNEAL_FRAC  = 0.5
const A_FINAL      = -1.0          # deep double well (wells at +-1 with c=1)
const C_FIXED      = 1.0           # quartic coeff b, held fixed (temperature acts on a)
const BETA         = 0.1
const LR           = 0.02
const TEST_RANGE   = 1.5
const N_TEST_INIT  = 25
const SOLVE_THRESH = 0.90

# (label, a_start).  a_start > 0 -> start ABOVE Tc (monostable);
#                    a_start < 0 -> start below Tc (shallow double well, our earlier scheme).
const CONFIGS = [
    ("below-Tc start a:-0.3->-1", -0.3),
    ("Landau  a: +0.5 -> -1",      0.5),
]

data   = Float64[-1 -1; -1 1; 1 -1; 1 1]
target = reshape(Float64[-1, 1, 1, -1], 4, 1)

# Anneal the temperature: a goes a_start -> A_FINAL over the first ANNEAL_FRAC of
# training (linearly, passing through a=0 = Tc if a_start>0), then holds at A_FINAL.
function a_at(epoch, a_start)
    ANNEAL_FRAC <= 0.0 && return A_FINAL
    n = max(1, round(Int, ANNEAL_FRAC * N_EPOCH))
    epoch >= n && return A_FINAL
    return a_start + (A_FINAL - a_start) * (epoch - 1) / (n == 1 ? 1 : n - 1)
end

function train!(net, input_idx, var_idx, a_start; rng)
    N = net.N
    s_W = zeros(N, N); r_W = zeros(N, N); s_h = zeros(N); r_h = zeros(N)
    best_cost = Inf; best_W = copy(net.W); best_h = copy(net.h); ch = zeros(N_EPOCH)
    nrow = 4 * MINIT
    for epoch in 1:N_EPOCH
        net.a = a_at(epoch, a_start); net.c = C_FIXED
        x0 = zeros(nrow, N); tgt = zeros(nrow, 1); row = 1
        for p in 1:4, _ in 1:MINIT
            x0[row, input_idx] .= data[p, :]
            x0[row, var_idx]   .= TEST_RANGE .* (2 .* rand(rng, length(var_idx)) .- 1)
            tgt[row, 1] = target[p, 1]; row += 1
        end
        gW, gh, cost, _ = EP_param_gradient(net, x0, tgt, BETA; symmetric=false)
        net.W, s_W, r_W = adam_update(net.W, gW, LR, epoch, s_W, r_W)
        net.W = (net.W + net.W') / 2; net.W[diagind(net.W)] .= 0
        net.h, s_h, r_h = adam_update(net.h, gh, LR, epoch, s_h, r_h)
        ch[epoch] = cost
        if net.a == A_FINAL && cost < best_cost  # checkpoint at the deep well only
            best_cost = cost; best_W = copy(net.W); best_h = copy(net.h)
        end
    end
    net.a = A_FINAL; net.c = C_FIXED; net.W = best_W; net.h = best_h
    return ch
end

function robust_acc(net, input_idx, var_idx, out_idx, a_eval, rng)
    net.a = a_eval; net.c = C_FIXED
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

println("Landau / deterministic-annealing EP-Duffing -- robust XOR under full-range test")
println(length(SEEDS), " seeds x ", N_TEST_INIT, " draws, Minit=", MINIT, ", ",
        N_EPOCH, " epochs, anneal frac ", ANNEAL_FRAC, ", c=", C_FIXED,
        " fixed; solve>=", Int(100SOLVE_THRESH), "% @ deep (a=-1)\n")
@printf("%-28s | %-6s %-5s %-5s %-8s | %-8s | %s\n",
        "schedule", "mean", "max", "min", "solved", "a=-0.5", "med cost")
println("-"^74)

for (label, a_start) in CONFIGS
    accD = Float64[]; accM = Float64[]; fc = Float64[]
    for seed in SEEDS
        rng = MersenneTwister(seed)
        net = DuffingNetwork(Nn, input_idx, out_idx; a=A_FINAL, c=C_FIXED, delta=1.0)
        random_init!(net; rng=rng)
        ch = train!(net, input_idx, var_idx, a_start; rng=rng)
        push!(accD, robust_acc(net, input_idx, var_idx, out_idx, -1.0, MersenneTwister(9000+seed)))
        push!(accM, robust_acc(net, input_idx, var_idx, out_idx, -0.5, MersenneTwister(9000+seed)))
        push!(fc, ch[end])
    end
    solved = count(>=(SOLVE_THRESH), accD)
    @printf("%-28s | %-6.0f %-5.0f %-5.0f %d/%-6d | %-8.0f | %.3f\n",
            label, 100mean(accD), 100maximum(accD), 100minimum(accD),
            solved, length(SEEDS), 100mean(accM), median(fc))
end

println("\nReference (minima-fixed anneal, Minit=50): basin-avg + anneal = 81-84% mean, 3/6 solved")
