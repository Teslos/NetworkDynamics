# Layered EP-Duffing + Landau cooling on XOR: the "everything combined" test.
#
# The 95% XOR breakthrough (results/ep_duffing_layered.md) used layered structure +
# MINIMA-FIXED annealing (a=-s, c=s, wells pinned at +-1, only barrier deepens).
# Landau cooling (anneal a from +0.5 -> -1 with c fixed, passing through Tc =
# monostable -> bistable) was only ever tested in the ALL-TO-ALL net (79%,
# results/ep_duffing_landau_anneal.md). This runs the untested combination:
# layered + Landau cooling, A/B against layered + minima-fixed annealing, to see
# whether cooling through the phase transition (where the hidden layer supplies the
# symmetry-breaking field) improves on 95% / pushes to a clean all-seed solve.
#
# Both paths end at the same deep well (a=-1, c=1, wells +-1); only the annealing
# PATH differs. Basin-averaging, layered mask, honest full-range test.
#
# Usage: julia --project=. scripts/duffing_layered_landau.jl

using Random, Printf, Statistics, LinearAlgebra

EP_DUFFING_SKIP_RUN = true
include(joinpath(@__DIR__, "..", "notebooks", "EP-Duffing-Network.jl"))

const SEEDS        = 1:6
const N_EPOCH      = 1000
const N_HIDDEN     = 12            # layered, N = 2 + 12 + 1 = 15 (the 95% config)
const MINIT        = 40
const ANNEAL_FRAC  = 0.5
const S_START      = 0.3           # minima-fixed: s start
const A_START_L    = 0.5           # Landau: a start (> 0 => above Tc, monostable)
const BETA         = 0.1
const LR           = 0.02
const TEST_RANGE   = 1.5
const N_TEST_INIT  = 25
const SOLVE_THRESH = 0.90

# (label, mode)  mode = :minima_fixed  or  :landau
const CONFIGS = [("layered + minima-fixed", :minima_fixed), ("layered + Landau cool", :landau)]

data   = Float64[-1 -1; -1 1; 1 -1; 1 1]
target = reshape(Float64[-1, 1, 1, -1], 4, 1)

# Returns (a, c) for the given epoch and schedule. Both end at (a=-1, c=1).
function ac_at(epoch, mode)
    n = max(1, round(Int, ANNEAL_FRAC * N_EPOCH))
    prog = epoch >= n ? 1.0 : (epoch - 1) / (n == 1 ? 1 : n - 1)
    if mode === :minima_fixed
        s = S_START + (1.0 - S_START) * prog     # 0.3 -> 1
        return (-s, s)                            # wells pinned at +-1
    else # :landau  -- c fixed, a: +0.5 -> -1 (through 0 = Tc)
        a = A_START_L + (-1.0 - A_START_L) * prog
        return (a, 1.0)
    end
end

Nn = 2 + N_HIDDEN + 1
const INPUT = [1, 2]; const HIDDEN = collect(3:2+N_HIDDEN); const OUT = [Nn]
const VAR = setdiff(1:Nn, INPUT)
const MASK = let M = zeros(Nn, Nn)   # layered: input<->hidden, hidden<->output
    for i in INPUT, j in HIDDEN; M[i,j]=1.0; M[j,i]=1.0; end
    for i in HIDDEN, j in OUT;   M[i,j]=1.0; M[j,i]=1.0; end
    M
end

function train!(net, mode; rng)
    N = net.N
    s_W = zeros(N,N); r_W = zeros(N,N); s_h = zeros(N); r_h = zeros(N)
    best_cost = Inf; best_W = copy(net.W); best_h = copy(net.h); ch = zeros(N_EPOCH)
    nrow = 4 * MINIT
    for epoch in 1:N_EPOCH
        a, c = ac_at(epoch, mode); net.a = a; net.c = c
        x0 = zeros(nrow, N); tgt = zeros(nrow, 1); row = 1
        for p in 1:4, _ in 1:MINIT
            x0[row, INPUT] .= data[p, :]
            x0[row, VAR]   .= TEST_RANGE .* (2 .* rand(rng, length(VAR)) .- 1)
            tgt[row, 1] = target[p, 1]; row += 1
        end
        gW, gh, cost, _ = EP_param_gradient(net, x0, tgt, BETA; symmetric=false)
        net.W, s_W, r_W = adam_update(net.W, gW, LR, epoch, s_W, r_W)
        net.W = (net.W + net.W') / 2; net.W .*= MASK
        net.h, s_h, r_h = adam_update(net.h, gh, LR, epoch, s_h, r_h)
        ch[epoch] = cost
        if net.a == -1.0 && net.c == 1.0 && cost < best_cost
            best_cost = cost; best_W = copy(net.W); best_h = copy(net.h)
        end
    end
    net.a = -1.0; net.c = 1.0; net.W = best_W; net.h = best_h
    return ch
end

function robust_acc(net, rng)
    net.a = -1.0; net.c = 1.0
    correct = 0
    for i in 1:4, _ in 1:N_TEST_INIT
        x0 = zeros(1, net.N); x0[1, INPUT] .= data[i, :]
        x0[1, VAR] .= TEST_RANGE .* (2 .* rand(rng, length(VAR)) .- 1)
        eq = relax_batch(net, x0, reshape(target[i, :], 1, :), 0.0)
        sign(eq[1, OUT[1]]) == sign(target[i, 1]) && (correct += 1)
    end
    return correct / (4 * N_TEST_INIT)
end

println("Layered EP-Duffing: Landau cooling vs minima-fixed annealing on XOR")
println(length(SEEDS), " seeds x ", N_TEST_INIT, " draws, layered H=", N_HIDDEN,
        " (N=", Nn, "), Minit=", MINIT, ", ", N_EPOCH, " epochs, full-range test\n")
@printf("%-24s | %-6s %-5s %-5s %-8s | %s\n", "schedule", "mean", "max", "min", "solved", "med cost")
println("-"^62)

for (label, mode) in CONFIGS
    accs = Float64[]; fc = Float64[]
    for seed in SEEDS
        rng = MersenneTwister(seed)
        net = DuffingNetwork(Nn, INPUT, OUT; a=-1.0, c=1.0, delta=1.0)
        random_init!(net; rng=rng)
        net.W = (net.W + net.W') / 2; net.W .*= MASK
        ch = train!(net, mode; rng=rng)
        push!(accs, robust_acc(net, MersenneTwister(9000+seed)))
        push!(fc, ch[end])
    end
    solved = count(>=(SOLVE_THRESH), accs)
    @printf("%-24s | %-6.0f %-5.0f %-5.0f %d/%-6d | %.3f\n",
            label, 100mean(accs), 100maximum(accs), 100minimum(accs), solved, length(SEEDS), median(fc))
end

println("\nRef: layered H=12 + minima-fixed (5 seeds): 95% mean, 85% floor, 4/5 solved, cost 0.034")
