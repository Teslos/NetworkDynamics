# Layered EP-Duffing: test the capacity hypothesis for the bistable XOR.
#
# Meta-finding across 5 mechanisms (results/ep_duffing_landau_anneal.md): all
# plateau at ~79-84% mean / 2-3 of 6 seeds solved / best seed ~100% on the N=5,
# 2-hidden, all-to-all bistable Duffing -- pointing at CAPACITY (not the training
# trick) as the residual limit. Two capacity/architecture levers to test:
#
#   * More hidden units.
#   * LAYERED (feedforward-symmetric) connectivity: coupling only between adjacent
#     layers (input<->hidden, hidden<->output), none within a layer or input<->output.
#     Symmetric, so the energy is still a gradient and EP applies. This structures
#     information flow input->hidden->output, so the OUTPUT's effective field comes
#     from the (input-driven) hidden layer -- which can supply a nonzero
#     symmetry-breaking field to the output even for the XOR-hard (+1,+1) corner
#     (the corner that sits at the pitchfork singularity in the all-to-all net).
#     Layered connectivity also lifted XY from 80->94% at the digit scale.
#
# Best recipe held fixed (basin-averaging + minima-fixed annealing a=-s,c=s
# s:0.3->1); the only variable is architecture. Honest full-range test.
#
# Usage: julia --project=. scripts/duffing_layered.jl

using Random, Printf, Statistics, LinearAlgebra

EP_DUFFING_SKIP_RUN = true
include(joinpath(@__DIR__, "..", "notebooks", "EP-Duffing-Network.jl"))

const SEEDS        = 1:5
const N_EPOCH      = 1000
const MINIT        = 40
const ANNEAL_FRAC  = 0.5
const S_START      = 0.3
const BETA         = 0.1
const LR           = 0.02
const TEST_RANGE   = 1.5
const N_TEST_INIT  = 25
const SOLVE_THRESH = 0.90

# (label, n_hidden, layered?)
const CONFIGS = [
    ("all-to-all H=2",  2,  false),
    ("layered   H=6",   6,  true),
    ("layered   H=12",  12, true),
]

data   = Float64[-1 -1; -1 1; 1 -1; 1 1]
target = reshape(Float64[-1, 1, 1, -1], 4, 1)

s_at(epoch) = ANNEAL_FRAC <= 0.0 ? 1.0 :
    (n = max(1, round(Int, ANNEAL_FRAC * N_EPOCH));
     epoch >= n ? 1.0 : S_START + (1.0 - S_START) * (epoch - 1) / (n == 1 ? 1 : n - 1))

# Symmetric allowed-coupling mask. all-to-all: everything off-diagonal. layered:
# only input<->hidden and hidden<->output adjacent-layer blocks.
function coupling_mask(N, input_idx, hidden_idx, output_idx, layered)
    M = zeros(N, N)
    if !layered
        M .= 1.0; M[diagind(M)] .= 0.0
        return M
    end
    for i in input_idx, j in hidden_idx;  M[i,j] = 1.0; M[j,i] = 1.0; end
    for i in hidden_idx, j in output_idx; M[i,j] = 1.0; M[j,i] = 1.0; end
    return M
end

function train!(net, input_idx, var_idx, M; rng)
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
        gW, gh, cost, _ = EP_param_gradient(net, x0, tgt, BETA; symmetric=false)
        net.W, s_W, r_W = adam_update(net.W, gW, LR, epoch, s_W, r_W)
        net.W = (net.W + net.W') / 2
        net.W .*= M                      # enforce (layered) connectivity
        net.h, s_h, r_h = adam_update(net.h, gh, LR, epoch, s_h, r_h)
        ch[epoch] = cost
        if net.a == -1.0 && cost < best_cost
            best_cost = cost; best_W = copy(net.W); best_h = copy(net.h)
        end
    end
    net.a = -1.0; net.c = 1.0; net.W = best_W; net.h = best_h
    return ch
end

function robust_acc(net, input_idx, var_idx, out_idx, rng)
    net.a = -1.0; net.c = 1.0
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

println("Layered EP-Duffing -- robust XOR under full-range test init")
println(length(SEEDS), " seeds x ", N_TEST_INIT, " draws, Minit=", MINIT, ", ",
        N_EPOCH, " epochs, basin-avg + anneal; solve>=", Int(100SOLVE_THRESH), "%\n")
@printf("%-18s | %-4s | %-6s %-5s %-5s %-8s | %s\n",
        "architecture", "N", "mean", "max", "min", "solved", "med cost")
println("-"^62)

for (label, nh, layered) in CONFIGS
    N = 2 + nh + 1
    input_idx = [1, 2]; hidden_idx = collect(3:2+nh); out_idx = [N]
    var_idx = setdiff(1:N, input_idx)
    M = coupling_mask(N, input_idx, hidden_idx, out_idx, layered)
    accs = Float64[]; fc = Float64[]
    for seed in SEEDS
        rng = MersenneTwister(seed)
        net = DuffingNetwork(N, input_idx, out_idx; a=-1.0, c=1.0, delta=1.0)
        random_init!(net; rng=rng)
        net.W = (net.W + net.W') / 2; net.W .*= M      # symmetric, masked init
        ch = train!(net, input_idx, var_idx, M; rng=rng)
        push!(accs, robust_acc(net, input_idx, var_idx, out_idx, MersenneTwister(9000+seed)))
        push!(fc, ch[end])
    end
    solved = count(>=(SOLVE_THRESH), accs)
    @printf("%-18s | %-4d | %-6.0f %-5.0f %-5.0f %d/%-6d | %.3f\n",
            label, N, 100mean(accs), 100maximum(accs), 100minimum(accs), solved, length(SEEDS), median(fc))
end

println("\nReference (all-to-all H=2, basin-avg+anneal): ~81-84% mean, 3/6 solved, best ~100%")
