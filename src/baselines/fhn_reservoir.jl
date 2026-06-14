# Reusable FHN reservoir, factored out of scripts/run_fhn_digits.jl so the
# diagnostics (B7 separability, B9 edge-of-chaos) can reuse the exact dynamics.
#
# Complete-graph FitzHugh-Nagumo reservoir, one node per input sample, diffusive
# all-to-all coupling of strength `sigma`. The coupling is a dense W*u matvec, so
# even the 1797-node solve is tractable.
#
#   du_i/dt = g_i(t) + u_i - u_i^3/3 - v_i + sum_j W_ij (u_j - u_i)
#   dv_i/dt = (g_i(t)*R0 + u_i - a) * eps
#
# `S` is the per-node drive matrix (N, T): row i is sample i's input sequence.

module FHNReservoir

using OrdinaryDiffEq
using LinearAlgebra
using Random
using Distributions
using Statistics
using Graphs

export fhn_states, fhn_esp_divergence, ws_adjacency

# Watts-Strogatz adjacency as a dense symmetric 0/1 mask (zero diagonal), for use
# as the `mask` argument to fhn_states. `k` = mean degree (even), `beta` = rewire.
function ws_adjacency(n::Int, k::Int, beta::Real; rng=Random.default_rng())
    g = watts_strogatz(n, k, beta; rng=rng)
    return Float64.(Matrix(adjacency_matrix(g)))
end

const EPS = 0.05
const A = 0.5
const R0 = 0.5

# complete-graph diffusive coupling matrix; weights ~ sigma * Normal-pdf(U[-1,1])
function _coupling(N, sigma, rng)
    W = [pdf(Normal(), r) for r in (2 .* rand(rng, N, N) .- 1)]
    W = sigma .* (W .+ W') ./ 2
    W[diagind(W)] .= 0
    return W, vec(sum(W, dims=2))
end

function _solve(S, W, rowsum, z0; a=A, eps=EPS, r0=R0)
    N, T = size(S)
    gbuf = zeros(N)
    function drive!(out, t)
        if t <= 1
            @inbounds out .= @view S[:, 1]
        else
            i = min(floor(Int, t), T - 1); f = t - i
            @inbounds out .= (1 - f) .* @view(S[:, i]) .+ f .* @view(S[:, i + 1])
        end
        return out
    end
    function rhs!(dz, z, p, t)
        u = @view z[1:N]; v = @view z[N+1:2N]
        du = @view dz[1:N]; dv = @view dz[N+1:2N]
        g = drive!(gbuf, t)
        coupling = W * u .- rowsum .* u
        @. du = g + u - u^3 / 3 - v + coupling
        @. dv = (g * r0 + u - a) * eps
        return nothing
    end
    prob = ODEProblem(rhs!, z0, (0.0, Float64(T)))
    sol = solve(prob, Tsit5(); saveat=1.0:1.0:T, save_idxs=1:N)
    return Array(sol)   # (N, T) node u-trajectories
end

# Reservoir node trajectories U (N, T) for drive S and coupling strength sigma.
# `a` is the FHN excitability threshold: |a|<1 = self-oscillating limit cycle
# (the manuscript's a=0.5), |a|>1 = excitable/quiescent (needed for input-driven
# avalanche propagation, Beattie-style criticality). `mask` optionally restricts
# the complete-graph coupling to a given adjacency (e.g. Watts-Strogatz).
function fhn_states(S, sigma; seed=1, a=A, eps=EPS, mask=nothing)
    rng = Xoshiro(seed)
    W, _ = _coupling(size(S, 1), sigma, rng)
    mask !== nothing && (W = W .* mask)
    rowsum = vec(sum(W, dims=2))
    z0 = rand(rng, 2 * size(S, 1))
    return _solve(S, W, rowsum, z0; a=a, eps=eps)
end

"""
Echo-state-property probe: drive the same reservoir from two random initial
states and return the mean final |u1-u2| across nodes, normalized by the typical
final amplitude. Small => states forget initial conditions (ESP holds); growing
with sigma marks the edge of chaos.
"""
function fhn_esp_divergence(S, sigma; seed=1)
    rng = Xoshiro(seed)
    N = size(S, 1)
    W, rowsum = _coupling(N, sigma, rng)
    U1 = _solve(S, W, rowsum, rand(rng, 2N))
    U2 = _solve(S, W, rowsum, rand(rng, 2N))
    scale = mean(abs.(U1[:, end])) + mean(abs.(U2[:, end])) + 1e-9
    return mean(abs.(U1[:, end] .- U2[:, end])) / scale
end

end # module FHNReservoir
