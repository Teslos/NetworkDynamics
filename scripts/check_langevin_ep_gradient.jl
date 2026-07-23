# Gradient-fidelity check for FINITE-TEMPERATURE (Langevin-sampled) Equilibrium
# Propagation on the Duffing network.
#
# The finite-T EP contrast estimates the gradient of the THERMAL-AVERAGED cost
#   <C>_0 = < 1/2 sum_out (x_out - target)^2 >_free
# (not the "cost of the mean"), because the nudge adds beta*C(x) to the energy
# instantaneously. So the finite-difference reference below differentiates <C>_0,
# computed from the sampled first and second moments.
#
# Two regimes are reported:
#   * MONOSTABLE (a>0, single well): sampling is unimodal, common random numbers
#     across the +-beta phases cancel cleanly, and the estimator is faithful
#     (cos(g_EP, g_FD) ~ 0.98, stable across beta) -- this validates the code.
#   * BISTABLE (a<0, double well): the +-beta paths cross the barrier at different
#     times, so the correlation difference is dominated by crossing-time jitter,
#     not the beta-drift; the gradient is high-variance (cos low / unstable).
#     This is a genuine property of thermal sampling on a multimodal landscape,
#     not a bug -- it is the reason a temperature/well-depth window matters.
#
# Usage:  julia --project=. scripts/check_langevin_ep_gradient.jl
#
# Uses common random numbers (fixed seed per sampler call) throughout to isolate
# the drift signal from sampling noise.

using LinearAlgebra
using Random
using Statistics
using Printf

EP_DUFFING_LANGEVIN_SKIP_RUN = true
include(joinpath(@__DIR__, "..", "notebooks", "EP-Duffing-Langevin.jl"))

const N = 5
const INPUT_IDX = [1, 2]
const OUTPUT_IDX = [5]
const VAR_IDX = setdiff(1:N, INPUT_IDX)
const BETAS = [0.1, 0.05, 0.02]
const FD_STEP = 2e-3
const nW = length([1 for i in 1:N for j in i+1:N])

upper_tri(M) = [M[i, j] for i in 1:N for j in i+1:N]
function fill_sym(ut)
    W = zeros(N, N); k = 1
    @inbounds for i in 1:N, j in i+1:N
        W[i, j] = ut[k]; W[j, i] = ut[k]; k += 1
    end
    return W
end
cossim(a, b) = dot(a, b) / (norm(a) * norm(b) + 1e-30)

# One fidelity report at a given operating point / regime.
function report(label; a, T, target, seed, sampler_seed,
                dt=0.02, n_burn=6000, n_sample=40000)
    make_net(th) = begin
        net = DuffingNetwork(N, INPUT_IDX, OUTPUT_IDX; a=a, c=1.0)
        net.W = fill_sym(th[1:nW]); net.h = th[nW+1:nW+N]; net
    end
    X0 = zeros(4, N); X0[:, INPUT_IDX] .= DATA; X0[:, VAR_IDX] .= 0.05

    # Thermal-averaged free cost <C>_0 from mean + second moments.
    meanC = function (th)
        net = make_net(th)
        mx, corr, _ = langevin_sample_batch(net, X0, target, 0.0, T;
            dt=dt, n_burn=n_burn, n_sample=n_sample, rng=MersenneTwister(sampler_seed))
        c = 0.0
        for d in 1:4, o in OUTPUT_IDX
            c += 0.5 * (corr[d, o, o] - 2 * target[d, 1] * mx[d, o] + target[d, 1]^2)
        end
        return c / 4
    end
    ep_grad = function (th, beta)
        net = make_net(th)
        gW, gh, _, _ = EP_langevin_gradient(net, X0, target, beta, T;
            symmetric=true, dt=dt, n_burn=n_burn, n_sample=n_sample,
            rng=MersenneTwister(sampler_seed))
        return vcat(upper_tri(gW), gh)
    end
    fd_grad = function (th; step=FD_STEP)
        g = similar(th)
        for k in eachindex(th)
            tp = copy(th); tp[k] += step
            tm = copy(th); tm[k] -= step
            g[k] = (meanC(tp) - meanC(tm)) / (2step)
        end
        return g
    end

    rng = MersenneTwister(seed)
    net = DuffingNetwork(N, INPUT_IDX, OUTPUT_IDX; a=a, c=1.0); random_init!(net; rng=rng)
    th = vcat(upper_tri(net.W), net.h)

    gfd = fd_grad(th); nfd = norm(gfd)
    println("\n=== $label ===")
    @printf("  ||g_FD|| = %.3e   (true gradient of <C>_0)\n", nfd)
    println("   beta     cos(g_EP,g_FD)   ||g_EP||/||g_FD||   rel.L2 err")
    println("  ------    --------------   ----------------    ----------")
    for beta in BETAS
        gep = ep_grad(th, beta)
        @printf("  %5.3f     %12.6f     %12.4f      %10.4f\n",
                beta, cossim(gep, gfd), norm(gep) / nfd, norm(gep - gfd) / nfd)
    end
end

const DATA = Float64[-1 -1; -1 1; 1 -1; 1 1]
const TARGET_PM1 = reshape(Float64[-1, 1, 1, -1], 4, 1)

println("Finite-T Langevin EP gradient fidelity -- Duffing network (XOR), N=$N")

# (A) Monostable, unimodal: validates the estimator code (expect cos ~ 0.98).
report("MONOSTABLE (a=+1, single well), T=0.10, soft targets"; a=1.0, T=0.10,
       target=TARGET_PM1 ./ 2, seed=3, sampler_seed=999)

# (B) Bistable, double well: expect high-variance / crossing-jitter-limited.
report("BISTABLE (a=-1, double well), T=0.15"; a=-1.0, T=0.15,
       target=TARGET_PM1, seed=7, sampler_seed=12345)
