# Gradient-fidelity check for Equilibrium Propagation on oscillator networks.
#
# Compares the EP parameter-gradient estimate against the TRUE gradient of the
# loss, obtained by central finite differences of the actual free-equilibrium
# cost L(theta) = cost(relax_to_fixed_point(theta)). Finite differences are the
# gold-standard reference here: they need no autodiff through the steady-state
# callback and make no assumption about the EP derivation.
#
# EP theory predicts the estimate -> true gradient as the nudge beta -> 0, with
# an O(beta) (one-sided) / O(beta^2) (symmetric) bias. So we sweep beta and
# report, per beta:
#   * cosine similarity   cos(g_EP, g_FD)   -> 1   (direction correct)
#   * magnitude ratio     ||g_EP|| / ||g_FD||  -> 1 (scale correct)
#   * relative L2 error   ||g_EP - g_FD|| / ||g_FD|| -> 0
#
# Usage:
#   julia --project=. scripts/check_ep_gradient_fidelity.jl duffing
#   julia --project=. scripts/check_ep_gradient_fidelity.jl xy
#
# (Run separately: the two model files define clashing global names.)

using LinearAlgebra
using Random
using Printf
using Statistics

const SUBSTRATE = length(ARGS) >= 1 ? ARGS[1] : "duffing"
const N = 5
const INPUT_IDX = [1, 2]
const OUTPUT_IDX = [5]
const VAR_IDX = setdiff(1:N, INPUT_IDX)
const BETAS = [0.2, 0.1, 0.05, 0.02, 0.01]
const FD_STEP = 1e-3

# Upper-triangle (i<j) <-> symmetric-matrix helpers. We treat each symmetric
# weight pair W_ij = W_ji as ONE parameter; the shipped EP weight gradient is
# already symmetric and corresponds to the derivative w.r.t. that single pair,
# so the FD perturbation must move both entries together.
upper_tri(M) = [M[i, j] for i in 1:N for j in i+1:N]
function fill_sym!(W, ut)
    W .= 0.0
    k = 1
    @inbounds for i in 1:N, j in i+1:N
        W[i, j] = ut[k]; W[j, i] = ut[k]; k += 1
    end
    return W
end

# Central-difference gradient of loss_fn over the flat parameter vector theta.
function fd_gradient(theta, loss_fn; step=FD_STEP)
    g = similar(theta)
    for k in eachindex(theta)
        tp = copy(theta); tp[k] += step
        tm = copy(theta); tm[k] -= step
        g[k] = (loss_fn(tp) - loss_fn(tm)) / (2step)
    end
    return g
end

cossim(a, b) = dot(a, b) / (norm(a) * norm(b) + 1e-30)

function report(label, theta, ep_grad_fn, loss_fn; blocks=nothing)
    g_fd = fd_gradient(theta, loss_fn)
    nfd = norm(g_fd)
    println("\n=== $label ===")
    @printf("  ||g_FD|| = %.3e   (true gradient norm)\n", nfd)
    println("   beta     cos(g_EP,g_FD)   ||g_EP||/||g_FD||   rel.L2 err")
    println("  ------    --------------   ----------------    ----------")
    for beta in BETAS
        g_ep = ep_grad_fn(theta, beta)
        c = cossim(g_ep, g_fd)
        ratio = norm(g_ep) / nfd
        rel = norm(g_ep - g_fd) / nfd
        @printf("  %5.3f     %12.6f     %12.4f      %10.4f\n", beta, c, ratio, rel)
    end
    if blocks !== nothing
        beta = minimum(BETAS)
        g_ep = ep_grad_fn(theta, beta)
        println("  per-block at beta=$beta  (name: cos, ||g_EP||, ||g_FD||)")
        for (name, idx) in blocks
            @printf("    %-8s cos=%+.4f   ||g_EP||=%.3e   ||g_FD||=%.3e\n",
                    name, cossim(g_ep[idx], g_fd[idx]), norm(g_ep[idx]), norm(g_fd[idx]))
        end
    end
end

# ============================================================================
if SUBSTRATE == "duffing"
    EP_DUFFING_SKIP_RUN = true
    include(joinpath(@__DIR__, "..", "notebooks", "EP-Duffing-Network.jl"))

    target = reshape(Float64[-1, 1, 1, -1], 4, 1)
    data   = Float64[-1 -1; -1 1; 1 -1; 1 1]

    function make_net(theta)
        net = DuffingNetwork(N, INPUT_IDX, OUTPUT_IDX)
        W = zeros(N, N)
        fill_sym!(W, theta[1:length(upper_tri(W))])
        net.W = W
        net.h = theta[end-N+1:end]
        return net
    end

    # Fixed deterministic warm start (inputs clamped, free cells seeded), then
    # relaxed to the free equilibrium so FD perturbations stay basin-stable.
    function warm_start(net)
        x0 = zeros(4, N)
        x0[:, INPUT_IDX] .= data
        x0[:, VAR_IDX]   .= 0.05
        return relax_batch(net, x0, target, 0.0)
    end

    function eval_point(label, theta0)
        net0 = make_net(theta0)
        X0 = warm_start(net0)            # frozen across all evaluations
        loss_fn = function (theta)
            net = make_net(theta)
            xeq = relax_batch(net, X0, target, 0.0)
            return batch_cost(xeq, target, OUTPUT_IDX)[1]
        end
        ep_grad_fn = function (theta, beta)
            net = make_net(theta)
            gW, gh, _, _ = EP_param_gradient(net, X0, target, beta; symmetric=true)
            return vcat(upper_tri(gW), gh)
        end
        report(label, theta0, ep_grad_fn, loss_fn)
    end

    println("EP gradient fidelity -- damped Duffing network (XOR), N=$N")

    # (i) random init
    rng = MersenneTwister(7)
    net = DuffingNetwork(N, INPUT_IDX, OUTPUT_IDX); random_init!(net; rng=rng)
    theta_rand = vcat(upper_tri(net.W), net.h)
    eval_point("Operating point A: random initialization", theta_rand)

    # (ii) lightly trained (still nonzero gradient)
    train!(net, data, target, 0.05, 0.02, 300; print_every=10^9, rng=rng)
    theta_trained = vcat(upper_tri(net.W), net.h)
    eval_point("Operating point B: after 300 training epochs", theta_trained)

# ============================================================================
elseif SUBSTRATE == "xy"
    EP_XY_SKIP_RUN = true
    include(joinpath(@__DIR__, "..", "notebooks", "EP-XY-Network-Claude.jl"))

    const N_EV = 200
    const DT = 0.1
    const T = N_EV * DT
    data   = (π / 2) * Float64[-1 -1; -1 1; 1 -1; 1 1]
    target = (π / 2) * reshape(Float64[-1, 1, 1, -1], 4, 1)

    n_w = length(upper_tri(zeros(N, N)))   # weights in theta
    # theta layout: [W upper-tri ; bias row1 (h) ; bias row2 (psi)]
    function unpack(theta)
        W = zeros(N, N); fill_sym!(W, theta[1:n_w])
        bias = zeros(2, N)
        bias[1, :] = theta[n_w+1 : n_w+N]
        bias[2, :] = theta[n_w+N+1 : n_w+2N]
        return W, bias
    end

    function warm_start(W, bias)
        phase0 = zeros(4, N)
        phase0[:, INPUT_IDX] .= data
        phase0[:, VAR_IDX]   .= 0.05
        return run_network_batch(phase0, T, W, bias, target, 0.0, INPUT_IDX, OUTPUT_IDX)
    end

    XY_BLOCKS = [("weights", 1:n_w), ("h", n_w+1:n_w+N), ("psi", n_w+N+1:n_w+2N)]

    function eval_point(label, theta0)
        W0, bias0 = unpack(theta0)
        P0 = warm_start(W0, bias0)
        loss_fn = function (theta)
            W, bias = unpack(theta)
            eq = run_network_batch(P0, T, W, bias, target, 0.0, INPUT_IDX, OUTPUT_IDX)
            return batch_costs(eq, target, OUTPUT_IDX)[1]
        end
        ep_grad_fn = function (theta, beta)
            W, bias = unpack(theta)
            gW, gbias, _, _ = EP_param_gradient(W, bias, P0, target, beta,
                                                N_EV, DT, INPUT_IDX, VAR_IDX, OUTPUT_IDX;
                                                symmetric=true)
            return vcat(upper_tri(gW), gbias[1, :], gbias[2, :])
        end
        report(label, theta0, ep_grad_fn, loss_fn; blocks=XY_BLOCKS)
    end

    println("EP gradient fidelity -- XY / Kuramoto phase network (XOR), N=$N")

    rng = MersenneTwister(7)
    net = SP_XY_Network(N, N_EV, DT, INPUT_IDX, OUTPUT_IDX)
    random_state_initiation!(net)
    net.weights *= 0.1; net.bias *= 0.1
    theta_rand = vcat(upper_tri(net.weights), net.bias[1, :], net.bias[2, :])
    eval_point("Operating point A: random initialization", theta_rand)

    # Lightly trained operating point (away from random-init degeneracies).
    Wt, bt, _ = train_network(net.weights, net.bias, data, target, 0.01, 0.05,
                              300, 4, N_EV, DT, INPUT_IDX, net.variable_index, OUTPUT_IDX;
                              print_every=10^9)
    theta_trained = vcat(upper_tri(Wt), bt[1, :], bt[2, :])
    eval_point("Operating point B: after 300 training epochs", theta_trained)

    # Confirmatory test: STRONGLY-coupled init. If the weak-coupling low-curvature
    # diagnosis is right, the weight-block fidelity should sharpen markedly.
    rng2 = MersenneTwister(7)
    Wstrong = 1.5 * randn(rng2, N, N); Wstrong = (Wstrong + Wstrong') / 2
    Wstrong[diagind(Wstrong)] .= 0
    bstrong = copy(net.bias)
    theta_strong = vcat(upper_tri(Wstrong), bstrong[1, :], bstrong[2, :])
    eval_point("Operating point C: strongly-coupled init (W ~ 1.5)", theta_strong)

else
    error("unknown substrate $SUBSTRATE (use 'duffing' or 'xy')")
end
