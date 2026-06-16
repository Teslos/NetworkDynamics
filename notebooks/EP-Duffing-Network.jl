# Equilibrium Propagation on a network of damped, UNFORCED, symmetrically
# coupled Duffing oscillators, trained on XOR.
#
# This is the second-oscillator generality test alongside the XY/phase network
# in EP-XY-Network-Claude.jl. EP works here for the same reason it works there:
# the free dynamics is a *dissipative gradient system* that relaxes to a static
# fixed point of a symmetric energy, so the EP two-equilibria estimator is exact
# at that fixed point.
#
# Substrate -- one Duffing unit per neuron, double-well potential:
#
#   V(x) = (1/4) c x^4 + (1/2) a x^2 ,   a < 0, c > 0  ->  minima at x = +-sqrt(-a/c)
#
# With a = -1, c = 1 the two stable wells sit at x = +-1, so the well bottoms ARE
# the +-1 logic levels (the amplitude analogue of the 0/pi phase encoding used in
# the Kuramoto-Hopfield model).
#
# Network energy (symmetric W, no self-coupling, bias h):
#
#   E(x) = sum_i V(x_i)  -  (1/2) sum_ij W_ij x_i x_j  -  sum_i h_i x_i
#
# Damped second-order ("Duffing") dynamics, F_i = -dE/dx_i:
#
#   xdot_i = v_i
#   vdot_i = -delta v_i  -  ( c x_i^3 + a x_i )  +  sum_j W_ij x_j  +  h_i
#
# Why EP still applies despite being second order: at equilibrium xdot = vdot = 0,
# so v = 0 and the fixed point is a stationary point of E(x) alone. Inertia only
# shapes the transient, not where the network lands, and the EP gradient is
# evaluated purely at the equilibrium positions x*. Damping (delta > 0) is what
# makes the relaxation actually settle so the steady-state callback can fire.
#
# Caveats (inherent to the double well, not to the code):
#   * Multistability: the free equilibrium of a free cell is basin-dependent, so
#     hidden/output cells are seeded with small noise to break the x = 0 saddle,
#     and the nudge must not kick the state across a basin boundary.
#   * delta must be large enough (overdamped-ish) for the relaxation to settle.

using LinearAlgebra
using Statistics
using Random
using OrdinaryDiffEq
using SciMLBase: get_du

# ----------------------------------------------------------------------------
# Network
# ----------------------------------------------------------------------------

mutable struct DuffingNetwork
    N::Int
    input_index::Vector{Int}
    output_index::Vector{Int}
    variable_index::Vector{Int}
    T::Float64                # max relaxation time span (callback usually ends earlier)
    a::Float64                # quadratic potential coeff (< 0 for a double well)
    c::Float64                # quartic potential coeff (> 0)
    delta::Float64            # damping
    W::Matrix{Float64}        # symmetric coupling, zero diagonal
    W_0::Matrix{Float64}      # initial coupling (kept for diagnostics)
    h::Vector{Float64}        # bias / local field
    h_0::Vector{Float64}
end

function DuffingNetwork(N::Int, input_index::Vector{Int}, output_index::Vector{Int};
                        T=40.0, a=-1.0, c=1.0, delta=1.0)
    variable_index = setdiff(1:N, input_index)
    W = zeros(N, N)
    h = zeros(N)
    return DuffingNetwork(N, input_index, output_index, variable_index,
                          T, a, c, delta, W, copy(W), h, copy(h))
end

function random_init!(net::DuffingNetwork; w_scale=0.1, h_scale=0.05, rng=Random.default_rng())
    W = w_scale * randn(rng, net.N, net.N)
    W = (W + W') / 2                 # symmetric coupling -> energy is a gradient
    W[diagind(W)] .= 0               # no self-coupling
    net.W = W
    net.W_0 = copy(W)
    net.h = h_scale * randn(rng, net.N)
    net.h_0 = copy(net.h)
    return net
end

# ----------------------------------------------------------------------------
# Dynamics
# ----------------------------------------------------------------------------
# State layout: u[1:N] = positions x, u[N+1:2N] = velocities v.
# In-place and allocation-free: this is the hot path of the whole run.

function duffing_force!(du, u, p, t)
    N = p.N
    W, h, a, c, delta = p.W, p.h, p.a, p.c, p.delta
    @inbounds for i in 1:N
        xi = u[i]
        acc = 0.0
        for j in 1:N
            acc += W[i, j] * u[j]          # W symmetric; coupling = +sum_j W_ij x_j
        end
        F = -(c * xi^3 + a * xi) + acc + h[i]   # F_i = -dE/dx_i
        du[i] = u[N + i]                   # xdot = v
        du[N + i] = -delta * u[N + i] + F  # vdot = -delta v + F
    end
    # Cost nudge on the output cells: C = (1/2) sum (x_j - target)^2, adds beta*C
    # to the energy, i.e. force -beta * dC/dx_j = -beta (x_j - target).
    if p.beta != 0.0
        @inbounds for (m, j) in enumerate(p.output_index)
            du[N + j] -= p.beta * (u[j] - p.target[m])
        end
    end
    # Clamp inputs: freeze position (and velocity) at the supplied value.
    @inbounds for j in p.input_index
        du[j] = 0.0
        du[N + j] = 0.0
    end
    return nothing
end

duffing_params(net::DuffingNetwork, target, beta) =
    (N=net.N, W=net.W, h=net.h, a=net.a, c=net.c, delta=net.delta,
     beta=beta, target=target, input_index=net.input_index, output_index=net.output_index)

# Stop a relaxation once both x and v stop moving (max|du| small). Because du
# includes vdot, this also guarantees v -> 0, i.e. a genuine static fixed point.
const STEADY_TOL = 1e-5
steady_state_callback() = DiscreteCallback(
    (u, t, integrator) -> maximum(abs, get_du(integrator)) < STEADY_TOL,
    terminate!; save_positions=(false, false))

const SOLVER_KWARGS = (reltol=1e-6, abstol=1e-8, maxiters=10^5,
                       save_everystep=false, save_start=false, verbose=false)

# Relax a batch of samples (one trajectory each) from given start positions
# (velocities zero) and return the equilibrium POSITIONS, N_batch x N.
function relax_batch(net::DuffingNetwork, x0_batch, target_batch, beta)
    N_batch, N = size(x0_batch)
    eq = zeros(N_batch, N)
    for d in 1:N_batch
        u0 = zeros(2N)
        @views u0[1:N] .= x0_batch[d, :]
        p = duffing_params(net, view(target_batch, d, :), beta)
        prob = ODEProblem(duffing_force!, u0, (0.0, net.T), p)
        sol = solve(prob, Tsit5(); callback=steady_state_callback(), SOLVER_KWARGS...)
        @views eq[d, :] .= sol.u[end][1:N]
    end
    return eq
end

# ----------------------------------------------------------------------------
# Cost and EP gradients
# ----------------------------------------------------------------------------
# Quadratic readout cost on the output cells.

function batch_cost(x_eq, target_batch, output_index; tol=0.2)
    dev = (x_eq[:, output_index] .- target_batch) .^ 2
    cost = mean(vec(sum(dev, dims=2)) ./ 2)
    q_cost = mean(vec(sum(dev .> tol, dims=2)))   # # of outputs off by > sqrt(tol)
    return cost, q_cost
end

# EP gradient of the energy parameters. For E = sum V - 1/2 sum W_ij x_i x_j - sum h_i x_i:
#   dE/dW_ij = -x_i x_j ,   dE/dh_j = -x_j
# Estimator: dL/dtheta ~ (1/scale)[ dE/dtheta(x_nudge) - dE/dtheta(x_free) ].
function weights_gradient(x_nudge, x_free)
    N_data, N = size(x_free)
    g = zeros(N, N)
    @inbounds for d in 1:N_data, i in 1:N, j in 1:N
        g[i, j] += (x_free[d, i] * x_free[d, j] - x_nudge[d, i] * x_nudge[d, j]) / N_data
    end
    g[diagind(g)] .= 0          # no self-coupling
    return g                    # symmetric by construction (x_i x_j)
end

function bias_gradient(x_nudge, x_free)
    N_data, N = size(x_free)
    g = zeros(N)
    @inbounds for d in 1:N_data, j in 1:N
        g[j] += (x_free[d, j] - x_nudge[d, j]) / N_data
    end
    return g
end

# Free relaxation, then nudged relaxation(s) started from the free equilibrium.
# symmetric=true uses the centered (+-beta) estimate (3 relaxations); false uses
# the one-sided estimate (2 relaxations, noisier).
function EP_param_gradient(net::DuffingNetwork, x0_batch, target_batch, beta; symmetric=true)
    x_zero = relax_batch(net, x0_batch, target_batch, 0.0)
    x_nudge = relax_batch(net, x_zero, target_batch, beta)
    if symmetric
        x_free = relax_batch(net, x_zero, target_batch, -beta)
        scale = 2 * beta
    else
        x_free = x_zero
        scale = beta
    end
    cost, q_cost = batch_cost(x_zero, target_batch, net.output_index)
    gW = weights_gradient(x_nudge, x_free) ./ scale
    gh = bias_gradient(x_nudge, x_free) ./ scale
    return gW, gh, cost, q_cost
end

# ----------------------------------------------------------------------------
# Adam
# ----------------------------------------------------------------------------

function adam_update(param, grad, lr, t, s, r; b1=0.9, b2=0.999, eps=1e-8)
    grad = clamp.(grad, -1.0, 1.0)
    s = b1 .* s .+ (1 - b1) .* grad
    r = b2 .* r .+ (1 - b2) .* grad .^ 2
    s_hat = s ./ (1 - b1^t)
    r_hat = r ./ (1 - b2^t)
    param = param .- lr .* s_hat ./ (sqrt.(r_hat) .+ eps)
    return param, s, r
end

# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------

function train!(net::DuffingNetwork, data, target, beta, lr, N_epoch;
                symmetric=true, noise=0.1, print_every=200, rng=Random.default_rng())
    N = net.N
    N_data = size(data, 1)
    cost_history = zeros(N_epoch)

    s_W = zeros(N, N); r_W = zeros(N, N)
    s_h = zeros(N);    r_h = zeros(N)

    # Best-state checkpoint: the double well is multistable, so the free cost can
    # bounce as patterns occasionally settle into the wrong basin. Keep the best
    # weights seen and restore them at the end (cf. the XY trainer).
    best_cost = Inf
    best_W = copy(net.W)
    best_h = copy(net.h)

    for epoch in 1:N_epoch
        # Inputs clamped to the data; free cells seeded with small noise so they
        # leave the x = 0 saddle of the double well.
        x0 = zeros(N_data, N)
        x0[:, net.input_index] .= data
        x0[:, net.variable_index] .= noise * randn(rng, N_data, length(net.variable_index))

        gW, gh, cost, _ = EP_param_gradient(net, x0, target, beta; symmetric=symmetric)

        net.W, s_W, r_W = adam_update(net.W, gW, lr, epoch, s_W, r_W)
        net.W = (net.W + net.W') / 2          # keep coupling symmetric
        net.W[diagind(net.W)] .= 0
        net.h, s_h, r_h = adam_update(net.h, gh, lr, epoch, s_h, r_h)

        cost_history[epoch] = cost
        if cost < best_cost
            best_cost = cost
            best_W = copy(net.W)
            best_h = copy(net.h)
        end
        if epoch == 1 || epoch % print_every == 0
            println("Epoch $epoch: cost = $(round(cost, sigdigits=4)) (best $(round(best_cost, sigdigits=4)))")
        end
        if cost < 1e-5
            println("Converged at epoch $epoch")
            cost_history = cost_history[1:epoch]
            break
        end
    end

    net.W = best_W                # restore the best checkpoint
    net.h = best_h
    println("Best free-phase cost over training: ", round(best_cost, sigdigits=4))
    return cost_history
end

# ----------------------------------------------------------------------------
# Experiment
# ----------------------------------------------------------------------------

function run_experiment(; N=5, beta=0.1, lr=0.02, N_epoch=3000, seed=1,
                        a=-1.0, c=1.0, delta=1.0, T=40.0)
    rng = MersenneTwister(seed)
    input_index = [1, 2]
    output_index = [N]

    net = DuffingNetwork(N, input_index, output_index; T=T, a=a, c=c, delta=delta)
    random_init!(net; rng=rng)

    # XOR encoded on the +-1 wells (same / differ -> -1 / +1).
    data   = Float64[-1 -1; -1 1; 1 -1; 1 1]
    target = reshape(Float64[-1, 1, 1, -1], 4, 1)

    cost_history = train!(net, data, target, beta, lr, N_epoch; rng=rng)

    # Test with the free dynamics (beta = 0).
    test_out = zeros(4)
    for i in 1:4
        x0 = zeros(1, N)
        x0[1, input_index] .= data[i, :]
        x0[1, net.variable_index] .= 0.1 * randn(rng, length(net.variable_index))
        eq = relax_batch(net, x0, reshape(target[i, :], 1, :), 0.0)
        test_out[i] = eq[1, output_index[1]]
    end

    println("\nTraining complete. Best cost: ", round(minimum(cost_history), sigdigits=4),
            " (weights restored to this checkpoint)")
    for i in 1:4
        println("Input $(Int.(data[i, :])) => target $(Int(target[i,1])), ",
                "output $(round(test_out[i], digits=3)), ",
                "sign-correct: $(sign(test_out[i]) == sign(target[i,1]))")
    end
    return net, cost_history, test_out
end

# ----------------------------------------------------------------------------
# Script entry point (define EP_DUFFING_SKIP_RUN = true before include() to only
# load the definitions).
# ----------------------------------------------------------------------------
if !@isdefined(EP_DUFFING_SKIP_RUN)
    Random.seed!(1)
    @time net, cost_history, test_out = run_experiment()

    using CairoMakie
    FIGDIR = joinpath(@__DIR__, "..", "results", "figures")
    isdir(FIGDIR) || mkpath(FIGDIR)

    fig = Figure(size=(700, 450))
    ax = Axis(fig[1, 1], xlabel="Epoch", ylabel="Cost  ⟨½ Σ(x_out − target)²⟩",
              title="Equilibrium Propagation on a damped Duffing network (XOR)",
              yscale=log10)
    lines!(ax, 1:length(cost_history), max.(cost_history, 1e-12), color=:black)
    save(joinpath(FIGDIR, "ep_duffing_cost_history.png"), fig)

    labels = ["(-1,-1)", "(-1,+1)", "(+1,-1)", "(+1,+1)"]
    tgt = Float64[-1, 1, 1, -1]
    fig2 = Figure(size=(600, 420))
    ax2 = Axis(fig2[1, 1], xlabel="Input pattern", ylabel="Output position x_out",
               xticks=(1:4, labels), title="Trained Duffing outputs vs XOR targets")
    hlines!(ax2, [0.0], color=:gray, linestyle=:dot)
    scatter!(ax2, 1:4, tgt, marker=:hline, markersize=30, color=:black, label="target")
    scatter!(ax2, 1:4, test_out, color=:crimson, markersize=14, label="network output")
    axislegend(ax2, position=:ct)
    save(joinpath(FIGDIR, "ep_duffing_outputs.png"), fig2)

    println("\nFigures saved to ", abspath(FIGDIR))
end
