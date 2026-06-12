# Equilibrium Propagation on an XY (phase-oscillator) network, trained on XOR.
#
# Performance/correctness notes relative to the previous version of this file:
#   * The ODE right-hand side (`xy_force!`) is in-place and allocation-free.
#     The old one built several NxN temporaries per call, and the solver calls
#     it millions of times over a training run.
#   * Relaxations terminate as soon as the network reaches equilibrium
#     (steady-state callback) instead of always integrating the full span, and
#     the solver only stores the final state.
#   * The symmetric (+-beta) gradient estimate is scaled by 1/(2*beta); the
#     cheaper one-sided variant (2 relaxations instead of 3) is available with
#     `symmetric=false`.
#   * Testing after training runs the free dynamics (beta = 0); previously the
#     nudge force was still on during evaluation.

using LinearAlgebra
using Statistics
using OrdinaryDiffEq
using SciMLBase: get_du

mutable struct SP_XY_Network
    N::Int                # Number of neurons
    N_ev::Int             # Maximum number of updates
    dt::Float64           # Time step
    input_index::Vector{Int}    # Indices of input cells
    output_index::Vector{Int}   # Indices of output cells
    variable_index::Vector{Int} # Indices of variable cells
    N_input::Int          # Number of input cells
    N_output::Int         # Number of output cells
    T::Float64            # Total time
    weights::Array{Float64, 2}   # Weight matrix
    weights_0::Array{Float64, 2} # Initial weight matrix
    bias::Array{Float64, 2}      # Bias matrix
    bias_0::Array{Float64, 2}    # Initial bias matrix
    beta::Float64         # Nudge strength
    phase_0::Array{Float64, 1}   # Initial phase
end

function SP_XY_Network(N::Int64, N_ev::Int64, dt::Float64, input_index::Vector{Int64}, output_index::Vector{Int64})
    variable_index = setdiff(1:N, input_index)
    N_input = length(input_index)
    N_output = length(output_index)
    T = dt * N_ev
    weights = zeros(N, N)
    weights_0 = zeros(N, N)
    bias = zeros(2, N)
    bias_0 = zeros(2, N)
    beta = 0.001
    phase_0 = zeros(N)

    return SP_XY_Network(N, N_ev, dt, input_index, output_index, variable_index, N_input, N_output, T, weights, weights_0, bias, bias_0, beta, phase_0)
end

function random_state_initiation!(network::SP_XY_Network)
    # Randomly set weights with smaller initialization
    network.weights_0 = 0.1 * randn(network.N, network.N)  # Reduced scale for stability
    for k in 1:network.N
        network.weights_0[k, k] = 0  # No self-connections
    end
    network.weights_0 = (network.weights_0 + network.weights_0') / 2  # Symmetric weights
    network.weights = network.weights_0

    # Randomly set bias with smaller initialization
    bias = 0.1 * rand(2, network.N)  # Reduced scale for stability
    bias[1, :] .-= 0.05  # Reduced offset
    bias[2, :] .= 2π * (bias[2, :] .- 0.5)
    network.bias_0 = bias
    network.bias = bias

    # Randomly set phase_0 with smaller initialization
    network.phase_0 = 0.5 * π * rand(network.N) .- π / 4  # Reduced scale
end

function get_beta!(network::SP_XY_Network, beta::Float64)
    network.beta = beta
end

# ----------------------------------------------------------------------------
# Dynamics
# ----------------------------------------------------------------------------

# In-place force F = -dE/dphase with the cost term scaled by beta and input
# cells clamped. Allocation-free: this is the hot path of the whole script.
function xy_force!(F, phase, p, t)
    W, h, psi = p.W, p.h, p.psi
    N = length(phase)
    @inbounds for j in 1:N
        pj = phase[j]
        acc = 0.0
        for k in 1:N
            acc += W[k, j] * sin(pj - phase[k])  # W is symmetric; column access
        end
        F[j] = -acc - h[j] * sin(pj - psi[j])
    end
    if p.beta != 0.0
        @inbounds for (m, j) in enumerate(p.output_index)
            d = phase[j] - p.target[m]
            F[j] -= p.beta * sin(d) / (1.0 + cos(d) + 1e-10)
        end
    end
    @inbounds for j in p.input_index
        F[j] = 0.0
    end
    return nothing
end

force_params(W, bias, target, beta, input_index, output_index) =
    (W=W, h=bias[1, :], psi=bias[2, :], target=target, beta=beta,
     input_index=input_index, output_index=output_index)

# Stop a relaxation as soon as the phases stop moving. The tolerance only
# needs to be small relative to the O(beta) nudge displacement.
const STEADY_TOL = 1e-5
steady_state_callback() = DiscreteCallback(
    (u, t, integrator) -> maximum(abs, get_du(integrator)) < STEADY_TOL,
    terminate!; save_positions=(false, false))

# maxiters is a bail-out: the nudge force -sin(d)/(1+cos(d)) is singular at
# d = pi, and a trajectory that lands near it would otherwise stall the solver.
const SOLVER_KWARGS = (reltol=1e-4, abstol=1e-6, maxiters=10^5,
                       save_everystep=false, save_start=false, verbose=false)

# Relax a single network state to equilibrium.
function run_network(network::SP_XY_Network, phase_0, target; beta=network.beta)
    p = force_params(network.weights, network.bias, target, beta,
                     network.input_index, network.output_index)
    prob = ODEProblem(xy_force!, collect(Float64, phase_0), (0.0, network.T), p)
    sol = solve(prob, Tsit5(); callback=steady_state_callback(), SOLVER_KWARGS...)
    return sol.u[end]
end

# Relax a batch of states (one ODE trajectory per sample, threaded).
function run_network_batch(phase_0_batch, T, W, bias, target_batch, beta, input_index, output_index)
    N_batch, N = size(phase_0_batch)
    p = force_params(W, bias, view(target_batch, 1, :), beta, input_index, output_index)
    prob = ODEProblem(xy_force!, phase_0_batch[1, :], (0.0, T), p)

    prob_func(prob, i, repeat) =
        remake(prob; u0=phase_0_batch[i, :], p=merge(p, (target=view(target_batch, i, :),)))

    ensemble = EnsembleProblem(prob; prob_func=prob_func)
    ensemble_alg = Threads.nthreads() > 1 ? EnsembleThreads() : EnsembleSerial()
    sol = solve(ensemble, Tsit5(), ensemble_alg; trajectories=N_batch,
                callback=steady_state_callback(), SOLVER_KWARGS...)

    equilibria = zeros(N_batch, N)
    for i in 1:N_batch
        equilibria[i, :] = sol[i].u[end]
    end
    return equilibria
end

# ----------------------------------------------------------------------------
# Cost and EP gradients
# ----------------------------------------------------------------------------

function batch_costs(equilibria, target_batch, output_index; tol=0.1)
    deviation = 1.0 .- cos.(equilibria[:, output_index] .- target_batch)
    cost = mean(vec(sum(deviation, dims=2)) ./ 2)
    q_cost = mean(vec(sum(deviation .> tol, dims=2)) ./ 2)
    return cost, q_cost
end

function weights_gradient(equi_nudge, equi_free)
    N_data, N = size(equi_free)
    gradient = zeros(N, N)
    @inbounds for i in 1:N_data, j in 1:N, k in 1:N
        nudge_diff = equi_nudge[i, j] - equi_nudge[i, k]
        free_diff = equi_free[i, j] - equi_free[i, k]
        gradient[j, k] += (cos(free_diff) - cos(nudge_diff)) / N_data
    end
    return gradient
end

function bias_gradient(equi_nudge, equi_free, bias)
    N_data, N = size(equi_free)
    h = bias[1, :]
    psi = bias[2, :]
    g = zeros(2, N)
    @inbounds for i in 1:N_data, j in 1:N
        g[1, j] += (cos(equi_free[i, j] - psi[j]) - cos(equi_nudge[i, j] - psi[j])) / N_data
        g[2, j] += h[j] * (sin(equi_free[i, j] - psi[j]) - sin(equi_nudge[i, j] - psi[j])) / N_data
    end
    return g
end

paras_gradient(equi_nudge, equi_free, bias) =
    (weights_gradient(equi_nudge, equi_free), bias_gradient(equi_nudge, equi_free, bias))

# EP gradient: free relaxation from phase_0, then nudged relaxation(s) started
# from the free equilibrium. `symmetric=true` uses the centered (+-beta)
# estimate (3 relaxations); `symmetric=false` uses the one-sided estimate
# (2 relaxations, ~1/3 faster but a noisier gradient).
function EP_param_gradient(W, bias, phase_0, target_batch, beta,
                           N_ev, dt, input_index, variable_index, output_index;
                           symmetric=true)
    T = N_ev * dt

    equi_zero = run_network_batch(phase_0, T, W, bias, target_batch, 0.0, input_index, output_index)
    equi_nudge = run_network_batch(equi_zero, T, W, bias, target_batch, beta, input_index, output_index)
    if symmetric
        equi_free = run_network_batch(equi_zero, T, W, bias, target_batch, -beta, input_index, output_index)
        scale = 2 * beta
    else
        equi_free = equi_zero
        scale = beta
    end

    cost, q_cost = batch_costs(equi_zero, target_batch, output_index)
    gW, gh = paras_gradient(equi_nudge, equi_free, bias)
    return gW ./ scale, gh ./ scale, cost, q_cost
end

# ----------------------------------------------------------------------------
# Optimizers
# ----------------------------------------------------------------------------

function gradient_descent_update(paras, g_paras, study_rate_0, itr_time; study_rate_f=0.0001, decay=0.0001)
    eta = max(study_rate_0 * exp(-decay * itr_time), study_rate_f)
    g_paras_clipped = clamp.(g_paras, -1.0, 1.0)  # Clip to prevent exploding gradients
    return paras .- eta .* g_paras_clipped
end

function Adam_update(param, grad, study_rate, epoch, s, r, beta1=0.9, beta2=0.999, epsilon=1e-8)
    grad_clipped = clamp.(grad, -1.0, 1.0)

    s = beta1 * s .+ (1 - beta1) * grad_clipped
    r = beta2 * r .+ (1 - beta2) * (grad_clipped .^ 2)

    s_hat = s ./ (1 - beta1^epoch)
    r_hat = r ./ (1 - beta2^epoch)

    param = param .- study_rate * s_hat ./ (sqrt.(r_hat) .+ epsilon)
    return param, s, r
end

# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------

function train_network(W_0, bias_0, training_data, training_target, beta, study_rate,
                       N_epoch, batch_size, N_ev, dt, input_index, variable_index, output_index;
                       use_adam=true, symmetric=true, patience=N_epoch, print_every=100)
    N = size(W_0, 1)
    N_data = size(training_data, 1)

    W = copy(W_0)
    bias = copy(bias_0)

    cost_history = zeros(N_epoch)
    best_cost = Inf
    best_W = copy(W)
    best_bias = copy(bias)
    patience_counter = 0
    epochs_run = N_epoch

    # Adam state
    s_W = zeros(size(W))
    r_W = zeros(size(W))
    s_bias = zeros(size(bias))
    r_bias = zeros(size(bias))

    for epoch in 1:N_epoch
        if batch_size == 0 || batch_size > N_data
            batch_indices = [1:N_data]
        else
            num_batches = ceil(Int, N_data / batch_size)
            batch_indices = [((i-1)*batch_size+1):min(i*batch_size, N_data) for i in 1:num_batches]
        end

        epoch_cost = 0.0

        for batch_idx in batch_indices
            batch_data = training_data[batch_idx, :]
            batch_target = training_target[batch_idx, :]

            # Inputs clamped to the data, small random noise on the free cells
            phase_0 = zeros(length(batch_idx), N)
            phase_0[:, input_index] .= batch_data
            phase_0[:, variable_index] .= 0.1 * randn(length(batch_idx), length(variable_index))

            gW, gbias, batch_cost, q_cost = EP_param_gradient(
                W, bias, phase_0, batch_target, beta,
                N_ev, dt, input_index, variable_index, output_index;
                symmetric=symmetric
            )

            if use_adam
                W, s_W, r_W = Adam_update(W, gW, study_rate, epoch, s_W, r_W)
                bias, s_bias, r_bias = Adam_update(bias, gbias, study_rate, epoch, s_bias, r_bias)
            else
                W = gradient_descent_update(W, gW, study_rate, epoch)
                bias = gradient_descent_update(bias, gbias, study_rate, epoch)
            end

            epoch_cost += batch_cost * length(batch_idx)
        end

        epoch_cost /= N_data
        cost_history[epoch] = epoch_cost

        if epoch == 1 || epoch % print_every == 0
            println("Epoch $epoch: Cost = $epoch_cost")
        end

        if epoch_cost < best_cost
            best_cost = epoch_cost
            best_W = copy(W)
            best_bias = copy(bias)
            patience_counter = 0
        else
            patience_counter += 1
            if patience_counter >= patience
                println("Early stopping at epoch $epoch")
                epochs_run = epoch
                break
            end
        end

        if epoch_cost < 1e-6
            println("Converged at epoch $epoch")
            epochs_run = epoch
            break
        end
    end

    return best_W, best_bias, cost_history[1:epochs_run]
end

# ----------------------------------------------------------------------------
# Experiment
# ----------------------------------------------------------------------------

function run_experiment(; N=5, N_ev=200, dt=0.1, beta=0.01,
                        N_epoch=20000, study_rate=0.05, batch_size=4)
    input_index = [1, 2]
    output_index = [N]

    network = SP_XY_Network(N, N_ev, dt, input_index, output_index)
    get_beta!(network, beta)

    # XOR problem
    training_data = (π / 2) * [-1 -1; -1 1; 1 -1; 1 1]
    training_target = (π / 2) * [-1; 1; 1; -1]

    random_state_initiation!(network)
    network.weights_0 *= 0.1
    network.bias_0 *= 0.1

    variable_index = network.variable_index
    W_final, bias_final, cost_history = train_network(
        network.weights_0, network.bias_0,
        training_data, training_target,
        beta, study_rate, N_epoch, batch_size,
        N_ev, dt, input_index, variable_index, output_index
    )

    network.weights = W_final
    network.bias = bias_final

    # Test on the training data with the free dynamics (beta = 0)
    test_results = zeros(size(training_data, 1))
    for i in 1:size(training_data, 1)
        phase_0 = zeros(network.N)
        phase_0[input_index] .= training_data[i, :]

        final_phase = run_network(network, phase_0, training_target[i]; beta=0.0)
        test_results[i] = final_phase[output_index[1]]
    end

    println("Training complete!")
    println("Final cost: ", cost_history[end])
    println("Test results:")
    for i in 1:size(training_data, 1)
        input_str = join(training_data[i, :] ./ (π/2), ", ")
        target = training_target[i] ./ (π/2)
        result = test_results[i] ./ (π/2)
        error = abs(result - target)
        println("Input: [$input_str] => Target: $target, Result: $result, Error: $error")
    end

    return network, cost_history, test_results
end

# ----------------------------------------------------------------------------
# Script entry point
# (define EP_XY_SKIP_RUN = true before include() to only load the definitions)
# ----------------------------------------------------------------------------
if !@isdefined(EP_XY_SKIP_RUN)
    @time network, cost_history, test_results = run_experiment()

    using CairoMakie

    fig = Figure(size=(700, 450))
    ax = Axis(fig[1, 1], xlabel="Epoch", ylabel="Distance",
              title="Training Distance History", yscale=log10)
    lines!(ax, 1:length(cost_history), max.(cost_history, 1e-12),
           color=:black, linewidth=0.5)
    save("./cost_history.pdf", fig)

    fig2 = Figure(size=(600, 420))
    ax2 = Axis(fig2[1, 1], xlabel="Sample", ylabel="Phase", title="Test Results")
    scatterlines!(ax2, 1:length(test_results), test_results)
    save("./test_results.pdf", fig2)

    println(network.weights)
    println(network.bias)
    println(network.phase_0)
end
