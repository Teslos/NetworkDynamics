mutable struct SP_XY_Network  
    N::Int                # Number of neurons  
    N_ev::Int             # Maximum number of updates  
    dt::Float64           # Time step  
    input_index::Vector{Int}  # Indices of input cells  
    output_index::Vector{Int} # Indices of output cells  
    variable_index::Vector{Int} # Indices of variable cells  
    N_input::Int          # Number of input cells  
    N_output::Int         # Number of output cells  
    T::Float64            # Total time  
    weights::Array{Float64, 2} # Weight matrix  
    weights_0::Array{Float64, 2} # Initial weight matrix  
    bias::Array{Float64, 2} # Bias matrix  
    bias_0::Array{Float64, 2} # Initial bias matrix  
    beta::Float64         # Regularization parameter  
    phase_0::Array{Float64, 1} # Initial phase
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
    beta = 0.001  # Reduced beta for better stability
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

function internal_energy(W::Array{Float64, 2}, phase::Array{Float64, 2})  
    dphase = phase .- phase'  
    E_list = 0.5 * sum(W .* cos.(dphase), dims=(1, 2))  
    return E_list  
end  

function bias_term(bias::Array{Float64, 2}, phase::Array{Float64, 2})  
    h = bias[1, :]  
    psi = bias[2, :]  
    N_data = size(phase, 1)  
    psi_mat = repeat(psi', N_data, 1)  
    E_list = sum(h .* cos.(phase .- psi_mat), dims=2)  
    return E_list  
end  

function cost_function(phase::Array{Float64, 2}, target::Array{Float64, 2}, output_index::Vector{Int})  
    doutput = phase[:, output_index] .- target  
    cost_mat = ones(size(doutput)) .- cos.(doutput)  
    cost_list = sum(cost_mat, dims=2)  
    return cost_list  
end 

using OrdinaryDiffEq  

# Unified force calculation function to replace multiple inconsistent versions
function calculate_force(phase, W, bias, target, beta, input_index, output_index)
    # Calculate internal force (from weights)
    N = size(phase, 1)
    dphase = phase .- phase'
    F0 = sum(W .* sin.(dphase), dims=2)
    
    # Calculate bias force
    h = bias[1, :]
    psi = bias[2, :]
    F1 = -h .* sin.(phase .- psi)
    
    # Calculate cost force
    F3 = zeros(size(phase))
    output_phase = phase[output_index]
    M1 = -sin.(output_phase .- target)
    M2 = ones(size(output_phase)) .+ cos.(output_phase .- target)
    F3[output_index] .= M1 ./ (M2 .+ 1e-10) # Avoid division by zero
    
    # Total force
    F = -F0 + F1 + beta .* F3
    
    # Zero force at input indices
    F[input_index] .= 0
    
    return vec(F)
end

function total_force(t, con_phase, network::SP_XY_Network, target)  
    phase = reshape(con_phase, (network.N,))
    F = calculate_force(phase, network.weights, network.bias, target, network.beta, network.input_index, network.output_index)
    return F
end  

function run_network(network::SP_XY_Network, phase_0, target)  
    tspan = (0.0, network.T)  
    prob = ODEProblem((y,p,t) -> total_force(t, y, network, target), phase_0, tspan)  
    sol = solve(prob, Tsit5(), reltol=1e-10, abstol=1e-10, maxiters=10^8)  # Improved solver parameters
    return sol[end]
end  

using LinearAlgebra  
using TensorOperations

# Improved phase difference calculation
function cal_dphase(phase)  
    N_data = size(phase, 1)
    N = size(phase, 2)
    
    # More direct and stable approach
    dphase = zeros(N_data, N, N)
    for i in 1:N_data
        for j in 1:N
            for k in 1:N
                dphase[i, j, k] = phase[i, j] - phase[i, k]
            end
        end
    end
    
    return dphase
end 

function run_network_batch(phase_0_batch, T, W, bias, target_batch, beta, input_index, output_index)
    N_batch = size(phase_0_batch, 1)

    # Define a force function expecting parameters as a NamedTuple or tuple.
    function force_fn(phase, p, t)
        return calculate_force(
            phase,
            p.W,
            p.bias,
            p.target,
            p.beta,
            p.input_index,
            p.output_index
        )
    end

    # Create a dummy initial condition and dummy parameter
    dummy_phase0 = phase_0_batch[1, :]
    dummy_param = (
        W = W,
        bias = bias,
        target = target_batch[1, :],
        beta = beta,
        input_index = input_index,
        output_index = output_index
    )

    # Define a base problem (will be updated for each trajectory)
    tspan = (0.0, T)
    prob = ODEProblem(force_fn, dummy_phase0, tspan, dummy_param)

    # Define the prob_func to set initial conditions & targets for each trajectory
    function prob_func(prob, i, repeat)
        new_param = (
            W = W,
            bias = bias,
            target = target_batch[i, :],
            beta = beta,
            input_index = input_index,
            output_index = output_index
        )
        return remake(prob; u0 = phase_0_batch[i, :], p = new_param)
    end

    # Create an EnsembleProblem for parallel solving
    ensemble_prob = EnsembleProblem(prob; prob_func = prob_func)

    # Solve with multiple trajectories (one per initial condition)
    ensemble_sol = solve(
        ensemble_prob,
        Tsit5(),
        EnsembleThreads();
        trajectories = N_batch
    )

    # Collect the final states into a matrix
    N = length(dummy_phase0)
    result_matrix = zeros(N_batch, N)
    for i in 1:N_batch
        result_matrix[i, :] = ensemble_sol[i](ensemble_sol[i].t[end])
    end

    return result_matrix
end



# Improved run network function with better batch handling
function run_network_batch_serial(phase_0_batch, T, W, bias, target_batch, beta, input_index, output_index)
    N_batch = size(phase_0_batch, 1)
    results = Array{Any}(undef, N_batch)
    
    for i in 1:N_batch
        phase_0 = phase_0_batch[i, :]
        target = target_batch[i, :]
        
        # Define ODE problem
        function force_fn(phase, p, t)
            return calculate_force(phase, W, bias, target, beta, input_index, output_index)
        end
        
        tspan = (0.0, T)
        prob = ODEProblem(force_fn, phase_0, tspan)
        sol = solve(prob, Tsit5(), reltol=1e-2, abstol=1e-2, maxiters=10^5)
        
        results[i] = sol[end]
    end
    
    # Convert results to matrix
    N = length(results[1])
    result_matrix = zeros(N_batch, N)
    for i in 1:N_batch
        result_matrix[i, :] = results[i]
    end
    
    return result_matrix
end

# Improved weights gradient calculation
function weights_gradient(equi_nudge, equi_free, beta)  
    N_data = size(equi_free, 1)
    N = size(equi_free, 2)
    
    gradient = zeros(N, N)
    
    for i in 1:N_data
        for j in 1:N
            for k in 1:N
                nudge_diff = equi_nudge[i, j] - equi_nudge[i, k]
                free_diff = equi_free[i, j] - equi_free[i, k]
                gradient[j, k] += (-cos(nudge_diff) + cos(free_diff)) / N_data
            end
        end
    end
    
    return gradient
end  

# Improved bias gradient calculation
function bias_gradient(equi_nudge, equi_free, bias, beta)  
    N_data = size(equi_free, 1)
    N = size(equi_free, 2)
    
    h = bias[1, :]
    psi = bias[2, :]
    
    g_h = zeros(1, N)
    g_psi = zeros(1, N)
    
    for i in 1:N_data
        for j in 1:N
            g_h[j] += (cos(equi_free[i, j] - psi[j]) - cos(equi_nudge[i, j] - psi[j])) / N_data
            g_psi[j] += h[j] * (sin(equi_free[i, j] - psi[j]) - sin(equi_nudge[i, j] - psi[j])) / N_data
        end
    end
    
    return [g_h; g_psi]
end  

# Combined gradient calculation
function paras_gradient(equi_nudge, equi_free, bias, beta)  
    gW = weights_gradient(equi_nudge, equi_free, beta)  
    gh = bias_gradient(equi_nudge, equi_free, bias, beta)  
    return gW, gh  
end  

using Statistics
# Improved EP parameter gradient calculation
function EP_param_gradient(W_0, bias_0, phase_0, training_target, beta, N_ev, dt, input_index, variable_index, output_index)  
    N_data = size(training_target, 1)  
    N = size(W_0, 2)  
    T = N_ev * dt  

    # Prepare batch training target
    batch_size = size(phase_0, 1)  
    batch_training_target = zeros(batch_size, size(training_target, 2))
    for i in 1:batch_size
        batch_training_target[i, :] = training_target[i, :]
    end

    # Run network for equilibrium states with improved parameters
    equi_zero = run_network_batch(phase_0, T, W_0, bias_0, batch_training_target, 0.0, input_index, output_index)  
    equi_nudge = run_network_batch(equi_zero, T, W_0, bias_0, batch_training_target, beta, input_index, output_index)  
    equi_free = run_network_batch(equi_zero, T, W_0, bias_0, batch_training_target, -beta, input_index, output_index)

    # Calculate costs
    cost_list = zeros(batch_size)
    for i in 1:batch_size
        output_phase = equi_zero[i, output_index]
        target = batch_training_target[i, :]
        doutput = output_phase - target
        cost_list[i] = sum(1.0 .- cos.(doutput)) / 2
    end
    cost = mean(cost_list)

    # Calculate qualitative cost (distance within tolerance)
    tol = 0.1
    q_cost_list = zeros(batch_size)
    for i in 1:batch_size
        output_phase = equi_zero[i, output_index]
        target = batch_training_target[i, :]
        doutput = output_phase - target
        q_cost_list[i] = sum((1.0 .- cos.(doutput)) .> tol) / 2
    end
    q_cost = mean(q_cost_list)

    # Calculate gradients
    gW, gh = paras_gradient(equi_nudge, equi_free, bias_0, beta)  

    # Return gradients and costs
    return gW / beta, gh / beta, cost, q_cost  
end  

# Improved gradient descent with adaptive learning rate
function gradient_descent_update(paras, g_paras, study_rate_0, itr_time; study_rate_f=0.0001, decay=0.0001)  
    # Compute the learning rate with decay
    eta = max(study_rate_0 * exp(-decay * itr_time), study_rate_f)
    
    # Clip gradients to prevent exploding gradients
    max_grad = 1.0
    g_paras_clipped = clamp.(g_paras, -max_grad, max_grad)
    
    # Perform the gradient descent update
    return paras .- eta .* g_paras_clipped
end  

# Improved Adam optimizer
function Adam_update(param, grad, study_rate, epoch, s, r, beta1=0.9, beta2=0.999, epsilon=1e-8)  
    # Clip gradients
    max_grad = 1.0
    grad_clipped = clamp.(grad, -max_grad, max_grad)
    
    # Update momentum and RMS
    s = beta1 * s .+ (1 - beta1) * grad_clipped  
    r = beta2 * r .+ (1 - beta2) * (grad_clipped .^ 2)  
    
    # Bias correction
    s_hat = s ./ (1 - beta1^epoch)  
    r_hat = r ./ (1 - beta2^epoch)  
    
    # Update parameters
    param = param .- study_rate * s_hat ./ (sqrt.(r_hat) .+ epsilon)  
    
    return param, s, r  
end 

# Improved training function with early stopping
function train_network(W_0, bias_0, training_data, training_target, beta, study_rate, N_epoch, batch_size, N_ev, dt, input_index, variable_index, output_index)  
    N = size(W_0, 1)
    N_data = size(training_data, 1)
    T = N_ev * dt
    
    # Initialize weights and biases
    W = copy(W_0)
    bias = copy(bias_0)
    
    # Storage for training history
    cost_history = zeros(N_epoch)
    best_cost = Inf
    best_W = copy(W)
    best_bias = copy(bias)
    patience = 20000  # Early stopping patience
    patience_counter = 0
    
    # Prepare for Adam optimizer
    s_W = zeros(size(W))
    r_W = zeros(size(W))
    s_bias = zeros(size(bias))
    r_bias = zeros(size(bias))
    
    # Use Adam optimizer
    use_adam = true
    
    # Training loop
    for epoch in 1:N_epoch
        # Create mini-batches
        if batch_size == 0 || batch_size > N_data
            batch_indices = [1:N_data]
        else
            num_batches = ceil(Int, N_data / batch_size)
            batch_indices = [((i-1)*batch_size+1):min(i*batch_size, N_data) for i in 1:num_batches]
        end
        
        epoch_cost = 0.0
        
        # Process each mini-batch
        for batch_idx in batch_indices
            # Prepare batch data
            batch_data = training_data[batch_idx, :]
            batch_target = training_target[batch_idx, :]
            
            # Initialize phase
            phase_0 = zeros(length(batch_idx), N)
            phase_0[:, input_index] .= batch_data
            
            # Add small random noise to non-input neurons for better exploration
            phase_0[:, variable_index] .= 0.1 * randn(length(batch_idx), length(variable_index))
            
            # Calculate gradients and cost
            gW, gbias, batch_cost, q_cost = EP_param_gradient(
                W, bias, phase_0, batch_target, beta, 
                N_ev, dt, input_index, variable_index, output_index
            )
            
            # Update parameters
            if use_adam
                # Adam updates
                W, s_W, r_W = Adam_update(W, gW, study_rate, epoch, s_W, r_W)
                bias, s_bias, r_bias = Adam_update(bias, gbias, study_rate, epoch, s_bias, r_bias)
            else
                # Standard gradient descent
                W = gradient_descent_update(W, gW, study_rate, epoch)
                bias = gradient_descent_update(bias, gbias, study_rate, epoch)
            end
            
            epoch_cost += batch_cost * length(batch_idx)
        end
        
        # Calculate average cost for the epoch
        epoch_cost /= N_data
        cost_history[epoch] = epoch_cost
        
        # Print progress
        if epoch % 10 == 0
            println("Epoch $epoch: Cost = $epoch_cost")
        end
        
        # Early stopping check
        if epoch_cost < best_cost
            best_cost = epoch_cost
            best_W = copy(W)
            best_bias = copy(bias)
            patience_counter = 0
        else
            patience_counter += 1
            if patience_counter >= patience
                println("Early stopping at epoch $epoch")
                break
            end
        end
        
        # Check for convergence
        if epoch_cost < 1e-6
            println("Converged at epoch $epoch")
            break
        end
    end
    
    return best_W, best_bias, cost_history
end

# Main execution
function run_experiment()
    # Define parameters with improved values
    N = 5 # Increased network size
    N_ev = 20 # Increased number of time steps
    dt = 0.1
    input_index = [1, 2]
    output_index = [N]
    
    # Create network
    network = SP_XY_Network(N, N_ev, dt, input_index, output_index)
    
    # Set beta to a smaller value for better stability
    beta = 0.01
    get_beta!(network, beta)
    
    # Define training parameters
    N_epoch = 20000  # Increased epochs
    study_rate = 0.05  # Reduced learning rate
    batch_size = 4  # Full batch for XOR
    
    # Define training data (XOR problem)
    training_data = (π / 2) * [-1 -1; -1 1; 1 -1; 1 1]
    training_target = (π / 2) * [-1; 1; 1; -1]
    
    # Initialize weights with smaller values
    random_state_initiation!(network)
    
    # Scale down initial weights further
    network.weights_0 *= 0.1
    network.bias_0 *= 0.1
    
    # Train network
    variable_index = network.variable_index
    W_final, bias_final, cost_history = train_network(
        network.weights_0, network.bias_0, 
        training_data, training_target, 
        beta, study_rate, N_epoch, batch_size,
        N_ev, dt, input_index, variable_index, output_index
    )
    
    # Update network with final weights
    network.weights = W_final
    network.bias = bias_final
    
    # Test network on training data
    test_results = zeros(size(training_data, 1))
    for i in 1:size(training_data, 1)
        phase_0 = zeros(network.N)
        phase_0[input_index] .= training_data[i, :]
        
        final_phase = run_network(network, phase_0, training_target[i])
        test_results[i] = final_phase[output_index[1]]
    end
    
    # Print results
    println("Training complete!")
    println("Final costs: ", cost_history[cost_history .!= 0])
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

# Run the experiment
@time network, cost_history, test_results = run_experiment()

# Plot cost history
using Plots
plot(cost_history, 
     xlabel="Epoch", 
     ylabel="Distance", 
     title="Training Distance History", 
     legend=false, 
     lw=0.5)
savefig("./cost_history.pdf")
# Plot test results
plot(test_results, 
     xlabel="Sample", 
     ylabel="Phase", 
     title="Test Results", 
     legend=false, 
     lw=1)
print(network.weights)
print(network.bias)
print(network.phase_0)