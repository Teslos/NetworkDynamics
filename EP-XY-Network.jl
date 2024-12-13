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
    beta = 0.001  
    phase_0 = zeros(N)

    return SP_XY_Network(N, N_ev, dt, input_index, output_index, variable_index, N_input, N_output, T, weights, weights_0, bias, bias_0, beta, phase_0)  
end 

 
function random_state_initiation!(network::SP_XY_Network)  
    # Randomly set weights  
    network.weights_0 = randn(network.N, network.N)  
    for k in 1:network.N  
        network.weights_0[k, k] = 0  
    end  
    network.weights_0 = (network.weights_0 + network.weights_0') / 2  
    network.weights = network.weights_0  

    # Randomly set bias  
    bias = rand(2, network.N)  
    bias[1, :] .-= 0.5  
    bias[2, :] .= 2π * (bias[2, :] .- 0.5)  
    network.bias_0 = bias  
    network.bias = bias  

    # Randomly set phase_0  
    network.phase_0 = π * rand(network.N) .- π / 2  
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

function total_force(t, con_phase, network::SP_XY_Network, target)  
    Nh = div(length(con_phase), network.N)  
    phase = reshape(con_phase, (Nh, network.N))  

    F0 = internal_force(network.weights, phase)  
    F1 = bias_force(network.bias, phase)  
    F2 = network.beta * cost_force(phase, target, network.output_index)  

    F = -F0 + F1 + F2  
    F[:, network.input_index] .= 0  

    return vec(F)  
end  

function run_network(network::SP_XY_Network, phase_0, target)  
    con_phase = vec(phase_0)  
    tspan = (0.0, network.T)  
    prob = ODEProblem((t, y) -> total_force(t, y, network, target), con_phase, tspan)  
    sol = solve(prob, Tsit5())  
    return reshape(sol[end], size(phase_0))  
end  

# CPU version of the training function probably better to put in different file
using LinearAlgebra  
#using DifferentialEquations  
using TensorOperations

# Calculate dphase[i,j] = phase[i] - phase[j]  

function sp_cal_dphase(phase)  
    # Calculate dphase[i, j] = phase[i] - phase[j]  

    N = size(phase, 2)  # Number of columns in the input array  
    aux_ones = ones(N)  # Create a vector of ones of length N  
    
    @tensor phase_mat[a1,a2,a3] := aux_ones[a1]*phase[a2,a3]
    @tensor phase_i[a1,a2,a3] := phase_mat[a2,a3,a1]
    @tensor phase_j[a1,a2,a3] := phase_i[a2,a1,a3]  
    dphase  = phase_i .- phase_j

    return dphase  
end   

function cal_dphase(phase)  
    # Calculate dphase[i, j] = phase[i] - phase[j]  

    N = size(phase, 2)  # Number of columns in the input array  
    aux_ones = ones(N)  
    
    @tensor phase_mat[a1,a2,a3] := aux_ones[a1]*phase[a2,a3]
    @tensor phase_i[a1,a2,a3] := phase_mat[a2,a3,a1]
    @tensor phase_j[a1,a2,a3] := phase_i[a2,a1,a3]  
    dphase  = phase_i .- phase_j

    return dphase  
end 

# Total force function  
function sp_total_force(t, con_phase, W, bias, target, beta, input_index, output_index)  
    N = size(W, 1)  
    Nh = div(length(con_phase), N)  
    phase = reshape(con_phase, (Nh, N))  

    dphase = sp_cal_dphase(phase)  
    F0 = sum(W .* sin.(dphase), dims=3)  

    h = bias[1, :]  
    psi = bias[2, :]  
    N_data = size(phase, 1)  
    #psi = ones(N_data) * psi'
    @tensor psi_mat[a1,a2,a3] := ones(a1)*psi[a2,a3]  
    print("Size phase: $(size(phase))")
    print("Size target: $(size(target))")
    output_phase = reshape(phase[:, output_index], size(target))  

    F1 = -h .* sin.(phase .- psi)  

    F2 = zeros(size(phase))  
    temp_F2 = -sin.(output_phase .- target)  
    F2[:, output_index] .= temp_F2  

    F3 = zeros(size(phase))  
    M1 = -sin.(output_phase .- target)  
    M2 = 1.0 .+ cos.(output_phase .- target)  
    F3[:, output_index] .= M1 ./ M2  

    F = -F0 + F1 + 0 * beta * F2 + beta * F3  
    F[:, input_index] .= 0  

    return vec(F)  
end  

# Wrapper for ODE solver  
function sp_ode_total_force(con_phase, t, W, bias, target, beta, input_index, output_index)  
    sp_total_force(t, con_phase, W, bias, target, beta, input_index, output_index)  
end  

# Run the network
function sp_run_network(phase_0, T, W, bias, target, beta, input_index, output_index)  
    # Set initial conditions  
    print(size(phase_0))
    N_data, N = size(phase_0)  
    con_phase_0 = vec(phase_0)  

    # Time span  
    t_span = (0.0, T)  

    # Solve ODE  
    prob = ODEProblem((u,p,t) -> sp_ode_total_force(u, t, W, bias, target, beta, input_index, output_index),  
                      con_phase_0, t_span)  
    solution = solve(prob, Tsit5(), reltol=1.4e-8, abstol=1.4e-8)  

    # Extract final phase  
    con_phase = solution.u[end]  
    phase = reshape(con_phase, (N_data, N))  

    return phase  
end  

# Calculate the gradient using EP method
# Weights gradient  
function sp_weights_gradient(equi_nudge, equi_free, beta)  
    N_data = size(equi_free, 1)  
    nudge_dphase = sp_cal_dphase(equi_nudge)  
    free_dphase = sp_cal_dphase(equi_free)  
    gradient_list = -cos.(nudge_dphase) .+ cos.(free_dphase)  
    gradient = mean(gradient_list, dims=1)  
    return gradient  
end  

# Bias gradient  
function sp_bias_gradient(equi_nudge, equi_free, bias, beta)  
    N_sample = size(equi_free, 1)  
    h = bias[1, :]  
    psi = bias[2, :]  

    h = ones(N_sample) * h'  
    psi = ones(N_sample) * psi'  

    g_h = cos.(equi_free .- psi) .- cos.(equi_nudge .- psi)  
    g_psi = h .* sin.(equi_free .- psi) .- h .* sin.(equi_nudge .- psi)  

    g_h = mean(g_h, dims=1)  
    g_psi = mean(g_psi, dims=1)  

    return [g_h; g_psi]  
end  

# Combined gradient  
function sp_paras_gradient(equi_nudge, equi_free, bias, beta)  
    gW = sp_weights_gradient(equi_nudge, equi_free, beta)  
    gh = sp_bias_gradient(equi_nudge, equi_free, bias, beta)  
    return gW, gh  
end  

# training function
function sp_train_network(W_0, bias_0, training_data, training_target, training_paras, model_paras;
    ext_init_phase_0=0, random_flag=false)  
    # Unpack parameters  
    N_epoch, beta, study_rate = training_paras  
    N, N_ev, dt, input_index, output_index = model_paras  
    N_data = size(training_data, 1)  
    T = N_ev * dt  

    W = W_0  
    bias = bias_0  

    cost = zeros(N_epoch)  

    # Training loop  
    for k in 1:N_epoch  
        phase_0 = zeros(N_data, N) .+ ext_init_phase_0 .+ random_flag * 2π * (rand(N_data, N) .- 0.5)  
        #println("Size of phase_0: $(size(phase_0))")
        #println("Size of training_data: $(size(training_data))")
        #println("Input index: $(input_index)")
        phase_0[:, input_index] .= training_data  

        # Calculate cost function  
        equi_zero = sp_run_network(phase_0, T, W, bias, training_target, 0, input_index, output_index)  
        cost_list = cost_function(equi_zero, training_target, output_index)  
        cost[k] = sum(cost_list) / N_data  

        # Run free and nudge phases  
        equi_free = sp_run_network(phase_0, T, W, bias, training_target, -beta, input_index, output_index)  
        equi_nudge = sp_run_network(phase_0, T, W, bias, training_target, beta, input_index, output_index)  

        # Calculate gradients  
        gW, gh = sp_paras_gradient(equi_nudge, equi_zero, bias, beta)  

        # Update weights and biases  
        W .-= study_rate * gW / (2 * beta)  
        bias .-= study_rate * gh / (2 * beta)  
    end  

    return cost, W, bias  
end  

function single_force(t, phase, W, bias, target, input_index, output_index, beta)  
    # This calculates the gradient of the energy (-F_total)  

    # Convert phase to a 2D array for matrix operations  
    #    phase_mat = reshape(phase, :, 1)  # Equivalent to jnp.asarray([phase])  
    phase_mat = phase
    dphase = -phase_mat .+ phase_mat'  # Compute pairwise differences  

    output_phase = phase[output_index]  # Extract output phase  

    h = bias[1, :]  # First row of bias  
    psi = bias[2, :]  # Second row of bias  
    N_cell = length(phase)  # Number of cells  
    N_temp = N_cell - length(output_index)  # Temporary variable (not used further) 
    
    #println("Size of dphase: $(size(dphase))")
    #println("Size of W: $(size(W))")

    # Compute F0  
    F0 = sum(W .* sin.(dphase), dims=2)  # Sum along rows (axis=1 in Python)  

    #println("Size of h: $(size(h))")
    #println("Size of psi: $(size(psi))")
    #println("Size of phase: $(size(phase))")
    #println("Size of h: $(size(h))")
    # Compute F1  
    F1 = -h .* sin.(phase - psi)  

    # Compute F3  
    F3 = zeros(size(phase))  # Initialize F3 with zeros  
    M1 = -sin.(output_phase .- target)  
    M2 = ones(size(output_phase)) .+ cos.(output_phase .- target)  
    F3[output_index] .= M1 ./ M2  # Update F3 at output_index  
    # Compute total force  
    F = -F0 + F1 + beta .* F3   

    # Set force to 0 at input_index  
    F[input_index] .= 0  
    F = vec(F)  # Convert F to a 1D array
    return F  # Return the force as a column vector  
end  


function single_run_network(phase_0, T, W, bias, target, beta, input_index, output_index)  
    # Define the ODE function  
    function odefunc!(phase, p, t)  
        single_force(t, phase, W, bias, target, input_index, output_index, beta)  
    end  

    # Set the time span  
    t_span = (0.0, T)  

    # Define the ODE problem  
    prob = ODEProblem(odefunc!, phase_0, t_span)  

    # Solve the ODE using a 4th-order Runge-Kutta method (Tsit5)  
    solution = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8, maxiters=10^7)  

    # Extract the final phase  
    phase = solution[:, end]  

    return phase  
end  

function train_US_model(model, N_epoch, batch_size, study_rate, training_data, training_target)  
    # Initialize variables  
    from_zero = false  
    if batch_size == 0  
        batch_size = 1  
        from_zero = true  
    end  
  
    N_data = size(training_data, 1)  
    batch_form = zeros(batch_size)  
    phase_form = zeros(N_data, model.N)  
  
    training_paras = (model.beta, study_rate)  
    model_paras = (model.N_ev, model.dt, model.input_index, model.variable_index, model.output_index)  
  
    phase_0 = julia_init_phase(model.N, model.input_index, training_data, batch_size)  
    W_0 = model.weights_0  
    bias_0 = model.bias_0  
  
    WL = zeros(N_epoch + 1, model.N, model.N)  
    biasL = zeros(N_epoch + 1, 2, model.N)  
    costL = zeros(N_epoch)  
  
    W = W_0  
    bias = bias_0  
  
    WL[1, :, :] .= W_0  
    biasL[1, :, :] .= bias_0  
  
    # Prepare for Adam optimizer  
    s_W = zeros(size(W))  
    r_W = zeros(size(W))  
    s_h = zeros(size(bias))  
    r_h = zeros(size(bias))  
  
    for k in 1:N_epoch  
        phase_0 = julia_init_phase(phase_form, model.input_index, training_data, batch_form, k, from_zero)  
        gW, gh, cost, q_cost = EP_param_gradient(W, bias, phase_0, training_target, training_paras, model_paras)  
  
        # Gradient descent update  
        W = gradient_descent_update(W, gW, study_rate, k)  
        bias = gradient_descent_update(bias, gh, study_rate, k)  
  
        # Uncomment for Adam optimizer  
        # W, s_W, r_W = Adam_update(W, gW, study_rate, k, s_W, r_W)  
        # bias, s_h, r_h = Adam_update(bias, gh, study_rate, k, s_h, r_h)  
  
        costL[k] = cost  
    end  
  
    return WL, biasL, costL  
end  

# Define parameters  
N_ev = 1000  
dt = 0.1  

study_rate = 0.1  
beta = 0.1  

# ---------------- Define the network with square lattice architecture ----------------  
dimension = (3, 3)  # Square lattice dimensions  
input_size = 2  
output_size = 1  

# Create the square lattice network  
#lattice_model = SP_XY_SquareLattice_Network(dimension, N_ev, dt, input_size, output_size)  
#random_state_initiation!(lattice_model)  # Initialize the network state  

# ---------------- Define the network with all-to-all connection ----------------  
N = 5  
input_index = [1, 2]  # Julia uses 1-based indexing  
output_index = [N]    # Last neuron as output  

# Create the all-to-all connected network  
UA_model = SP_XY_Network(N, N_ev, dt, input_index, output_index)  
random_state_initiation!(UA_model)  # Initialize the network state  

# ---------------- Define the network with layered structure ----------------  
structure_list = [2, 5, 5, 5, 1]  # Number of neurons in each layer  
N = sum(structure_list)           # Total number of neurons  

# Create the layered network  
#layer_model = SP_XY_Layer_Network(N, N_ev, dt, structure_list)  
#random_state_initiation!(layer_model)  # Initialize the network state  
 
# Set experimental parameters  
N_task = 100  
N_epoch = 1000  
study_rate = 0.1  
batch_size = 1  

# Update the beta parameter for each model  
get_beta!(UA_model, beta)  
#get_beta!(layer_model, beta)  
#get_beta!(lattice_model, beta)  

function single_train_US_model01(W_0, bias_0, model_paras, training_paras, N_epoch, batch_size, study_rate, training_data, training_target)  
    # Initialize variables  
    from_zero = false  
    if batch_size == 0  
        batch_size = 1  
        from_zero = true  
    end  

    beta, study_rate = training_paras  
    N_ev, dt, input_index, variable_index, output_index = model_paras  

    N = size(input_index, 1) + size(variable_index, 1)  
    N_data = size(training_data, 1)  

    batch_form = zeros(batch_size)  
    phase_form = zeros(N_data, N)  

    WL = zeros(N_epoch + 1, N, N)  
    biasL = zeros(N_epoch + 1, 2, N)  
    costL = zeros(N_epoch)  

    W = W_0  
    bias = bias_0  

    WL[1, :, :] .= W_0  
    biasL[1, :, :] .= bias_0  

    # Prepare for Adam (if needed later)  
    s_W = zeros(size(W))  
    r_W = zeros(size(W))  
    s_h = zeros(size(bias))  
    r_h = zeros(size(bias))  

    for k in 1:N_epoch  
        # Initialize phase  
        phase_0 = julia_init_phase(phase_form, input_index, training_data, batch_form, k, from_zero)  

        # Compute gradients and cost  
        gW, gh, cost, q_cost = EP_param_gradient(W, bias, phase_0, training_target, training_paras, model_paras)  

        # Update weights and biases using gradient descent  
        W = gradient_descent_update(W, gW, study_rate, k)  
        bias = gradient_descent_update(bias, gh, study_rate, k)  

        # Optional: Adam optimizer (commented out in the original code)  
        # W, s_W, r_W = Adam_update(W, gW, study_rate, k, s_W, r_W)  
        # bias, s_h, r_h = Adam_update(bias, gh, study_rate, k, s_h, r_h)  

        # Store cost  
        costL[k] = cost  
    end  

    return WL, biasL, costL  
end  


function single_train_US_model(W_0, bias_0, model_paras, training_paras, N_epoch, batch_size, study_rate, training_data, training_target)  
    # Unpack parameters  
    beta, study_rate = training_paras  
    N_ev, dt, input_index, variable_index, output_index = model_paras  

    N = size(input_index,1) + size(variable_index,1)  
    N_data = size(training_data, 1)  

    # Initialize variables  
    from_zero = false  
    if batch_size == 0  
        batch_size = 1  
        from_zero = true  
    end  

    batch_form = zeros(batch_size)  
    phase_form = zeros(N_data, N)  

    WL = zeros(N_epoch + 1, N, N)  
    biasL = zeros(N_epoch + 1, 2, N)  
    costL = zeros(N_epoch)  

    W = copy(W_0)  
    bias = copy(bias_0)  

    WL[1, :, :] .= W_0  
    biasL[1, :, :] .= bias_0  

    # Prepare for Adam optimizer  
    s_W = zeros(size(W))  
    r_W = zeros(size(W))  
    s_h = zeros(size(bias))  
    r_h = zeros(size(bias))  

    # Training loop  
    for k in 1:N_epoch  
        # Initialize phase  
        phase_0 = julia_init_phase(phase_form, input_index, training_data, batch_form, k, from_zero)  
        #println("Epoch: $(k)")  
        #println("Size of phase_0: $(size(phase_0))")
        #println("Size of W: $(size(W))")
        # Compute gradients and cost  
        gW, gh, cost, q_cost = EP_param_gradient(W, bias, phase_0, training_target, training_paras, model_paras)  
        #println("Size of W after EP_param_gradient: $(size(W))")
        # Gradient descent update  
        W = gradient_descent_update(W, gW, study_rate, k)  
        bias = gradient_descent_update(bias, gh, study_rate, k)  

        #println("Size of W after gradient_descent_update: $(size(W))")
        # Uncomment for Adam optimizer  
        #W, s_W, r_W = Adam_update(W, gW, study_rate, k, s_W, r_W)  
        #bias, s_h, r_h = Adam_update(bias, gh, study_rate, k, s_h, r_h)  

        # Update cost and save weights and biases  
        costL[k] = cost  
    end  

    return WL, biasL, costL  
end  

function init_phase(phase_form, input_index, training_data, batch_form, epoch, from_zero)  
    # Initialize the phase based on the input data and other parameters  
    # Placeholder implementation  
    phase_0 = copy(phase_form)  
    phase_0[:, input_index] .= training_data[:, input_index]  
    return phase_0  
end 

# Initialize phase
using Random  
using Distributions
function julia_init_phase(phase_form, input_index, input_data, batch_form, seed, from_zero=false)  
    # Extract dimensions  
    N_data, N = size(phase_form)  
    batch_size = size(batch_form, 1)  

    # Set the random seed  
    Random.seed!(seed)  

    # Compute batch size (ensure it's at least 1)  
    BS = max(1, batch_size)  

    # Initialize phase_0 uniformly at random in [-π/2, π/2]
    phase_0 = rand(Uniform(-π/2, π/2), batch_size, N_data, N)

    # Set the input data at the specified indices  
    phase_0[:, :, input_index] = input_data  

    return phase_0  
end  

# function implementing the EP method for calculating gradients and cost

function cost_function(phase, target, output_index)  
    # Reshape the output phase to match the shape of the target  
    output_phase = reshape(phase[:, output_index], size(target))  

    # Calculate the difference between output and target  
    doutput = output_phase .- target  

    # Compute the cost matrix  
    cost_mat = ones(size(doutput)) .- cos.(doutput)  

    # Sum over the second dimension and divide by 2 to get the cost list  
    cost_list = sum(cost_mat, dims=2) ./ 2  

    return vec(cost_list)  # Return as a 1D array  
end  

function qualitative_cost(phase, target, output_index, tol)  
    # Reshape the output phase to match the shape of the target  
    output_phase = reshape(phase[:, output_index], size(target))  

    # Calculate the difference between output and target  
    doutput = output_phase .- target  

    # Compute the cost matrix  
    cost_mat = ones(size(doutput)) .- cos.(doutput)  

    # Check where cost_mat exceeds the tolerance and sum the results  
    cost_list = sum((cost_mat .> tol) ./ 2)  

    return cost_list  
end  

# calculate weights gradient  
function weights_gradient(equi_nudge, equi_free, beta)  
    # Calculate the phase differences
    #println("Size of equi_nudge: $(size(equi_nudge))")
    #println("Size of equi_free: $(size(equi_free))")  
    nudge_dphase = cal_dphase(equi_nudge)  
    free_dphase = cal_dphase(equi_free)  
    #println("Size of nudge_dphase: $(size(nudge_dphase))")
    #println("Size of free_dphase: $(size(free_dphase))")
    # Compute the gradient list  
    gradient_list = -cos.(nudge_dphase) .+ cos.(free_dphase)  
    #println("Size of gradient_list: $(size(gradient_list))")
    # Take the mean along the first dimension  
    gradient = mean(gradient_list, dims=3)  
    #println("Size of gradient: $(size(gradient))")
    return gradient  
end  

# calculate bias gradient  
function bias_gradient(equi_nudge, equi_free, bias, beta)  
    # Extract h and psi from the bias  
    h = bias[1, :]  
    psi = bias[2, :]  

    # Expand h and psi to match the shape of equi_free  
    h = ones(size(equi_free, 1)) * h'  
    psi = ones(size(equi_free, 1)) * psi'  

    # Compute gradients for h and psi  
    g_h = cos.(equi_free .- psi) .- cos.(equi_nudge .- psi)  
    g_psi = h .* sin.(equi_free .- psi) .- h .* sin.(equi_nudge .- psi)  

    # Take the mean along the first dimension  
    g_h = mean(g_h, dims=1)  
    g_psi = mean(g_psi, dims=1)  

    return [g_h; g_psi]  # Combine gradients into a single array  
end  
  
#v_run_network(arg1_batch, arg2, arg3, arg4, arg5_batch, arg6, arg7, arg8) =
#    map((a1, a5) -> single_run_network(a1, arg2, arg3, arg4, a5, arg6, arg7, arg8), arg1_batch, arg5_batch)
function v_run_network(arg1_batch, arg2, arg3, arg4, arg5_batch, arg6, arg7, arg8)
        results = []
        #println("Size of arg1_batch: $(size(arg1_batch))")
        #println("Size of arg5_batch: $(size(arg5_batch))")

        # comapact all dimensions that are 1 in tensor
        if size(arg1_batch, 1) == 1
            arg1_batch = dropdims(arg1_batch, dims=1)
        end
        
        i = 0
        for (a1, a5) in zip(eachrow(arg1_batch), eachrow(arg5_batch))
            #println("loop i: $(i)")
            result = single_run_network(a1, arg2, arg3, arg4, a5, arg6, arg7, arg8)
            #println("Size of result: $(size(result))")
            push!(results, result)
            i += 1
        end
        return hcat(results...)'
end

function EP_param_gradient(W_0, bias_0, phase_0, training_target, training_paras, model_paras)  
    # Unpack training and model parameters  
    beta, study_rate = training_paras  
    N_ev, dt, input_index, variable_index, output_index = model_paras  
    N_data = size(training_target, 1)  
    N = size(W_0, 2)  
    T = N_ev * dt  

    # Initial phase  
    batch_size = size(phase_0, 1)  
    aux_ones = ones(batch_size)
    batch_training_target = aux_ones .* training_target'

    # Run the network for equilibrium states  
    #println("Size of W_0: $(size(W_0))")
    equi_zero = v_run_network(phase_0, T, W_0, bias_0, vec(batch_training_target), 0, input_index, output_index)  
    equi_nudge = v_run_network(equi_zero, T, W_0, bias_0, vec(batch_training_target), beta, input_index, output_index)  
    #println("finished the  EP step")
    # Calculate costs  
    cost_list = cost_function(equi_zero, vec(batch_training_target), output_index)  
    cost = mean(cost_list)  

    q_cost_list = qualitative_cost(equi_zero, vec(batch_training_target), output_index, 0.1)  
    q_cost = mean(q_cost_list)  

    # Calculate gradients  
    gW, gh = paras_gradient(equi_nudge, equi_zero, bias_0, beta)  

    # Return gradients and costs  
    return gW / beta, gh / beta, cost, q_cost  
end  

# combined params gradient  
function paras_gradient(equi_nudge, equi_free, bias, beta)  
    # Compute gradients for weights and biases  
    gW = weights_gradient(equi_nudge, equi_free, beta)  
    gb = bias_gradient(equi_nudge, equi_free, bias, beta)  

    return gW, gb  
end  

function gradient_descent_update_k(param, grad, study_rate, epoch)  
    return param .- study_rate * grad  
end  

function gradient_descent_update(paras, g_paras, study_rate_0, itr_time; study_rate_f=0.001, decay=0.0)  
    # Compute the learning rate with decay  
    eta = max(study_rate_0 - decay * itr_time, study_rate_f)  
    #println("Learning rate: $(eta)")
    #println("Size of paras: $(size(paras))")
    #println("Size of g_paras: $(size(g_paras))")
    # Perform the gradient descent update  
    return paras .- eta .* g_paras  
end  

function Adam_update(param, grad, study_rate, epoch, s, r, beta1=0.9, beta2=0.999, epsilon=1e-8)  
    s = beta1 * s .+ (1 - beta1) * grad  
    r = beta2 * r .+ (1 - beta2) * (grad .^ 2)  
    s_hat = s ./ (1 - beta1^epoch)  
    r_hat = r ./ (1 - beta2^epoch)  
    param = param .- study_rate * s_hat ./ (sqrt.(r_hat) .+ epsilon)  
    return param, s, r  
end 
  
# Define training_data  
training_data = (π / 2) * [-1 -1; -1 1; 1 -1; 1 1]  
  
# Define training_target  
training_target = (π / 2) * [-1; 1; 1; -1]  
  
# Extract initial weights and biases from the UA_model  
W_0 = UA_model.weights_0  
bias_0 = UA_model.bias_0  

# Extract variable indices from the UA_model  
variable_index = UA_model.variable_index  

# Define training parameters  
training_paras = (beta, study_rate)  

# Define model parameters  
model_paras = (N_ev, dt, input_index, variable_index, output_index)  

# Define the training function  
train_fun = (W_0, bias_0) -> single_train_US_model(W_0, bias_0, model_paras, training_paras, N_epoch, batch_size, study_rate, training_data, training_target)  

# Vectorized training using `map` or broadcasting  
# v_train = map(train_fun, W_0_list, bias_0_list)  # Apply train_fun to each pair of W_0 and bias_0
# Set experimental parameters
N_task = 10
N_epoch = 100
study_rate = 0.1
batch_size = 1
# Define parameters  
N = 5  
input_index = [1, 2]  # Julia uses 1-based indexing  
output_index = [N]    # Last neuron as output  
  
# Create the all-to-all connected network  
UA_model = SP_XY_Network(N, N_ev, dt, input_index, output_index)  
  
# Initialize the network state  
random_state_initiation!(UA_model)  
  
# Set the beta parameter  
get_beta!(UA_model, beta)  
  
# Initialize a matrix to store costs for multiple tasks and epochs  
all_cost_US3 = zeros(N_task, N_epoch)  


using Random  
using JLD2  # For saving data  
using LinearAlgebra  
using TimerOutputs  # For timing  
using Zygote  # For automatic differentiation (if needed for training)  

# Initialize weight and bias arrays  
W0L = zeros(N_task, N, N)  
bias0L = zeros(N_task, 2, N)  

# Extract model parameters  
W_0 = UA_model.weights_0  
bias_0 = UA_model.bias_0  

variable_index = UA_model.variable_index  

# Training and model parameters  
training_paras = (N_epoch, beta, study_rate)  
model_paras = (N_ev, dt, input_index, variable_index, output_index)  

# Random initialization for each task  
for k in 1:N_task  
    random_state_initiation!(UA_model)  
    W0L[k, :, :] .= UA_model.weights_0  
    bias0L[k, :, :] .= UA_model.bias_0  
end  

# Convert to arrays (already arrays in Julia, so no need for conversion)  
# W0L and bias0L are already in the correct format.  

# Define the training function  
train_fun = function(W_0, bias_0)  
    #println("Size W0 in train: $(size(W_0))")
    single_train_US_model(W_0, bias_0, model_paras, training_paras, N_epoch, batch_size, study_rate, training_data, training_target)  
end  

println("Starting training...")
println("Size of W0L: $(size(W0L))")
# Vectorized training using `map` or broadcasting
t0 = time()
results = map(k->train_fun(W0L[k, :, :], bias0L[k, :, :]), 1:N_task)
t1 = time()

# Extract results
A = [res[1] for res in results]
B = [res[2] for res in results]
all_cost_US3 = [res[3] for res in results]

# Save results  
@save "all_cost_US$(N).jld2" all_cost_US3  

# Print results  
println("N = $N, Time = $(t1-t0)) seconds")  

# Initialize weight and bias arrays  
W0L = zeros(N, N)  
bias0L = zeros(2, N) 
# Random initialization for single task  
random_state_initiation!(UA_model)  
W0L[:, :] .= UA_model.weights_0  
bias0L[:, :] .= UA_model.bias_0  
sp_train_network(W_0, bias_0, training_data, training_target, training_paras, model_paras)

using Plots  

plt = plot()
# Loop over tasks and plot  
for k in 1:N_task  
    plot!(plt,collect(0:(N_epoch-1)), all_cost_US3[k], color=:black, linewidth=0.1, label="")  
end  
display(plt)