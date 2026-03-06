import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time

class SP_XY_Network:
    def __init__(self, N, N_ev, dt, input_index, output_index):
        self.N = N                      # Number of neurons
        self.N_ev = N_ev                # Maximum number of updates
        self.dt = dt                    # Time step
        self.input_index = input_index  # Indices of input cells
        self.output_index = output_index  # Indices of output cells
        self.variable_index = [i for i in range(N) if i not in input_index]  # Indices of variable cells
        self.N_input = len(input_index)  # Number of input cells
        self.N_output = len(output_index)  # Number of output cells
        self.T = dt * N_ev              # Total time
        self.weights = np.zeros((N, N))  # Weight matrix
        self.weights_0 = np.zeros((N, N))  # Initial weight matrix
        self.bias = np.zeros((2, N))    # Bias matrix
        self.bias_0 = np.zeros((2, N))  # Initial bias matrix
        self.beta = 0.001               # Regularization parameter
        self.phase_0 = np.zeros(N)      # Initial phase

    def random_state_initiation(self):
        # Randomly set weights with smaller initialization
        self.weights_0 = 0.1 * np.random.randn(self.N, self.N)
        # No self-connections
        np.fill_diagonal(self.weights_0, 0)
        # Symmetric weights
        self.weights_0 = (self.weights_0 + self.weights_0.T) / 2
        self.weights = self.weights_0.copy()

        # Randomly set bias with smaller initialization
        bias = 0.1 * np.random.rand(2, self.N)
        bias[0, :] -= 0.05  # Reduced offset
        bias[1, :] = 2 * np.pi * (bias[1, :] - 0.5)
        self.bias_0 = bias.copy()
        self.bias = bias.copy()

        # Randomly set phase_0 with smaller initialization
        self.phase_0 = 0.5 * np.pi * np.random.rand(self.N) - np.pi / 4

    def set_beta(self, beta):
        self.beta = beta

# Calculate force for ODE solver
def calculate_force(phase, W, bias, target, beta, input_index, output_index):
    # Calculate internal force (from weights)
    N = len(phase)
    dphase = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            dphase[i, j] = phase[i] - phase[j]
    
    F0 = np.sum(W * np.sin(dphase), axis=1)
    
    # Calculate bias force
    h = bias[0, :]
    psi = bias[1, :]
    F1 = -h * np.sin(phase - psi)
    
    # Calculate cost force
    F3 = np.zeros_like(phase)
    output_phase = phase[output_index]
    if not isinstance(output_phase, np.ndarray):
        output_phase = np.array([output_phase])
    if not isinstance(target, np.ndarray):
        target = np.array([target])
    
    M1 = -np.sin(output_phase - target)
    M2 = np.ones(np.shape(output_phase)) + np.cos(output_phase - target)
    
    for i, idx in enumerate(output_index):
        F3[idx] = M1[i] / (M2[i]+1e-10)
    
    # Total force
    F = -F0 + F1 + beta * F3
    
    # Zero force at input indices
    F[input_index] = 0
    
    return F

# Run network for a single phase
def run_network(phase_0, T, W, bias, target, beta, input_index, output_index):
    def force_fn(t, phase):
        return calculate_force(phase, W, bias, target, beta, input_index, output_index)
    
    # Solve ODE
    tspan = (0.0, T)
    sol = solve_ivp(
        force_fn, 
        tspan, 
        phase_0, 
        method='RK45', 
        rtol=1e-10, 
        atol=1e-10,
        max_step=T/1000  # Ensure enough steps for accuracy
    )
    
    return sol.y[:, -1]  # Return final state

# Run network for a batch of phases
def run_network_batch(phase_0_batch, T, W, bias, target_batch, beta, input_index, output_index):
    N_batch = len(phase_0_batch)
    N = len(phase_0_batch[0])
    result_matrix = np.zeros((N_batch, N))
    
    for i in range(N_batch):
        phase_0 = phase_0_batch[i]
        target = target_batch[i]
        result_matrix[i] = run_network(phase_0, T, W, bias, target, beta, input_index, output_index)
    
    return result_matrix

# Calculate cost function
def cost_function(phase_batch, target_batch, output_index):
    N_batch = len(phase_batch)
    cost_list = np.zeros(N_batch)
    
    for i in range(N_batch):
        phase = phase_batch[i]
        target = target_batch[i]
        output_phase = phase[output_index]
        
        if not isinstance(output_phase, np.ndarray):
            output_phase = np.array([output_phase])
        if not isinstance(target, np.ndarray):
            target = np.array([target])
        
        doutput = output_phase - target
        cost_mat = 1.0 - np.cos(doutput)
        cost_list[i] = np.sum(cost_mat) / 2
    
    return cost_list

# Calculate weights gradient
def weights_gradient(equi_nudge, equi_free, beta):
    N_data = len(equi_free)
    N = len(equi_free[0])
    
    gradient = np.zeros((N, N))
    
    for i in range(N_data):
        for j in range(N):
            for k in range(N):
                nudge_diff = equi_nudge[i, j] - equi_nudge[i, k]
                free_diff = equi_free[i, j] - equi_free[i, k]
                gradient[j, k] += (-np.cos(nudge_diff) + np.cos(free_diff)) / N_data
    
    return gradient

# Calculate bias gradient
def bias_gradient(equi_nudge, equi_free, bias, beta):
    N_data = len(equi_free)
    N = len(equi_free[0])
    
    h = bias[0, :]
    psi = bias[1, :]
    
    g_h = np.zeros((1, N))
    g_psi = np.zeros((1, N))
    
    for i in range(N_data):
        for j in range(N):
            g_h[0, j] += (np.cos(equi_free[i, j] - psi[j]) - np.cos(equi_nudge[i, j] - psi[j])) / N_data
            g_psi[0, j] += h[j] * (np.sin(equi_free[i, j] - psi[j]) - np.sin(equi_nudge[i, j] - psi[j])) / N_data
    
    return np.vstack([g_h, g_psi])

# Combined gradient calculation
def paras_gradient(equi_nudge, equi_free, bias, beta):
    gW = weights_gradient(equi_nudge, equi_free, beta)
    gb = bias_gradient(equi_nudge, equi_free, bias, beta)
    return gW, gb

# EP parameter gradient calculation
def EP_param_gradient(W_0, bias_0, phase_0, training_target, beta, N_ev, dt, input_index, variable_index, output_index):
    N_data = len(training_target)
    N = W_0.shape[0]
    T = N_ev * dt
    
    # Run network for equilibrium states
    equi_zero = run_network_batch(phase_0, T, W_0, bias_0, training_target, 0.0, input_index, output_index)
    equi_nudge = run_network_batch(equi_zero, T, W_0, bias_0, training_target, beta, input_index, output_index)
    equi_free = run_network_batch(equi_zero, T, W_0, bias_0, training_target, -beta, input_index, output_index)
    
    # Calculate costs
    cost_list = cost_function(equi_zero, training_target, output_index)
    cost = np.mean(cost_list)
    
    # Calculate qualitative cost (distance within tolerance)
    tol = 0.1
    q_cost_list = np.zeros(N_data)
    for i in range(N_data):
        output_phase = equi_zero[i, output_index]
        target = training_target[i]
        
        if not isinstance(output_phase, np.ndarray):
            output_phase = np.array([output_phase])
        if not isinstance(target, np.ndarray):
            target = np.array([target])
        
        doutput = output_phase - target
        q_cost_list[i] = np.sum((1.0 - np.cos(doutput)) > tol) / 2
    
    q_cost = np.mean(q_cost_list)
    
    # Calculate gradients
    gW, gh = paras_gradient(equi_nudge, equi_free, bias_0, beta)
    
    # Return gradients and costs
    return gW / beta, gh / beta, cost, q_cost

# Adam optimizer
def Adam_update(param, grad, study_rate, epoch, s, r, beta1=0.9, beta2=0.999, epsilon=1e-8):
    # Clip gradients
    max_grad = 1.0
    grad_clipped = np.clip(grad, -max_grad, max_grad)
    
    # Update momentum and RMS
    s = beta1 * s + (1 - beta1) * grad_clipped
    r = beta2 * r + (1 - beta2) * (grad_clipped ** 2)
    
    # Bias correction
    s_hat = s / (1 - beta1 ** epoch)
    r_hat = r / (1 - beta2 ** epoch)
    
    # Update parameters
    param = param - study_rate * s_hat / (np.sqrt(r_hat) + epsilon)
    
    return param, s, r

# Train network
def train_network(W_0, bias_0, training_data, training_target, beta, study_rate, N_epoch, batch_size, N_ev, dt, input_index, variable_index, output_index):
    N = W_0.shape[0]
    N_data = len(training_data)
    T = N_ev * dt
    
    # Initialize weights and biases
    W = W_0.copy()
    bias = bias_0.copy()
    
    # Storage for training history
    cost_history = np.zeros(N_epoch)
    best_cost = float('inf')
    best_W = W.copy()
    best_bias = bias.copy()
    patience = 20  # Early stopping patience
    patience_counter = 0
    
    # Prepare for Adam optimizer
    s_W = np.zeros_like(W)
    r_W = np.zeros_like(W)
    s_bias = np.zeros_like(bias)
    r_bias = np.zeros_like(bias)
    
    # Use Adam optimizer
    use_adam = True
    
    # Training loop
    for epoch in range(1, N_epoch + 1):
        # Create mini-batches
        if batch_size == 0 or batch_size > N_data:
            batch_indices = [list(range(N_data))]
        else:
            num_batches = int(np.ceil(N_data / batch_size))
            batch_indices = [list(range((i-1)*batch_size, min(i*batch_size, N_data))) for i in range(1, num_batches+1)]
        
        epoch_cost = 0.0
        
        # Process each mini-batch
        for batch_idx in batch_indices:
            # Prepare batch data
            batch_data = [training_data[i] for i in batch_idx]
            batch_target = [training_target[i] for i in batch_idx]
            
            # Initialize phase
            phase_0 = np.zeros((len(batch_idx), N))
            for i, idx in enumerate(batch_idx):
                phase_0[i, input_index] = training_data[idx]
            
            # Add small random noise to non-input neurons for better exploration
            phase_0[:, variable_index] = 0.1 * np.random.randn(len(batch_idx), len(variable_index))
            
            # Calculate gradients and cost
            gW, gbias, batch_cost, q_cost = EP_param_gradient(
                W, bias, phase_0, batch_target, beta, 
                N_ev, dt, input_index, variable_index, output_index
            )
            
            # Update parameters
            if use_adam:
                # Adam updates
                W, s_W, r_W = Adam_update(W, gW, study_rate, epoch, s_W, r_W)
                bias, s_bias, r_bias = Adam_update(bias, gbias, study_rate, epoch, s_bias, r_bias)
            else:
                # Standard gradient descent with adaptive learning rate
                eta = max(study_rate * np.exp(-0.0001 * epoch), 0.0001)
                W = W - eta * gW
                bias = bias - eta * gbias
            
            epoch_cost += batch_cost * len(batch_idx)
        
        # Calculate average cost for the epoch
        epoch_cost /= N_data
        cost_history[epoch-1] = epoch_cost
        
        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Cost = {epoch_cost:.6f}")
        
        # Early stopping check
        if epoch_cost < best_cost:
            best_cost = epoch_cost
            best_W = W.copy()
            best_bias = bias.copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Check for convergence
        if epoch_cost < 1e-6:
            print(f"Converged at epoch {epoch}")
            break
    
    return best_W, best_bias, cost_history

# Test case for 3-node XY network
def test_3node_network():
    print("Testing 3-node XY network...")
    
    # Define parameters
    N = 3  # 3-node network
    N_ev = 200  # Number of time steps
    dt = 0.1
    input_index = [0]  # First node is input
    output_index = [2]  # Last node is output
    variable_index = [1]  # Middle node is variable
    
    # Create network
    network = SP_XY_Network(N, N_ev, dt, input_index, output_index)
    
    # Set beta to a smaller value for better stability
    beta = 0.01
    network.set_beta(beta)
    
    # Define training parameters
    N_epoch = 3
    study_rate = 0.005
    batch_size = 2
    
    # Define training data (simple binary classification)
    training_data = np.array([
        [0],      # Input 0 -> Output 0
        [np.pi]   # Input π -> Output π
    ])
    
    training_target = np.array([
        [0],      # Target 0
        [np.pi]   # Target π
    ])
    
    # Initialize weights with smaller values
    network.random_state_initiation()
    
    # Scale down initial weights further
    network.weights_0 *= 0.1
    network.bias_0 *= 0.1
    
    print("Initial weights:")
    print(network.weights_0)
    print("Initial bias:")
    print(network.bias_0)
    
    # Train network
    start_time = time.time()
    W_final, bias_final, cost_history = train_network(
        network.weights_0, network.bias_0, 
        training_data, training_target, 
        beta, study_rate, N_epoch, batch_size,
        N_ev, dt, input_index, variable_index, output_index
    )
    training_time = time.time() - start_time
    
    # Update network with final weights
    network.weights = W_final
    network.bias = bias_final
    
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print("Final weights:")
    print(W_final)
    print("Final bias:")
    print(bias_final)
    
    # Test network on training data
    test_results = np.zeros(len(training_data))
    for i in range(len(training_data)):
        phase_0 = np.zeros(network.N)
        phase_0[input_index] = training_data[i]
        
        final_phase = run_network(phase_0, network.T, W_final, bias_final, training_target[i], beta, input_index, output_index)
        test_results[i] = final_phase[output_index[0]]
    
    # Print results
    print("\nTest results:")
    for i in range(len(training_data)):
        input_val = training_data[i][0]
        target = training_target[i][0]
        result = test_results[i]
        error = abs(result - target)
        print(f"Input: {input_val:.2f} => Target: {target:.2f}, Result: {result:.2f}, Error: {error:.6f}")
    
    # Plot cost history
    plt.figure(figsize=(10, 6))
    valid_costs = cost_history[cost_history != 0]
    plt.plot(valid_costs, 'b-', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Cost', fontsize=12)
    plt.title('Training Cost History', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale to better see convergence
    plt.tight_layout()
    plt.show()
    
    # Visualize the network dynamics
    visualize_network_dynamics(W_final, bias_final, beta, input_index, output_index)
    
    return network, cost_history, test_results

# Visualize network dynamics
def visualize_network_dynamics(W, bias, beta, input_index, output_index):
    N = W.shape[0]
    
    # Create a grid of initial phases
    n_points = 20
    x = np.linspace(-np.pi, np.pi, n_points)
    y = np.linspace(-np.pi, np.pi, n_points)
    X, Y = np.meshgrid(x, y)
    
    # For 3-node network, we fix the input and see how the variable node affects the output
    if N == 3 and len(input_index) == 1 and len(output_index) == 1:
        variable_index = [i for i in range(N) if i not in input_index and i not in output_index][0]
        
        plt.figure(figsize=(12, 10))
        
        # Test with two different inputs
        for input_val, target_val, subplot_idx in zip([0, np.pi], [0, np.pi], [1, 2]):
            Z = np.zeros_like(X)
            
            for i in range(n_points):
                for j in range(n_points):
                    # Set initial phase
                    phase_0 = np.zeros(N)
                    phase_0[input_index[0]] = input_val
                    phase_0[variable_index] = X[i, j]
                    phase_0[output_index[0]] = Y[i, j]
                    
                    # Run network
                    T = 20  # Shorter time for visualization
                    final_phase = run_network(phase_0, T, W, bias, target_val, beta, input_index, output_index)
                    
                    # Calculate energy (negative of force)
                    force = calculate_force(final_phase, W, bias, target_val, beta, input_index, output_index)
                    energy = -np.sum(np.abs(force))
                    Z[i, j] = energy
            
            plt.subplot(1, 2, subplot_idx)
            plt.contourf(X, Y, Z, 50, cmap='viridis')
            plt.colorbar(label='Energy')
            plt.xlabel(f'Phase of Node {variable_index}', fontsize=12)
            plt.ylabel(f'Phase of Node {output_index[0]}', fontsize=12)
            plt.title(f'Energy Landscape (Input={input_val:.2f}, Target={target_val:.2f})', fontsize=14)
            
            # Mark the target
            plt.scatter([0], [target_val], color='red', s=100, marker='*', label='Target')
            plt.legend()
        
        plt.tight_layout()
        plt.show()

# Run the test
if __name__ == "__main__":
    network, cost_history, test_results = test_3node_network()