module ChaosFHN
using DynamicalSystems
using ChaosTools
using OrdinaryDiffEq
using Graphs
using CairoMakie
using CUDA
using Statistics
using LinearAlgebra
using FFTW

export calculate_lyapunov_spectrum, analyze_phase_space, analyze_phase_space_statistics

function calculate_lyapunov_spectrum(g_directed, edge_weights, g_input, tspan, x0, σ, fhn_network!)
    """
    Calculate Lyapunov spectrum for FHN network
    """
    N = nv(g_directed)
    p = (g_input, σ * edge_weights)
    
    # Define the dynamics function for DynamicalSystems.jl
    function fhn_ds!(du, u, p, t)
        fhn_network!(du, u, p, t)
    end
    
    # Create dynamical system
    ds = CoupledODEs(fhn_ds!, x0, p)
    total_time = tspan[2] - tspan[1]
    dt = 0.1
    Y,t = trajectory(ds, total_time; Δt=dt)
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="Time", ylabel="u")
    #for var in columns(Y)
    #    println(size(var))
    #    lines!(ax, t, var[:,2], alpha=0.3)
    #end
    for i in 1:N
        lines!(ax, t, Y[:, i], color=:blue, alpha=0.3)
        lines!(ax, t, Y[:, N + i], color=:red, alpha=0.3)
    end
 

    CairoMakie.save("fhn_chaos_trajectory.png", fig)
    # make the raster plot
    # Create the heatmap
    Y_matrix = Matrix(Y)
    println(size(Y_matrix))
    Y_subset = Y_matrix[:, 1:2:20]  # Take every second variable for better visibility
    fig2 = Figure()
    ax2 = Axis(fig2[1, 1], 
        xlabel="Time", 
        ylabel="Variable Index",
        title="FHN Chaos Trajectory Heatmap")

    # Create a custom colormap: blue for below threshold, yellow for above
    hm = heatmap!(ax2, t, 1:size(Y_subset, 2), Y_subset',
    colormap = :viridis,  # or use a custom colormap
    colorrange = (minimum(Y_subset), maximum(Y_subset)))

    # Add colorbar
    Colorbar(fig2[1, 2], hm, label="u value")

    CairoMakie.save("fhn_chaos_trajectory.png", fig)
    CairoMakie.save("fhn_chaos_raster_plot.png", fig2)
    println("Raster plot saved to fhn_chaos_raster_plot.png")
    println("Trajectory plot saved to fhn_chaos_trajectory.png")
    # Calculate Lyapunov spectrum
    # For large systems, calculate only the largest few exponents
    λs = lyapunovspectrum(ds, 5000)
    
    return λs
end


function analyze_phase_space(sol, node_indices=1:5; colormap=:viridis, use_gpu=true, tsteps=nothing)
    """
    Analyze phase space structure for multiple oscillators using GPU-accelerated operations
    
    Parameters:
    - sol: ODE solution (can be CuArray or regular Array)
    - node_indices: Vector or range of node indices to plot
    - colormap: Color scheme for trajectories
    - use_gpu: Whether to use GPU acceleration for computations
    """
    tsteps = isnothing(tsteps) ? sol.t : tsteps
    n_nodes = length(node_indices)
    n_timesteps = length(tsteps)
    
    # Convert solution to matrix format and move to GPU if requested
    # Extract all u and v values at once
    u_indices = [2*i-1 for i in node_indices]
    v_indices = [2*i for i in node_indices]
    
    if use_gpu && CUDA.functional()
        # Extract data and move to GPU
        sol_array = sol |> Array  # Ensure it's an array first
        u_matrix = CuArray(sol_array[u_indices, :])  # (n_nodes, n_timesteps)
        v_matrix = CuArray(sol_array[v_indices, :])  # (n_nodes, n_timesteps)
    else
        sol_array = sol |> Array
        u_matrix = sol_array[u_indices, :]
        v_matrix = sol_array[v_indices, :]
    end
    
    # Create figure with subplots
    fig = Figure(size=(1600, 1000))
    
    # 1. Individual phase portraits (2x3 grid for first 6 nodes)
    n_individual = min(6, n_nodes)
    for idx in 1:n_individual
        row = (idx - 1) ÷ 3 + 1
        col = (idx - 1) % 3 + 1
        
        # Move data back to CPU for plotting
        u = Array(u_matrix[idx, :])
        v = Array(v_matrix[idx, :])
        
        ax = Axis(fig[row, col], 
                  xlabel="u (voltage)", 
                  ylabel="v (recovery)",
                  title="Oscillator $(node_indices[idx])")
        lines!(ax, u, v, color=1:length(u), colormap=colormap, linewidth=1.5)
    end
    
    # 2. Overlay of all trajectories
    ax_overlay = Axis(fig[3, 1:2], 
                      xlabel="u (voltage)", 
                      ylabel="v (recovery)",
                      title="Overlay of All Trajectories")
    
    colors = cgrad(colormap, n_nodes, categorical=true)
    
    # Move data to CPU for plotting
    u_cpu = Array(u_matrix)
    v_cpu = Array(v_matrix)
    
    for idx in 1:n_nodes
        lines!(ax_overlay, u_cpu[idx, :], v_cpu[idx, :], 
               color=colors[idx], alpha=0.6, 
               linewidth=1, label="Osc $(node_indices[idx])")
    end
    
    if n_nodes <= 10
        axislegend(ax_overlay, position=:rt, nbanks=2)
    end
    
    # 3. Time series for all oscillators
    ax_time = Axis(fig[3, 3], 
                   xlabel="Time", 
                   ylabel="u (voltage)",
                   title="Time Series")
    
    t_array = Array(tsteps)
    for idx in 1:n_nodes
        lines!(ax_time, t_array, u_cpu[idx, :], 
               color=colors[idx], alpha=0.7, linewidth=1)
    end
    
    # 4. Poincaré sections for all oscillators (GPU-accelerated)
    ax_poincare = Axis(fig[4, 1:2], 
                       xlabel="u", 
                       ylabel="v",
                       title="Poincaré Sections (v=0, dv/dt>0)")
    
    # Compute all Poincaré sections at once using GPU
    poincare_points = compute_poincare_sections_gpu(u_matrix, v_matrix, use_gpu)
    
    for idx in 1:n_nodes
        if !isempty(poincare_points[idx])
            scatter!(ax_poincare, poincare_points[idx][:, 1], 
                    poincare_points[idx][:, 2], 
                    color=colors[idx], markersize=8.0, alpha=0.7,
                    label="Osc $(node_indices[idx])")
        end
    end
    
    if n_nodes <= 10
        axislegend(ax_poincare, position=:rt, nbanks=2)
    end
    
    # 5. 3D phase space with time derivative
    if n_nodes >= 1
        u_first = u_cpu[1, :]
        v_first = v_cpu[1, :]
        
        # Compute derivative using GPU
        if use_gpu && CUDA.functional()
            du = Array(diff(CuArray(u_first)))
        else
            du = diff(u_first)
        end
        
        ax_3d = Axis3(fig[4, 3], 
                      xlabel="u", 
                      ylabel="v", 
                      zlabel="du/dt",
                      title="3D Phase Space (Osc $(node_indices[1]))",
                      azimuth=1.5π)
        
        lines!(ax_3d, u_first[1:end-1], v_first[1:end-1], du, 
               color=1:length(du), colormap=colormap, linewidth=2)
    end
    
    fig
end

function compute_poincare_sections_gpu(u_matrix, v_matrix, use_gpu=true)
    """
    Compute Poincaré sections for all oscillators using GPU acceleration
    Returns vector of matrices, each containing (u, v) crossing points
    """
    n_nodes = size(u_matrix, 1)
    n_timesteps = size(u_matrix, 2)
    
    poincare_points = Vector{Matrix{Float64}}(undef, n_nodes)
    
    # Move to CPU for processing (GPU kernel would be more complex)
    u_cpu = Array(u_matrix)
    v_cpu = Array(v_matrix)
    
    for idx in 1:n_nodes
        v = v_cpu[idx, :]
        u = u_cpu[idx, :]
        
        # Vectorized zero-crossing detection
        dv = diff(v)
        # Find where v crosses zero with positive derivative
        crossings = findall(i -> v[i] < 0 && v[i+1] >= 0 && dv[i] > 0, 
                           1:length(v)-1)
        
        if !isempty(crossings)
            # Vectorized interpolation
            u_cross = u[crossings] .+ (u[crossings.+1] .- u[crossings]) .* 
                      (-v[crossings]) ./ (v[crossings.+1] .- v[crossings])
            v_cross = zeros(length(crossings))
            
            poincare_points[idx] = hcat(u_cross, v_cross)
        else
            poincare_points[idx] = zeros(0, 2)
        end
    end
    
    return poincare_points
end

function analyze_phase_space_statistics(sol, node_indices=1:10; use_gpu=true)
    """
    Analyze phase space with statistical measures using GPU-accelerated matrix operations
    """
    n_nodes = length(node_indices)
    n_timesteps = length(sol.t)
    
    # Extract data as matrices
    u_indices = [2*i-1 for i in node_indices]
    v_indices = [2*i for i in node_indices]
    
    sol_array = sol |> Array
    
    if use_gpu && CUDA.functional()
        u_matrix = CuArray(sol_array[u_indices, :])  # (n_nodes, n_timesteps)
        v_matrix = CuArray(sol_array[v_indices, :])  # (n_nodes, n_timesteps)
    else
        u_matrix = sol_array[u_indices, :]
        v_matrix = sol_array[v_indices, :]
    end
    
    fig = Figure(size=(1600, 1200))
    
    # 1. Compute statistics using GPU
    u_mean = vec(mean(u_matrix, dims=1))  # Mean across oscillators
    v_mean = vec(mean(v_matrix, dims=1))
    u_std = vec(std(u_matrix, dims=1))
    v_std = vec(std(v_matrix, dims=1))
    
    # Move to CPU for plotting
    u_cpu = Array(u_matrix)
    v_cpu = Array(v_matrix)
    u_mean_cpu = Array(u_mean)
    v_mean_cpu = Array(v_mean)
    u_std_cpu = Array(u_std)
    t_array = Array(sol.t)
    
    # 1. Phase portraits with mean trajectory
    ax1 = Axis(fig[1, 1:2], 
               xlabel="u (voltage)", 
               ylabel="v (recovery)",
               title="Phase Portraits with Mean Trajectory")
    
    colors = cgrad(:viridis, n_nodes, categorical=true)
    
    # Plot individual trajectories
    for idx in 1:n_nodes
        lines!(ax1, u_cpu[idx, :], v_cpu[idx, :], 
               color=(colors[idx], 0.3), linewidth=1)
    end
    
    # Plot mean trajectory
    lines!(ax1, u_mean_cpu, v_mean_cpu, color=:red, linewidth=3, label="Mean")
    axislegend(ax1)
    
    # 2. Standard deviation envelope
    ax2 = Axis(fig[1, 3], 
               xlabel="Time", 
               ylabel="u (voltage)",
               title="Mean ± Std Dev")
    
    band!(ax2, t_array, u_mean_cpu .- u_std_cpu, u_mean_cpu .+ u_std_cpu, 
          color=(:blue, 0.3), label="±1σ")
    lines!(ax2, t_array, u_mean_cpu, color=:blue, linewidth=2, label="Mean")
    axislegend(ax2)
    
    # 3. Amplitude distribution (GPU-accelerated)
    ax3 = Axis(fig[2, 1], 
               xlabel="Amplitude", 
               ylabel="Density",
               title="Amplitude Distribution")
    
    # Vectorized amplitude computation
    amplitudes = Array(maximum(u_matrix, dims=2) .- minimum(u_matrix, dims=2))
    hist!(ax3, vec(amplitudes), bins=20, color=(:blue, 0.5), normalization=:pdf)
    
    # 4. Frequency distribution
    ax4 = Axis(fig[2, 2], 
               xlabel="Frequency (Hz)", 
               ylabel="Density",
               title="Oscillation Frequency")
    
    frequencies = compute_frequencies_vectorized(u_cpu, t_array)
    
    if !isempty(frequencies)
        hist!(ax4, frequencies, bins=20, color=(:green, 0.5), normalization=:pdf)
    end
    
    # 5. Correlation matrix (GPU-accelerated)
    ax5 = Axis(fig[2, 3], 
               xlabel="Oscillator", 
               ylabel="Oscillator",
               title="Correlation Matrix",
               aspect=DataAspect())
    
    # Compute correlation using GPU
    corr_matrix = compute_correlation_gpu(u_matrix, use_gpu)
    
    hm = heatmap!(ax5, corr_matrix, colormap=:RdBu, colorrange=(-1, 1))
    Colorbar(fig[2, 4], hm, label="Correlation")
    
    # 6. Trajectory divergence over time (GPU-accelerated)
    ax6 = Axis(fig[3, 1:2], 
               xlabel="Time", 
               ylabel="Mean Pairwise Distance",
               title="Trajectory Divergence")
    
    distances = compute_pairwise_distances_gpu(u_matrix, v_matrix, use_gpu)
    
    lines!(ax6, t_array, Array(distances), color=:purple, linewidth=2)
    
    # 7. Return map for first oscillator
    ax7 = Axis(fig[3, 3], 
               xlabel="u_n", 
               ylabel="u_{n+1}",
               title="Return Map (Osc $(node_indices[1]))")
    
    peaks = findlocalmaxima(u_cpu[1, :])
    if length(peaks) > 1
        scatter!(ax7, u_cpu[1, peaks[1:end-1]], u_cpu[1, peaks[2:end]], 
                markersize=8.0, color=:orange)
    end
    
    # 8. Power spectrum (GPU-accelerated FFT)
    ax8 = Axis(fig[4, 1:2], 
               xlabel="Frequency (Hz)", 
               ylabel="Power",
               title="Average Power Spectrum",
               yscale=log10)
    
    freqs, power_avg = compute_power_spectrum_gpu(u_matrix, t_array, use_gpu)
    
    lines!(ax8, Array(freqs), Array(power_avg), color=:darkblue, linewidth=2)
    
    # 9. Phase coherence over time
    ax9 = Axis(fig[4, 3], 
               xlabel="Time", 
               ylabel="Phase Coherence",
               title="Kuramoto Order Parameter")
    
    order_param = compute_order_parameter_gpu(u_matrix, v_matrix, use_gpu)
    
    lines!(ax9, t_array, Array(order_param), color=:red, linewidth=2)
    hlines!(ax9, [0.5], color=:black, linestyle=:dash, label="R=0.5")
    axislegend(ax9)
    
    fig
end

function compute_correlation_gpu(u_matrix, use_gpu=true)
    """
    Compute correlation matrix using GPU acceleration
    """
    if use_gpu && CUDA.functional()
        # Standardize data
        u_centered = u_matrix .- mean(u_matrix, dims=2)
        u_std = std(u_matrix, dims=2)
        u_normalized = u_centered ./ (u_std .+ 1e-10)
        
        # Compute correlation: C = (1/n) * X * X^T
        n_timesteps = size(u_matrix, 2)
        corr_matrix = (u_normalized * u_normalized') ./ n_timesteps
        
        return Array(corr_matrix)
    else
        return cor(u_matrix')
    end
end

function compute_pairwise_distances_gpu(u_matrix, v_matrix, use_gpu=true)
    """
    Compute mean pairwise Euclidean distance at each timestep using GPU
    """
    n_nodes, n_timesteps = size(u_matrix)
    
    if use_gpu && CUDA.functional()
        distances = CuArray{Float64}(undef, n_timesteps)
        
        # For each timestep, compute pairwise distances
        for t in 1:n_timesteps
            u_t = u_matrix[:, t]
            v_t = v_matrix[:, t]
            
            # Compute pairwise distances using broadcasting
            # dist[i,j] = sqrt((u[i]-u[j])^2 + (v[i]-v[j])^2)
            u_diff = u_t .- u_t'  # (n_nodes, n_nodes)
            v_diff = v_t .- v_t'
            dist_matrix = sqrt.(u_diff.^2 .+ v_diff.^2)
            
            # Mean of upper triangle (excluding diagonal)
            mask = CuArray(triu(ones(n_nodes, n_nodes), 1))
            distances[t] = sum(dist_matrix .* mask) / sum(mask)
        end
        
        return distances
    else
        distances = zeros(n_timesteps)
        u_cpu = Array(u_matrix)
        v_cpu = Array(v_matrix)
        
        for t in 1:n_timesteps
            dist_sum = 0.0
            count = 0
            for i in 1:n_nodes-1
                for j in i+1:n_nodes
                    dist_sum += sqrt((u_cpu[i,t] - u_cpu[j,t])^2 + 
                                   (v_cpu[i,t] - v_cpu[j,t])^2)
                    count += 1
                end
            end
            distances[t] = dist_sum / count
        end
        
        return distances
    end
end

function compute_power_spectrum_gpu(u_matrix, t_array, use_gpu=true)
    """
    Compute average power spectrum across all oscillators using GPU FFT
    """
    
    n_nodes, n_timesteps = size(u_matrix)
    dt = t_array[2] - t_array[1]
    
    if use_gpu && CUDA.functional()
        # Compute FFT for all oscillators at once
        u_fft = fft(Array(u_matrix), 2)  # FFT along time dimension
        power = abs2.(u_fft)
        power_avg = vec(mean(power, dims=1))
        
        # Frequency array
        freqs = fftfreq(n_timesteps, 1/dt)
        
        # Take positive frequencies only
        pos_idx = 1:(n_timesteps÷2)
        return freqs[pos_idx], power_avg[pos_idx]
    else
        u_cpu = Array(u_matrix)
        u_fft = fft(u_cpu, 2)
        power = abs2.(u_fft)
        power_avg = vec(mean(power, dims=1))
        
        freqs = fftfreq(n_timesteps, 1/dt)
        pos_idx = 1:(n_timesteps÷2)
        return freqs[pos_idx], power_avg[pos_idx]
    end
end

function compute_order_parameter_gpu(u_matrix, v_matrix, use_gpu=true)
    """
    Compute Kuramoto order parameter using GPU
    R(t) = |⟨exp(iθ)⟩| where θ = atan(v, u)
    """
    n_nodes, n_timesteps = size(u_matrix)
    
    if use_gpu && CUDA.functional()
        # Compute phases
        phases = atan.(v_matrix, u_matrix)
        
        # Compute complex order parameter
        z = exp.(im .* phases)
        z_mean = mean(z, dims=1)
        order_param = abs.(z_mean)
        
        return vec(order_param)
    else
        phases = atan.(Array(v_matrix), Array(u_matrix))
        z = exp.(im .* phases)
        z_mean = mean(z, dims=1)
        order_param = abs.(z_mean)
        
        return vec(order_param)
    end
end

function compute_frequencies_vectorized(u_matrix, t_array)
    """
    Compute oscillation frequencies for all oscillators using vectorized operations
    """
    n_nodes = size(u_matrix, 1)
    frequencies = Float64[]
    
    dt = t_array[2] - t_array[1]
    
    for idx in 1:n_nodes
        u = u_matrix[idx, :]
        peaks = findlocalmaxima(u)
        
        if length(peaks) > 1
            mean_period = mean(diff(peaks)) * dt
            push!(frequencies, 1.0 / mean_period)
        end
    end
    
    return frequencies
end

function findlocalmaxima(x)
    """
    Find local maxima in a 1D array
    """
    maxima = Int[]
    for i in 2:length(x)-1
        if x[i] > x[i-1] && x[i] > x[i+1]
            push!(maxima, i)
        end
    end
    return maxima
end

end # module ChaosFHN