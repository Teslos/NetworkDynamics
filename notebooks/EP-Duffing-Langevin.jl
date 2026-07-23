# Finite-temperature THERMODYNAMIC Equilibrium Propagation on a network of
# symmetrically coupled Duffing oscillators, trained on XOR.
#
# This is the stochastic-sampling counterpart of the deterministic relaxer in
# EP-Duffing-Network.jl. Instead of integrating the damped 2nd-order ODE to a
# single fixed point, we sample the OVERDAMPED LANGEVIN dynamics
#
#     dx_i = F_i(x) dt + sqrt(2T) dW_i ,   F_i = -dE/dx_i
#
# whose stationary law is the Boltzmann distribution p(x) ~ exp(-E(x)/T)
# (conditioned on the clamped input cells). Thermal noise lets the state cross
# the double-well barrier and sample BOTH wells, so the output becomes an
# input-determined distribution rather than an init-pinned point -- a principled
# replacement for the basin-averaging / Landau-annealing band-aids.
#
# Energy (identical to EP-Duffing-Network.jl):
#
#   E(x) = sum_i V(x_i) - (1/2) sum_ij W_ij x_i x_j - sum_i h_i x_i
#   V(x) = (1/4) c x^4 + (1/2) a x^2      (a<0,c>0 -> wells at x=+-sqrt(-a/c))
#   F_i  = -(c x_i^3 + a x_i) + sum_j W_ij x_j + h_i
#
# Finite-T EP gradient (thermal-average / Boltzmann-machine contrast), matching
# the sign convention of weights_gradient() in EP-Duffing-Network.jl:
#
#   dL/dW_ij ~ (1/scale) ( <x_i x_j>_free  - <x_i x_j>_nudge )
#   dL/dh_j  ~ (1/scale) ( <x_j>_free      - <x_j>_nudge     )
#
# where <.> are TIME AVERAGES over a post-burn-in Langevin window, and
# symmetric=true uses the centered +-beta estimator (scale = 2 beta).
#
# Reuses DuffingNetwork, random_init!, adam_update, batch_cost from the
# deterministic notebook.

using LinearAlgebra
using Statistics
using Random

const _HERE = @__DIR__
if !@isdefined(DuffingNetwork)
    EP_DUFFING_SKIP_RUN = true
    include(joinpath(_HERE, "EP-Duffing-Network.jl"))
end

# ----------------------------------------------------------------------------
# Overdamped Langevin sampler (hand-rolled Euler-Maruyama)
# ----------------------------------------------------------------------------
# Samples a batch of data rows simultaneously. Returns, per data row:
#   mean_x :: N_batch x N          first moments  <x_i>
#   corr   :: N_batch x N x N       second moments <x_i x_j>
#   cost   :: scalar               1/2 <(<x_out> - target)^2> over the batch
# Input cells are clamped to x0_batch; output cells receive the beta nudge.

function langevin_sample_batch(net::DuffingNetwork, x0_batch, target_batch, beta, T;
                               dt=0.02, n_burn=1000, n_sample=2000,
                               rng=Random.default_rng())
    N_batch, N = size(x0_batch)
    a, c = net.a, net.c
    W = net.W
    hrow = reshape(net.h, 1, N)
    inp = net.input_index
    out = net.output_index

    X   = copy(x0_batch)
    Xin = x0_batch[:, inp]                 # clamped input values
    @views X[:, inp] .= Xin
    sq = sqrt(2 * T * dt)

    F  = zeros(N_batch, N)
    xi = zeros(N_batch, N)

    step! = function ()
        @. F = -(c * X^3 + a * X)          # on-site double-well force
        mul!(F, X, W, 1.0, 1.0)            # + coupling sum_j W_ij x_j  (W symmetric)
        F .+= hrow                         # + local field
        if beta != 0.0
            @views F[:, out] .-= beta .* (X[:, out] .- target_batch)
        end
        randn!(rng, xi)
        @. X += F * dt + sq * xi
        @views X[:, inp] .= Xin            # re-clamp inputs each step
        return nothing
    end

    for _ in 1:n_burn
        step!()
    end

    mean_x = zeros(N_batch, N)
    corr   = zeros(N_batch, N, N)
    for _ in 1:n_sample
        step!()
        mean_x .+= X
        @inbounds for d in 1:N_batch, i in 1:N, j in 1:N
            corr[d, i, j] += X[d, i] * X[d, j]
        end
    end
    mean_x ./= n_sample
    corr ./= n_sample

    cost, _ = batch_cost(mean_x, target_batch, out)
    return mean_x, corr, cost
end

# ----------------------------------------------------------------------------
# Finite-T EP gradient
# ----------------------------------------------------------------------------
# Free phase (beta=0), then nudged phase(s) warm-started from the free means for
# variance reduction. Contrast the thermal correlations.

function EP_langevin_gradient(net::DuffingNetwork, x0_batch, target_batch, beta, T;
                              symmetric=true, dt=0.02, n_burn=1000, n_sample=2000,
                              rng=Random.default_rng())
    N_batch, N = size(x0_batch)
    kw = (dt=dt, n_burn=n_burn, n_sample=n_sample)

    # COMMON RANDOM NUMBERS: every phase must see the identical noise realization,
    # otherwise the O(1/sqrt(n_sample)) sampling noise in the free vs nudged
    # correlations does not cancel and, divided by beta, blows up as beta -> 0.
    # Draw one base seed per call from the outer rng (so successive epochs still
    # differ) and reseed each phase from it.
    base = rand(rng, UInt32)

    meanF, corrF, _ = langevin_sample_batch(net, x0_batch, target_batch, 0.0, T;
                                            kw..., rng=MersenneTwister(base))

    # Warm-start the nudged phase(s) from the free equilibrium means.
    x_start = copy(x0_batch)
    @views x_start[:, net.variable_index] .= meanF[:, net.variable_index]

    meanP, corrP, _ = langevin_sample_batch(net, x_start, target_batch, beta, T;
                                            kw..., rng=MersenneTwister(base))
    if symmetric
        meanM, corrM, _ = langevin_sample_batch(net, x_start, target_batch, -beta, T;
                                                kw..., rng=MersenneTwister(base))
        dcorr = corrM .- corrP        # <xx>_{-beta} - <xx>_{+beta}
        dmean = meanM .- meanP
        scale = 2 * beta
    else
        dcorr = corrF .- corrP        # <xx>_free - <xx>_{+beta}
        dmean = meanF .- meanP
        scale = beta
    end

    gW = zeros(N, N)
    @inbounds for d in 1:N_batch, i in 1:N, j in 1:N
        gW[i, j] += dcorr[d, i, j] / N_batch
    end
    gW ./= scale
    gW[diagind(gW)] .= 0

    gh = zeros(N)
    @inbounds for d in 1:N_batch, j in 1:N
        gh[j] += dmean[d, j] / N_batch
    end
    gh ./= scale

    cost, q_cost = batch_cost(meanF, target_batch, net.output_index)
    return gW, gh, cost, q_cost
end

# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------

function train_langevin!(net::DuffingNetwork, data, target; beta=0.1, T=0.08, lr=0.02,
                         N_epoch=800, symmetric=true, init_noise=0.3,
                         dt=0.02, n_burn=1000, n_sample=2000, print_every=50,
                         rng=Random.default_rng())
    N = net.N
    N_data = size(data, 1)
    cost_history = zeros(N_epoch)

    s_W = zeros(N, N); r_W = zeros(N, N)
    s_h = zeros(N);    r_h = zeros(N)

    best_cost = Inf
    best_W = copy(net.W); best_h = copy(net.h)

    for epoch in 1:N_epoch
        x0 = zeros(N_data, N)
        x0[:, net.input_index] .= data
        x0[:, net.variable_index] .= init_noise * randn(rng, N_data, length(net.variable_index))

        gW, gh, cost, _ = EP_langevin_gradient(net, x0, target, beta, T;
            symmetric=symmetric, dt=dt, n_burn=n_burn, n_sample=n_sample, rng=rng)

        net.W, s_W, r_W = adam_update(net.W, gW, lr, epoch, s_W, r_W)
        net.W = (net.W + net.W') / 2
        net.W[diagind(net.W)] .= 0
        net.h, s_h, r_h = adam_update(net.h, gh, lr, epoch, s_h, r_h)

        cost_history[epoch] = cost
        if cost < best_cost
            best_cost = cost
            best_W = copy(net.W); best_h = copy(net.h)
        end
        if epoch == 1 || epoch % print_every == 0
            println("Epoch $epoch: free cost = $(round(cost, sigdigits=4)) (best $(round(best_cost, sigdigits=4)))")
        end
    end

    net.W = best_W; net.h = best_h
    println("Best free-phase cost over training: ", round(best_cost, sigdigits=4))
    return cost_history
end

# Evaluate sign accuracy on the four XOR patterns with the free (beta=0) sampler,
# using full-range random inits for the free cells.
function langevin_xor_accuracy(net::DuffingNetwork, data, target; T=0.08,
                               dt=0.02, n_burn=1000, n_sample=2000,
                               init_range=1.0, rng=Random.default_rng())
    n = size(data, 1)
    outs = zeros(n)
    correct = 0
    for i in 1:n
        x0 = zeros(1, net.N)
        x0[1, net.input_index] .= data[i, :]
        x0[1, net.variable_index] .= init_range .* (2 .* rand(rng, length(net.variable_index)) .- 1)
        meanx, _, _ = langevin_sample_batch(net, x0, reshape(target[i, :], 1, :), 0.0, T;
            dt=dt, n_burn=n_burn, n_sample=n_sample, rng=rng)
        outs[i] = meanx[1, net.output_index[1]]
        correct += (sign(outs[i]) == sign(target[i, 1])) ? 1 : 0
    end
    return correct / n, outs
end

# ----------------------------------------------------------------------------
# Layered (feedforward-symmetric) coupling mask
# ----------------------------------------------------------------------------
# Restrict the symmetric coupling to input<->hidden and hidden<->output edges
# only (no input-output, no within-layer edges). The energy stays a symmetric
# gradient system (EP/Langevin valid), but the output field now comes solely
# from the input-driven hidden layer, so the output well is input-determined --
# the remedy that makes the phase-network XOR robust.
function layered_mask(N, input_index, hidden_index, output_index)
    M = falses(N, N)
    @inbounds for i in input_index, j in hidden_index
        M[i, j] = true; M[j, i] = true
    end
    @inbounds for i in hidden_index, j in output_index
        M[i, j] = true; M[j, i] = true
    end
    return M
end

# ----------------------------------------------------------------------------
# Temperature annealing (hot -> cold)
# ----------------------------------------------------------------------------
# Geometric cooling schedule T(frac) = T_hi * (T_lo/T_hi)^frac, frac in [0,1].
anneal_T(T_hi, T_lo, frac) = T_hi * (T_lo / T_hi)^frac

# Annealed training: hot epochs mix / escape wells and give lower-variance
# gradients (good statistics), cold epochs sharpen the landscape. Otherwise
# identical to train_langevin!.
function train_langevin_anneal!(net::DuffingNetwork, data, target; beta=0.1,
                                T_hi=0.20, T_lo=0.06, lr=0.03, N_epoch=300,
                                symmetric=true, init_noise=0.5, dt=0.02,
                                n_burn=1200, n_sample=2500, print_every=10^9,
                                mask=nothing, rng=Random.default_rng())
    N = net.N
    N_data = size(data, 1)
    cost_history = zeros(N_epoch)
    s_W = zeros(N, N); r_W = zeros(N, N)
    s_h = zeros(N);    r_h = zeros(N)
    # Layered: keep only the allowed edges from the start.
    mask === nothing || (net.W .*= mask)
    best_cost = Inf; best_W = copy(net.W); best_h = copy(net.h)

    for epoch in 1:N_epoch
        frac = N_epoch == 1 ? 1.0 : (epoch - 1) / (N_epoch - 1)
        T = anneal_T(T_hi, T_lo, frac)

        x0 = zeros(N_data, N)
        x0[:, net.input_index] .= data
        x0[:, net.variable_index] .= init_noise * randn(rng, N_data, length(net.variable_index))

        gW, gh, cost, _ = EP_langevin_gradient(net, x0, target, beta, T;
            symmetric=symmetric, dt=dt, n_burn=n_burn, n_sample=n_sample, rng=rng)
        mask === nothing || (gW .*= mask)

        net.W, s_W, r_W = adam_update(net.W, gW, lr, epoch, s_W, r_W)
        net.W = (net.W + net.W') / 2
        net.W[diagind(net.W)] .= 0
        mask === nothing || (net.W .*= mask)
        net.h, s_h, r_h = adam_update(net.h, gh, lr, epoch, s_h, r_h)

        cost_history[epoch] = cost
        if cost < best_cost
            best_cost = cost; best_W = copy(net.W); best_h = copy(net.h)
        end
        if epoch % print_every == 0
            println("Epoch $epoch (T=$(round(T,digits=3))): free cost = $(round(cost, sigdigits=4))")
        end
    end

    net.W = best_W; net.h = best_h
    println("Best free-phase cost over annealed training: ", round(best_cost, sigdigits=4))
    return cost_history
end

# Annealed READOUT: ramp T from T_hi down to T_lo within a single run, then
# average positions over a final cold window. The warm start lets the trained
# field pull each output to the input-selected well; cooling then commits it.
# A fixed cold readout instead would re-pin in whatever well the init landed in.
function langevin_anneal_readout(net::DuffingNetwork, x0_batch, target_batch;
                                 T_hi=0.15, T_lo=0.05, n_ramp=4000, n_read=2000,
                                 dt=0.02, beta=0.0, rng=Random.default_rng())
    N_batch, N = size(x0_batch)
    a, c = net.a, net.c
    W = net.W
    hrow = reshape(net.h, 1, N)
    inp = net.input_index
    out = net.output_index

    X = copy(x0_batch)
    Xin = x0_batch[:, inp]
    @views X[:, inp] .= Xin
    F = zeros(N_batch, N)
    xi = zeros(N_batch, N)
    mean_x = zeros(N_batch, N)

    for step in 1:(n_ramp + n_read)
        frac = min(step, n_ramp) / n_ramp
        T = anneal_T(T_hi, T_lo, frac)
        @. F = -(c * X^3 + a * X)
        mul!(F, X, W, 1.0, 1.0)
        F .+= hrow
        if beta != 0.0
            @views F[:, out] .-= beta .* (X[:, out] .- target_batch)
        end
        randn!(rng, xi)
        @. X += F * dt + sqrt(2 * T * dt) * xi
        @views X[:, inp] .= Xin
        if step > n_ramp
            mean_x .+= X
        end
    end
    mean_x ./= n_read
    return mean_x
end

function langevin_anneal_xor_accuracy(net::DuffingNetwork, data, target;
                                      T_hi=0.15, T_lo=0.05, n_ramp=4000, n_read=2000,
                                      dt=0.02, init_range=1.0, rng=Random.default_rng())
    n = size(data, 1)
    outs = zeros(n); correct = 0
    for i in 1:n
        x0 = zeros(1, net.N)
        x0[1, net.input_index] .= data[i, :]
        x0[1, net.variable_index] .= init_range .* (2 .* rand(rng, length(net.variable_index)) .- 1)
        mx = langevin_anneal_readout(net, x0, reshape(target[i, :], 1, :);
            T_hi=T_hi, T_lo=T_lo, n_ramp=n_ramp, n_read=n_read, dt=dt, rng=rng)
        outs[i] = mx[1, net.output_index[1]]
        correct += (sign(outs[i]) == sign(target[i, 1])) ? 1 : 0
    end
    return correct / n, outs
end

# ----------------------------------------------------------------------------
# Script entry point (define EP_DUFFING_LANGEVIN_SKIP_RUN = true to only load).
# ----------------------------------------------------------------------------
function run_langevin_experiment(; N=5, beta=0.1, T=0.08, lr=0.02, N_epoch=800, seed=1,
                                 a=-1.0, c=1.0, dt=0.02, n_burn=1000, n_sample=2000)
    rng = MersenneTwister(seed)
    net = DuffingNetwork(N, [1, 2], [N]; a=a, c=c)
    random_init!(net; rng=rng)

    data   = Float64[-1 -1; -1 1; 1 -1; 1 1]
    target = reshape(Float64[-1, 1, 1, -1], 4, 1)

    cost_history = train_langevin!(net, data, target; beta=beta, T=T, lr=lr, N_epoch=N_epoch,
        dt=dt, n_burn=n_burn, n_sample=n_sample, rng=rng)

    acc, outs = langevin_xor_accuracy(net, data, target; T=T, dt=dt,
        n_burn=n_burn, n_sample=n_sample, rng=rng)
    println("\nRobust XOR accuracy (free sampler, full-range init): ", round(acc * 100, digits=1), "%")
    for i in 1:4
        println("Input $(Int.(data[i, :])) => target $(Int(target[i,1])), ",
                "output $(round(outs[i], digits=3)), sign-correct: $(sign(outs[i]) == sign(target[i,1]))")
    end
    return net, cost_history, outs, acc
end

if !@isdefined(EP_DUFFING_LANGEVIN_SKIP_RUN)
    @time run_langevin_experiment()
end
