# Does successful EP training actually remove the dependence on initialisation?
#
# The manuscript repeatedly appeals to "input-determined equilibria": the claim
# that a trained network relaxes to a fixed point set by the input rather than by
# where the relaxation happened to start. Accuracy alone cannot establish this.
# A network scored from a single initial condition can be perfectly accurate and
# still be choosing its answer from the initial state, and a bistable network
# that is right on average can be arbitrarily unreliable sample by sample.
#
# This script freezes trained networks and relaxes each input from many random
# initial states, recording four things per input:
#
#   * prediction consistency -- the fraction of initialisations that agree with
#     the most common predicted class for that input;
#   * accuracy across initialisations -- whether those predictions are correct;
#   * convergence rate -- the fraction that actually reach the residual
#     threshold rather than stopping when the integrator ran out of time;
#   * equilibrium variation -- how far apart the final states are, measured both
#     over all initialisations and over only those that agree on the prediction.
#
# The last pair is the point of the exercise. A single input-determined
# equilibrium and a consistent decision taken from many distinct equilibria look
# identical in an accuracy table and are different claims; separating them needs
# the spread of the final states, not just the spread of the decisions.
#
# As a local diagnostic the script also reports the smallest eigenvalue of the
# energy Hessian at the equilibria that were reached. This measures curvature at
# the attained fixed point: lambda_min > 0 confirms it is a proper local minimum
# rather than a saddle the integrator stalled on. It says nothing about whether
# the minimum is unique, and is not reported as if it did.
#
# Phase states are compared with circular distances throughout.
#
# Usage: julia --project=. -t auto scripts/init_dependence.jl

using Random, Printf, Statistics, LinearAlgebra

# The two notebooks both define `run_experiment`, `weights_gradient`,
# `bias_gradient`, `EP_param_gradient` and a *differing* `SOLVER_KWARGS`, so they
# cannot be included into the same namespace. One module each keeps both intact.
module XYEP
const EP_XY_SKIP_RUN = true
include(joinpath(@__DIR__, "..", "notebooks", "EP-XY-Network-Claude.jl"))
end

module DuffEP
const EP_DUFFING_SKIP_RUN = true
include(joinpath(@__DIR__, "..", "notebooks", "EP-Duffing-Network.jl"))
end

using .XYEP: SP_XY_Network, xy_force!, force_params
using .DuffEP: DuffingNetwork, random_init!, relax_batch, EP_param_gradient,
               adam_update

const RESULTDIR = joinpath(@__DIR__, "..", "results")
isdir(RESULTDIR) || mkpath(RESULTDIR)

# Overridable so the script can be smoke-tested cheaply before a full run.
const SEEDS       = 1:parse(Int, get(ENV, "ID_SEEDS", "5"))
const N_INIT      = parse(Int, get(ENV, "ID_NINIT", "50"))   # inits per pattern
const N_EPOCH_D   = parse(Int, get(ENV, "ID_EPOCH_D", "1000"))  # Duffing epochs
const N_EPOCH_XY  = parse(Int, get(ENV, "ID_EPOCH_XY", "5000")) # XY epochs
const RESIDUAL_TOL = 1e-4   # ||grad E|| below this counts as converged
# Relaxation budgets, as multiples of each substrate's published span (T = 40 for
# the Duffing networks, T = 20 for the XY network). Both substrates terminate
# their relaxation on a time limit rather than on convergence, so a single budget
# would conflate what the trained network does with what the integrator had time
# to do.
const TIME_SCALES = (1.0, 10.0)

const DATA   = Float64[-1 -1; -1 1; 1 -1; 1 1]
const TARGET = Float64[-1, 1, 1, -1]
const PATTERN_LABELS = ["(-1,-1)", "(-1,+1)", "(+1,-1)", "(+1,+1)"]

# ---------------------------------------------------------------------------
# Energy gradients and Hessians
# ---------------------------------------------------------------------------
# Duffing:  E = sum_i [c x_i^4/4 + a x_i^2/2] - (1/2) sum_ij W_ij x_i x_j
#               - sum_i h_i x_i
#           dE/dx_i    = c x_i^3 + a x_i - sum_j W_ij x_j - h_i
#           d2E/dx_i dx_j = delta_ij (3 c x_i^2 + a) - W_ij
#
# XY:       E = -(1/2) sum_jk W_jk cos(phi_j - phi_k)
#               - sum_j h_j cos(phi_j - psi_j)
#           dE/dphi_m  = sum_k W_mk sin(phi_m - phi_k) + h_m sin(phi_m - psi_m)
#           d2E/dphi_m dphi_n = -W_mn cos(phi_m - phi_n)          (n != m)
#           d2E/dphi_m^2      = sum_k W_mk cos(phi_m - phi_k)
#                               + h_m cos(phi_m - psi_m)
# Both are restricted to the variable (unclamped) indices: the clamped input
# cells are not degrees of freedom and would otherwise contribute spurious
# zero rows.

function duffing_gradient(x, W, h, a, c, var_idx)
    [c * x[i]^3 + a * x[i] - sum(W[i, j] * x[j] for j in eachindex(x)) - h[i]
     for i in var_idx]
end

function duffing_hessian(x, W, a, c, var_idx)
    n = length(var_idx)
    H = zeros(n, n)
    for (p, i) in enumerate(var_idx), (q, j) in enumerate(var_idx)
        H[p, q] = (i == j ? 3c * x[i]^2 + a : 0.0) - W[i, j]
    end
    return H
end

function xy_gradient(phi, W, h, psi, var_idx)
    [sum(W[m, k] * sin(phi[m] - phi[k]) for k in eachindex(phi)) +
     h[m] * sin(phi[m] - psi[m]) for m in var_idx]
end

function xy_hessian(phi, W, h, psi, var_idx)
    n = length(var_idx)
    H = zeros(n, n)
    for (p, m) in enumerate(var_idx), (q, nn) in enumerate(var_idx)
        if m == nn
            H[p, q] = sum(W[m, k] * cos(phi[m] - phi[k]) for k in eachindex(phi)) +
                      h[m] * cos(phi[m] - psi[m])
        else
            H[p, q] = -W[m, nn] * cos(phi[m] - phi[nn])
        end
    end
    return H
end

# Circular standard deviation: sqrt(-2 ln R) with R the mean resultant length.
# Reduces to the linear standard deviation for tightly clustered angles and
# stays bounded when they are spread over the circle.
function circular_std(angles)
    isempty(angles) && return NaN
    R = abs(sum(cis, angles)) / length(angles)
    R < 1e-12 && return sqrt(-2 * log(1e-12))
    return sqrt(-2 * log(min(R, 1.0)))
end

linear_std(values) = length(values) < 2 ? 0.0 : std(values)

# ---------------------------------------------------------------------------
# Duffing training recipes
# ---------------------------------------------------------------------------

function coupling_mask(N, input_idx, hidden_idx, output_idx, layered)
    M = zeros(N, N)
    if !layered
        M .= 1.0
        M[diagind(M)] .= 0.0
        return M
    end
    for i in input_idx, j in hidden_idx
        M[i, j] = 1.0; M[j, i] = 1.0
    end
    for i in hidden_idx, j in output_idx
        M[i, j] = 1.0; M[j, i] = 1.0
    end
    return M
end

# `basin_averaging` is Wang's remedy for EP under multistability: instead of one
# initial state per pattern, average the gradient over `minit` initial states
# drawn from the full state range, so the update reflects every basin the
# network can land in rather than the one it happens to start in. With it off,
# the network is trained from a small perturbation of the origin -- the "naive"
# configuration that fails on XOR.
function train_duffing!(net, input_idx, var_idx, M;
                        rng, n_epoch, beta=0.1, lr=0.02,
                        basin_averaging=true, minit=40, anneal_frac=0.5,
                        s_start=0.3, test_range=1.5)
    N = net.N
    s_W = zeros(N, N); r_W = zeros(N, N); s_h = zeros(N); r_h = zeros(N)
    best_cost = Inf; best_W = copy(net.W); best_h = copy(net.h)
    cost_history = zeros(n_epoch)
    rows = basin_averaging ? 4 * minit : 4

    s_at(epoch) = anneal_frac <= 0 ? 1.0 :
        (n = max(1, round(Int, anneal_frac * n_epoch));
         epoch >= n ? 1.0 : s_start + (1.0 - s_start) * (epoch - 1) / (n == 1 ? 1 : n - 1))

    for epoch in 1:n_epoch
        s = s_at(epoch); net.a = -s; net.c = s
        x0 = zeros(rows, N); tgt = zeros(rows, 1); row = 1
        for p in 1:4
            for _ in 1:(basin_averaging ? minit : 1)
                x0[row, input_idx] .= DATA[p, :]
                x0[row, var_idx] .= basin_averaging ?
                    test_range .* (2 .* rand(rng, length(var_idx)) .- 1) :
                    0.1 .* randn(rng, length(var_idx))
                tgt[row, 1] = TARGET[p]
                row += 1
            end
        end
        gW, gh, cost, _ = EP_param_gradient(net, x0, tgt, beta; symmetric=false)
        net.W, s_W, r_W = adam_update(net.W, gW, lr, epoch, s_W, r_W)
        net.W = (net.W + net.W') / 2
        net.W .*= M
        net.h, s_h, r_h = adam_update(net.h, gh, lr, epoch, s_h, r_h)
        cost_history[epoch] = cost
        if net.a == -1.0 && cost < best_cost
            best_cost = cost; best_W = copy(net.W); best_h = copy(net.h)
        end
    end
    net.a = -1.0; net.c = 1.0; net.W = best_W; net.h = best_h
    return cost_history
end

# ---------------------------------------------------------------------------
# Initialisation sweeps
# ---------------------------------------------------------------------------

struct PatternReport
    label::String
    modal_prediction::Int
    consistency::Float64
    accuracy::Float64
    convergence::Float64
    spread_all::Float64
    spread_modal::Float64
    hessian_min_eigenvalue::Float64
end

function sweep_duffing(net, input_idx, var_idx, out_idx; rng, test_range=1.5,
                       time_scale=1.0)
    base_T = net.T
    net.T = base_T * time_scale
    reports = PatternReport[]
    for p in 1:4
        predictions = zeros(Int, N_INIT)
        converged = falses(N_INIT)
        equilibria = zeros(N_INIT, length(var_idx))
        eigenvalues = Float64[]
        for k in 1:N_INIT
            x0 = zeros(1, net.N)
            x0[1, input_idx] .= DATA[p, :]
            x0[1, var_idx] .= test_range .* (2 .* rand(rng, length(var_idx)) .- 1)
            eq = relax_batch(net, x0, reshape([TARGET[p]], 1, :), 0.0)
            state = vec(eq)
            predictions[k] = Int(sign(state[out_idx[1]]))
            residual = norm(duffing_gradient(state, net.W, net.h, net.a, net.c, var_idx))
            converged[k] = residual < RESIDUAL_TOL
            equilibria[k, :] = state[var_idx]
            if converged[k]
                push!(eigenvalues,
                      minimum(eigvals(Symmetric(
                          duffing_hessian(state, net.W, net.a, net.c, var_idx)))))
            end
        end
        modal = argmax([count(==(v), predictions) for v in (-1, 0, 1)]) - 2
        consistency = count(==(modal), predictions) / N_INIT
        accuracy = count(==(Int(sign(TARGET[p]))), predictions) / N_INIT
        modal_rows = findall(==(modal), predictions)
        spread_all = mean(linear_std(equilibria[:, j]) for j in 1:length(var_idx))
        spread_modal = mean(linear_std(equilibria[modal_rows, j])
                            for j in 1:length(var_idx))
        push!(reports, PatternReport(
            PATTERN_LABELS[p], modal, consistency, accuracy,
            count(converged) / N_INIT, spread_all, spread_modal,
            isempty(eigenvalues) ? NaN : minimum(eigenvalues)))
    end
    net.T = base_T
    return reports
end

function sweep_xy(network; rng, time_scale=1.0)
    base_T = network.T
    network.T = base_T * time_scale
    W = network.weights
    h = network.bias[1, :]
    psi = network.bias[2, :]
    input_idx = network.input_index
    var_idx = network.variable_index
    out = network.output_index[1]
    reports = PatternReport[]
    for p in 1:4
        predictions = zeros(Int, N_INIT)
        converged = falses(N_INIT)
        equilibria = zeros(N_INIT, length(var_idx))
        eigenvalues = Float64[]
        for k in 1:N_INIT
            phase0 = zeros(network.N)
            phase0[input_idx] .= (π / 2) .* DATA[p, :]
            phase0[var_idx] .= 2π .* rand(rng, length(var_idx)) .- π
            state = XYEP.run_network(network, phase0, [(π / 2) * TARGET[p]]; beta=0.0)
            # Logic is encoded at phi = +-pi/2, so the decision is the sign of
            # sin(phi_out): the nearer of the two encoded phases on the circle.
            predictions[k] = Int(sign(sin(state[out])))
            residual = norm(xy_gradient(state, W, h, psi, var_idx))
            converged[k] = residual < RESIDUAL_TOL
            equilibria[k, :] = state[var_idx]
            if converged[k]
                push!(eigenvalues,
                      minimum(eigvals(Symmetric(xy_hessian(state, W, h, psi, var_idx)))))
            end
        end
        modal = argmax([count(==(v), predictions) for v in (-1, 0, 1)]) - 2
        consistency = count(==(modal), predictions) / N_INIT
        accuracy = count(==(Int(sign(TARGET[p]))), predictions) / N_INIT
        modal_rows = findall(==(modal), predictions)
        spread_all = mean(circular_std(equilibria[:, j]) for j in 1:length(var_idx))
        spread_modal = mean(circular_std(equilibria[modal_rows, j])
                            for j in 1:length(var_idx))
        push!(reports, PatternReport(
            PATTERN_LABELS[p], modal, consistency, accuracy,
            count(converged) / N_INIT, spread_all, spread_modal,
            isempty(eigenvalues) ? NaN : minimum(eigenvalues)))
    end
    network.T = base_T
    return reports
end

# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------

aggregate(reports) = (
    consistency = mean(r.consistency for r in reports),
    accuracy    = mean(r.accuracy for r in reports),
    convergence = mean(r.convergence for r in reports),
    spread_all  = mean(r.spread_all for r in reports),
    spread_modal = mean(r.spread_modal for r in reports),
    eigenvalue  = minimum(r.hessian_min_eigenvalue for r in reports),
)

function run_duffing_config(label, n_hidden, layered, basin_averaging;
                            n_epoch=N_EPOCH_D)
    N = 2 + n_hidden + 1
    input_idx = [1, 2]
    hidden_idx = collect(3:2 + n_hidden)
    out_idx = [N]
    var_idx = setdiff(1:N, input_idx)
    M = coupling_mask(N, input_idx, hidden_idx, out_idx, layered)

    per_scale = Dict(sc => Any[] for sc in TIME_SCALES)
    for seed in SEEDS
        rng = MersenneTwister(seed)
        net = DuffingNetwork(N, input_idx, out_idx; a=-1.0, c=1.0, delta=1.0)
        random_init!(net; rng=rng)
        net.W = (net.W + net.W') / 2
        net.W .*= M
        train_duffing!(net, input_idx, var_idx, M; rng=rng, n_epoch=n_epoch,
                       basin_averaging=basin_averaging)
        # One training run per seed, swept at each budget, so the only thing
        # differing between scales is how long the frozen network may relax.
        for sc in TIME_SCALES
            reports = sweep_duffing(net, input_idx, var_idx, out_idx;
                                    rng=MersenneTwister(9000 + seed), time_scale=sc)
            push!(per_scale[sc], aggregate(reports))
            @printf("  %-20s seed %d (T x%-4g): consistency %.3f acc %.3f conv %.3f\n",
                    label, seed, sc, per_scale[sc][end].consistency,
                    per_scale[sc][end].accuracy, per_scale[sc][end].convergence)
        end
        flush(stdout)
    end
    return per_scale
end

function run_xy_config(; n_epoch=N_EPOCH_XY)
    per_scale = Dict(sc => Any[] for sc in TIME_SCALES)
    for seed in SEEDS
        Random.seed!(seed)
        network, _, _ = XYEP.run_experiment(N=5, N_epoch=n_epoch)
        for sc in TIME_SCALES
            reports = sweep_xy(network; rng=MersenneTwister(9000 + seed), time_scale=sc)
            push!(per_scale[sc], aggregate(reports))
            @printf("  %-20s seed %d (T x%-4g): consistency %.3f acc %.3f conv %.3f\n",
                    "XY (phase)", seed, sc, per_scale[sc][end].consistency,
                    per_scale[sc][end].accuracy, per_scale[sc][end].convergence)
        end
        flush(stdout)
    end
    return per_scale
end

function summarise(io, label, scale, per_seed)
    m(f) = mean(getfield(s, f) for s in per_seed)
    sd(f) = length(per_seed) < 2 ? 0.0 : std(getfield(s, f) for s in per_seed)
    eigenvalues = filter(!isnan, [s.eigenvalue for s in per_seed])
    @printf(io, "| %-24s | x%-4g | %.3f ± %.3f | %.3f ± %.3f | %.3f ± %.3f | %.3f | %.3f | %s |\n",
            label, scale,
            m(:accuracy), sd(:accuracy),
            m(:consistency), sd(:consistency),
            m(:convergence), sd(:convergence),
            m(:spread_all), m(:spread_modal),
            isempty(eigenvalues) ? "n/a" : @sprintf("%+.3f", minimum(eigenvalues)))
end

const TABLE_HEADER =
    "| configuration | T | accuracy | consistency | convergence | spread(all) | spread(modal) | λ_min |"
const TABLE_RULE = "|---|---|---|---|---|---|---|---|"

function emit_table(io, results)
    println(io, TABLE_HEADER)
    println(io, TABLE_RULE)
    for (label, per_scale) in results, sc in TIME_SCALES
        summarise(io, label, sc, per_scale[sc])
    end
end

function main()
    println("Initialisation dependence of trained EP networks")
    println(length(SEEDS), " seeds x 4 XOR patterns x ", N_INIT,
            " random initial states\n")

    results = Pair{String,Any}[]

    println("XY (phase) network:")
    push!(results, "XY (phase), Wang recipe" => run_xy_config())

    println("\nDuffing networks:")
    push!(results, "Duffing, naive" =>
        run_duffing_config("Duffing naive", 2, false, false))
    push!(results, "Duffing, basin-averaged" =>
        run_duffing_config("Duffing basin-avg", 2, false, true))
    push!(results, "Duffing, layered H=6" =>
        run_duffing_config("Duffing layered", 6, true, true))

    path = joinpath(RESULTDIR, "init_dependence.md")
    open(path, "w") do io
        println(io, "# Initialisation dependence of trained EP networks\n")
        println(io, "Generated by `scripts/init_dependence.jl`. ",
                    length(SEEDS), " seeds, 4 XOR patterns, ", N_INIT,
                    " random initial states per pattern; mean ± std over seeds.\n")
        println(io, "`accuracy` is the fraction of initialisations giving the correct")
        println(io, "answer, `consistency` the fraction agreeing with the most common")
        println(io, "prediction for that input, `convergence` the fraction reaching")
        println(io, "‖∇E‖ < ", RESIDUAL_TOL, ". `spread(all)` and `spread(modal)` are the")
        println(io, "dispersion of the equilibria over all initialisations and over only")
        println(io, "those that agree on the prediction (circular s.d. for phases, linear")
        println(io, "for positions). `λ_min` is the smallest Hessian eigenvalue at the")
        println(io, "attained equilibria: a local curvature check, not a uniqueness proof;")
        println(io, "`n/a` means no relaxation met the residual threshold, so there was no")
        println(io, "equilibrium at which to evaluate it.\n")
        println(io, "`T` is the relaxation budget as a multiple of the published span")
        println(io, "(40 for the Duffing networks, 20 for the XY network). The extended")
        println(io, "row separates properties of the trained network from properties of")
        println(io, "the integrator having run out of time.\n")
        emit_table(io, results)
    end
    println("\nwrote ", path)
    println()
    emit_table(stdout, results)
end

main()
