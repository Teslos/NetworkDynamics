# Minimal linear-projection continuous-time echo state network (LPCTESN),
# following Anantharaman et al. 2021 ("Accelerating Simulation of Stiff Nonlinear
# Systems using Continuous-Time Echo State Networks", arXiv:2010.04004).
#
# Reservoir (continuous time):   r'(t) = -leak*r + tanh(A r + W_hyb x(t))
# Readout (linear projection):   x(t) = W_out r(t),  W_out by ridge least squares
#
# A is a fixed sparse random matrix (scaled to a target spectral radius), W_hyb a
# fixed dense random input matrix; only W_out is fit. The paper's bare form uses
# leak=0; we default to leak=1 (a stable leaky-integrator reservoir).
#
# For the Lorenz *forecasting* task we run it autonomously: teacher-force the
# reservoir on the training trajectory, fit W_out, then close the loop
# (drive the reservoir with its own prediction W_out r) and integrate forward.
# This is the continuous-time analogue of the discrete ESN in baseline_models.jl,
# returning the same fields for a direct comparison.

module LPCTESN

using OrdinaryDiffEq
using LinearAlgebra
using Random
using Statistics
using SparseArrays

export lpctesn_lorenz

function _build(NR, N; spectral_radius, density, input_scale, rng)
    A = Matrix(sprandn(rng, NR, NR, density))
    rho = maximum(abs, eigvals(A))
    rho > 0 && (A .*= spectral_radius / rho)
    Whyb = input_scale .* randn(rng, NR, N)
    return A, Whyb
end

# linear interpolation of a discretely-sampled driver D (N, T) at continuous t
function _driver(D, dt)
    T = size(D, 2)
    return function (t)
        x = t / dt + 1
        i = clamp(floor(Int, x), 1, T - 1); f = x - i
        return (1 - f) .* view(D, :, i) .+ f .* view(D, :, i + 1)
    end
end

"""
LPCTESN forecast of a Lorenz trajectory `data` (3, Ttotal). Teacher-forces the
reservoir over `train_len` steps (discarding `washout`), fits a ridge readout,
then forecasts `horizon` steps autonomously. Returns
(truth, pred, valid_steps, nrmse, err) with truth/pred de-normalized.
"""
function lpctesn_lorenz(data; NR=200, spectral_radius=1.1, density=0.1, input_scale=0.5,
                        leak=1.0, gamma=10.0, lambda=0.1, noise=1e-3, dt=0.02,
                        train_len=5000, horizon=2000, washout=200, valid_thresh=0.4,
                        rng=Random.default_rng())
    mu = vec(mean(data, dims=2)); sg = vec(std(data, dims=2))
    D = (data .- mu) ./ sg
    N = size(D, 1)
    A, Whyb = _build(NR, N; spectral_radius, density, input_scale, rng)

    # teacher-forced reservoir over the training window. `gamma` sets the
    # reservoir speed (1/time-constant) so it can track the Lorenz timescale.
    drive = _driver(D[:, 1:train_len], dt)
    Ttr = (train_len - 1) * dt
    fdrv!(dr, r, p, t) = (dr .= gamma .* (-leak .* r .+ tanh.(A * r .+ Whyb * drive(t))); nothing)
    tgrid = range(0.0, Ttr; length=train_len)
    solr = solve(ODEProblem(fdrv!, zeros(NR), (0.0, Ttr)), Tsit5();
                 saveat=tgrid, abstol=1e-8, reltol=1e-8)
    R = reduce(hcat, solr.u)                       # NR × train_len

    # ridge readout against the true (normalized) trajectory (drop washout).
    # Small state noise regularizes the autonomous feedback (standard ESN trick).
    Rw = R[:, washout+1:end] .+ noise .* randn(rng, NR, train_len - washout)
    Xw = D[:, washout+1:train_len]
    Wout = (Xw * Rw') / (Rw * Rw' + lambda * I)    # N × NR

    # autonomous closed-loop forecast from the end-of-training reservoir state
    r0 = R[:, end]
    fauto!(dr, r, p, t) = (dr .= gamma .* (-leak .* r .+ tanh.(A * r .+ Whyb * (Wout * r))); nothing)
    Th = (horizon - 1) * dt
    tgrid2 = range(0.0, Th; length=horizon)
    sola = solve(ODEProblem(fauto!, r0, (0.0, Th)), Tsit5();
                 saveat=tgrid2, abstol=1e-8, reltol=1e-8)
    pred = Wout * reduce(hcat, sola.u)             # N × horizon (normalized)

    # truth aligned so pred[:,1] ≈ truth[:,1] = D[:, train_len]
    last_col = min(train_len + horizon - 1, size(D, 2))
    truth = D[:, train_len:last_col]
    h = min(size(pred, 2), size(truth, 2))
    pred = pred[:, 1:h]; truth = truth[:, 1:h]

    err = vec(sqrt.(sum((pred .- truth) .^ 2, dims=1)) ./ sqrt(mean(sum(truth .^ 2, dims=1))))
    valid = findfirst(>(valid_thresh), err)
    valid_steps = valid === nothing ? h : valid - 1
    nrmse = sqrt(mean(sum((pred .- truth) .^ 2, dims=1)) / mean(sum(truth .^ 2, dims=1)))
    return (truth=truth .* sg .+ mu, pred=pred .* sg .+ mu,
            valid_steps=valid_steps, nrmse=nrmse, err=err)
end

end # module LPCTESN
