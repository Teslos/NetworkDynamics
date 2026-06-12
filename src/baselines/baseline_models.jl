# Baseline classifiers and an echo-state network, as required by the critique
# (docs/critique_chaotic_oscillator_networks.md, sections 1 and 6):
#   - multinomial logistic regression   (linear softmax + L2)
#   - linear SVM                         (linear + multiclass hinge + L2)
#   - small MLP                          (one hidden layer)
#   - standard tanh echo-state network   (reservoir + ridge readout)
#
# The same ESN is reused for the Lorenz time-series baseline.

module BaselineModels

using Flux
using LinearAlgebra
using Random
using Statistics

export train_logreg, train_linear_svm, train_mlp, predict_nn,
       ESN, build_esn, esn_features, esn_classify, lorenz_data, esn_lorenz

# ---------------------------------------------------------------------------
# Shared neural-net training (full-batch Adam with L2). X is (features, N),
# Y is one-hot (K, N). Returns a function mapping X -> class indices.
# ---------------------------------------------------------------------------

function _train!(model, X, Y, lossfn; epochs, lr, l2, rng)
    opt = Flux.setup(Flux.Adam(lr), model)
    Xf = Float32.(X)
    for _ in 1:epochs
        loss, grads = Flux.withgradient(model) do m
            ŷ = m(Xf)
            lossfn(ŷ, Y) + l2 * sum(p -> sum(abs2, p), Flux.trainables(m))
        end
        Flux.update!(opt, model, grads[1])
    end
    return model
end

predict_nn(model, X) = vec(map(argmax, eachcol(model(Float32.(X)))))

"Multinomial logistic regression (linear + softmax cross-entropy)."
function train_logreg(X, Y; epochs=400, lr=0.05, l2=1e-4, rng=Random.default_rng())
    d, K = size(X, 1), size(Y, 1)
    model = Dense(d => K)
    _train!(model, X, Y, Flux.logitcrossentropy; epochs, lr, l2, rng)
    return model
end

# Crammer-Singer multiclass hinge: mean over samples of
#   max(0, 1 + max_{j != y} s_j - s_y)
function _multiclass_hinge(scores, Y)
    sy = sum(scores .* Y, dims=1)                 # score of true class (1, N)
    masked = scores .- 1f6 .* Y                   # blank out true class
    smax = maximum(masked, dims=1)                # best wrong class (1, N)
    return mean(max.(0f0, 1f0 .+ smax .- sy))
end

"Linear SVM (linear scores + multiclass hinge + L2)."
function train_linear_svm(X, Y; epochs=400, lr=0.05, l2=1e-3, rng=Random.default_rng())
    d, K = size(X, 1), size(Y, 1)
    model = Dense(d => K)
    _train!(model, X, Y, _multiclass_hinge; epochs, lr, l2, rng)
    return model
end

"Small MLP: one hidden layer (relu) + softmax cross-entropy."
function train_mlp(X, Y; hidden=128, epochs=400, lr=0.01, l2=1e-4, rng=Random.default_rng())
    d, K = size(X, 1), size(Y, 1)
    model = Chain(Dense(d => hidden, relu), Dense(hidden => K))
    _train!(model, X, Y, Flux.logitcrossentropy; epochs, lr, l2, rng)
    return model
end

# ---------------------------------------------------------------------------
# Echo-state network (standard tanh reservoir, ridge readout).
# ---------------------------------------------------------------------------

struct ESN
    Win::Matrix{Float64}   # Nr x (din + 1)  (input + bias)
    Wr::Matrix{Float64}    # Nr x Nr
    leak::Float64
end

"""
Build a tanh ESN with `Nr` reservoir units, input dimension `din`.
`spectral_radius` rescales Wr (echo-state property: keep < 1), `density` is the
reservoir connection fraction, `input_scale` scales Win, `leak` is the leaking
rate. Randomness is seeded by `rng` (this is the per-seed source of dispersion).
"""
function build_esn(Nr, din; spectral_radius=0.9, density=0.1, input_scale=1.0,
                   leak=1.0, rng=Random.default_rng())
    Win = input_scale .* (2 .* rand(rng, Nr, din + 1) .- 1)
    Wr = randn(rng, Nr, Nr) .* (rand(rng, Nr, Nr) .< density)
    rho = maximum(abs, eigvals(Wr))
    rho > 0 && (Wr .*= spectral_radius / rho)
    return ESN(Win, Wr, leak)
end

# Drive the reservoir with input sequence U (din, T); return all states (Nr, T).
function _run(esn::ESN, U)
    Nr = size(esn.Wr, 1)
    T = size(U, 2)
    X = zeros(Nr, T)
    x = zeros(Nr)
    @inbounds for t in 1:T
        u = @view U[:, t]
        pre = esn.Wr * x .+ esn.Win * vcat(u, 1.0)
        x = (1 - esn.leak) .* x .+ esn.leak .* tanh.(pre)
        X[:, t] = x
    end
    return X
end

"""
Reservoir features for a batch of *static* samples X (features, N). Each sample
is presented as a length-`feat` scalar input sequence (one feature per tick);
the readout features are [final state; mean state; 1] per sample (size 2Nr+1).
This is the standard way to use an ESN as a classifier of vectors.
"""
function esn_features(esn::ESN, X)
    d, N = size(X)
    Nr = size(esn.Wr, 1)
    F = zeros(2Nr + 1, N)
    for j in 1:N
        S = _run(esn, reshape(X[:, j], 1, d))   # (Nr, d)
        F[1:Nr, j] = S[:, end]
        F[Nr+1:2Nr, j] = vec(mean(S, dims=2))
        F[end, j] = 1.0
    end
    return F
end

# Ridge readout, closed form: Wout = Y Fᵀ (F Fᵀ + λI)^{-1}.
function _ridge(F, Y; lambda=1e-4)
    G = F * F' + lambda * I
    return (Y * F') / G
end

"""
Train+evaluate the ESN classifier. Returns predicted class indices for the test
set. `Ytr` is one-hot (K, Ntr). Reservoir features are standardized using train
statistics before the ridge readout.
"""
function esn_classify(esn::ESN, Xtr, Ytr, Xte; lambda=1e-4)
    Ftr = esn_features(esn, Xtr)
    Fte = esn_features(esn, Xte)
    mu = vec(mean(Ftr, dims=2)); sg = vec(std(Ftr, dims=2)); sg[sg .== 0] .= 1
    Ftr = (Ftr .- mu) ./ sg; Ftr[end, :] .= 1.0
    Fte = (Fte .- mu) ./ sg; Fte[end, :] .= 1.0
    Wout = _ridge(Ftr, Float64.(Ytr); lambda=lambda)
    return vec(map(argmax, eachcol(Wout * Fte)))
end

# ---------------------------------------------------------------------------
# Lorenz system + ESN one-step / autonomous prediction baseline.
# ---------------------------------------------------------------------------

"Integrate the Lorenz '63 system (σ=10, ρ=28, β=8/3) with RK4. Returns (3, T)."
function lorenz_data(; dt=0.02, T=10000, x0=[1.0, 1.0, 1.0],
                     sigma=10.0, rho=28.0, beta=8/3, transient=1000)
    f(s) = [sigma * (s[2] - s[1]), s[1] * (rho - s[3]) - s[2], s[1] * s[2] - beta * s[3]]
    s = copy(x0)
    out = zeros(3, T)
    for t in 1:(T + transient)
        k1 = f(s); k2 = f(s .+ dt/2 .* k1); k3 = f(s .+ dt/2 .* k2); k4 = f(s .+ dt .* k3)
        s = s .+ dt/6 .* (k1 .+ 2k2 .+ 2k3 .+ k4)
        t > transient && (out[:, t - transient] = s)
    end
    return out
end

"""
Train an ESN to predict the Lorenz trajectory and roll it out autonomously.
Returns (truth, prediction, valid_time_steps, nrmse) where prediction is the
free-running forecast over the test horizon. `washout` discards initial
transient; readout is ridge on [state; input; 1].
"""
function esn_lorenz(data; Nr=400, spectral_radius=0.95, density=0.1, input_scale=0.5,
                    leak=1.0, lambda=1e-6, washout=200, train_len=5000, horizon=2000,
                    valid_thresh=0.4, rng=Random.default_rng())
    # normalize each coordinate
    mu = vec(mean(data, dims=2)); sg = vec(std(data, dims=2))
    D = (data .- mu) ./ sg

    esn = build_esn(Nr, 3; spectral_radius, density, input_scale, leak, rng)

    # teacher forcing over the training window
    Utr = D[:, 1:train_len]
    Ytr = D[:, 2:train_len+1]
    Str = _run(esn, Utr)
    Φ = vcat(Str, Utr, ones(1, train_len))[:, washout+1:end]
    Yt = Ytr[:, washout+1:end]
    Wout = (Yt * Φ') / (Φ * Φ' + lambda * I)

    # autonomous rollout from the end of training
    x = Str[:, end]
    u = D[:, train_len+1]
    truth = D[:, train_len+1:train_len+horizon]
    pred = zeros(3, horizon)
    for t in 1:horizon
        x = (1 - leak) .* x .+ leak .* tanh.(esn.Wr * x .+ esn.Win * vcat(u, 1.0))
        yhat = Wout * vcat(x, u, 1.0)
        pred[:, t] = yhat
        u = yhat
    end

    # metrics (in normalized coordinates)
    err = vec(sqrt.(sum((pred .- truth).^2, dims=1)) ./ sqrt(mean(sum(truth.^2, dims=1))))
    valid = findfirst(>(valid_thresh), err)
    valid_steps = valid === nothing ? horizon : valid - 1
    nrmse = sqrt(mean(sum((pred .- truth).^2, dims=1)) / mean(sum(truth.^2, dims=1)))

    # de-normalize for plotting
    return (truth = truth .* sg .+ mu, pred = pred .* sg .+ mu,
            valid_steps = valid_steps, nrmse = nrmse, err = err)
end

end # module BaselineModels
