# Gap-check for a synthetic k-parity task:
#   linear (ridge) vs random-features (random encoder) vs MLP (learned ceiling).
# We want a k where: linear ~ 50%, random-features leaves a gap, MLP ~ high.
using LinearAlgebra, Random, StableRNGs, Printf, Statistics
import Flux

logp(m) = (println(m); flush(stdout))

# k-parity: x in {-1,+1}^d, label = parity of the first k coords (rest are nuisance)
function gen_parity(d, k, n; rng, flip=0.0)
    X = Float32.(rand(rng, (-1.0f0, 1.0f0), d, n))
    y = [ (prod(@view X[1:k, j]) > 0) ? 1 : 0 for j in 1:n ]
    if flip > 0            # optional label noise
        for j in 1:n; rand(rng) < flip && (y[j] = 1 - y[j]); end
    end
    X, y
end

# closed-form ridge classifier on given features F (rows=features, cols=samples)
function ridge_acc(Ftr, ytr, Fte, yte; λ=1e-2)
    Ytr = zeros(2, length(ytr)); for (n,c) in enumerate(ytr); Ytr[c+1,n]=1.0; end
    W = (Ytr * Ftr') / (Ftr * Ftr' + λ*I)
    pred(F) = [argmax(W*F[:,n])-1 for n in 1:size(F,2)]
    mean(pred(Fte) .== yte)
end

lin_acc(Xtr,ytr,Xte,yte) = ridge_acc(vcat(Xtr,ones(1,size(Xtr,2))), ytr,
                                      vcat(Xte,ones(1,size(Xte,2))), yte)

function randfeat_acc(Xtr,ytr,Xte,yte,d; H=1024, rng)
    W = randn(rng, H, d) .* Float32(1/sqrt(d)); b = randn(rng, H)
    ϕ(X) = tanh.(W*X .+ b)
    ridge_acc(vcat(ϕ(Xtr),ones(1,size(Xtr,2))), ytr, vcat(ϕ(Xte),ones(1,size(Xte,2))), yte)
end

function mlp_acc(Xtr,ytr,Xte,yte,d; epochs=400)
    model = Flux.Chain(Flux.Dense(d=>128,Flux.relu), Flux.Dense(128=>128,Flux.relu), Flux.Dense(128=>2))
    Ytr = Flux.onehotbatch(ytr, 0:1)
    opt = Flux.setup(Flux.Adam(1f-3), model)
    for _ in 1:epochs
        g = Flux.gradient(m -> Flux.logitcrossentropy(m(Xtr), Ytr), model)[1]
        Flux.update!(opt, model, g)
    end
    mean(Flux.onecold(model(Xte),0:1) .== yte)
end

try
    d = 20; ntr = 10000; nte = 3000
    logp("Synthetic k-parity gap-check  (d=$d, ntr=$ntr, nte=$nte)")
    logp("k    linear%   randfeat%   MLP%")
    for k in [2, 3, 4]
        rng = StableRNG(1000+k)
        Xtr,ytr = gen_parity(d,k,ntr; rng=rng)
        Xte,yte = gen_parity(d,k,nte; rng=rng)
        al = lin_acc(Xtr,ytr,Xte,yte)
        ar = randfeat_acc(Xtr,ytr,Xte,yte,d; rng=rng)
        am = mlp_acc(Xtr,ytr,Xte,yte,d)
        logp(@sprintf("%d    %.1f      %.1f       %.1f", k, 100al, 100ar, 100am))
    end
    logp("GAP_DONE")
catch e
    logp("GAP_ERROR: " * sprint(showerror, e, catch_backtrace()))
end
