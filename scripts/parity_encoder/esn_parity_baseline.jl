# ESN-on-parity: does a RANDOM recurrent reservoir solve k=3 parity?
#
# Direct test of the thesis "learned oscillator encoder beats random reservoir":
# a random ESN is presented the d bits sequentially (one per timestep) and read
# out by ridge. Reservoir size is swept to show failure is NOT a capacity issue.
# Contrast (from parity_ude.jl, same data/seed): random-features 78.5%,
# UDE fixed-W_IN 50.8% (chance), UDE trainable-W_IN 100.0%, MLP ~100%.
#
# Run: julia --project=. scripts/parity_encoder/esn_parity_baseline.jl
using LinearAlgebra, Random, StableRNGs, Printf, Statistics
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(REPO, "src", "baselines", "baseline_models.jl"))
const BM = Main.BaselineModels
logp(m) = (println(m); flush(stdout))

const PD=20; const PK=3; const PNTR=6000; const PNTE=2000    # parity is representation-limited; fewer samples suffice
pgen(n; rng) = begin
    X=Float64.(rand(rng,(-1.0,1.0),PD,n)); y=[(prod(@view X[1:PK,j])>0) ? 1 : 0 for j in 1:n]; (X,y)
end

# random ESN features: feed the d bits as d timesteps of scalar input, then read
# out [final state; mean state; bias].
function esn_seq_features(esn, X)
    Nr=size(esn.Wr,1); d,N=size(X); F=zeros(2Nr+1,N)
    for j in 1:N
        r=zeros(Nr); S=zeros(Nr,d)
        for t in 1:d
            pre=esn.Wr*r .+ esn.Win*[X[t,j],1.0]
            r=(1-esn.leak).*r .+ esn.leak.*tanh.(pre); S[:,t]=r
        end
        F[1:Nr,j]=S[:,end]; F[Nr+1:2Nr,j]=vec(mean(S,dims=2)); F[end,j]=1.0
    end
    F
end
function ridge_acc(Ftr,ytr,Fte,yte; λ=1e-2)
    Y=zeros(2,length(ytr)); for (n,c) in enumerate(ytr); Y[c+1,n]=1.0; end
    W=(Y*Ftr')/(Ftr*Ftr'+λ*I); mean([argmax(W*Fte[:,n])-1 for n in 1:size(Fte,2)].==yte)
end

try
    rng=StableRNG(1003)                       # same split family as parity_ude.jl
    Xtr,ytr=pgen(PNTR;rng=rng); Xte,yte=pgen(PNTE;rng=rng)
    logp("k=$PK parity, d=$PD  — random recurrent ESN (sequential), 2 seeds/size")
    logp("N_RES    ESN acc %(mean±sd)")
    for N_RES in [100, 500, 1000, 2000]
        accs=Float64[]
        for seed in 1:2
            rl=StableRNG(seed)
            esn=BM.build_esn(N_RES, 1; spectral_radius=0.9, density=0.1, input_scale=1.0, leak=0.1, rng=rl)
            Ftr=esn_seq_features(esn,Xtr); Fte=esn_seq_features(esn,Xte)
            push!(accs, 100*ridge_acc(Ftr,ytr,Fte,yte))
        end
        logp(@sprintf("%5d      %.1f ± %.1f", N_RES, mean(accs), std(accs)))
    end
    logp("--- reference (parity_ude.jl, same task) ---")
    logp("random static features : 78.5%")
    logp("UDE, fixed random W_IN : 50.8%  (chance)")
    logp("UDE, trainable W_IN    : 100.0%")
    logp("MLP                    : ~100%")
    logp("ESN_PARITY_DONE")
catch e
    logp("ESN_PARITY_ERROR: "*sprint(showerror,e,catch_backtrace()))
end
flush(stdout)
