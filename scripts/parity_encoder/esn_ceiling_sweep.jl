# Ceiling experiment: sweep ESN size, hold the (already-trained) UDE fixed.
# Q: does the ESN+UDE gain grow as the plain ESN drops (more headroom)?
using ComponentArrays, LinearAlgebra, Random, StableRNGs, Printf, Statistics, JLD2
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(REPO, "src", "baselines", "baseline_utils.jl"))
include(joinpath(REPO, "src", "baselines", "baseline_models.jl"))
const BU = Main.BaselineUtils; const BM = Main.BaselineModels

logp(m) = (println(m); flush(stdout))

const N_IN=64; const N_ROWS=8; const N_COLS=8; const N_CLS=10; const N_OSC=10
const H_BIAS=0.5; const T_SETTLE=1.0; const DT_UDE=0.2; const NSTEPS=round(Int,T_SETTLE/DT_UDE)

# ---- UDE model (hoisted RHS, matching the committed file) ----
function kuramoto_rhs(φ, p, h_vec, ψ_vec)
    Tp=promote_type(eltype(φ),eltype(p)); W1=reshape(p.W1,16,4); b1=p.b1; W2=p.W2; b2=p.b2[1]
    out=Vector{Tp}(undef,N_OSC)
    @inbounds for i in 1:N_OSC
        si=sin(φ[i]); ci=cos(φ[i])
        bi=Vector{Tp}(undef,16); for h in 1:16; bi[h]=W1[h,3]*si+W1[h,4]*ci+b1[h]; end
        s=-h_vec[i]*sin(φ[i]-ψ_vec[i])
        for j in 1:N_OSC; j==i && continue
            sj=sin(φ[j]); cj=cos(φ[j]); w=b2
            for h in 1:16; z=W1[h,1]*sj+W1[h,2]*cj+bi[h]; w+=W2[h]*tanh(z); end
            s+=w*sin(φ[j]-φ[i])
        end
        out[i]=s
    end
    out
end
function settle_phases(ψ_vec, p)
    h=fill(H_BIAS,N_OSC); φ=zeros(eltype(p),N_OSC)
    for _ in 1:NSTEPS
        k1=kuramoto_rhs(φ,p,h,ψ_vec); k2=kuramoto_rhs(φ.+0.5*DT_UDE.*k1,p,h,ψ_vec)
        k3=kuramoto_rhs(φ.+0.5*DT_UDE.*k2,p,h,ψ_vec); k4=kuramoto_rhs(φ.+DT_UDE.*k3,p,h,ψ_vec)
        φ=φ.+(DT_UDE/6.0).*(k1.+2.0.*k2.+2.0.*k3.+k4)
    end
    φ
end
function ude_phase_features(p, W_IN, X)
    Nn=size(X,2); sf=zeros(N_OSC,Nn); cf=zeros(N_OSC,Nn)
    for n in 1:Nn; φ=settle_phases(W_IN*X[:,n],p); sf[:,n]=sin.(φ); cf[:,n]=cos.(φ); end
    vcat(sf,cf)
end

# ---- ESN feature extraction (matching the committed file) ----
function esn_features_rows(esn, X)
    Nr=size(esn.Wr,1); Nn=size(X,2); F=zeros(2Nr+1,Nn)
    for j in 1:Nn
        U=reshape(X[:,j],N_COLS,N_ROWS); r=zeros(Nr); S=zeros(Nr,N_ROWS)
        for t in 1:N_ROWS; pre=esn.Wr*r.+esn.Win*vcat(U[:,t],1.0); r=(1-esn.leak).*r.+esn.leak.*tanh.(pre); S[:,t]=r; end
        F[1:Nr,j]=S[:,end]; F[Nr+1:2Nr,j]=vec(mean(S,dims=2)); F[end,j]=1.0
    end
    F
end
function esn_ude_features(esn, X, F_ude, W_ctx)
    Nr=size(esn.Wr,1); Nn=size(X,2); F=zeros(2Nr+1,Nn)
    for n in 1:Nn
        ctx=W_ctx*F_ude[:,n]; U=reshape(X[:,n],N_COLS,N_ROWS); r=zeros(Nr); S=zeros(Nr,N_ROWS)
        for t in 1:N_ROWS; pre=esn.Wr*r.+esn.Win*vcat(U[:,t],1.0).+ctx; r=(1-esn.leak).*r.+esn.leak.*tanh.(pre); S[:,t]=r; end
        F[1:Nr,n]=S[:,end]; F[Nr+1:2Nr,n]=vec(mean(S,dims=2)); F[end,n]=1.0
    end
    F
end
function ridge_fit(F,y;λ=1e-3)
    Y=zeros(N_CLS,length(y)); for (n,c) in enumerate(y); Y[c+1,n]=1.0; end
    (Y*F')/(F*F'+λ*I)
end
rpred(W,F)=[argmax(W*F[:,n])-1 for n in 1:size(F,2)]
accf(ŷ,y)=mean(ŷ.==y)

try
    rng=StableRNG(42)
    X_all,y_all=BU.load_digits()
    tr,te=BU.stratified_split(y_all,0.8;rng=rng)
    sc=BU.standardize_fit(X_all[:,tr]); X_train=BU.standardize_apply(X_all[:,tr],sc); X_test=BU.standardize_apply(X_all[:,te],sc)
    y_train=y_all[tr]; y_test=y_all[te]
    logp("train=$(length(tr)) test=$(length(te))")

    # load already-trained UDE (params + input projection).
    # NOTE: requires running src/models/UDE-SubReservoir.jl first (gitignored output).
    jld = joinpath(REPO, "results", "models", "ude_subreservoir.jld2")
    isfile(jld) || error("missing $jld — run `julia --project=. src/models/UDE-SubReservoir.jl` first")
    m=JLD2.load(jld)
    trained_p=m["trained_p"]; W_IN_UDE=m["W_IN_UDE"]
    F_ude_tr=ude_phase_features(trained_p,W_IN_UDE,X_train)
    F_ude_te=ude_phase_features(trained_p,W_IN_UDE,X_test)
    # UDE-only accuracy (constant across ESN sizes)
    Wu=ridge_fit(vcat(F_ude_tr,ones(1,size(F_ude_tr,2))),y_train)
    ude_te=accf(rpred(Wu,vcat(F_ude_te,ones(1,size(F_ude_te,2)))),y_test)
    logp(@sprintf("UDE-only test acc = %.2f%% (fixed)", 100ude_te))
    logp("N_RES   plainESN%(mean±sd)   ESN+UDE%(mean±sd)    gain(pp, mean±sd)   [5 ESN seeds]")

    for N_RES in [30, 60, 120, 250, 500, 1000]
        plains=Float64[]; combs=Float64[]; gains=Float64[]
        for seed in 1:5
            rl=StableRNG(seed)
            esn=BM.build_esn(N_RES,N_COLS; spectral_radius=0.9, density=0.1, input_scale=1.0, leak=0.1, rng=rl)
            W_ctx=randn(rl,N_RES,2*N_OSC).*0.05
            Fp_tr=esn_features_rows(esn,X_train); Fp_te=esn_features_rows(esn,X_test)
            Fc_tr=esn_ude_features(esn,X_train,F_ude_tr,W_ctx); Fc_te=esn_ude_features(esn,X_test,F_ude_te,W_ctx)
            Wp=ridge_fit(Fp_tr,y_train); Wc=ridge_fit(Fc_tr,y_train)
            ap=accf(rpred(Wp,Fp_te),y_test); ac=accf(rpred(Wc,Fc_te),y_test)
            push!(plains,100ap); push!(combs,100ac); push!(gains,100(ac-ap))
        end
        logp(@sprintf("%5d    %.2f ± %.2f       %.2f ± %.2f        %+.2f ± %.2f",
            N_RES, mean(plains),std(plains), mean(combs),std(combs), mean(gains),std(gains)))
    end
    logp("EXP_DONE")
catch e
    logp("EXP_ERROR: "*sprint(showerror,e,catch_backtrace()))
end
flush(stdout)
