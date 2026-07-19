# Dynamics-bypass CONTROL for the parity claim.
#
# The UDE result (trainable W_IN -> 100% on k=3 parity) is only meaningful if the
# oscillator DYNAMICS do the work. But parity of 3 +/-1 bits is a single Fourier
# mode: sin(pi/2 * (x1+x2+x3)) separates the classes. So a trainable linear
# projection + sin/cos readout might solve it with NO dynamics at all.
#
# This script removes the RK4 settling entirely (phases = W_IN*x directly) and
# asks whether trainable-W_IN + sin/cos still hits ~100%. If yes, the honest claim
# shrinks from "oscillator dynamics are a capable encoder" to "trainable input +
# sinusoidal nonlinearity solves parity" (textbook). Also checks whether random
# FOURIER (sin) features beat the tanh random-features baseline (78.5%).
#
# Run: julia --project=. scripts/parity_encoder/dynamics_bypass_control.jl
using ComponentArrays, LinearAlgebra, Random, StableRNGs, Printf, Statistics
import Enzyme
logp(m) = (println(m); flush(stdout))

const PD=20; const PK=3; const PNOSC=8; const PNTR=10000; const PNTE=3000; const PBATCH=256
pgen(n; rng) = begin
    X=Float64.(rand(rng,(-1.0,1.0),PD,n)); y=[(prod(@view X[1:PK,j])>0) ? 1 : 0 for j in 1:n]; (X,y)
end

# NO dynamics: "phases" are the linear projection W_IN*x; features = sin/cos of it.
function bypass_loss(p, W_IN, X, y)
    B=size(X,2); W_cls=reshape(p.W_cls,2,2*PNOSC); loss=zero(promote_type(eltype(p),eltype(W_IN)))
    for n in 1:B
        φ=W_IN*(@view X[:,n]); feat=vcat(sin.(φ),cos.(φ)); lg=W_cls*feat
        m=maximum(lg); lse=m+log(sum(exp.(lg.-m))); loss+=lse-lg[y[n]+1]
    end
    loss/B
end
function bypass_eval(p, W_IN, X, y)
    W_cls=reshape(p.W_cls,2,2*PNOSC); nc=0
    for n in 1:size(X,2)
        φ=W_IN*(@view X[:,n]); lg=W_cls*vcat(sin.(φ),cos.(φ)); (argmax(lg)-1)==y[n] && (nc+=1)
    end
    nc/size(X,2)
end
function bypass_train(Xtr,ytr,Xte,yte; iters=800, lr1=0.02, lr2=0.005, clip=8.0, seed=13)
    rng=StableRNG(seed); p=ComponentArray(W_cls=zeros(2*2*PNOSC)); W_IN=randn(rng,PNOSC,PD)./sqrt(PD)
    β1=0.9;β2=0.999;ϵ=1e-8; mp=zero(p);vp=zero(p); mW=zero(W_IN);vW=zero(W_IN); t=0
    stage(lr,nit)=for it in 1:nit
        idx=rand(rng,1:size(Xtr,2),PBATCH); Xb=Xtr[:,idx]; yb=ytr[idx]
        gp,gW=Enzyme.gradient(Enzyme.Reverse,bypass_loss,p,W_IN,Enzyme.Const(Xb),Enzyme.Const(yb))[1:2]
        gn=norm(vec(gp)); gn>clip && (gp.*=clip/gn); gnW=norm(vec(gW)); gnW>clip && (gW.*=clip/gnW)
        t+=1
        @. mp=β1*mp+(1-β1)*gp; @. vp=β2*vp+(1-β2)*gp*gp; @. p = p - lr*(mp/(1-β1^t))/(sqrt(vp/(1-β2^t))+ϵ)
        @. mW=β1*mW+(1-β1)*gW; @. vW=β2*vW+(1-β2)*gW*gW; @. W_IN = W_IN - lr*(mW/(1-β1^t))/(sqrt(vW/(1-β2^t))+ϵ)
        it%200==0 && logp(@sprintf("   [bypass-train] it %4d loss=%.4f", t, bypass_loss(p,W_IN,Xb,yb)))
    end
    stage(lr1,iters÷2); stage(lr2,iters-iters÷2)
    bypass_eval(p,W_IN,Xte,yte)
end

# fixed-feature ridge helper
function ridge_feat(ϕ, Xtr,ytr,Xte,yte; λ=1e-2)
    Ftr=vcat(ϕ(Xtr),ones(1,size(Xtr,2))); Fte=vcat(ϕ(Xte),ones(1,size(Xte,2)))
    Y=zeros(2,length(ytr)); for (n,c) in enumerate(ytr); Y[c+1,n]=1.0; end
    W=(Y*Ftr')/(Ftr*Ftr'+λ*I); mean([argmax(W*Fte[:,n])-1 for n in 1:size(Fte,2)].==yte)
end

try
    rng=StableRNG(1003)  # same task/split as parity_ude.jl
    Xtr,ytr=pgen(PNTR;rng=rng); Xte,yte=pgen(PNTE;rng=rng)
    logp("k=$PK parity, d=$PD  — DYNAMICS-BYPASS CONTROL")

    # tanh random features (the baseline used in parity_ude, H=1024)
    let rf=StableRNG(7); W=randn(rf,1024,PD)./sqrt(PD); b=randn(rf,1024)
        logp(@sprintf("tanh random features (H=1024, fixed)     : %.1f%%",
            100*ridge_feat(X->tanh.(W*X.+b), Xtr,ytr,Xte,yte)))
    end
    # random FOURIER features (sin/cos), same width — is tanh baseline flattering?
    let rf=StableRNG(7); W=randn(rf,512,PD)./sqrt(PD)
        logp(@sprintf("random FOURIER features (sin/cos, H=1024) : %.1f%%",
            100*ridge_feat(X->vcat(sin.(W*X),cos.(W*X)), Xtr,ytr,Xte,yte)))
    end
    # dynamics-bypass, FIXED random W_IN (N_OSC=8) + ridge on sin/cos
    let rf=StableRNG(13); W=randn(rf,PNOSC,PD)./sqrt(PD)
        logp(@sprintf("bypass, FIXED random W_IN (sin/cos, ridge): %.1f%%",
            100*ridge_feat(X->vcat(sin.(W*X),cos.(W*X)), Xtr,ytr,Xte,yte)))
    end
    # THE CONTROL: dynamics-bypass, TRAINABLE W_IN (no RK4, no oscillators)
    logp("training dynamics-bypass with TRAINABLE W_IN (no dynamics)...")
    acc=bypass_train(Xtr,ytr,Xte,yte)
    logp(@sprintf("bypass, TRAINABLE W_IN (sin/cos, no dynamics): %.1f%%", 100acc))
    logp("--- reference (with dynamics, parity_ude.jl) ---")
    logp("UDE trainable W_IN (WITH dynamics): 100.0%")
    logp("BYPASS_DONE")
catch e
    logp("BYPASS_ERROR: "*sprint(showerror,e,catch_backtrace()))
end
flush(stdout)
