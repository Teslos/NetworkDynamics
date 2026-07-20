# k=3 parity: UDE encoder with FIXED vs TRAINABLE input projection W_IN,
# against random-feature and (known) MLP baselines.
# Q: can a learned oscillator encoder solve parity that random features cannot,
#    and does making the input projection trainable unlock it?
using ComponentArrays, LinearAlgebra, Random, StableRNGs, Printf, Statistics
import Enzyme

logp(m) = (println(m); flush(stdout))

const PD=20; const PK=3; const PNOSC=8; const PH=16          # dims, parity order, #osc, NN width
const PHBIAS=0.5; const PDT=0.2; const PNS=5                 # bias, RK4 step, #steps
const PNTR=10000; const PNTE=3000
const PBATCH=256

pgen(n; rng) = begin
    X=Float64.(rand(rng,(-1.0,1.0),PD,n)); y=[(prod(@view X[1:PK,j])>0) ? 1 : 0 for j in 1:n]; (X,y)
end
pinit(rng)=ComponentArray(W1=randn(rng,Float64,PH*4).*0.1, b1=zeros(PH), W2=randn(rng,Float64,PH).*0.1,
                          b2=zeros(1), W_cls=zeros(2*2*PNOSC))   # head: 2 classes x 2*PNOSC feats

# scalar hoisted RHS
function p_rhs(φ, p, h_vec, ψ)
    Tp=promote_type(eltype(φ),eltype(p)); W1=reshape(p.W1,PH,4); b1=p.b1; W2=p.W2; b2=p.b2[1]
    out=Vector{Tp}(undef,PNOSC)
    @inbounds for i in 1:PNOSC
        si=sin(φ[i]); ci=cos(φ[i])
        bi=Vector{Tp}(undef,PH); for h in 1:PH; bi[h]=W1[h,3]*si+W1[h,4]*ci+b1[h]; end
        s=-h_vec[i]*sin(φ[i]-ψ[i])
        for j in 1:PNOSC; j==i && continue
            sj=sin(φ[j]); cj=cos(φ[j]); w=b2
            for h in 1:PH; z=W1[h,1]*sj+W1[h,2]*cj+bi[h]; w+=W2[h]*tanh(z); end
            s+=w*sin(φ[j]-φ[i])
        end
        out[i]=s
    end
    out
end
function p_settle(ψ, p)
    T=promote_type(eltype(ψ),eltype(p)); h=fill(PHBIAS,PNOSC); φ=zeros(T,PNOSC)
    for _ in 1:PNS
        k1=p_rhs(φ,p,h,ψ); k2=p_rhs(φ.+0.5*PDT.*k1,p,h,ψ); k3=p_rhs(φ.+0.5*PDT.*k2,p,h,ψ); k4=p_rhs(φ.+PDT.*k3,p,h,ψ)
        φ=φ.+(PDT/6.0).*(k1.+2.0.*k2.+2.0.*k3.+k4)
    end
    φ
end
# shared loss: differentiate w.r.t. p only (fixed W_IN) or w.r.t. p AND W_IN (trainable)
function p_loss(p, W_IN, X, y)
    B=size(X,2); W_cls=reshape(p.W_cls,2,2*PNOSC); loss=zero(promote_type(eltype(p),eltype(W_IN)))
    for n in 1:B
        ψ=W_IN*(@view X[:,n]); φ=p_settle(ψ,p); feat=vcat(sin.(φ),cos.(φ)); lg=W_cls*feat
        m=maximum(lg); lse=m+log(sum(exp.(lg.-m))); loss+=lse-lg[y[n]+1]
    end
    loss/B
end
function p_eval(p, W_IN, X, y)
    W_cls=reshape(p.W_cls,2,2*PNOSC); nc=0
    for n in 1:size(X,2)
        ψ=W_IN*(@view X[:,n]); φ=p_settle(ψ,p); lg=W_cls*vcat(sin.(φ),cos.(φ))
        (argmax(lg)-1)==y[n] && (nc+=1)
    end
    nc/size(X,2)
end

# random-feature baseline (same split)
function p_randfeat(Xtr,ytr,Xte,yte; H=1024, rng)
    W=randn(rng,H,PD)./sqrt(PD); b=randn(rng,H); ϕ(X)=tanh.(W*X.+b)
    Ftr=vcat(ϕ(Xtr),ones(1,size(Xtr,2))); Fte=vcat(ϕ(Xte),ones(1,size(Xte,2)))
    Y=zeros(2,length(ytr)); for (n,c) in enumerate(ytr); Y[c+1,n]=1.0; end
    Wr=(Y*Ftr')/(Ftr*Ftr'+1e-2*I); mean([argmax(Wr*Fte[:,n])-1 for n in 1:size(Fte,2)].==yte)
end

# train UDE; train_win toggles whether W_IN is learned
function p_train(Xtr,ytr,Xte,yte; train_win::Bool, iters=1200, lr1=0.01, lr2=0.002, clip=8.0, seed=13)
    rng=StableRNG(seed); p=pinit(rng); W_IN=randn(rng,PNOSC,PD)./sqrt(PD)
    β1=0.9;β2=0.999;ϵ=1e-8; mp=zero(p);vp=zero(p); mW=zero(W_IN);vW=zero(W_IN); t=0
    sample()=(idx=rand(rng,1:PNTR,PBATCH); (Xtr[:,idx],ytr[idx]))
    stage(lr,nit)=for it in 1:nit
        Xb,yb=sample()
        if train_win
            gp,gW=Enzyme.gradient(Enzyme.Reverse,p_loss,p,W_IN,Enzyme.Const(Xb),Enzyme.Const(yb))[1:2]
        else
            gp=Enzyme.gradient(Enzyme.Reverse,p_loss,p,Enzyme.Const(W_IN),Enzyme.Const(Xb),Enzyme.Const(yb))[1]
            gW=nothing
        end
        gn=norm(vec(gp)); gn>clip && (gp.*=clip/gn)
        t+=1
        @. mp=β1*mp+(1-β1)*gp; @. vp=β2*vp+(1-β2)*gp*gp
        @. p = p - lr*(mp/(1-β1^t))/(sqrt(vp/(1-β2^t))+ϵ)
        if train_win
            gnW=norm(vec(gW)); gnW>clip && (gW.*=clip/gnW)
            @. mW=β1*mW+(1-β1)*gW; @. vW=β2*vW+(1-β2)*gW*gW
            @. W_IN = W_IN - lr*(mW/(1-β1^t))/(sqrt(vW/(1-β2^t))+ϵ)
        end
        it%150==0 && logp(@sprintf("   [%s] it %4d loss=%.4f", train_win ? "train-Win" : "fixed-Win", t, p_loss(p,W_IN,Xb,yb)))
    end
    stage(lr1, iters÷2); stage(lr2, iters-iters÷2)
    p_eval(p,W_IN,Xte,yte)
end

try
    rng=StableRNG(1003)   # matches gap-check k=3 seed family
    Xtr,ytr=pgen(PNTR;rng=rng); Xte,yte=pgen(PNTE;rng=rng)
    logp("k=$PK parity, d=$PD, N_OSC=$PNOSC  (train=$PNTR test=$PNTE)")
    rf=p_randfeat(Xtr,ytr,Xte,yte;rng=StableRNG(7))
    logp(@sprintf("random-features: %.1f%%   (linear=~50%%, MLP=~100%% from gap-check)", 100rf))
    logp("training UDE (fixed random W_IN)...")
    a=p_train(Xtr,ytr,Xte,yte; train_win=false)
    logp(@sprintf("UDE fixed-Win  test acc: %.1f%%", 100a))
    logp("training UDE (trainable W_IN)...")
    b=p_train(Xtr,ytr,Xte,yte; train_win=true)
    logp(@sprintf("UDE train-Win  test acc: %.1f%%", 100b))
    logp(@sprintf("SUMMARY  randfeat=%.1f  UDE(fixedWin)=%.1f  UDE(trainWin)=%.1f", 100rf,100a,100b))
    logp("PARITY_DONE")
catch e
    logp("PARITY_ERROR: "*sprint(showerror,e,catch_backtrace()))
end
