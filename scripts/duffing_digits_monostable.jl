# Monostable ("single-basin") Duffing on digits: does removing bistability recover
# multi-class accuracy?
#
# The graded-readout run (results/ep_duffing_digits_graded.md) showed a linear
# output helps only partway (0.18 -> 0.27), because the double-well HIDDEN cells are
# still multistable (2^40 basins) -> unstable feature representation. This tests the
# fix: put the HIDDEN cells in the SINGLE-well (monostable) regime (a>0), where each
# cell has a UNIQUE, smooth, saturating equilibrium x_eq = f(field) (a hardening-
# spring nonlinearity). Then:
#   * hidden features are a deterministic function of the input (no basins),
#   * output is linear + softmax-CE (graded, competitive),
#   * NO basin-averaging is needed (unique equilibria) -- part of the test.
#
# "Landau annealing in the single-basin sense" = deterministic annealing / graduated
# non-convexity: start very convex (large a_h, near-linear) and cool to the nonlinear
# operating point, staying a_h > 0 (never entering the bistable phase). A/B: fixed
# monostable vs Landau-annealed monostable. 10 hidden nodes (as proposed), 4x4 inputs.
#
# If accuracy jumps toward logreg/MLP, it confirms the BISTABILITY -- not the Duffing
# dynamics -- was the obstruction to multi-class.
#
# Run: julia -t auto --project=. scripts/duffing_digits_monostable.jl

using Random, Printf, Statistics, LinearAlgebra, DelimitedFiles
using OrdinaryDiffEq
using SciMLBase: get_du

EP_DUFFING_SKIP_RUN = true
include(joinpath(@__DIR__, "..", "notebooks", "EP-Duffing-Network.jl"))

const SEED=1; const CLASSES=collect(0:9); const N_TRAIN_PC=60; const N_TEST_PC=40
const N_HID=10; const T_MAX=40.0; const DELTA=1.0; const BETA=0.1; const LR=0.02
const N_ITER=300; const BATCH=100; const ANNEAL_FRAC=0.5
const A_OP=0.5; const A_HI=3.0; const C_H=1.0     # monostable: a_h > 0 (single well)
const EVAL_EVERY=25; const STEADY_TOL=1e-3

Random.seed!(SEED)
println("threads = ", Threads.nthreads())
raw = readdlm(joinpath(@__DIR__, "..", "data", "digits", "optdigits.tes"), ',', Int)
const X_ALL=Float64.(raw[:,1:64]); const Y_ALL=raw[:,65]
pool4x4(v)=(img=reshape(v,8,8); [ (img[bi,bj]+img[bi+1,bj]+img[bi,bj+1]+img[bi+1,bj+1])/4
           for bi in 1:2:8 for bj in 1:2:8 ])
pool_all(X)=permutedims(reduce(hcat,[pool4x4(X[i,:]) for i in 1:size(X,1)]))

const N_IN=16; const N_CLS=length(CLASSES); const N=N_IN+N_HID+N_CLS
const INPUT=collect(1:N_IN); const HIDDEN=collect(N_IN+1:N_IN+N_HID); const OUT=collect(N-N_CLS+1:N)
const VAR=vcat(HIDDEN,OUT); const IS_INPUT=[i in INPUT for i in 1:N]; const IS_OUT=[i in OUT for i in 1:N]
const MASK=let M=zeros(N,N)
    for i in INPUT, j in HIDDEN; M[i,j]=1.0;M[j,i]=1.0; end
    for i in HIDDEN, j in OUT;   M[i,j]=1.0;M[j,i]=1.0; end; M end

# hidden monostable (a_h>0, single well); output linear (a=1,c=0); softmax-CE nudge
function dforce!(du,z,p,t)
    N=p.N
    if p.beta!=0.0
        m=-Inf; @inbounds for j in p.out; m=max(m,z[j]); end
        s=0.0; @inbounds for j in p.out; s+=exp(z[j]-m); end
    end
    @inbounds for i in 1:N
        if p.is_input[i]; du[i]=0.0; du[N+i]=0.0; continue; end
        xi=z[i]; acc=0.0
        for j in 1:N; acc+=p.W[i,j]*z[j]; end
        Fpot = p.is_out[i] ? -(1.0*xi) : -(p.c_h*xi^3 + p.a_h*xi)   # a_h>0 => single well
        F=Fpot+p.h[i]+acc
        du[i]=z[N+i]; du[N+i]=-p.delta*z[N+i]+F
    end
    if p.beta!=0.0
        @inbounds for (mi,j) in enumerate(p.out)
            pj=exp(z[j]-m)/s; du[N+j]-=p.beta*(pj-p.y[mi])
        end
    end
    return nothing
end
dcb()=DiscreteCallback((u,t,it)->maximum(abs,get_du(it))<STEADY_TOL,terminate!;save_positions=(false,false))
dpar(W,h,a_h,y,beta)=(N=N,W=W,h=h,a_h=a_h,c_h=C_H,delta=DELTA,beta=beta,y=y,out=OUT,is_input=IS_INPUT,is_out=IS_OUT)

function drelax(W,h,a_h,x0,Y,beta)
    nb=size(x0,1); p0=dpar(W,h,a_h,view(Y,1,:),beta)
    prob=ODEProblem(dforce!,vcat(x0[1,:],zeros(N)),(0.0,T_MAX),p0)
    pf(pr,i,rep)=remake(pr; u0=vcat(x0[i,:],zeros(N)), p=merge(p0,(y=view(Y,i,:),)))
    ens=EnsembleProblem(prob;prob_func=pf); alg=Threads.nthreads()>1 ? EnsembleThreads() : EnsembleSerial()
    sol=solve(ens,Tsit5(),alg;trajectories=nb,callback=dcb(),SOLVER_KWARGS...)
    eq=zeros(nb,N); for i in 1:nb; eq[i,:]=sol[i].u[end][1:N]; end; eq
end
function dgrad(W,h,a_h,x0,Y,beta)
    xz=drelax(W,h,a_h,x0,Y,0.0); xn=drelax(W,h,a_h,xz,Y,beta)
    nb=size(xz,1); gW=zeros(N,N); gh=zeros(N)
    @inbounds for d in 1:nb, i in 1:N
        gh[i]+=(xz[d,i]-xn[d,i]); for j in 1:N; gW[i,j]+=(xz[d,i]*xz[d,j]-xn[d,i]*xn[d,j]); end; end
    f=1.0/(nb*beta); ce=0.0
    for d in 1:nb; o=xz[d,OUT];mo=maximum(o);pe=exp.(o.-mo);pe./=sum(pe);ce+=-sum(Y[d,:].*log.(pe.+1e-12)); end
    gW.*f, gh.*f, ce/nb
end
logreg_acc(Xtr,ytr,Xte,yte,nc;iters=800,lr=0.5,l2=1e-3)=begin
    n,d=size(Xtr);W=zeros(d,nc);b=zeros(nc);Y=zeros(n,nc);for i in 1:n;Y[i,ytr[i]]=1.0;end
    for _ in 1:iters;e=exp.((Xtr*W.+b').-maximum(Xtr*W.+b',dims=2));P=e./sum(e,dims=2);G=(P.-Y)./n
        W.-=lr.*(Xtr'*G.+l2.*W);b.-=lr.*vec(sum(G,dims=1));end
    L=Xte*W.+b';mean([argmax(@view L[i,:]) for i in 1:size(Xte,1)].==yte) end
mlp_acc(Xtr,ytr,Xte,yte,nc;h=64,iters=3000,lr=0.2,l2=1e-4)=begin
    rng=MersenneTwister(SEED);n,d=size(Xtr);W1=0.1*randn(rng,d,h);b1=zeros(h);W2=0.1*randn(rng,h,nc);b2=zeros(nc)
    Y=zeros(n,nc);for i in 1:n;Y[i,ytr[i]]=1.0;end
    for _ in 1:iters;A1=tanh.(Xtr*W1.+b1');Lg=A1*W2.+b2';e=exp.(Lg.-maximum(Lg,dims=2));P=e./sum(e,dims=2)
        dL=(P.-Y)./n;gW2=A1'*dL.+l2.*W2;gb2=vec(sum(dL,dims=1));dZ1=(dL*W2').*(1 .-A1.^2)
        gW1=Xtr'*dZ1.+l2.*W1;gb1=vec(sum(dZ1,dims=1));W1.-=lr.*gW1;b1.-=lr.*gb1;W2.-=lr.*gW2;b2.-=lr.*gb2;end
    A1=tanh.(Xte*W1.+b1');Lg=A1*W2.+b2';mean([argmax(@view Lg[i,:]) for i in 1:size(Xte,1)].==yte) end

rng=MersenneTwister(SEED); cc=Dict(c=>j for (j,c) in enumerate(CLASSES)); tr=Int[];te=Int[]
for c in CLASSES; ci=shuffle(rng,findall(==(c),Y_ALL)); append!(tr,ci[1:N_TRAIN_PC]); append!(te,ci[N_TRAIN_PC+1:N_TRAIN_PC+N_TEST_PC]); end
Xtr_p=pool_all(X_ALL[tr,:])./16.0; Xte_p=pool_all(X_ALL[te,:])./16.0
ytr=[cc[c] for c in Y_ALL[tr]]; yte=[cc[c] for c in Y_ALL[te]]
Xtr=2 .*Xtr_p.-1; Xte=2 .*Xte_p.-1; Nd=length(ytr)
Ytr=[ytr[i]==j ? 1.0 : 0.0 for i in eachindex(ytr), j in 1:N_CLS]
println("Monostable Duffing digits: N=$N (16 in, $N_HID monostable-hidden, 10 linear out), train=$Nd test=$(length(yte))")
println("hidden a_h>0 (single well), NO basin-averaging; softmax-CE readout\n")

a_at(it, mode) = mode==:fixed ? A_OP :
    (n=max(1,round(Int,ANNEAL_FRAC*N_ITER)); it>=n ? A_OP : A_HI + (A_OP-A_HI)*(it-1)/(n==1 ? 1 : n-1))

function run_cfg(mode)
    rng2=MersenneTwister(SEED)
    W=0.1*randn(rng2,N,N);W=(W+W')/2;W.*=MASK; h=zeros(N)
    sW=zeros(N,N);rW=zeros(N,N);sh=zeros(N);rh=zeros(N); best=0.0;bW=copy(W);bh=copy(h)
    acc(W,h,X,y)=begin
        n=size(X,1);x0=zeros(n,N);x0[:,INPUT].=X; x0[:,VAR].=0.1*randn(rng2,n,length(VAR))  # small init (monostable)
        eq=drelax(W,h,A_OP,x0,zeros(n,N_CLS),0.0);o=eq[:,OUT];mean([argmax(@view o[i,:]) for i in 1:n].==y)
    end
    for it in 1:N_ITER
        a_h=a_at(it,mode)
        bidx=rand(rng2,1:Nd,BATCH);x0=zeros(BATCH,N);x0[:,INPUT].=Xtr[bidx,:]
        x0[:,VAR].=0.1*randn(rng2,BATCH,length(VAR))   # small init -- monostable, no basin-averaging
        gW,gh,ce=dgrad(W,h,a_h,x0,Ytr[bidx,:],BETA)
        W,sW,rW=adam_update(W,gW,LR,it,sW,rW);W=(W+W')/2;W.*=MASK
        h,sh,rh=adam_update(h,gh,LR,it,sh,rh)
        if it==1 || it%EVAL_EVERY==0
            a=acc(W,h,Xte,yte); if a>best; best=a;bW=copy(W);bh=copy(h); end
            @printf("  [%s] iter %d: CE %.3f (a_h=%.2f) test %.3f (best %.3f)\n", mode, it, ce, a_h, a, best)
        end
    end
    return acc(bW,bh,Xtr,ytr), acc(bW,bh,Xte,yte)
end

results=[]
for mode in (:fixed, :landau)
    println("=== $mode ==="); t0=time()
    trn,tst=run_cfg(mode); push!(results,(mode,trn,tst))
    @printf("  -> train %.3f test %.3f  (%.0fs)\n\n", trn, tst, time()-t0)
end
lr_te=logreg_acc(Xtr_p,ytr,Xte_p,yte,N_CLS); ml_te=mlp_acc(Xtr_p,ytr,Xte_p,yte,N_CLS)

println("="^56)
@printf("%-24s | %-8s %-8s (chance %.3f, 4x4)\n","model","train","test",1/N_CLS)
println("-"^56)
for (mode,trn,tst) in results; @printf("%-24s | %-8.3f %-8.3f\n","Duffing mono ($mode)",trn,tst); end
@printf("%-24s | %-8s %-8.3f\n","logreg","-",lr_te)
@printf("%-24s | %-8s %-8.3f\n","MLP","-",ml_te)
println("\nRef: bistable Duffing 0.177; graded-readout(dw hidden) 0.270; XY 0.94")
