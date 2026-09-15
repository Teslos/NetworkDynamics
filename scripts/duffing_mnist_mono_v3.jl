# Monostable Duffing on MNIST, v3: push the cheap levers (more data + iterations).
#
# v2 reached 0.904 at 14x14 (2000 train, 500 iters) with the test still rising at
# the final iteration. This bumps the two cheap levers to gauge the near-term
# ceiling at this resolution: 500 train / 100 test per class (5000/1000) and 800
# iterations. Substrate and other hyperparameters unchanged (monostable hidden,
# softmax readout, symmetric gradient, Landau anneal, 60 hidden, batch 150,
# LR 0.006, best-checkpoint). Still 14x14-pooled (feature resolution is the ceiling).
#
# Run: julia -t auto --project=. scripts/duffing_mnist_mono_v3.jl

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
using Random, Printf, Statistics, LinearAlgebra
using OrdinaryDiffEq
using SciMLBase: get_du
using MLDatasets

EP_DUFFING_SKIP_RUN = true
include(joinpath(@__DIR__, "..", "notebooks", "EP-Duffing-Network.jl"))
include(joinpath(@__DIR__, "..", "src", "utils", "eval_protocol.jl"))
using .EvalProtocol

# Evaluation protocol (revised 2026-09-12): the checkpoint is selected on a
# stratified 20% validation split carved out of the training subset, the test
# subset (MNIST own test split) is evaluated once per seed on checkpoints fixed
# in advance, and the result is a mean over seeds. See src/utils/eval_protocol.jl.
# The earlier single-seed, test-selected number for this script was 0.917.

const SEEDS=1:parse(Int,get(ENV,"MN_V3_SEEDS","3")); const VAL_FRAC=0.2
const CLASSES=collect(0:9); const N_TRAIN_PC=500; const N_TEST_PC=100
const N_HID=60; const T_MAX=40.0; const DELTA=1.0; const BETA=0.1; const LR=0.006
const N_ITER=parse(Int,get(ENV,"MN_V3_ITER","800")); const BATCH=150; const ANNEAL_FRAC=0.30
const A_OP=0.5; const A_HI=3.0; const C_H=1.0
const OUTFILE=joinpath(@__DIR__,"..","results","ep_duffing_mnist_mono_v3_seeds.json")
const EVAL_EVERY=40; const STEADY_TOL=1e-3

println("threads = ", Threads.nthreads(), ", MNIST v3: 500tr/cls (5000), 800 iters, 60 hidden, batch ", BATCH)

function pool2x2_28(img)
    out = Matrix{Float64}(undef, 14, 14)
    @inbounds for i in 1:14, j in 1:14
        out[i,j] = (img[2i-1,2j-1]+img[2i,2j-1]+img[2i-1,2j]+img[2i,2j])/4
    end
    return vec(out)
end
tr_raw = MNIST(split=:train); te_raw = MNIST(split=:test)
function subset(ds, npc, rng)
    X = Float64.(ds.features); y = ds.targets
    feats = Vector{Float64}[]; labs = Int[]
    for c in CLASSES
        idx = shuffle(rng, findall(==(c), y))[1:npc]
        for i in idx; push!(feats, pool2x2_28(@view X[:,:,i])); push!(labs, c); end
    end
    return permutedims(reduce(hcat, feats)), labs
end
const CC = Dict(c=>j for (j,c) in enumerate(CLASSES))
# Train/validation come from MNIST's train split, test from MNIST's test split.
# The seed moves which images are drawn as well as the initialisation.
function make_split(seed)
    rng = MersenneTwister(1000 + seed)
    Xtr_all, ytr_all = subset(tr_raw, N_TRAIN_PC, rng)
    Xte_p, yte0 = subset(te_raw, N_TEST_PC, rng)
    n_val = max(1, round(Int, VAL_FRAC * N_TRAIN_PC))
    va = Int[]; tr = Int[]
    for c in CLASSES                      # `subset` returns each class contiguously
        idx = findall(==(c), ytr_all)
        append!(va, idx[1:n_val]); append!(tr, idx[n_val+1:end])
    end
    lab(v) = [CC[c] for c in v]
    return (Xtr_p=Xtr_all[tr,:], Xva_p=Xtr_all[va,:], Xte_p=Xte_p,
            Xtr=2 .* Xtr_all[tr,:] .- 1, Xva=2 .* Xtr_all[va,:] .- 1, Xte=2 .* Xte_p .- 1,
            ytr=lab(ytr_all[tr]), yva=lab(ytr_all[va]), yte=lab(yte0))
end

const N_IN=196; const N_CLS=length(CLASSES); const N=N_IN+N_HID+N_CLS
const INPUT=collect(1:N_IN); const HIDDEN=collect(N_IN+1:N_IN+N_HID); const OUT=collect(N-N_CLS+1:N)
const VAR=vcat(HIDDEN,OUT); const IS_INPUT=[i in INPUT for i in 1:N]; const IS_OUT=[i in OUT for i in 1:N]
const MASK=let M=zeros(N,N)
    for i in INPUT,j in HIDDEN;M[i,j]=1.0;M[j,i]=1.0;end
    for i in HIDDEN,j in OUT;M[i,j]=1.0;M[j,i]=1.0;end;M end
println("N=$N (196 in, $N_HID monostable-hidden, 10 linear out)")

function dforce!(du,z,p,t)
    N=p.N
    if p.beta!=0.0; m=-Inf;@inbounds for j in p.out;m=max(m,z[j]);end; s=0.0;@inbounds for j in p.out;s+=exp(z[j]-m);end; end
    @inbounds for i in 1:N
        if p.is_input[i];du[i]=0.0;du[N+i]=0.0;continue;end
        xi=z[i];acc=0.0;for j in 1:N;acc+=p.W[i,j]*z[j];end
        Fpot = p.is_out[i] ? -(1.0*xi) : -(p.c_h*xi^3+p.a_h*xi)
        F=Fpot+p.h[i]+acc; du[i]=z[N+i]; du[N+i]=-p.delta*z[N+i]+F
    end
    if p.beta!=0.0; @inbounds for (mi,j) in enumerate(p.out); pj=exp(z[j]-m)/s; du[N+j]-=p.beta*(pj-p.y[mi]); end; end
    return nothing
end
dcb()=DiscreteCallback((u,t,it)->maximum(abs,get_du(it))<STEADY_TOL,terminate!;save_positions=(false,false))
dpar(W,h,a_h,y,beta)=(N=N,W=W,h=h,a_h=a_h,c_h=C_H,delta=DELTA,beta=beta,y=y,out=OUT,is_input=IS_INPUT,is_out=IS_OUT)
function drelax(W,h,a_h,x0,Y,beta)
    nb=size(x0,1);p0=dpar(W,h,a_h,view(Y,1,:),beta)
    prob=ODEProblem(dforce!,vcat(x0[1,:],zeros(N)),(0.0,T_MAX),p0)
    pf(pr,i,rep)=remake(pr;u0=vcat(x0[i,:],zeros(N)),p=merge(p0,(y=view(Y,i,:),)))
    ens=EnsembleProblem(prob;prob_func=pf);alg=Threads.nthreads()>1 ? EnsembleThreads() : EnsembleSerial()
    sol=solve(ens,Tsit5(),alg;trajectories=nb,callback=dcb(),SOLVER_KWARGS...)
    eq=zeros(nb,N);for i in 1:nb;eq[i,:]=sol[i].u[end][1:N];end;eq
end
function dgrad(W,h,a_h,x0,Y,beta)
    xz=drelax(W,h,a_h,x0,Y,0.0); xp=drelax(W,h,a_h,xz,Y,beta); xm=drelax(W,h,a_h,xz,Y,-beta)
    nb=size(xz,1);f=1.0/(nb*2beta);gW=zeros(N,N);gh=zeros(N)
    @inbounds for d in 1:nb,i in 1:N
        gh[i]+=(xm[d,i]-xp[d,i]); for j in 1:N;gW[i,j]+=(xm[d,i]*xm[d,j]-xp[d,i]*xp[d,j]);end;end
    ce=0.0;for d in 1:nb;o=xz[d,OUT];mo=maximum(o);pe=exp.(o.-mo);pe./=sum(pe);ce+=-sum(Y[d,:].*log.(pe.+1e-12));end
    gW.*f,gh.*f,ce/nb
end
logreg_acc(Xtr,ytr,Xte,yte,nc;iters=800,lr=0.5,l2=1e-3)=begin
    n,d=size(Xtr);W=zeros(d,nc);b=zeros(nc);Y=zeros(n,nc);for i in 1:n;Y[i,ytr[i]]=1.0;end
    for _ in 1:iters;e=exp.((Xtr*W.+b').-maximum(Xtr*W.+b',dims=2));P=e./sum(e,dims=2);G=(P.-Y)./n;W.-=lr.*(Xtr'*G.+l2.*W);b.-=lr.*vec(sum(G,dims=1));end
    L=Xte*W.+b';mean([argmax(@view L[i,:]) for i in 1:size(Xte,1)].==yte) end
mlp_acc(Xtr,ytr,Xte,yte,nc,seed;h=64,iters=3000,lr=0.2,l2=1e-4)=begin
    rng=MersenneTwister(seed);n,d=size(Xtr);W1=0.1*randn(rng,d,h);b1=zeros(h);W2=0.1*randn(rng,h,nc);b2=zeros(nc);Y=zeros(n,nc);for i in 1:n;Y[i,ytr[i]]=1.0;end
    for _ in 1:iters;A1=tanh.(Xtr*W1.+b1');Lg=A1*W2.+b2';e=exp.(Lg.-maximum(Lg,dims=2));P=e./sum(e,dims=2);dL=(P.-Y)./n;gW2=A1'*dL.+l2.*W2;gb2=vec(sum(dL,dims=1));dZ1=(dL*W2').*(1 .-A1.^2);gW1=Xtr'*dZ1.+l2.*W1;gb1=vec(sum(dZ1,dims=1));W1.-=lr.*gW1;b1.-=lr.*gb1;W2.-=lr.*gW2;b2.-=lr.*gb2;end
    A1=tanh.(Xte*W1.+b1');Lg=A1*W2.+b2';mean([argmax(@view Lg[i,:]) for i in 1:size(Xte,1)].==yte) end

a_at(it)= it>=max(1,round(Int,ANNEAL_FRAC*N_ITER)) ? A_OP : A_HI+(A_OP-A_HI)*(it-1)/(max(1,round(Int,ANNEAL_FRAC*N_ITER))-1)
# `var_init` is a frozen small draw, so the score depends only on (W,h).
acc(W,h,X,y,var_init)=begin n=size(X,1);x0=zeros(n,N);x0[:,INPUT].=X;x0[:,VAR].=var_init;eq=drelax(W,h,A_OP,x0,zeros(n,N_CLS),0.0);o=eq[:,OUT];mean([argmax(@view o[i,:]) for i in 1:n].==y) end

function run_seed(seed)
    s=make_split(seed);Nd=length(s.ytr)
    Ytr=[s.ytr[i]==j ? 1.0 : 0.0 for i in eachindex(s.ytr),j in 1:N_CLS]
    rng=MersenneTwister(seed)
    W=0.1*randn(rng,N,N);W=(W+W')/2;W.*=MASK;h=zeros(N)
    init_tr=frozen_init(7000+seed,Nd,length(VAR);scale=0.1)
    init_va=frozen_init(8000+seed,length(s.yva),length(VAR);scale=0.1)
    init_te=frozen_init(9000+seed,length(s.yte),length(VAR);scale=0.1)
    sW=zeros(N,N);rW=zeros(N,N);sh=zeros(N);rh=zeros(N)
    best_va=-1.0;bW=copy(W);bh=copy(h);best_it=0
    @printf("=== seed %d: train=%d val=%d test=%d ===\n",seed,Nd,length(s.yva),length(s.yte))
    t0=time()
    for it in 1:N_ITER
        a_h=a_at(it);bidx=rand(rng,1:Nd,BATCH);x0=zeros(BATCH,N);x0[:,INPUT].=s.Xtr[bidx,:];x0[:,VAR].=0.1*randn(rng,BATCH,length(VAR))
        gW,gh,ce=dgrad(W,h,a_h,x0,Ytr[bidx,:],BETA)
        W,sW,rW=adam_update(W,gW,LR,it,sW,rW);W=(W+W')/2;W.*=MASK
        h,sh,rh=adam_update(h,gh,LR,it,sh,rh)
        if it==1||it%EVAL_EVERY==0
            va=acc(W,h,s.Xva,s.yva,init_va)
            if va>best_va;best_va=va;bW=copy(W);bh=copy(h);best_it=it;end
            @printf("  iter %d: CE %.3f (a_h=%.2f) val %.3f (best %.3f @ %d) [%.0fs]\n",it,ce,a_h,va,best_va,best_it,time()-t0)
        end
    end
    secs=time()-t0
    # The test subset is evaluated here only, on checkpoints fixed in advance.
    te_sel=acc(bW,bh,s.Xte,s.yte,init_te);te_fin=acc(W,h,s.Xte,s.yte,init_te)
    tr_acc=acc(bW,bh,s.Xtr,s.ytr,init_tr)
    lr_te=logreg_acc(s.Xtr_p,s.ytr,s.Xte_p,s.yte,N_CLS)
    ml_te=mlp_acc(s.Xtr_p,s.ytr,s.Xte_p,s.yte,N_CLS,seed)
    @printf("  seed %d done in %.0fs: train %.3f | val %.3f @ iter %d | test %.3f (final iterate %.3f) | logreg %.3f | MLP %.3f\n\n",
            seed,secs,tr_acc,best_va,best_it,te_sel,te_fin,lr_te,ml_te)
    return (seed=seed,train=tr_acc,val=best_va,selected_iter=best_it,test=te_sel,
            test_final=te_fin,logreg=lr_te,mlp=ml_te,seconds=secs)
end

results=[run_seed(seed) for seed in SEEDS]
du=[r.test for r in results];duf=[r.test_final for r in results];dutr=[r.train for r in results]
lrb=[r.logreg for r in results];mlb=[r.mlp for r in results];gap=100 .*(du.-lrb)
println("="^68)
@printf("%d seeds, 14x14 MNIST, chance %.3f. Test evaluated once per seed.\n",length(results),1/N_CLS)
println("-"^68)
@printf("%-28s | %-16s %-16s\n","model","train","test")
@printf("%-28s | %-16s %-16s\n","Duffing mono v3 (MNIST) (val-sel)",msfmt(dutr),msfmt(du))
@printf("%-28s | %-16s %-16s\n","  (final iterate)","-",msfmt(duf))
@printf("%-28s | %-16s %-16s\n","logreg 14x14","-",msfmt(lrb))
@printf("%-28s | %-16s %-16s\n","MLP 14x14","-",msfmt(mlb))
println("-"^68)
@printf("per-seed test: %s\n",join((@sprintf("%.3f",a) for a in du),", "))
@printf("Duffing - logreg, paired: %s pp (ahead on %d/%d seeds)\n",msfmt(gap;digits=2),count(>(0),gap),length(gap))
write_seed_record(OUTFILE,Dict("seeds"=>collect(SEEDS),"iterations"=>N_ITER,"hidden"=>N_HID,
    "validation_fraction"=>VAL_FRAC,"train_per_class"=>N_TRAIN_PC,"test_per_class"=>N_TEST_PC,
    "beta"=>BETA,"inputs"=>"14x14 pooled MNIST"),results)
println("\nwrote ",OUTFILE)
