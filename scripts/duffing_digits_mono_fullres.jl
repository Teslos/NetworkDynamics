# Monostable Duffing digits at FULL 64px resolution: the last loose end.
#
# mono v2 (results/ep_duffing_digits_mono_v2.md) reached 0.84 = logreg on 4x4-pooled
# inputs. This runs the identical, stable monostable setup on the FULL 64 pixels (no
# pooling) -- equal footing with XY Stage 3 (which used full 64px) -- to see how far
# the monostable Duffing climbs at full resolution. Same substrate/training as mono
# v2: single-well hidden (a>0), linear/softmax readout, symmetric +-beta gradient,
# Landau annealing. Only the input resolution changes.
#
# EVALUATION PROTOCOL (revised). The first version of this script selected its
# "best checkpoint" by repeatedly scoring the TEST set and then reported that
# maximum, from a single seed, while the manuscript states every accuracy is a
# mean over seeds. Both are fixed here, following src/hybrid/readout_ablation.py
# and matching scripts/xy_digits_stage3.jl so the two substrates stay comparable:
#   * a stratified 20% VALIDATION split is carved out of the 100-image/class
#     training partition (training sees 80/class), and the checkpoint is selected
#     on validation accuracy;
#   * the test partition (70/class) is evaluated exactly once per seed, at the
#     end, for two checkpoints fixed in advance -- the validation-selected one and
#     the final iterate;
#   * SEEDS resample the split, the initialization and the batch order, and the
#     result is reported as mean +/- std; logreg/MLP are refit per seed on the
#     same reduced training split.
#
# Run: julia -t auto --project=. scripts/duffing_digits_mono_fullres.jl
#      DUF_FR_SEEDS=1 DUF_FR_ITER=50 julia -t auto --project=. scripts/duffing_digits_mono_fullres.jl

using Random, Printf, Statistics, LinearAlgebra, DelimitedFiles, JSON
using OrdinaryDiffEq
using SciMLBase: get_du

EP_DUFFING_SKIP_RUN = true
include(joinpath(@__DIR__, "..", "notebooks", "EP-Duffing-Network.jl"))

const SEEDS=1:parse(Int, get(ENV,"DUF_FR_SEEDS","5"))
const CLASSES=collect(0:9); const N_TRAIN_PC=100; const N_TEST_PC=70   # Wang split
const VAL_FRAC=0.2                                                     # readout_ablation.py
const N_HID=40; const T_MAX=40.0; const DELTA=1.0; const BETA=0.1; const LR=0.008
const N_ITER=parse(Int, get(ENV,"DUF_FR_ITER","400")); const BATCH=100; const ANNEAL_FRAC=0.4
const A_OP=0.5; const A_HI=3.0; const C_H=1.0
const EVAL_EVERY=25; const STEADY_TOL=1e-3
const OUTFILE=joinpath(@__DIR__,"..","results","ep_duffing_digits_mono_fullres_seeds.json")

println("threads = ", Threads.nthreads(), ", FULL 64px monostable Duffing, symmetric grad, hidden=", N_HID)
println("seeds = ", collect(SEEDS), ", validation fraction = ", VAL_FRAC,
        " (checkpoint selected on validation, test evaluated once per seed)")
raw=readdlm(joinpath(@__DIR__,"..","data","digits","optdigits.tes"),',',Int)
const X_ALL=Float64.(raw[:,1:64]); const Y_ALL=raw[:,65]

const N_IN=64; const N_CLS=length(CLASSES); const N=N_IN+N_HID+N_CLS
const INPUT=collect(1:N_IN); const HIDDEN=collect(N_IN+1:N_IN+N_HID); const OUT=collect(N-N_CLS+1:N)
const VAR=vcat(HIDDEN,OUT); const IS_INPUT=[i in INPUT for i in 1:N]; const IS_OUT=[i in OUT for i in 1:N]
const CLASSCOL=Dict(c=>j for (j,c) in enumerate(CLASSES))
const MASK=let M=zeros(N,N)
    for i in INPUT,j in HIDDEN;M[i,j]=1.0;M[j,i]=1.0;end
    for i in HIDDEN,j in OUT;M[i,j]=1.0;M[j,i]=1.0;end;M end

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
function dgrad(W,h,a_h,x0,Y,beta)   # symmetric +-beta
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

# 100/70-per-class split, with a stratified validation hold-out carved out of the
# training partition. The seed moves the split as well as the initialization.
function make_split(seed)
    rng=MersenneTwister(1000+seed); n_val=max(1,round(Int,VAL_FRAC*N_TRAIN_PC))
    tr=Int[];va=Int[];te=Int[]
    for c in CLASSES
        ci=shuffle(rng,findall(==(c),Y_ALL))
        append!(va,ci[1:n_val]); append!(tr,ci[n_val+1:N_TRAIN_PC])
        append!(te,ci[N_TRAIN_PC+1:N_TRAIN_PC+N_TEST_PC])
    end
    pixels(idx)=X_ALL[idx,:]./16.0; labels(idx)=[CLASSCOL[c] for c in Y_ALL[idx]]
    return (Xtr_raw=pixels(tr),Xva_raw=pixels(va),Xte_raw=pixels(te),
            Xtr=2 .*pixels(tr).-1,Xva=2 .*pixels(va).-1,Xte=2 .*pixels(te).-1,
            ytr=labels(tr),yva=labels(va),yte=labels(te))
end

a_at(it)= it>=max(1,round(Int,ANNEAL_FRAC*N_ITER)) ? A_OP : A_HI+(A_OP-A_HI)*(it-1)/(max(1,round(Int,ANNEAL_FRAC*N_ITER))-1)

# `var_init` is a frozen 0.1*randn draw so the score depends only on (W,h).
function acc(W,h,X,y,var_init)
    n=size(X,1);x0=zeros(n,N);x0[:,INPUT].=X;x0[:,VAR].=var_init
    eq=drelax(W,h,A_OP,x0,zeros(n,N_CLS),0.0);o=eq[:,OUT]
    mean([argmax(@view o[i,:]) for i in 1:n].==y)
end

function run_seed(seed)
    s=make_split(seed);Nd=length(s.ytr)
    Ytr=[s.ytr[i]==j ? 1.0 : 0.0 for i in eachindex(s.ytr),j in 1:N_CLS]
    rng=MersenneTwister(seed)
    W=0.1*randn(rng,N,N);W=(W+W')/2;W.*=MASK;h=zeros(N)
    init_tr=0.1*randn(MersenneTwister(7000+seed),Nd,length(VAR))
    init_va=0.1*randn(MersenneTwister(8000+seed),length(s.yva),length(VAR))
    init_te=0.1*randn(MersenneTwister(9000+seed),length(s.yte),length(VAR))
    sW=zeros(N,N);rW=zeros(N,N);sh=zeros(N);rh=zeros(N)
    best_va=-1.0;bW=copy(W);bh=copy(h);best_it=0;history=Any[]
    @printf("=== seed %d: N=%d (64 in, %d monostable-hidden, 10 linear out), train=%d val=%d test=%d ===\n",
            seed,N,N_HID,Nd,length(s.yva),length(s.yte))
    t0=time()
    for it in 1:N_ITER
        a_h=a_at(it);bidx=rand(rng,1:Nd,BATCH)
        x0=zeros(BATCH,N);x0[:,INPUT].=s.Xtr[bidx,:];x0[:,VAR].=0.1*randn(rng,BATCH,length(VAR))
        gW,gh,ce=dgrad(W,h,a_h,x0,Ytr[bidx,:],BETA)
        W,sW,rW=adam_update(W,gW,LR,it,sW,rW);W=(W+W')/2;W.*=MASK
        h,sh,rh=adam_update(h,gh,LR,it,sh,rh)
        if it==1||it%EVAL_EVERY==0
            va=acc(W,h,s.Xva,s.yva,init_va)
            if va>best_va;best_va=va;bW=copy(W);bh=copy(h);best_it=it;end
            push!(history,Dict("iter"=>it,"cross_entropy"=>ce,"a_h"=>a_h,"val"=>va))
            @printf("  iter %d: CE %.3f (a_h=%.2f) val %.3f (best %.3f @ %d) [%.0fs]\n",
                    it,ce,a_h,va,best_va,best_it,time()-t0)
        end
    end
    secs=time()-t0
    # The test partition is touched here only, for checkpoints fixed in advance.
    du_te_sel=acc(bW,bh,s.Xte,s.yte,init_te)
    du_te_fin=acc(W,h,s.Xte,s.yte,init_te)
    du_tr=acc(bW,bh,s.Xtr,s.ytr,init_tr)
    lr_te=logreg_acc(s.Xtr_raw,s.ytr,s.Xte_raw,s.yte,N_CLS)
    ml_te=mlp_acc(s.Xtr_raw,s.ytr,s.Xte_raw,s.yte,N_CLS,seed)
    @printf("  seed %d done in %.0fs: train %.3f | val %.3f @ iter %d | test %.3f (final iterate %.3f) | logreg %.3f | MLP %.3f\n\n",
            seed,secs,du_tr,best_va,best_it,du_te_sel,du_te_fin,lr_te,ml_te)
    return (seed=seed,train=du_tr,val=best_va,selected_iter=best_it,
            test=du_te_sel,test_final=du_te_fin,logreg=lr_te,mlp=ml_te,
            seconds=secs,history=history)
end

results=[run_seed(seed) for seed in SEEDS]

ms(v)=length(v)>1 ? @sprintf("%.3f +/- %.3f",mean(v),std(v)) : @sprintf("%.3f",only(v))
du=[r.test for r in results];duf=[r.test_final for r in results];dutr=[r.train for r in results]
lrb=[r.logreg for r in results];mlb=[r.mlp for r in results];gap=100 .*(du.-lrb)

println("="^72)
@printf("%d seeds, full 64px, chance %.3f. Test evaluated once per seed.\n",length(results),1/N_CLS)
println("-"^72)
@printf("%-28s | %-18s %-18s\n","model","train","test")
@printf("%-28s | %-18s %-18s\n","Duffing mono 64px (val-sel)",ms(dutr),ms(du))
@printf("%-28s | %-18s %-18s\n","Duffing (final iterate)","-",ms(duf))
@printf("%-28s | %-18s %-18s\n","logreg 64px","-",ms(lrb))
@printf("%-28s | %-18s %-18s\n","MLP 64px","-",ms(mlb))
println("-"^72)
@printf("per-seed test: %s\n",join((@sprintf("%.3f",a) for a in du),", "))
@printf("Duffing - logreg, paired: %s pp (Duffing ahead on %d/%d seeds)\n",
        ms(gap),count(>(0),gap),length(gap))
println("\nRef: Duffing mono 4x4 = 0.84")

open(OUTFILE,"w") do io
    JSON.print(io,Dict(
        "seeds"=>collect(SEEDS),"iterations"=>N_ITER,"hidden"=>N_HID,
        "validation_fraction"=>VAL_FRAC,"train_per_class"=>N_TRAIN_PC,
        "test_per_class"=>N_TEST_PC,"beta"=>BETA,"a_op"=>A_OP,"a_hi"=>A_HI,
        "per_seed"=>[Dict(string(k)=>getfield(r,k) for k in keys(r)) for r in results],
        "summary"=>Dict(
            "duffing_test_mean"=>mean(du),
            "duffing_test_std"=>length(du)>1 ? std(du) : NaN,
            "duffing_test_final_mean"=>mean(duf),
            "duffing_train_mean"=>mean(dutr),
            "logreg_test_mean"=>mean(lrb),"mlp_test_mean"=>mean(mlb),
            "duffing_minus_logreg_pp_mean"=>mean(gap)),
    ),2)
end
println("\nwrote ",OUTFILE)
