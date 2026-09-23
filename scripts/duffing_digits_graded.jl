# Graded-readout layered Duffing on digits: is the READOUT the culprit?
#
# Layered Duffing with a BISTABLE readout failed on 10-class digits (test ~0.18,
# cost stuck ~3, results/ep_duffing_digits_layered.md): 10 independent double-well
# output cells = 2^10 output basins, no competition, no graded error signal. This
# tests the diagnosis by changing ONLY the output layer, keeping everything else
# identical (layered, double-well HIDDEN cells, basin-averaging, annealing):
#
#   * HIDDEN cells: Duffing double wells (a=-s, c=s annealed) -- nonlinear features,
#     basin-averaged (full-range init).
#   * OUTPUT cells: LINEAR / graded (a=+1, c=0) -> equilibrium x_out = field, a
#     continuous readout of the hidden representation.
#   * Readout/cost: softmax over the 10 output positions + cross-entropy, so the
#     outputs COMPETE (one up, rest down) with a smooth error signal -- exactly the
#     ingredient the bistable readout lacked. EP nudge on output i: -beta (p_i - y_i).
#
# If accuracy jumps toward the logreg/MLP baselines, the readout -- not the Duffing
# dynamics -- was the bottleneck. Threaded relaxation. 4x4 inputs.
#
# Run: julia -t auto --project=. scripts/duffing_digits_graded.jl

using Random, Printf, Statistics, LinearAlgebra, DelimitedFiles
using OrdinaryDiffEq
using SciMLBase: get_du

EP_DUFFING_SKIP_RUN = true
include(joinpath(@__DIR__, "..", "notebooks", "EP-Duffing-Network.jl"))
include(joinpath(@__DIR__, "..", "src", "utils", "eval_protocol.jl"))
using .EvalProtocol

# Evaluation protocol (revised 2026-09-12): checkpoint selected on a stratified
# 20% validation split carved out of the training partition, test evaluated once
# per seed on checkpoints fixed in advance, result reported as a mean over seeds.
# See src/utils/eval_protocol.jl. The earlier single-seed, test-selected number
# for this script was 0.27.
const SEEDS=1:parse(Int,get(ENV,"DUF_GR_SEEDS","5")); const VAL_FRAC=0.2
const CLASSES=collect(0:9); const N_TRAIN_PC=60; const N_TEST_PC=40
const N_HID=40; const T_MAX=40.0; const DELTA=1.0; const BETA=0.1; const LR=0.02
const N_ITER=parse(Int,get(ENV,"DUF_GR_ITER","200")); const BATCH=100; const ANNEAL_FRAC=0.5; const S_START=0.3
const TEST_RANGE=1.5; const EVAL_EVERY=25; const STEADY_TOL=1e-3
const OUTFILE=joinpath(@__DIR__,"..","results","ep_duffing_digits_graded_seeds.json")

println("threads = ", Threads.nthreads(), ", seeds ", collect(SEEDS),
        " (checkpoint selected on validation, test evaluated once per seed)")
raw = readdlm(joinpath(@__DIR__, "..", "data", "digits", "optdigits.tes"), ',', Int)
const X_ALL = Float64.(raw[:,1:64]); const Y_ALL = raw[:,65]
pool4x4(v) = (img=reshape(v,8,8); [ (img[bi,bj]+img[bi+1,bj]+img[bi,bj+1]+img[bi+1,bj+1])/4
             for bi in 1:2:8 for bj in 1:2:8 ])
pool_all(X) = permutedims(reduce(hcat, [pool4x4(X[i,:]) for i in 1:size(X,1)]))

const N_IN=16; const N_CLS=length(CLASSES); const N=N_IN+N_HID+N_CLS
const INPUT=collect(1:N_IN); const HIDDEN=collect(N_IN+1:N_IN+N_HID); const OUT=collect(N-N_CLS+1:N)
const VAR=vcat(HIDDEN,OUT); const IS_INPUT=[i in INPUT for i in 1:N]; const IS_OUT=[i in OUT for i in 1:N]
const MASK = let M=zeros(N,N)
    for i in INPUT, j in HIDDEN; M[i,j]=1.0;M[j,i]=1.0; end
    for i in HIDDEN, j in OUT;   M[i,j]=1.0;M[j,i]=1.0; end
    M
end

# per-cell potential coeffs: hidden double-well (a_h,c_h annealed); output linear (a=1,c=0)
function dforce!(du, z, p, t)
    N=p.N
    # softmax over current output positions (for the CE nudge)
    if p.beta != 0.0
        m = -Inf; @inbounds for j in p.out; m = max(m, z[j]); end
        s = 0.0; @inbounds for j in p.out; s += exp(z[j]-m); end
    end
    @inbounds for i in 1:N
        if p.is_input[i]; du[i]=0.0; du[N+i]=0.0; continue; end
        xi=z[i]; acc=0.0
        for j in 1:N; acc += p.W[i,j]*z[j]; end
        Fpot = p.is_out[i] ? -(1.0*xi) : -(p.c_h*xi^3 + p.a_h*xi)   # output linear, hidden double-well
        F = Fpot + p.h[i] + acc
        du[i]=z[N+i]; du[N+i] = -p.delta*z[N+i] + F
    end
    if p.beta != 0.0     # softmax-CE nudge on outputs: -beta (p_i - y_i)
        @inbounds for (mi,j) in enumerate(p.out)
            pj = exp(z[j]-m)/s
            du[N+j] -= p.beta * (pj - p.y[mi])
        end
    end
    return nothing
end

dcb() = DiscreteCallback((u,t,it)->maximum(abs,get_du(it))<STEADY_TOL, terminate!; save_positions=(false,false))
dpar(W,h,a_h,c_h,y,beta) = (N=N,W=W,h=h,a_h=a_h,c_h=c_h,delta=DELTA,beta=beta,y=y,out=OUT,
                            is_input=IS_INPUT,is_out=IS_OUT)

function drelax(W,h,a_h,c_h, x0, Y, beta)
    nb=size(x0,1)
    p0=dpar(W,h,a_h,c_h, view(Y,1,:), beta)
    prob=ODEProblem(dforce!, vcat(x0[1,:],zeros(N)), (0.0,T_MAX), p0)
    pf(pr,i,rep)=remake(pr; u0=vcat(x0[i,:],zeros(N)), p=merge(p0,(y=view(Y,i,:),)))
    ens=EnsembleProblem(prob; prob_func=pf)
    alg = Threads.nthreads()>1 ? EnsembleThreads() : EnsembleSerial()
    sol=solve(ens,Tsit5(),alg; trajectories=nb, callback=dcb(), SOLVER_KWARGS...)
    eq=zeros(nb,N); for i in 1:nb; eq[i,:]=sol[i].u[end][1:N]; end
    eq
end

function dgrad(W,h,a_h,c_h, x0, Y, beta)
    xz=drelax(W,h,a_h,c_h,x0,Y,0.0); xn=drelax(W,h,a_h,c_h,xz,Y,beta)
    nb=size(xz,1); gW=zeros(N,N); gh=zeros(N)
    @inbounds for d in 1:nb, i in 1:N
        gh[i]+=(xz[d,i]-xn[d,i])
        for j in 1:N; gW[i,j]+=(xz[d,i]*xz[d,j]-xn[d,i]*xn[d,j]); end
    end
    f=1.0/(nb*beta)
    # cross-entropy of free-phase outputs, for reporting
    ce=0.0
    for d in 1:nb
        o=xz[d,OUT]; mo=maximum(o); pe=exp.(o.-mo); pe./=sum(pe)
        ce += -sum(Y[d,:].*log.(pe.+1e-12))
    end
    return gW.*f, gh.*f, ce/nb
end

logreg_acc(Xtr,ytr,Xte,yte,nc;iters=800,lr=0.5,l2=1e-3)=begin
    n,d=size(Xtr);W=zeros(d,nc);b=zeros(nc);Y=zeros(n,nc);for i in 1:n;Y[i,ytr[i]]=1.0;end
    for _ in 1:iters; e=exp.((Xtr*W.+b').-maximum(Xtr*W.+b',dims=2));P=e./sum(e,dims=2);G=(P.-Y)./n
        W.-=lr.*(Xtr'*G.+l2.*W);b.-=lr.*vec(sum(G,dims=1)); end
    L=Xte*W.+b';mean([argmax(@view L[i,:]) for i in 1:size(Xte,1)].==yte) end
mlp_acc(Xtr,ytr,Xte,yte,nc,seed;h=64,iters=3000,lr=0.2,l2=1e-4)=begin
    rng=MersenneTwister(seed);n,d=size(Xtr);W1=0.1*randn(rng,d,h);b1=zeros(h);W2=0.1*randn(rng,h,nc);b2=zeros(nc)
    Y=zeros(n,nc);for i in 1:n;Y[i,ytr[i]]=1.0;end
    for _ in 1:iters; A1=tanh.(Xtr*W1.+b1');Lg=A1*W2.+b2';e=exp.(Lg.-maximum(Lg,dims=2));P=e./sum(e,dims=2)
        dL=(P.-Y)./n;gW2=A1'*dL.+l2.*W2;gb2=vec(sum(dL,dims=1));dZ1=(dL*W2').*(1 .-A1.^2)
        gW1=Xtr'*dZ1.+l2.*W1;gb1=vec(sum(dZ1,dims=1));W1.-=lr.*gW1;b1.-=lr.*gb1;W2.-=lr.*gW2;b2.-=lr.*gb2; end
    A1=tanh.(Xte*W1.+b1');Lg=A1*W2.+b2';mean([argmax(@view Lg[i,:]) for i in 1:size(Xte,1)].==yte) end

const CC=Dict(c=>j for (j,c) in enumerate(CLASSES))
function make_split(seed)
    tr,va,te=class_split(Y_ALL,CLASSES,N_TRAIN_PC,N_TEST_PC;val_frac=VAL_FRAC,seed=seed)
    feat(idx)=pool_all(X_ALL[idx,:])./16.0; lab(idx)=[CC[c] for c in Y_ALL[idx]]
    ftr,fva,fte=feat(tr),feat(va),feat(te)
    return (Xtr_p=ftr,Xva_p=fva,Xte_p=fte,Xtr=2 .*ftr.-1,Xva=2 .*fva.-1,Xte=2 .*fte.-1,
            ytr=lab(tr),yva=lab(va),yte=lab(te))
end
println("Graded-readout layered Duffing: N=$N (16 in, $N_HID dw-hidden, 10 LINEAR out)\n")

s_at(it)= it>=max(1,round(Int,ANNEAL_FRAC*N_ITER)) ? 1.0 : S_START+(1.0-S_START)*(it-1)/(max(1,round(Int,ANNEAL_FRAC*N_ITER))-1)

# `hid_init` is a frozen full-range basin draw, so the score depends only on (W,h).
function acc(W,h,X,y,hid_init)
    n=size(X,1);x0=zeros(n,N);x0[:,INPUT].=X
    x0[:,HIDDEN].=hid_init                                          # basin init for hidden dw
    eq=drelax(W,h,-1.0,1.0,x0,zeros(n,N_CLS),0.0); o=eq[:,OUT]
    mean([argmax(@view o[i,:]) for i in 1:n].==y)
end

function run_seed(seed)
    s=make_split(seed); Nd=length(s.ytr)
    Ytr=[s.ytr[i]==j ? 1.0 : 0.0 for i in eachindex(s.ytr), j in 1:N_CLS]
    rng=MersenneTwister(seed)
    W=0.1*randn(rng,N,N);W=(W+W')/2;W.*=MASK; h=zeros(N)
    hid_init(sd,n)=TEST_RANGE.*(2 .*rand(MersenneTwister(sd),n,length(HIDDEN)).-1)
    init_tr=hid_init(7000+seed,Nd); init_va=hid_init(8000+seed,length(s.yva)); init_te=hid_init(9000+seed,length(s.yte))
    sW=zeros(N,N);rW=zeros(N,N);sh=zeros(N);rh=zeros(N)
    best_va=-1.0;bW=copy(W);bh=copy(h);best_it=0
    @printf("=== seed %d: train=%d val=%d test=%d ===\n",seed,Nd,length(s.yva),length(s.yte))
    t0=time()
    for it in 1:N_ITER
        sa=s_at(it); a_h=-sa; c_h=sa
        bidx=rand(rng,1:Nd,BATCH); x0=zeros(BATCH,N); x0[:,INPUT].=s.Xtr[bidx,:]
        x0[:,HIDDEN].=TEST_RANGE.*(2 .*rand(rng,BATCH,length(HIDDEN)).-1)   # full-range hidden basin init
        gW,gh,ce=dgrad(W,h,a_h,c_h,x0,Ytr[bidx,:],BETA)
        W,sW,rW=adam_update(W,gW,LR,it,sW,rW); W=(W+W')/2; W.*=MASK
        h,sh,rh=adam_update(h,gh,LR,it,sh,rh)
        if it==1 || it%EVAL_EVERY==0
            va=acc(W,h,s.Xva,s.yva,init_va)
            if va>best_va; best_va=va; bW=copy(W); bh=copy(h); best_it=it; end
            @printf("  iter %d: CE %.3f (s=%.2f)  val %.3f (best %.3f @ %d) [%.0fs]\n", it, ce, sa, va, best_va, best_it, time()-t0)
        end
    end
    secs=time()-t0
    # The test partition is evaluated here only, on checkpoints fixed in advance.
    te_sel=acc(bW,bh,s.Xte,s.yte,init_te); te_fin=acc(W,h,s.Xte,s.yte,init_te)
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
println("="^64)
@printf("%d seeds, 4x4-pooled, chance %.3f. Test evaluated once per seed.\n",length(results),1/N_CLS)
println("-"^64)
@printf("%-24s | %-16s %-16s\n","model","train","test")
@printf("%-24s | %-16s %-16s\n","Duffing graded (val-sel)",msfmt(dutr),msfmt(du))
@printf("%-24s | %-16s %-16s\n","  (final iterate)","-",msfmt(duf))
@printf("%-24s | %-16s %-16s\n","logreg","-",msfmt([r.logreg for r in results]))
@printf("%-24s | %-16s %-16s\n","MLP","-",msfmt([r.mlp for r in results]))
println("-"^64)
@printf("per-seed test: %s\n",join((@sprintf("%.3f",a) for a in du),", "))
println("\nRef: bistable-readout Duffing (same net) = 0.177 test (~chance)")
write_seed_record(OUTFILE,Dict("seeds"=>collect(SEEDS),"iterations"=>N_ITER,"hidden"=>N_HID,
    "validation_fraction"=>VAL_FRAC,"train_per_class"=>N_TRAIN_PC,"test_per_class"=>N_TEST_PC,
    "beta"=>BETA,"inputs"=>"4x4 pooled","readout"=>"graded softmax, double-well hidden"),results)
println("\nwrote ",OUTFILE)
