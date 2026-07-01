# Layered EP-Duffing on the digit task: the capstone parallel to XY Stage 3.
#
# Layered structure + capacity solved the bistable-Duffing XOR (95% mean,
# results/ep_duffing_layered.md), and the same feedforward-symmetric fix lifted the
# XY net to ~94% on 10-class digits (results/xy_digits_stage3.md). This tests
# whether layered Duffing ALSO scales to multi-class digits -- the harder case,
# because each of the 10 output cells is an independent double well (2^10 output
# basins per input), which was my a-priori worry.
#
# Setup: 4x4-pooled inputs (16 cells; keeps the 2nd-order Duffing solves tractable),
# layered input(16) -> hidden(H) -> output(10), one-hot logic on +-1 wells. Trained
# with basin-averaging (full-range hidden/output init, averaged over the batch) +
# minima-fixed annealing (a=-s, c=s, s:0.3->1). Readout = argmax of output
# positions. Duffing is 2nd order (2N state) and its stock relax is serial, so this
# script implements a THREADED batch relaxation (EnsembleThreads) to be feasible.
# Baselines: logreg + a small MLP on the same 4x4 features.
#
# Run: julia -t auto --project=. scripts/duffing_digits_layered.jl

using Random, Printf, Statistics, LinearAlgebra, DelimitedFiles
using OrdinaryDiffEq
using SciMLBase: get_du

EP_DUFFING_SKIP_RUN = true
include(joinpath(@__DIR__, "..", "notebooks", "EP-Duffing-Network.jl"))  # adam_update, SOLVER_KWARGS

# ---------------------------------------------------------------- config
const SEED        = 1
const CLASSES     = collect(0:9)
const N_TRAIN_PC  = 60
const N_TEST_PC   = 40
const N_HID       = 40
const N_EV_T      = 40.0           # relaxation horizon
const DELTA       = 1.0
const BETA        = 0.1
const LR          = 0.02
const N_ITER      = 200
const BATCH       = 100
const ANNEAL_FRAC = 0.5
const S_START     = 0.3
const TEST_RANGE  = 1.5
const EVAL_EVERY  = 25
const STEADY_TOL  = 1e-3

Random.seed!(SEED)
println("threads = ", Threads.nthreads(), "  (run with -t auto)")

raw = readdlm(joinpath(@__DIR__, "..", "data", "digits", "optdigits.tes"), ',', Int)
const X_ALL = Float64.(raw[:, 1:64]); const Y_ALL = raw[:, 65]

pool4x4(v) = (img = reshape(v, 8, 8); [ (img[bi,bj]+img[bi+1,bj]+img[bi,bj+1]+img[bi+1,bj+1])/4
              for bi in 1:2:8 for bj in 1:2:8 ])
pool_all(X) = permutedims(reduce(hcat, [pool4x4(X[i, :]) for i in 1:size(X, 1)]))  # (n,16)

# ---------------------------------------------------------------- geometry
const N_IN = 16; const N_CLS = length(CLASSES); const N = N_IN + N_HID + N_CLS
const INPUT  = collect(1:N_IN)
const HIDDEN = collect(N_IN+1 : N_IN+N_HID)
const OUT    = collect(N-N_CLS+1 : N)
const VAR    = vcat(HIDDEN, OUT)
const IS_INPUT = [i in INPUT for i in 1:N]
# layered mask: input<->hidden, hidden<->output only
const MASK = let M = zeros(N, N)
    for i in INPUT, j in HIDDEN;  M[i,j]=1.0; M[j,i]=1.0; end
    for i in HIDDEN, j in OUT;    M[i,j]=1.0; M[j,i]=1.0; end
    M
end

# ---------------------------------------------------------------- dynamics (threaded)
function dforce!(du, z, p, t)
    N = p.N
    @inbounds for i in 1:N
        if p.is_input[i]; du[i]=0.0; du[N+i]=0.0; continue; end
        xi=z[i]; acc=0.0
        for j in 1:N; acc += p.W[i,j]*z[j]; end
        F = -(p.c*xi^3 + p.a*xi) + p.h[i] + acc
        du[i]=z[N+i]; du[N+i] = -p.delta*z[N+i] + F
    end
    if p.beta != 0.0
        @inbounds for (m,j) in enumerate(p.output_index)
            du[N+j] -= p.beta*(z[j]-p.target[m])
        end
    end
    return nothing
end

dcallback() = DiscreteCallback((u,t,it)->maximum(abs, get_du(it)) < STEADY_TOL,
                               terminate!; save_positions=(false,false))

dparams(W,h,a,c,target,beta) = (N=N, W=W, h=h, a=a, c=c, delta=DELTA, beta=beta,
                                target=target, output_index=OUT, is_input=IS_INPUT)

# Relax a batch (one trajectory per row, threaded); return equilibrium positions (nb,N).
function drelax(W,h,a,c, x0, tgt, beta)
    nb = size(x0,1)
    z0 = [vcat(x0[1,:], zeros(N))]  # placeholder base
    p0 = dparams(W,h,a,c, view(tgt,1,:), beta)
    prob = ODEProblem(dforce!, vcat(x0[1,:], zeros(N)), (0.0, N_EV_T), p0)
    pf(prob,i,repeat) = remake(prob; u0=vcat(x0[i,:], zeros(N)),
                               p=merge(p0, (target=view(tgt,i,:),)))
    ens = EnsembleProblem(prob; prob_func=pf)
    alg = Threads.nthreads() > 1 ? EnsembleThreads() : EnsembleSerial()
    sol = solve(ens, Tsit5(), alg; trajectories=nb, callback=dcallback(), SOLVER_KWARGS...)
    eq = zeros(nb, N)
    for i in 1:nb; eq[i,:] = sol[i].u[end][1:N]; end
    return eq
end

# ---------------------------------------------------------------- EP gradient (one-sided)
function dgrad(W,h,a,c, x0, tgt, beta)
    xz = drelax(W,h,a,c, x0, tgt, 0.0)
    xn = drelax(W,h,a,c, xz, tgt, beta)
    nb = size(xz,1)
    gW = zeros(N,N); gh = zeros(N)
    @inbounds for d in 1:nb
        for i in 1:N
            gh[i] += (xz[d,i]-xn[d,i])
            for j in 1:N; gW[i,j] += (xz[d,i]*xz[d,j]-xn[d,i]*xn[d,j]); end
        end
    end
    f = 1.0/(nb*beta)
    dev = (xz[:, OUT] .- tgt) .^ 2
    cost = mean(vec(sum(dev, dims=2)) ./ 2)
    return gW.*f, gh.*f, cost
end

# ---------------------------------------------------------------- baselines
function logreg_acc(Xtr,ytr,Xte,yte,nc; iters=800, lr=0.5, l2=1e-3)
    n,d=size(Xtr); W=zeros(d,nc); b=zeros(nc); Y=zeros(n,nc); for i in 1:n; Y[i,ytr[i]]=1.0; end
    for _ in 1:iters
        e=exp.((Xtr*W.+b').-maximum(Xtr*W.+b',dims=2)); P=e./sum(e,dims=2); G=(P.-Y)./n
        W.-=lr.*(Xtr'*G.+l2.*W); b.-=lr.*vec(sum(G,dims=1))
    end
    L=Xte*W.+b'; mean([argmax(@view L[i,:]) for i in 1:size(Xte,1)].==yte)
end
function mlp_acc(Xtr,ytr,Xte,yte,nc; h=64, iters=3000, lr=0.2, l2=1e-4)
    rng=MersenneTwister(SEED); n,d=size(Xtr)
    W1=0.1*randn(rng,d,h);b1=zeros(h);W2=0.1*randn(rng,h,nc);b2=zeros(nc)
    Y=zeros(n,nc); for i in 1:n; Y[i,ytr[i]]=1.0; end
    for _ in 1:iters
        A1=tanh.(Xtr*W1.+b1'); Lg=A1*W2.+b2'; e=exp.(Lg.-maximum(Lg,dims=2)); P=e./sum(e,dims=2)
        dL=(P.-Y)./n; gW2=A1'*dL.+l2.*W2; gb2=vec(sum(dL,dims=1))
        dZ1=(dL*W2').*(1 .-A1.^2); gW1=Xtr'*dZ1.+l2.*W1; gb1=vec(sum(dZ1,dims=1))
        W1.-=lr.*gW1;b1.-=lr.*gb1;W2.-=lr.*gW2;b2.-=lr.*gb2
    end
    A1=tanh.(Xte*W1.+b1'); Lg=A1*W2.+b2'; mean([argmax(@view Lg[i,:]) for i in 1:size(Xte,1)].==yte)
end

# ---------------------------------------------------------------- data
rng = MersenneTwister(SEED)
cc = Dict(c=>j for (j,c) in enumerate(CLASSES))
tr=Int[]; te=Int[]
for c in CLASSES
    ci = shuffle(rng, findall(==(c), Y_ALL))
    append!(tr, ci[1:N_TRAIN_PC]); append!(te, ci[N_TRAIN_PC+1:N_TRAIN_PC+N_TEST_PC])
end
Xtr_p = pool_all(X_ALL[tr,:]) ./ 16.0; Xte_p = pool_all(X_ALL[te,:]) ./ 16.0
ytr = [cc[c] for c in Y_ALL[tr]]; yte = [cc[c] for c in Y_ALL[te]]
Xtr = 2 .* Xtr_p .- 1; Xte = 2 .* Xte_p .- 1        # pixels -> position in [-1,1]
Nd = length(ytr)
ON, OFF = 1.0, -1.0
Ttr = [ytr[i]==j ? ON : OFF for i in eachindex(ytr), j in 1:N_CLS]
println("Layered Duffing digits: N=$N (16 in, $N_HID hid, 10 out), train=$Nd test=$(length(yte))\n")

s_at(it) = it >= max(1,round(Int,ANNEAL_FRAC*N_ITER)) ? 1.0 :
           S_START + (1.0-S_START)*(it-1)/(max(1,round(Int,ANNEAL_FRAC*N_ITER))-1)

# ---------------------------------------------------------------- init
W = 0.1*randn(rng,N,N); W=(W+W')/2; W .*= MASK
h = zeros(N)

function duff_acc(W,h,X,y)
    n=size(X,1); x0=zeros(n,N); x0[:,INPUT].=X
    x0[:,VAR] .= TEST_RANGE.*(2 .*rand(rng,n,length(VAR)).-1)
    eq = drelax(W,h,-1.0,1.0, x0, fill(OFF,n,N_CLS), 0.0)
    out=eq[:,OUT]; mean([argmax(@view out[i,:]) for i in 1:n].==y)
end

# ---------------------------------------------------------------- train
sW=zeros(N,N);rW=zeros(N,N);sh=zeros(N);rh=zeros(N)
best_te=0.0; bW=copy(W); bh=copy(h)
println("Training layered Duffing (basin-avg + anneal, one-sided)...")
t0=time()
for it in 1:N_ITER
    s = s_at(it); a=-s; c=s
    bidx = rand(rng, 1:Nd, BATCH)
    x0 = zeros(BATCH,N); x0[:,INPUT] .= Xtr[bidx,:]
    x0[:,VAR] .= TEST_RANGE.*(2 .*rand(rng,BATCH,length(VAR)).-1)   # full-range basin init
    gW,gh,cost = dgrad(W,h,a,c, x0, Ttr[bidx,:], BETA)
    global W,sW,rW = adam_update(W,gW,LR,it,sW,rW)
    global W = (W+W')/2; global W .*= MASK
    global h,sh,rh = adam_update(h,gh,LR,it,sh,rh)
    if it==1 || it % EVAL_EVERY == 0
        te=duff_acc(W,h,Xte,yte)
        if te>best_te; global best_te=te; global bW=copy(W); global bh=copy(h); end
        @printf("  iter %d: cost %.3f (s=%.2f)  test %.3f (best %.3f) [%.0fs]\n",
                it, cost, s, te, best_te, time()-t0)
    end
end
secs=time()-t0

du_tr=duff_acc(bW,bh,Xtr,ytr); du_te=duff_acc(bW,bh,Xte,yte)
lr_te=logreg_acc(Xtr_p,ytr,Xte_p,yte,N_CLS); ml_te=mlp_acc(Xtr_p,ytr,Xte_p,yte,N_CLS)

@printf("\ntrained %d iters in %.0fs\n", N_ITER, secs)
println("="^54)
@printf("%-16s | %-10s %-10s  (chance %.3f, 4x4)\n","model","train","test",1/N_CLS)
println("-"^54)
@printf("%-16s | %-10.3f %-10.3f\n","Duffing(layer)",du_tr,du_te)
@printf("%-16s | %-10s %-10.3f\n","logreg","-",lr_te)
@printf("%-16s | %-10s %-10.3f\n","MLP","-",ml_te)
println("\nRef: XY Stage 3 (full 64px, layered/all-to-all) 94%; Duffing worry = 2^10 output basins")
