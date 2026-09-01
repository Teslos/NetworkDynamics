# Temporal / delayed parity — puts the DYNAMICS on trial (unlike static parity).
#
# k bits are presented SEQUENTIALLY, each to its own input node for one time
# window, then RELEASED (clamp off). The label is the parity of all k bits, read
# out at the end from the network's phases. Because each input is transient, the
# only thing carrying its value to readout time is the network's internal state —
# so the task REQUIRES memory. The static-parity "dynamics-bypass" cannot work
# here: a memoryless model (seeing only the current/last bit) is at chance by
# construction, and a model seeing all bits at once trivially solves it, so the
# task's entire difficulty is the temporal-memory requirement.
#
# Sharp comparison: does TRAINING the coupling let the oscillator dynamics
# accumulate parity over time, where RANDOM coupling cannot?
#   * UDE            : train coupling NN + linear readout.
#   * random reservoir: FIX coupling at random init, train only the readout.
# Both read out a 2-class parity head on the final [sin phi; cos phi].
#
# Run: julia --project=. scripts/parity_encoder/temporal_parity.jl
using ComponentArrays, LinearAlgebra, Random, StableRNGs, Printf, Statistics
import Enzyme
logp(m) = (println(m); flush(stdout))

const K=3                       # parity order (2^K patterns)
const NHID=3                    # hidden oscillators
const N=K+NHID                  # nodes 1..K = transient inputs, K+1..N = hidden
const H=16                      # coupling-NN width
const PHI0=-π/2; const PHI1=π/2
const HCLAMP=0.5
const DT=0.1; const STEPS_WIN=30   # window length T_win = 3.0

# hoisted scalar RHS (shared coupling NN)
function trhs(φ, p, h_vec, ψ_vec)
    Tp=promote_type(eltype(φ),eltype(p)); W1=reshape(p.W1,H,4); b1=p.b1; W2=p.W2; b2=p.b2[1]
    out=Vector{Tp}(undef,N)
    @inbounds for i in 1:N
        si=sin(φ[i]); ci=cos(φ[i])
        bi=Vector{Tp}(undef,H); for h in 1:H; bi[h]=W1[h,3]*si+W1[h,4]*ci+b1[h]; end
        s=-h_vec[i]*sin(φ[i]-ψ_vec[i])
        for j in 1:N; j==i && continue
            sj=sin(φ[j]); cj=cos(φ[j]); w=b2
            for h in 1:H; z=W1[h,1]*sj+W1[h,2]*cj+bi[h]; w+=W2[h]*tanh(z); end
            s+=w*sin(φ[j]-φ[i])
        end
        out[i]=s
    end
    out
end
function rk4win(φ, p, h_vec, ψ_vec, nsteps)
    for _ in 1:nsteps
        k1=trhs(φ,p,h_vec,ψ_vec); k2=trhs(φ.+0.5*DT.*k1,p,h_vec,ψ_vec)
        k3=trhs(φ.+0.5*DT.*k2,p,h_vec,ψ_vec); k4=trhs(φ.+DT.*k3,p,h_vec,ψ_vec)
        φ=φ.+(DT/6.0).*(k1.+2.0.*k2.+2.0.*k3.+k4)
    end
    φ
end
# present bits sequentially (node t clamped to bit_t during window t, else free),
# then one free "recall" window; return final phases.
function run_seq(bits, p)
    φ=zeros(eltype(p), N)
    for t in 1:K
        h=zeros(N); ψ=zeros(N); h[t]=HCLAMP; ψ[t]= bits[t]==1 ? PHI1 : PHI0
        φ=rk4win(φ,p,h,ψ,STEPS_WIN)
    end
    rk4win(φ,p,zeros(N),zeros(N),STEPS_WIN)   # recall window: all inputs released
end
patbits(pat)=ntuple(t->(pat>>(t-1))&1, K)
parity(bits)=reduce(⊻, bits)

function tloss(p)
    W_cls=reshape(p.W_cls,2,2N); loss=zero(eltype(p))
    for pat in 0:(2^K-1)
        bits=patbits(pat); φ=run_seq(bits,p)
        feat=vcat(sin.(φ),cos.(φ)); lg=W_cls*feat
        m=maximum(lg); lse=m+log(sum(exp.(lg.-m))); loss += lse - lg[parity(bits)+1]
    end
    loss/(2^K)
end
function accuracy(p)
    W_cls=reshape(p.W_cls,2,2N); nc=0
    for pat in 0:(2^K-1)
        bits=patbits(pat); φ=run_seq(bits,p); lg=W_cls*vcat(sin.(φ),cos.(φ))
        (argmax(lg)-1)==parity(bits) && (nc+=1)
    end
    nc
end

pinit(rng)=ComponentArray(W1=randn(rng,Float64,H*4).*0.1, b1=zeros(H), W2=randn(rng,Float64,H).*0.1,
                          b2=zeros(1), W_cls=zeros(2*2N))

# train; freeze_coupling=true => random reservoir (only W_cls learns)
function train(; freeze_coupling::Bool, iters=2000, lr1=0.01, lr2=0.002, clip=8.0, seed=1)
    rng=StableRNG(seed); p=pinit(rng)
    β1=0.9;β2=0.999;ϵ=1e-8; m=zero(p); v=zero(p); t=0
    stage(lr,nit)=for it in 1:nit
        g=Enzyme.gradient(Enzyme.Reverse, tloss, p)[1]
        if freeze_coupling            # zero coupling grad -> only readout learns
            g.W1.=0; g.b1.=0; g.W2.=0; g.b2.=0
        end
        gn=norm(vec(g)); gn>clip && (g.*=clip/gn); t+=1
        @. m=β1*m+(1-β1)*g; @. v=β2*v+(1-β2)*g*g
        @. p = p - lr*(m/(1-β1^t))/(sqrt(v/(1-β2^t))+ϵ)
        it%250==0 && logp(@sprintf("   [%s] it %4d loss=%.4f acc=%d/%d",
            freeze_coupling ? "reservoir" : "UDE", t, tloss(p), accuracy(p), 2^K))
    end
    stage(lr1,iters÷2); stage(lr2,iters-iters÷2)
    accuracy(p)
end

try
    logp("Temporal parity: k=$K bits presented sequentially (transient), N=$N, $(2^K) patterns")
    logp("training RANDOM RESERVOIR (fixed coupling, readout only)...")
    ar=train(freeze_coupling=true)
    logp(@sprintf("random reservoir  accuracy: %d/%d", ar, 2^K))
    logp("training UDE (trained coupling)...")
    au=train(freeze_coupling=false)
    logp(@sprintf("UDE (trained dyn) accuracy: %d/%d", au, 2^K))
    logp(@sprintf("SUMMARY  random-reservoir=%d/%d   UDE=%d/%d   (memoryless model = chance = 4/8 by construction)", ar,2^K,au,2^K))
    logp("TEMPPAR_DONE")
catch e
    logp("TEMPPAR_ERROR: "*sprint(showerror,e,catch_backtrace()))
end
flush(stdout)
