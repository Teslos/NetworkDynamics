# Controlled temporal-parity UDE with ADAPTIVE solver (Tsit5) + SciMLSensitivity
# adjoint + Zygote, low LR, small coupling penalty. Question: is the trained-
# coupling training now STABLE (vs the fixed-step RK4 bounce of 8->6->7->5->6/8)?
using OrdinaryDiffEq, SciMLSensitivity
import Zygote
using ComponentArrays, LinearAlgebra, Random, StableRNGs, Printf, Statistics
logp(m) = (println(m); flush(stdout))

const N=6; const K=3; const H=16
const PHI0=-π/2; const PHI1=π/2; const HCLAMP=0.5; const TWIN=3.0
const LREG=1e-4

# non-mutating Kuramoto RHS (Zygote-friendly); θ carries coupling NN (+ unused W_cls)
function krhs(φ, θ, h_vec, ψ_vec)
    W1=reshape(θ.W1,H,4); b1=θ.b1; W2=θ.W2; b2=θ.b2[1]
    [ (-h_vec[i]*sin(φ[i]-ψ_vec[i]) +
        sum(j==i ? 0.0 :
            (dot(W2, tanh.(W1*[sin(φ[j]),cos(φ[j]),sin(φ[i]),cos(φ[i])] .+ b1))+b2)*sin(φ[j]-φ[i])
            for j in 1:N))
      for i in 1:N ]
end
const SENSE = InterpolatingAdjoint(autojacvec=ZygoteVJP())
function solve_win(φ0, θ, h_vec, ψ_vec)
    prob=ODEProblem((u,p,t)->krhs(u,p,h_vec,ψ_vec), φ0, (0.0,TWIN), θ)
    solve(prob, Tsit5(); sensealg=SENSE, save_everystep=false, abstol=1e-6, reltol=1e-6).u[end]
end
patbits(pat)=ntuple(t->(pat>>(t-1))&1, K); parity(bits)=reduce(⊻, bits)
zerof()=zeros(N)
function run_seq(θ, bits)
    φ=zerof()
    for t in 1:K
        h=[m==t ? HCLAMP : 0.0 for m in 1:N]; ψ=[m==t ? (bits[t]==1 ? PHI1 : PHI0) : 0.0 for m in 1:N]
        φ=solve_win(φ,θ,h,ψ)
    end
    solve_win(φ,θ,zerof(),zerof())   # recall window
end
function loss(θ)
    W_cls=reshape(θ.W_cls,2,2N); L=zero(eltype(θ))
    for pat in 0:(2^K-1)
        bits=patbits(pat); φ=run_seq(θ,bits); feat=vcat(sin.(φ),cos.(φ)); lg=W_cls*feat
        m=maximum(lg); L += (m+log(sum(exp.(lg.-m)))) - lg[parity(bits)+1]
    end
    L/(2^K) + LREG*(sum(abs2,θ.W1)+sum(abs2,θ.W2))
end
function accuracy(θ)
    W_cls=reshape(θ.W_cls,2,2N); nc=0
    for pat in 0:(2^K-1)
        bits=patbits(pat); lg=W_cls*vcat(sin.(run_seq(θ,bits)),cos.(run_seq(θ,bits)))
        (argmax(lg)-1)==parity(bits) && (nc+=1)
    end
    nc
end
θinit(rng)=ComponentArray(W1=randn(rng,Float64,H*4).*0.1, b1=zeros(H), W2=randn(rng,Float64,H).*0.1, b2=zeros(1), W_cls=zeros(2*2N))

try
    logp("Temporal parity K=$K, N=$N — ADAPTIVE Tsit5 + adjoint, lr=1e-3, coupling L2=$LREG")
    θ=θinit(StableRNG(1))
    β1=0.9;β2=0.999;ϵ=1e-8; m=zero(θ); v=zero(θ); t=0; lr=1e-3; clip=2.0; warm=30
    logp(@sprintf("init loss=%.4f acc=%d/%d", loss(θ), accuracy(θ), 2^K))
    for it in 1:250
        g=Zygote.gradient(loss, θ)[1]
        gn=norm(vec(g)); gn>clip && (g.*=clip/gn)
        t+=1; lrt = lr*min(1.0, t/warm)
        @. m=β1*m+(1-β1)*g; @. v=β2*v+(1-β2)*g*g
        @. θ = θ - lrt*(m/(1-β1^t))/(sqrt(v/(1-β2^t))+ϵ)
        it%25==0 && logp(@sprintf("it %4d  loss=%.4f  acc=%d/%d  |g|=%.2f", t, loss(θ), accuracy(θ), 2^K, gn))
    end
    logp(@sprintf("FINAL adaptive-UDE acc=%d/%d  (fixed-step RK4 was unstable 8->6->7->5->6/8; random reservoir=7/8)", accuracy(θ), 2^K))
    logp("ADAPT_DONE")
catch e
    logp("ADAPT_ERR: "*sprint(showerror,e,catch_backtrace()))
end
flush(stdout)
