# Compare adjoint VJP backends on the Kuramoto-UDE: does EnzymeVJP work (Julia 1.12),
# is it faster than ZygoteVJP, and does GaussAdjoint help? Also checks gradients agree.
using OrdinaryDiffEq, SciMLSensitivity
import Zygote, Enzyme
using ComponentArrays, LinearAlgebra, Printf
logp(m) = (println(m); flush(stdout))

function run_probe()
    Np=3; Hp=8
    nn=ComponentArray(W1=randn(Hp*4).*0.1, b1=zeros(Hp), W2=randn(Hp).*0.1, b2=zeros(1))
    ann(p,x)=dot(p.W2, tanh.(reshape(p.W1,Hp,4)*x .+ p.b1))+p.b2[1]
    ff(φ,p,t)=[ sum(j==i ? 0.0 : ann(p,[sin(φ[j]),cos(φ[j]),sin(φ[i]),cos(φ[i])])*sin(φ[j]-φ[i]) for j in 1:Np) for i in 1:Np ]
    φ0=[0.1,0.2,0.3]
    mkloss(sa)= p->begin
        prob=ODEProblem(ff, φ0, (0.0,3.0), p)
        sum(abs2, solve(prob, Tsit5(); sensealg=sa, save_everystep=false).u[end])
    end
    cfgs = [("Interp+Zygote", InterpolatingAdjoint(autojacvec=ZygoteVJP())),
            ("Interp+Enzyme", InterpolatingAdjoint(autojacvec=EnzymeVJP())),
            ("Gauss+Enzyme",  GaussAdjoint(autojacvec=EnzymeVJP()))]
    ref=nothing
    for (name,sa) in cfgs
        L=mkloss(sa)
        try
            g=Zygote.gradient(L, nn)[1]                       # warmup / compile
            t=@elapsed (for _ in 1:10; Zygote.gradient(L, nn); end)
            gv=vec(g); d = ref===nothing ? 0.0 : maximum(abs.(gv.-ref))
            ref===nothing && (ref=copy(gv))
            logp(@sprintf("%-14s  %.4f s/grad   |g|=%.4f   maxdiff-vs-Zygote=%.2e", name, t/10, norm(gv), d))
        catch e
            logp(@sprintf("%-14s  FAILED: %s", name, first(split(sprint(showerror,e), '\n'))))
        end
    end
end
try
    logp("VJP-backend probe (Interp/Gauss x Zygote/Enzyme)...")
    run_probe()
    logp("VJPPROBE_DONE")
catch e
    logp("VJPPROBE_ERR: " * sprint(showerror, e, catch_backtrace()))
end
flush(stdout)
