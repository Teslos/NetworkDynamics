# Phase-field / general-quartic EP-Duffing: does an asymmetric, input-tilted well
# (with the symmetric double well as a special case) address the bistable XOR?
#
# Motivation (see the analysis behind this): the symmetric well V = 1/4 c x^4 +
# 1/2 a x^2 welds "expressivity" (nonlinearity) to "deep symmetric bistability"
# (the init-dependent basin selection that breaks robust XOR). A general quartic
# unwelds them and adds two levers this script tests:
#
#   1. INPUT-CONTROLLED TILT (phase-field driving force). A dedicated, DIRECTED,
#      trainable input->cell projection B drives each free cell with a field
#      sum_k B_ik u_k. A strong enough tilt pushes a symmetric double well past
#      its saddle-node bifurcation (|tilt| > 2/(3 sqrt3) ~ 0.385 for the +-1 well),
#      leaving ONE input-determined well -> forward inference follows the input
#      (the XY-like behavior), rather than the initialization.
#   2. LEARNABLE POTENTIAL SHAPE. Per-cell trainable c3 (cubic) and c2 (quadratic),
#      c4 fixed > 0 (boundedness guaranteed). EP trains these locally
#      (dE/dc3 = x^3/3, dE/dc2 = x^2/2) -- a learnable activation baked into the
#      physics; c3=0, c2=-1 recovers the symmetric +-1 well.
#
# Structure vs the plain Duffing net: recurrent free-free coupling W is symmetric
# (energy is a gradient); the input drive B is a SEPARATE directed matrix (inputs
# are clamped external fields, so no symmetry needed) that can grow large enough
# to reach the bifurcation. Trained with basin-averaging (full-range init +
# gradient averaged over Minit basins) and evaluated under the honest full-range
# test. Compares: input-tilt with a FIXED symmetric well vs + learnable shape.
#
# Usage: julia --project=. scripts/duffing_phasefield.jl

using Random, Printf, Statistics, LinearAlgebra
using OrdinaryDiffEq
using SciMLBase: get_du

EP_DUFFING_SKIP_RUN = true
include(joinpath(@__DIR__, "..", "notebooks", "EP-Duffing-Network.jl"))  # reuse adam_update, callback, SOLVER_KWARGS

# ---------------------------------------------------------------- config
const SEEDS        = 1:5
const N_EPOCH      = 1000
const N_HIDDEN     = 2
const MINIT        = 20
const BETA         = 0.1
const LR           = 0.02
const DELTA        = 1.0
const C4           = 1.0
const T_MAX        = 40.0
const TEST_RANGE   = 1.5
const N_TEST_INIT  = 20
const SOLVE_THRESH = 0.90
const C2_RANGE     = (-3.0, 0.5)   # clamps for learnable shape
const C3_RANGE     = (-2.0, 2.0)

const CONFIGS = [("input-tilt, fixed sym well", false), ("input-tilt + learnable shape", true)]

data   = Float64[-1 -1; -1 1; 1 -1; 1 1]
target = reshape(Float64[-1, 1, 1, -1], 4, 1)

Nn = 2 + N_HIDDEN + 1
const INPUT = [1, 2]; const OUT = [Nn]; const VAR = setdiff(1:Nn, INPUT)
const IS_INPUT = [i in INPUT for i in 1:Nn]

# ---------------------------------------------------------------- dynamics
# State z = [x(1:N); v(1:N)].  Free cell i: vdot = -delta v + F_i, with
# F_i = -(c4 x^3 + c3_i x^2 + c2_i x) + h_i + sum_j (W_ij + B_ij) x_j  (+ nudge on outputs).
function pf_force!(du, z, p, t)
    N = p.N
    @inbounds for i in 1:N
        if p.is_input[i]
            du[i] = 0.0; du[N+i] = 0.0; continue
        end
        xi = z[i]; acc = 0.0
        for j in 1:N
            acc += (p.W[i, j] + p.B[i, j]) * z[j]
        end
        F = -(p.c4 * xi^3 + p.c3[i] * xi^2 + p.c2[i] * xi) + p.h[i] + acc
        du[i] = z[N+i]
        du[N+i] = -p.delta * z[N+i] + F
    end
    if p.beta != 0.0
        @inbounds for (m, j) in enumerate(p.output_index)
            du[N+j] -= p.beta * (z[j] - p.target[m])
        end
    end
    return nothing
end

pf_params(W, B, h, c2, c3, target, beta) =
    (N=Nn, W=W, B=B, h=h, c2=c2, c3=c3, c4=C4, delta=DELTA, beta=beta,
     target=target, output_index=OUT, is_input=IS_INPUT)

function pf_relax(W, B, h, c2, c3, x0_batch, target_batch, beta)
    nb = size(x0_batch, 1); eq = zeros(nb, Nn)
    for d in 1:nb
        u0 = zeros(2Nn); @views u0[1:Nn] .= x0_batch[d, :]
        p = pf_params(W, B, h, c2, c3, view(target_batch, d, :), beta)
        prob = ODEProblem(pf_force!, u0, (0.0, T_MAX), p)
        sol = solve(prob, Tsit5(); callback=steady_state_callback(), SOLVER_KWARGS...)
        @views eq[d, :] .= sol.u[end][1:Nn]
    end
    return eq
end

# ---------------------------------------------------------------- EP gradients
# g_theta = mean_d[ dE/dtheta(x_nudge) - dE/dtheta(x_free) ] / scale, with
# dE/dW=dE/dB=-x_i x_j, dE/dh=-x_i, dE/dc2=+x_i^2/2, dE/dc3=+x_i^3/3.
function pf_gradients(W, B, h, c2, c3, x0, tgt, beta)
    xz = pf_relax(W, B, h, c2, c3, x0, tgt, 0.0)
    xn = pf_relax(W, B, h, c2, c3, xz, tgt, beta)
    xf = xz; scale = beta
    nb = size(xf, 1)
    gW = zeros(Nn, Nn); gB = zeros(Nn, Nn); gh = zeros(Nn); gc2 = zeros(Nn); gc3 = zeros(Nn)
    @inbounds for d in 1:nb
        for i in 1:Nn
            gh[i]  += (xf[d,i] - xn[d,i])
            gc2[i] += 0.5 * (xn[d,i]^2 - xf[d,i]^2)
            gc3[i] += (xn[d,i]^3 - xf[d,i]^3) / 3
            for j in 1:Nn
                gW[i,j] += (xf[d,i]*xf[d,j] - xn[d,i]*xn[d,j])
                gB[i,j] += (xf[d,i]*xf[d,j] - xn[d,i]*xn[d,j])
            end
        end
    end
    f = 1.0 / (nb * scale)
    dev = (xz[:, OUT] .- tgt) .^ 2
    cost = mean(vec(sum(dev, dims=2)) ./ 2)
    return gW.*f, gB.*f, gh.*f, gc2.*f, gc3.*f, cost
end

# ---------------------------------------------------------------- training
function train_pf!(W, B, h, c2, c3, learn_shape; rng)
    aW=(zeros(Nn,Nn),zeros(Nn,Nn)); aB=(zeros(Nn,Nn),zeros(Nn,Nn))
    ah=(zeros(Nn),zeros(Nn)); a2=(zeros(Nn),zeros(Nn)); a3=(zeros(Nn),zeros(Nn))
    best=Inf; bW=copy(W); bB=copy(B); bh=copy(h); b2=copy(c2); b3=copy(c3); ch=zeros(N_EPOCH)
    nrow = 4*MINIT
    for epoch in 1:N_EPOCH
        x0 = zeros(nrow, Nn); tg = zeros(nrow, 1); r = 1
        for p in 1:4, _ in 1:MINIT
            x0[r, INPUT] .= data[p, :]
            x0[r, VAR]   .= TEST_RANGE .* (2 .* rand(rng, length(VAR)) .- 1)
            tg[r,1] = target[p,1]; r += 1
        end
        gW,gB,gh,gc2,gc3,cost = pf_gradients(W,B,h,c2,c3,x0,tg,BETA)
        # W: symmetric, free-free only
        W,aW1,aW2 = adam_update(W, gW, LR, epoch, aW[1], aW[2]); aW=(aW1,aW2)
        W = (W+W')/2; W[diagind(W)] .= 0
        for i in 1:Nn, j in 1:Nn; if IS_INPUT[i]||IS_INPUT[j]; W[i,j]=0.0; end; end
        # B: directed, free-row / input-col only
        B,aB1,aB2 = adam_update(B, gB, LR, epoch, aB[1], aB[2]); aB=(aB1,aB2)
        for i in 1:Nn, j in 1:Nn; if IS_INPUT[i] || !IS_INPUT[j]; B[i,j]=0.0; end; end
        h,ah1,ah2 = adam_update(h, gh, LR, epoch, ah[1], ah[2]); ah=(ah1,ah2)
        for i in INPUT; h[i]=0.0; end
        if learn_shape
            c2,a21,a22 = adam_update(c2, gc2, LR, epoch, a2[1], a2[2]); a2=(a21,a22)
            c3,a31,a32 = adam_update(c3, gc3, LR, epoch, a3[1], a3[2]); a3=(a31,a32)
            c2 .= clamp.(c2, C2_RANGE...); c3 .= clamp.(c3, C3_RANGE...)
            for i in INPUT; c2[i]=-1.0; c3[i]=0.0; end
        end
        ch[epoch] = cost
        if cost < best; best=cost; bW=copy(W);bB=copy(B);bh=copy(h);b2=copy(c2);b3=copy(c3); end
    end
    return bW,bB,bh,b2,b3,ch
end

function robust_acc(W,B,h,c2,c3, rng)
    correct = 0
    for i in 1:4, _ in 1:N_TEST_INIT
        x0 = zeros(1, Nn); x0[1, INPUT] .= data[i, :]
        x0[1, VAR] .= TEST_RANGE .* (2 .* rand(rng, length(VAR)) .- 1)
        eq = pf_relax(W,B,h,c2,c3, x0, reshape(target[i,:],1,:), 0.0)
        sign(eq[1, OUT[1]]) == sign(target[i,1]) && (correct += 1)
    end
    return correct / (4*N_TEST_INIT)
end

# ---------------------------------------------------------------- run
println("Phase-field / general-quartic EP-Duffing -- robust XOR under full-range test init")
println(length(SEEDS), " seeds x ", N_TEST_INIT, " draws, Minit=", MINIT, ", ",
        N_EPOCH, " epochs, one-sided grad; directed input tilt B; solve>=",
        Int(100SOLVE_THRESH), "%\n")
@printf("%-30s | %-6s %-5s %-5s %-8s | %s\n", "config", "mean", "max", "min", "solved", "med cost")
println("-"^72)

for (label, learn_shape) in CONFIGS
    accs = Float64[]; fc = Float64[]
    for seed in SEEDS
        rng = MersenneTwister(seed)
        W = zeros(Nn,Nn); B = zeros(Nn,Nn); h = zeros(Nn)
        c2 = fill(-1.0, Nn); c3 = zeros(Nn)
        # init free-free W (symmetric) and free<-input B (directed, larger scale)
        for i in VAR, j in VAR; if i<j; w=0.1*randn(rng); W[i,j]=w; W[j,i]=w; end; end
        for i in VAR, j in INPUT; B[i,j] = 0.3*randn(rng); end
        bW,bB,bh,b2,b3,ch = train_pf!(W,B,h,c2,c3, learn_shape; rng=rng)
        push!(accs, robust_acc(bW,bB,bh,b2,b3, MersenneTwister(9000+seed)))
        push!(fc, ch[end])
    end
    solved = count(>=(SOLVE_THRESH), accs)
    @printf("%-30s | %-6.0f %-5.0f %-5.0f %d/%-6d | %.3f\n",
            label, 100mean(accs), 100maximum(accs), 100minimum(accs), solved, length(SEEDS), median(fc))
end

println("\nReference: symmetric-well basin-averaging (Minit=20, no input-tilt): 76% mean, best 100%")
println("           basin-avg + anneal (Minit=50): 81-84% mean, 3/6 solved")
