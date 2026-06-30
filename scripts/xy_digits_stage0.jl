# EP-XY digits scale-up, STAGE 0: does a ~100-cell XY network relax to a static
# fixed point (the precondition for EP), and does the digits pipeline plumb end
# to end?
#
# The XY phase net solves XOR robustly (10/10 seeds), but XOR uses N=5. EP needs
# the FREE dynamics to settle to a STATIC equilibrium; a large XY net with random
# symmetric coupling can frustrate and limit-cycle instead, in which case the
# steady-state callback never fires and EP is ill-defined. This script tests that
# make-or-break property BEFORE committing to the full pipeline, and smoke-tests
# the input encoding / argmax readout / accuracy plumbing on a small 3-class task.
#
# This is a feasibility probe, NOT a tuned classifier -- accuracy here is
# secondary to (a) "does it settle?" and (b) "does the plumbing run and the cost
# move?".
#
# Run (threads matter -- one ODE solve per sample, batched):
#   julia -t auto --project=. scripts/xy_digits_stage0.jl

using LinearAlgebra, Statistics, Random, Printf, DelimitedFiles
using OrdinaryDiffEq
using SciMLBase: get_du, successful_retcode

EP_XY_SKIP_RUN = true
include(joinpath(@__DIR__, "..", "notebooks", "EP-XY-Network-Claude.jl"))

# ---------------------------------------------------------------- config
const SEED       = 1
const CLASSES    = [0, 1, 2]      # 3-class subset
const N_TRAIN_PC = 50             # train samples per class
const N_TEST_PC  = 30             # test samples per class
const N_HIDDEN   = 20
const N_EV       = 300            # T = N_EV * dt (relaxation horizon)
const DT         = 0.1
const BETA       = 0.01
const STUDY_RATE = 0.05
const N_EPOCH    = 60             # short -- smoke test, not convergence to optimum
const BATCH      = 30
const W_SCALE    = 0.1            # coupling init scale (small -> less frustration)
const ON, OFF    = π/2, -π/2      # one-hot phase target levels

Random.seed!(SEED)
println("threads = ", Threads.nthreads(), "  (run with `julia -t auto` for speed)\n")

# ---------------------------------------------------------------- data
function load_digits(; path=joinpath(@__DIR__, "..", "data", "digits", "optdigits.tes"))
    raw = readdlm(path, ',', Int)
    return Float64.(raw[:, 1:64]), raw[:, 65]   # (Nall,64) pixels in 0..16, labels
end

X_all, y_all = load_digits()
rng = MersenneTwister(SEED)

# Balanced subset: N_TRAIN_PC + N_TEST_PC samples per class, split train/test.
train_idx = Int[]; test_idx = Int[]
for c in CLASSES
    ci = shuffle(rng, findall(==(c), y_all))
    append!(train_idx, ci[1:N_TRAIN_PC])
    append!(test_idx,  ci[N_TRAIN_PC+1 : N_TRAIN_PC+N_TEST_PC])
end
shuffle!(rng, train_idx); shuffle!(rng, test_idx)

# Pixels 0..16 -> phase in [-pi/2, pi/2]; class label -> column index 1..n_class.
encode(X) = (X ./ 16.0 .- 0.5) .* π
classcol = Dict(c => j for (j, c) in enumerate(CLASSES))

Xtr = encode(X_all[train_idx, :]);  ytr = [classcol[c] for c in y_all[train_idx]]
Xte = encode(X_all[test_idx, :]);   yte = [classcol[c] for c in y_all[test_idx]]

const N_IN  = size(Xtr, 2)              # 64
const N_CLS = length(CLASSES)
const N     = N_IN + N_HIDDEN + N_CLS
const input_index    = collect(1:N_IN)
const output_index   = collect(N-N_CLS+1:N)
const variable_index = setdiff(1:N, input_index)
println("Network: N=$N  (inputs=$N_IN, hidden=$N_HIDDEN, outputs=$N_CLS)")
println("Train=$(length(ytr))  Test=$(length(yte))  classes=$CLASSES\n")

# One-hot phase targets, (N_data, N_CLS): ON for the true class, OFF elsewhere.
onehot_phase(y) = [y[i] == j ? ON : OFF for i in eachindex(y), j in 1:N_CLS]
Ttr = onehot_phase(ytr)

# ---------------------------------------------------------------- net init
W0 = W_SCALE * randn(rng, N, N); W0 = (W0 + W0') / 2; W0[diagind(W0)] .= 0
bias0 = zeros(2, N)
bias0[1, :] .= 0.1 * rand(rng, N)                    # field magnitude h
bias0[2, :] .= 2π .* (rand(rng, N) .- 0.5)           # field phase psi

# ---------------------------------------------------------------- diagnostics
# Free relaxation of one clamped input with per-solve telemetry: did it reach a
# static fixed point (callback terminated early) or run the full horizon / fail?
function relax_diag(W, bias, phase0, target)
    p = force_params(W, bias, target, 0.0, input_index, output_index)
    prob = ODEProblem(xy_force!, collect(Float64, phase0), (0.0, N_EV * DT), p)
    sol = solve(prob, Tsit5(); callback=steady_state_callback(), SOLVER_KWARGS...)
    du = similar(sol.u[end]); xy_force!(du, sol.u[end], p, 0.0)
    return (t=sol.t[end], rc=sol.retcode, maxdu=maximum(abs, du), u=sol.u[end])
end

# Settled := terminated strictly before the horizon with a near-zero vector field.
function convergence_report(label, W, bias; n=45, settle_tol=1e-3)
    settled = 0; full = 0; failed = 0; ts = Float64[]
    for i in 1:n
        phase0 = zeros(N)
        phase0[input_index] .= Xtr[i, :]
        phase0[variable_index] .= 0.1 * randn(rng, length(variable_index))
        d = relax_diag(W, bias, phase0, view(Ttr, i, :))
        if !successful_retcode(d.rc)
            failed += 1
        elseif d.t < N_EV * DT - 1e-6 && d.maxdu < settle_tol
            settled += 1; push!(ts, d.t)
        else
            full += 1
        end
    end
    @printf("[%s] settled %d/%d  ran-full-horizon %d  failed %d", label, settled, n, full, failed)
    isempty(ts) ? println("  (no settling times)") :
        @printf("  | settle time: median %.1f, max %.1f (horizon %.0f)\n",
                median(ts), maximum(ts), N_EV * DT)
end

# ---------------------------------------------------------------- readout
# Predicted class = output cell whose phase is closest to the ON target.
function predict(W, bias, X)
    n = size(X, 1)
    phase0 = zeros(n, N)
    phase0[:, input_index] .= X
    phase0[:, variable_index] .= 0.1 * randn(rng, n, length(variable_index))
    dummy = fill(OFF, n, N_CLS)
    eq = run_network_batch(phase0, N_EV * DT, W, bias, dummy, 0.0, input_index, output_index)
    out = eq[:, output_index]
    return [argmax(@view out[i, :]) for i in 1:n]   # nearest to ON == max phase
end
accuracy(W, bias, X, y) = mean(predict(W, bias, X) .== y)

# ---------------------------------------------------------------- run
println("== PART A: fixed-point convergence (the make-or-break test) ==")
convergence_report("random-init", W0, bias0)

println("\n== PART B: pipeline smoke test (short EP training) ==")
@printf("chance accuracy ~ %.3f\n", 1 / N_CLS)
t0 = time()
Wf, biasf, ch = train_network(W0, bias0, Xtr, Ttr, BETA, STUDY_RATE,
                              N_EPOCH, BATCH, N_EV, DT,
                              input_index, variable_index, output_index;
                              use_adam=true, symmetric=true, print_every=10)
@printf("trained %d epochs in %.1f s\n", length(ch), time() - t0)
@printf("cost: first %.4f  ->  last %.4f  (min %.4f)\n", ch[1], ch[end], minimum(ch))
@printf("train accuracy: %.3f\n", accuracy(Wf, biasf, Xtr, ytr))
@printf("test  accuracy: %.3f\n", accuracy(Wf, biasf, Xte, yte))

println("\n== PART A again: convergence with TRAINED weights ==")
convergence_report("trained", Wf, biasf)
