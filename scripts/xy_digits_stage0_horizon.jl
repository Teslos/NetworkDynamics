# Stage 0 addendum: is the N=87 XY net's non-settling a horizon artifact (slow
# gradient flow) or a genuine wall (frustration / limit cycle)?
#
# The free XY dynamics with symmetric coupling is first-order gradient flow on a
# bounded-below energy, so it CANNOT limit-cycle -- it must descend to a fixed
# point. The Stage 0 probe showed 0/45 reached |du|<1e-3 within horizon T=30.
# This sweeps the relaxation horizon and reports how the residual vector-field
# magnitude |du| at the end shrinks with T. If it's slow gradient flow, median
# final |du| should fall monotonically and the settled fraction should rise.
#
# Run: julia -t auto --project=. scripts/xy_digits_stage0_horizon.jl

using LinearAlgebra, Statistics, Random, Printf, DelimitedFiles
using OrdinaryDiffEq
using SciMLBase: get_du, successful_retcode

EP_XY_SKIP_RUN = true
include(joinpath(@__DIR__, "..", "notebooks", "EP-XY-Network-Claude.jl"))

const SEED     = 1
const CLASSES  = [0, 1, 2]
const N_HIDDEN = 20
const DT       = 0.1
const W_SCALE  = 0.1
const ON, OFF  = π/2, -π/2
const HORIZONS = [300, 1000, 3000, 10000]   # N_ev values; T = N_ev*DT
const N_PROBE  = 30

Random.seed!(SEED)
rng = MersenneTwister(SEED)

raw = readdlm(joinpath(@__DIR__, "..", "data", "digits", "optdigits.tes"), ',', Int)
X_all = Float64.(raw[:, 1:64]); y_all = raw[:, 65]
sub = vcat([shuffle(rng, findall(==(c), y_all))[1:N_PROBE] for c in CLASSES]...)
X = (X_all[sub, :] ./ 16.0 .- 0.5) .* π

const N_IN = 64
const N_CLS = length(CLASSES)
const N = N_IN + N_HIDDEN + N_CLS
const input_index    = collect(1:N_IN)
const output_index   = collect(N-N_CLS+1:N)
const variable_index = setdiff(1:N, input_index)

W0 = W_SCALE * randn(rng, N, N); W0 = (W0 + W0') / 2; W0[diagind(W0)] .= 0
bias0 = zeros(2, N)
bias0[1, :] .= 0.1 * rand(rng, N)
bias0[2, :] .= 2π .* (rand(rng, N) .- 0.5)
target = fill(OFF, N_CLS)

# Relax one input for a given horizon; return (terminated_early, final |du|).
function relax(n_ev, xi)
    phase0 = zeros(N); phase0[input_index] .= xi
    phase0[variable_index] .= 0.1 * randn(rng, length(variable_index))
    p = force_params(W0, bias0, target, 0.0, input_index, output_index)
    prob = ODEProblem(xy_force!, phase0, (0.0, n_ev * DT), p)
    sol = solve(prob, Tsit5(); callback=steady_state_callback(), SOLVER_KWARGS...)
    du = similar(sol.u[end]); xy_force!(du, sol.u[end], p, 0.0)
    return (sol.t[end] < n_ev * DT - 1e-6, maximum(abs, du))
end

println("XY N=$N free-relaxation horizon sweep (random init, $N_PROBE inputs)\n")
@printf("%-10s %-10s | %-12s %-12s %-12s | %s\n",
        "N_ev", "T", "settled<1e-5", "med |du|_end", "max |du|_end", "min |du|_end")
println("-"^78)
for n_ev in HORIZONS
    res = [relax(n_ev, X[i, :]) for i in 1:size(X, 1)]
    term = count(first, res)
    dus = last.(res)
    @printf("%-10d %-10.0f | %-12s %-12.2e %-12.2e %.2e\n",
            n_ev, n_ev * DT, "$term/$(length(res))", median(dus), maximum(dus), minimum(dus))
end
