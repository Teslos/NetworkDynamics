# Locate the edge of chaos for the 64-node FHN reservoir used by the Dry-Bean
# classifier, as a function of the coupling strength sigma.
#
# Criterion: drive two copies of the reservoir with the SAME input from initial
# conditions differing by a small perturbation, and measure how the separation
# evolves. Ordered regime -> perturbations decay (ratio < 1); chaotic regime ->
# they grow (ratio > 1). The edge is where the ratio crosses 1, and reservoir
# computing is generally best just below it, where the response is maximally
# expressive while still being a function of the input rather than of the
# initial state.
#
# This replaces choosing sigma by accuracy hunting: the criterion is a property
# of the reservoir alone, independent of the readout or the labels.
#
# Usage: julia --project=. scripts/fhn_edge_of_chaos.jl

include(joinpath(@__DIR__, "..", "src", "utils", "spikerate.jl"))
include(joinpath(@__DIR__, "..", "src", "utils", "drybean.jl"))
using .spikerate, .drybean
using OrdinaryDiffEq, LinearAlgebra, Statistics, Random, Distributions, Printf

const NRES, NIN, NSTEPS = 64, 16, 32
const EPS, AFHN, R0 = 0.05, 0.5, 0.5
const SEED = 1234

db = drybean.read_drybean()
raw = Matrix(permutedims(db))
X = Float64.(raw[1:16, :])
rng = Xoshiro(SEED)
sample_ids = shuffle(rng, 1:size(X, 2))[1:40]        # average over 40 inputs
mu = mean(X, dims = 2); sd = std(X, dims = 2) .+ 1e-9
Z = (X[:, sample_ids] .- mu) ./ sd
Sp = Float64.(permutedims(spikerate.rate(permutedims(Z), NSTEPS), (2, 3, 1)))  # (40,16,T)

function run_pair(sigma, drive, rng)
    Wc = [pdf(Normal(), r) for r in (2 .* rand(rng, NRES, NRES) .- 1)]
    Wc = sigma .* (Wc .+ Wc') ./ 2
    Wc[diagind(Wc)] .= 0
    rowsum = vec(sum(Wc, dims = 2))
    gvec = zeros(NRES)
    function rhs!(dz, z, p, t)
        u = @view z[1:NRES]; v = @view z[NRES+1:2NRES]
        du = @view dz[1:NRES]; dv = @view dz[NRES+1:2NRES]
        i = clamp(floor(Int, t) + 1, 1, NSTEPS)
        @inbounds for k in 1:NRES
            gvec[k] = k <= NIN ? drive[k, i] : 0.0
        end
        coupling = Wc * u .- rowsum .* u
        @. du = gvec + u - u^3 / 3 - v + coupling
        @. dv = (gvec * R0 + u - AFHN) * EPS
        nothing
    end
    z0 = zeros(2NRES)
    d0 = 1e-6
    z0p = copy(z0); z0p[1] += d0
    tspan = (0.0, Float64(NSTEPS))
    sa = solve(ODEProblem(rhs!, z0,  tspan), Tsit5(); saveat = Float64(NSTEPS), abstol = 1e-9, reltol = 1e-9)
    sb = solve(ODEProblem(rhs!, z0p, tspan), Tsit5(); saveat = Float64(NSTEPS), abstol = 1e-9, reltol = 1e-9)
    sep = norm(sa.u[end] .- sb.u[end]) / d0
    # richness of the response: spread of node activity across the reservoir
    U = Array(solve(ODEProblem(rhs!, z0, tspan), Tsit5(); saveat = 1.0:1.0:NSTEPS,
                    save_idxs = 1:NRES, abstol = 1e-8, reltol = 1e-8))
    return sep, mean(std(U, dims = 2)), std(vec(mean(U, dims = 2)))
end

@printf("%8s %14s %14s %14s\n", "sigma", "sep ratio", "temporal std", "across-node std")
for sigma in (0.006, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2)
    seps = Float64[]; tstd = Float64[]; nstd = Float64[]
    for s in 1:size(Sp, 1)
        r, ts, ns = run_pair(sigma, @view(Sp[s, :, :]), Xoshiro(SEED + s))
        push!(seps, r); push!(tstd, ts); push!(nstd, ns)
    end
    @printf("%8.3f %14.4f %14.4f %14.4f\n", sigma, mean(seps), mean(tstd), mean(nstd))
end
println("\nsep ratio < 1: ordered (perturbations decay);  > 1: chaotic")
