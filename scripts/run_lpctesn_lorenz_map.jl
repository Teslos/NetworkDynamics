# Lorenz map (Lorenz 1963 return map of successive z-maxima) and attractor for
# the LPCTESN autonomous rollout, compared to ground truth. Even after the
# pointwise forecast loses phase, this tests whether the LPCTESN reproduces the
# attractor "climate" (invariant structure) -- the standard reservoir-computing
# long-term test.
#
# Usage: julia --project=. scripts/run_lpctesn_lorenz_map.jl [--seed N]

include(joinpath(@__DIR__, "..", "src", "baselines", "baseline_models.jl"))
include(joinpath(@__DIR__, "..", "src", "baselines", "lpctesn.jl"))
using .BaselineModels, .LPCTESN
using Random, Statistics, Printf, CairoMakie

sarg = findfirst(==("--seed"), ARGS)
const SEED = sarg === nothing ? 1 : parse(Int, ARGS[sarg + 1])
const FIGDIR = joinpath(@__DIR__, "..", "results", "figures")
isdir(FIGDIR) || mkpath(FIGDIR)

# successive local maxima of z(t)
function z_maxima(z)
    m = Float64[]
    @inbounds for i in 2:length(z)-1
        (z[i] > z[i-1] && z[i] >= z[i+1]) && push!(m, z[i])
    end
    return m
end

println("Generating Lorenz data and LPCTESN long rollout (seed $SEED)...")
data = lorenz_data(T=20000)
res = lpctesn_lorenz(data; NR=300, spectral_radius=1.1, gamma=10.0, lambda=0.1,
                     input_scale=0.5, train_len=6000, horizon=13000, rng=Xoshiro(SEED))

ztrue = z_maxima(data[3, :])
zpred = z_maxima(res.pred[3, :])
@printf("z-maxima: truth %d, LPCTESN %d\n", length(ztrue), length(zpred))

fig = Figure(size=(950, 420))
Label(fig[0, 1:2], "LPCTESN Lorenz climate (autonomous rollout, seed $SEED)",
      fontsize=17, font=:bold)

# (a) Lorenz return map: z_n vs z_{n+1}
ax1 = Axis(fig[1, 1], xlabel="zₙ (successive maxima)", ylabel="zₙ₊₁",
           title="Lorenz map", aspect=DataAspect())
scatter!(ax1, ztrue[1:end-1], ztrue[2:end], color=(:black, 0.5), markersize=5, label="truth")
scatter!(ax1, zpred[1:end-1], zpred[2:end], color=(:crimson, 0.5), markersize=5, label="LPCTESN")
axislegend(ax1, position=:lt)

# (b) x-z attractor projection
ax2 = Axis(fig[1, 2], xlabel="x", ylabel="z", title="attractor (x–z)")
lines!(ax2, data[1, 1:6000], data[3, 1:6000], color=(:black, 0.5), linewidth=0.4, label="truth")
lines!(ax2, res.pred[1, :], res.pred[3, :], color=(:crimson, 0.6), linewidth=0.4, label="LPCTESN")
axislegend(ax2, position=:lt)

save(joinpath(FIGDIR, "lpctesn_lorenz_map.png"), fig)
println("Wrote ", joinpath(FIGDIR, "lpctesn_lorenz_map.png"))
