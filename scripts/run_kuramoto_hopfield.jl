# Diagnostic figures for the oscillator-Hopfield (Kuramoto associative memory)
# in src/models/kuramoto_hopfield.jl.
#
# Produces, in results/figures/:
#   kh_patterns.png            stored 8x8 digit patterns
#   kh_coupling.png            Hebbian vs projection coupling magnitude |J|
#   kh_retrieval_dynamics.png  order parameter and energy vs time during recall
#   kh_basin.png               retrieval quality vs cue noise, both rules
#
# Usage: julia --project=. scripts/run_kuramoto_hopfield.jl

KH_SKIP_RUN = true
include(joinpath(@__DIR__, "..", "src", "models", "kuramoto_hopfield.jl"))

using CairoMakie
using OrdinaryDiffEq
using Statistics
using Random

const FIGDIR = joinpath(@__DIR__, "..", "results", "figures")
isdir(FIGDIR) || mkpath(FIGDIR)

const KEYS = ['0', '1', '2']
patterns = [bitmap_to_pattern(PATTERN_BITMAPS[k]) for k in KEYS]
phases = [binary_to_phase(p) for p in patterns]
J_heb = hebbian_coupling(phases)
J_prj = projection_coupling(phases)

asimg(p) = rotr90(reshape(p, 8, 8))

# ---------------------------------------------------------- stored patterns
fig1 = Figure(size=(720, 280))
Label(fig1[0, 1:3], "Stored patterns (8x8, ±1)", fontsize=17, font=:bold)
for (i, (k, p)) in enumerate(zip(KEYS, patterns))
    ax = Axis(fig1[1, i], title="'$k'", aspect=DataAspect())
    heatmap!(ax, asimg(p), colormap=:grays)
    hidedecorations!(ax)
end
save(joinpath(FIGDIR, "kh_patterns.png"), fig1)

# --------------------------------------------------- coupling matrices |J|
fig2 = Figure(size=(820, 380))
Label(fig2[0, 1:2], "Learned coupling magnitude |J_ij|  (64×64)", fontsize=17, font=:bold)
for (j, (name, J)) in enumerate((("Hebbian rule", J_heb), ("Projection rule", J_prj)))
    ax = Axis(fig2[1, j], title=name, aspect=DataAspect(),
              xlabel="neuron j", ylabel="neuron i")
    hm = heatmap!(ax, abs.(J), colormap=:viridis)
    Colorbar(fig2[2, j], hm, vertical=false, flipaxis=false)
end
save(joinpath(FIGDIR, "kh_coupling.png"), fig2)

# -------------------------------------------- retrieval dynamics over time
# Corrupt '2' by 12% and recall under the projection rule (seed 5 -> clean).
N = length(phases[1])
target_idx = 3
target = phases[target_idx]
cue = corrupt(target; frac=0.12, rng=MersenneTwister(5))

Tend = 60.0
prob = ODEProblem(phasor_rhs!, collect(float(cue)), (0.0, Tend),
                  (J=J_prj, omega=zeros(N)))
sol = solve(prob, Tsit5(); saveat=0:0.25:Tend, reltol=1e-8, abstol=1e-8)
ts = sol.t
ov = [[overlap(u, phases[m]) for u in sol.u] for m in 1:3]
en = [energy(J_prj, u) for u in sol.u]

fig3 = Figure(size=(820, 360))
Label(fig3[0, 1:2], "Recall of a 12%-corrupted '2' (projection rule)",
      fontsize=17, font=:bold)
ax3a = Axis(fig3[1, 1], xlabel="time", ylabel="overlap |m^μ|", title="order parameters")
colors = [:gray60, :gray40, :crimson]
for m in 1:3
    lines!(ax3a, ts, ov[m], color=colors[m], linewidth=m == target_idx ? 2.5 : 1.5,
           label="with '$(KEYS[m])'")
end
hlines!(ax3a, [1.0], color=:black, linestyle=:dot)
axislegend(ax3a, position=:rc)
ax3b = Axis(fig3[1, 2], xlabel="time", ylabel="energy E(θ)", title="energy descent")
lines!(ax3b, ts, en, color=:black, linewidth=2)
save(joinpath(FIGDIR, "kh_retrieval_dynamics.png"), fig3)

# ----------------------------------------------- basin: overlap vs noise
noise_levels = 0.0:0.05:0.4
nseed = 6
mean_ov = Dict(:hebbian => Float64[], :projection => Float64[])
for rule in (:hebbian, :projection)
    J = rule == :hebbian ? J_heb : J_prj
    for nf in noise_levels
        vals = Float64[]
        for k in 1:3, s in 1:nseed
            c = corrupt(phases[k]; frac=nf, rng=MersenneTwister(s + 100k))
            rec = retrieve(J, c; T=200.0)
            push!(vals, overlap(rec, phases[k]))
        end
        push!(mean_ov[rule], mean(vals))
    end
end

fig4 = Figure(size=(680, 420))
ax4 = Axis(fig4[1, 1], xlabel="cue noise (fraction of pixels flipped)",
           ylabel="mean final overlap",
           title="Retrieval quality vs cue corruption")
lines!(ax4, collect(noise_levels), mean_ov[:hebbian], color=:steelblue,
       linewidth=2.5, label="Hebbian rule")
scatter!(ax4, collect(noise_levels), mean_ov[:hebbian], color=:steelblue)
lines!(ax4, collect(noise_levels), mean_ov[:projection], color=:crimson,
       linewidth=2.5, label="Projection rule")
scatter!(ax4, collect(noise_levels), mean_ov[:projection], color=:crimson)
hlines!(ax4, [1.0], color=:black, linestyle=:dot)
axislegend(ax4, position=:lb)
save(joinpath(FIGDIR, "kh_basin.png"), fig4)

println("Saved to $(abspath(FIGDIR)):")
for f in ("kh_patterns.png", "kh_coupling.png", "kh_retrieval_dynamics.png", "kh_basin.png")
    println("  - ", f)
end
