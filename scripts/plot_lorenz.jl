"""
Plot the corrected Lorenz ESN simulation from `baseline_models.jl`.

The figure compares the true and autonomous ESN trajectories in 3D phase
space and in the x(t), y(t), and z(t) coordinates.  It also shows the
normalized forecast error and corrected valid-time threshold.

Run with:
    julia --project=. scripts/plot_lorenz.jl
    julia --project=. scripts/plot_lorenz.jl --full

The default uses the quick-baseline reservoir size (300); `--full` uses 500.
"""

include(joinpath(@__DIR__, "..", "src", "baselines", "baseline_models.jl"))
using .BaselineModels
using CairoMakie
using Random

const DT = 0.02
const N_POINTS = 12_000
const TRAIN_LEN = 6_000
const HORIZON = 3_000
const RESERVOIR_SIZE = "--full" in ARGS ? 500 : 300
const VALID_THRESHOLD = 0.4
const OUTDIR = joinpath(@__DIR__, "..", "results", "figures")
const OUTFILE = joinpath(OUTDIR, "lorenz_esn_attractor_coordinates.png")

isdir(OUTDIR) || mkpath(OUTDIR)

println("Running corrected Lorenz ESN simulation (reservoir size ", RESERVOIR_SIZE, ")...")
data = lorenz_data(T=N_POINTS)
result = esn_lorenz(data;
                    Nr=RESERVOIR_SIZE,
                    train_len=TRAIN_LEN,
                    horizon=HORIZON,
                    valid_thresh=VALID_THRESHOLD,
                    rng=Xoshiro(1))

nshow = size(result.truth, 2)
t = (1:nshow) .* DT
valid_time = result.valid_steps * DT

fig = Figure(size=(1_400, 1_000), fontsize=14)
Label(fig[0, 1:2], "Corrected Lorenz ESN forecast (seed 1)", fontsize=22, font=:bold)

# True and predicted attractors in 3D phase space.
ax3 = Axis3(fig[1:3, 1],
            xlabel="x", ylabel="y", zlabel="z",
            title="Lorenz attractor: truth vs autonomous ESN",
            azimuth=0.65π, elevation=0.25π)
lines!(ax3, result.truth[1, 1:nshow], result.truth[2, 1:nshow], result.truth[3, 1:nshow],
       color=(:black, 0.65), linewidth=1.2, label="truth")
lines!(ax3, result.pred[1, 1:nshow], result.pred[2, 1:nshow], result.pred[3, 1:nshow],
       color=(:crimson, 0.70), linewidth=1.2, label="ESN")
axislegend(ax3, position=:rt)

coord_names = ("x(t)", "y(t)", "z(t)")
for c in 1:3
    ax = Axis(fig[c, 2],
              xlabel=(c == 3 ? "time" : ""),
              ylabel=coord_names[c],
              title=(c == 1 ? "Coordinate forecasts" : ""))
    lines!(ax, t, result.truth[c, 1:nshow], color=:black, linewidth=1.5,
           label="truth")
    lines!(ax, t, result.pred[c, 1:nshow], color=:crimson, linestyle=:dash,
           linewidth=1.4, label="ESN")
    vlines!(ax, [valid_time], color=:gray45, linestyle=:dot, linewidth=1.5)
    c == 1 && axislegend(ax, position=:rt)
    c < 3 && hidexdecorations!(ax, grid=false)
end

axerr = Axis(fig[4, 1:2], xlabel="time", ylabel="normalized error",
             title="Forecast error and valid-time threshold")
lines!(axerr, t, result.err[1:nshow], color=:steelblue, linewidth=1.4)
hlines!(axerr, [VALID_THRESHOLD], color=:gray45, linestyle=:dash)
vlines!(axerr, [valid_time], color=:gray45, linestyle=:dot, linewidth=1.5)
text!(axerr, valid_time, VALID_THRESHOLD;
      text="  valid = $(result.valid_steps) steps ($(round(valid_time, digits=2)) time units)",
      align=(:left, :bottom), fontsize=12)

save(OUTFILE, fig)
println("Valid time: ", result.valid_steps, " steps (", round(valid_time, digits=2), " time units)")
println("Full-horizon NRMSE: ", round(result.nrmse, digits=4))
println("Wrote: ", OUTFILE)
