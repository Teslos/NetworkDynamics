# Train the Equilibrium Propagation XY network on XOR and save diagnostic
# figures to results/figures/.
#
# Usage: julia --project=. scripts/run_ep_xor.jl

EP_XY_SKIP_RUN = true
include(joinpath(@__DIR__, "..", "notebooks", "EP-XY-Network-Claude.jl"))

using Random
using CairoMakie

const FIGDIR = joinpath(@__DIR__, "..", "results", "figures")
isdir(FIGDIR) || mkpath(FIGDIR)

Random.seed!(1234)
network, cost_history, test_results = run_experiment(N_epoch=5000)

training_data = (π / 2) * [-1 -1; -1 1; 1 -1; 1 1]
training_target = (π / 2) * [-1.0, 1.0, 1.0, -1.0]
input_labels = ["(-1, -1)", "(-1, +1)", "(+1, -1)", "(+1, +1)"]

# ---------------------------------------------------------------- cost history
fig = Figure(size=(700, 450))
ax = Axis(fig[1, 1], xlabel="Epoch", ylabel="Cost  ⟨1 − cos(φ_out − target)⟩ / 2",
          title="Equilibrium Propagation training on XOR", yscale=log10)
lines!(ax, 1:length(cost_history), max.(cost_history, 1e-12), color=:black, linewidth=1)
save(joinpath(FIGDIR, "ep_xor_cost_history.png"), fig)

# ------------------------------------------- free relaxation per input pattern
fig2 = Figure(size=(950, 650))
ax1 = nothing
for i in 1:4
    r, c = divrem(i - 1, 2)
    local ax = Axis(fig2[r + 1, c + 1], xlabel="t", ylabel="phase / (π/2)",
                    title="Input $(input_labels[i])  →  target $(round(training_target[i] / (π/2), digits=0))")
    i == 1 && (global ax1 = ax)

    phase0 = zeros(network.N)
    phase0[network.input_index] .= training_data[i, :]
    p = force_params(network.weights, network.bias, [0.0], 0.0,
                     network.input_index, network.output_index)
    prob = ODEProblem(xy_force!, phase0, (0.0, network.T), p)
    sol = solve(prob, Tsit5(); reltol=1e-6, abstol=1e-8, saveat=0.02)

    for j in 1:network.N
        vals = [u[j] / (π / 2) for u in sol.u]
        if j == network.output_index[1]
            lines!(ax, sol.t, vals, color=:crimson, linewidth=2.5,
                   label=(i == 1 ? "output neuron" : nothing))
        elseif j in network.input_index
            lines!(ax, sol.t, vals, color=:steelblue, linewidth=1.5,
                   label=(i == 1 && j == 1 ? "input neurons (clamped)" : nothing))
        else
            lines!(ax, sol.t, vals, color=:gray60, linewidth=1.0,
                   label=(i == 1 && j == 3 ? "hidden neurons" : nothing))
        end
    end
    hlines!(ax, [training_target[i] / (π / 2)], color=:crimson, linestyle=:dash,
            label=(i == 1 ? "target" : nothing))
end
axislegend(ax1, position=:rc)
Label(fig2[0, :], "Free relaxation of the trained network (β = 0)", fontsize=18)
save(joinpath(FIGDIR, "ep_xor_relaxation.png"), fig2)

# ------------------------------------------------------- outputs vs targets
fig3 = Figure(size=(600, 420))
ax3 = Axis(fig3[1, 1], xlabel="Input pattern", ylabel="Output phase / (π/2)",
           xticks=(1:4, input_labels), title="Trained outputs vs XOR targets")
scatter!(ax3, 1:4, training_target ./ (π / 2), marker=:hline, markersize=30,
         color=:black, label="target")
scatter!(ax3, 1:4, test_results ./ (π / 2), color=:crimson, markersize=14,
         label="network output")
axislegend(ax3, position=:ct)
save(joinpath(FIGDIR, "ep_xor_outputs.png"), fig3)

# ----------------------------------------------------- weights before / after
fig4 = Figure(size=(900, 380))
wmax = maximum(abs, [network.weights_0; network.weights])
ax4a = Axis(fig4[1, 1], title="Initial weights", xlabel="neuron j", ylabel="neuron k")
heatmap!(ax4a, network.weights_0, colormap=:RdBu, colorrange=(-wmax, wmax))
ax4b = Axis(fig4[1, 2], title="Trained weights", xlabel="neuron j", ylabel="neuron k")
hm = heatmap!(ax4b, network.weights, colormap=:RdBu, colorrange=(-wmax, wmax))
Colorbar(fig4[1, 3], hm)
save(joinpath(FIGDIR, "ep_xor_weights.png"), fig4)

println()
println("Figures saved to $(abspath(FIGDIR)):")
for f in ("ep_xor_cost_history.png", "ep_xor_relaxation.png", "ep_xor_outputs.png", "ep_xor_weights.png")
    println("  - ", f)
end
