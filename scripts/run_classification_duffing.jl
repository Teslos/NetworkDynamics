"""
run_classification_duffing.jl

Duffing oscillator reservoir computing classification on the optdigits dataset.
Uses a network of coupled Duffing oscillators as a dynamical reservoir,
then trains a Flux dense-layer readout for 10-class digit classification.

Duffing model (per node):
    dxᵢ/dt = vᵢ
    dvᵢ/dt = -δ vᵢ - α xᵢ - β xᵢ³ + Iᵢ(t) + σ Σⱼ Aᵢⱼ (xⱼ − xᵢ)

With α = -1, β = 1 this is the double-well Duffing oscillator,
which exhibits rich nonlinear dynamics suitable for reservoir computing.

Usage:
    julia --project=. scripts/run_classification_duffing.jl
"""

const ROOT = dirname(@__DIR__)
const FIG_DIR  = joinpath(ROOT, "results", "figures")
const CONF_DIR = joinpath(ROOT, "results", "confusion_matrices")
mkpath(FIG_DIR)
mkpath(CONF_DIR)

# ── Configuration ─────────────────────────────────────────────────────────────
const N_RESERVOIR   = 16      # Duffing oscillator nodes in the reservoir
const N_TRAIN       = 3823    # use ALL training samples
const N_TEST        = 1797    # use ALL test samples
const SPIKE_STEPS   = 64      # time steps for spike rate encoding
const ODE_TSPAN     = 200.0   # ODE integration time per sample
const N_SNAPSHOTS   = 8       # temporal snapshots for feature extraction
const EPOCHS        = 200     # training epochs for readout layer
const BATCH_SIZE    = 64
const LEARNING_RATE = 0.001f0

println("=" ^ 60)
println("Duffing Reservoir Computing — Digit Classification")
println("  Reservoir nodes: $N_RESERVOIR")
println("  Train samples:   $N_TRAIN")
println("  Test samples:    $N_TEST")
println("  Epochs:          $EPOCHS")
println("=" ^ 60)

# ── Dependencies ──────────────────────────────────────────────────────────────
using DelimitedFiles
using Graphs
using OrdinaryDiffEq
using Interpolations
using Random
using Flux
using OneHotArrays
using CairoMakie
using Statistics
using LinearAlgebra

# ── 1. Load optdigits data ───────────────────────────────────────────────────
println("\n[1/5] Loading optdigits data...")

function load_optdigits()
    dir = joinpath(ROOT, "data", "digits")
    train_raw = readdlm(joinpath(dir, "optdigits.tra"), ',', Int)
    test_raw  = readdlm(joinpath(dir, "optdigits.tes"), ',', Int)

    train_x = Float64.(train_raw[:, 1:64]) ./ 16.0
    train_y = train_raw[:, 65]
    test_x  = Float64.(test_raw[:, 1:64]) ./ 16.0
    test_y  = test_raw[:, 65]

    println("  Training: $(size(train_raw, 1)) samples, Test: $(size(test_raw, 1)) samples")
    return train_x, train_y, test_x, test_y
end

train_x_full, train_y_full, test_x_full, test_y_full = load_optdigits()

rng = Xoshiro(42)
train_x = train_x_full
train_y = train_y_full
test_x  = test_x_full
test_y  = test_y_full

println("  Using $N_TRAIN train / $N_TEST test samples (all data)")

# ── 2. Build Duffing reservoir network ──────────────────────────────────────
println("\n[2/5] Building Duffing reservoir...")

# Duffing parameters (double-well regime for chaotic dynamics)
const α_duff  = -1.0     # negative linear stiffness (double-well)
const β_duff  =  1.0     # cubic nonlinearity
const δ_duff  =  0.3     # damping coefficient
const σ_coup  =  0.2     # diffusive coupling strength

# Watts-Strogatz small-world topology
const W_adj = Float64.(adjacency_matrix(watts_strogatz(N_RESERVOIR, 4, 0.3; rng=rng)))
println("  Reservoir: $N_RESERVOIR Duffing oscillators, Watts-Strogatz(k=4, β=0.3)")
println("  α=$α_duff, β=$β_duff, δ=$δ_duff, σ=$σ_coup (double-well regime)")

# Overlapping pixel assignments: each node gets 8 pixels with stride 4
const PIXEL_GROUPS = [collect(mod.((i-1)*4 .+ (0:7), 64) .+ 1) for i in 1:N_RESERVOIR]

# ── 3. Spike encoding & reservoir processing ────────────────────────────────
println("\n[3/5] Processing samples through reservoir...")

function encode_per_node(features::Vector{Float64}, n_steps::Int)
    spike_probs = clamp.(features, 0.0, 1.0)
    trains = zeros(Float64, n_steps, N_RESERVOIR)
    for node in 1:N_RESERVOIR
        pixels = PIXEL_GROUPS[node]
        for t in 1:n_steps
            trains[t, node] = mean(rand(rng) < spike_probs[p] ? 1.0 : 0.0 for p in pixels)
        end
    end
    return trains
end

function make_input_fns(trains::Matrix{Float64}, tspan_max::Float64)
    ts = range(0.0, tspan_max, length=size(trains, 1))
    fns = Vector{Any}(undef, N_RESERVOIR)
    for node in 1:N_RESERVOIR
        itp = interpolate(trains[:, node], BSpline(Linear()))
        sitp = Interpolations.scale(itp, ts)
        fns[node] = t -> sitp(clamp(t, first(ts), last(ts)))
    end
    return fns
end

# Per-node external input (set before each ODE solve)
const CURRENT_INPUTS = Vector{Any}(undef, N_RESERVOIR)
for i in 1:N_RESERVOIR; CURRENT_INPUTS[i] = t -> 0.0; end

# Duffing ODE:
#   dxᵢ/dt = vᵢ
#   dvᵢ/dt = -δ vᵢ - α xᵢ - β xᵢ³ + Iᵢ(t) + σ Σⱼ Aᵢⱼ (xⱼ − xᵢ)
# State layout: u[2i-1] = xᵢ (position), u[2i] = vᵢ (velocity)
function duffing_reservoir!(du, u, p, t)
    N = N_RESERVOIR
    for i in 1:N
        xi = u[2i - 1]
        vi = u[2i]
        I_ext = CURRENT_INPUTS[i](t)

        # Diffusive coupling on position variable
        coupling = 0.0
        for j in 1:N
            if W_adj[i, j] > 0
                coupling += (u[2j - 1] - xi)
            end
        end

        du[2i - 1] = vi
        du[2i]     = -δ_duff * vi - α_duff * xi - β_duff * xi^3 + I_ext + σ_coup * coupling
    end
    nothing
end

"""
Run one sample through the Duffing reservoir and extract features.
Duffing has 2 state vars per node (x, v), similar to FHN.
"""
function process_sample(features::Vector{Float64})
    trains = encode_per_node(features, SPIKE_STEPS)
    fns = make_input_fns(trains, ODE_TSPAN)
    for i in 1:N_RESERVOIR; CURRENT_INPUTS[i] = fns[i]; end

    # Initial conditions: small random perturbations near one of the wells
    x0 = zeros(2 * N_RESERVOIR)
    for i in 1:N_RESERVOIR
        x0[2i - 1] = 0.5 * randn(rng)   # position near origin
        x0[2i]     = 0.1 * randn(rng)    # small velocity
    end
    tspan = (0.0, ODE_TSPAN)
    saveat = range(0.0, ODE_TSPAN, length=N_SNAPSHOTS + 1)[2:end]

    prob = ODEProblem(duffing_reservoir!, x0, tspan)
    sol = solve(prob, Tsit5(); saveat=saveat, abstol=1e-6, reltol=1e-4, maxiters=100_000)

    if sol.retcode == ReturnCode.Success || sol.retcode == ReturnCode.Default
        states = hcat(sol.u...)  # (2N, N_SNAPSHOTS)

        # Snapshot features: all state variables at each timepoint
        snapshot_feat = vec(states)  # 2N * N_SNAPSHOTS

        # Per-node summary statistics
        x_states = states[1:2:end, :]   # (N, N_SNAPSHOTS) — positions
        v_states = states[2:2:end, :]   # (N, N_SNAPSHOTS) — velocities

        mean_feat = vcat(vec(mean(x_states, dims=2)), vec(mean(v_states, dims=2)))  # 2N
        var_feat  = vcat(vec(var(x_states, dims=2)),  vec(var(v_states, dims=2)))   # 2N
        max_feat  = vcat(vec(maximum(abs.(x_states), dims=2)),
                         vec(maximum(abs.(v_states), dims=2)))                      # 2N

        return Float32.(vcat(snapshot_feat, mean_feat, var_feat, max_feat))
    else
        n_feat = 2 * N_RESERVOIR * N_SNAPSHOTS + 6 * N_RESERVOIR
        return zeros(Float32, n_feat)
    end
end

# Feature dims: 2*N*S snapshots + 2*N mean + 2*N var + 2*N max + 64 raw pixels
reservoir_feat_dim = 2 * N_RESERVOIR * N_SNAPSHOTS + 6 * N_RESERVOIR
feature_dim = reservoir_feat_dim + 64
println("  Feature dimension: $feature_dim (reservoir: $reservoir_feat_dim + 64 raw pixels)")

train_features = zeros(Float32, N_TRAIN, feature_dim)
test_features  = zeros(Float32, N_TEST, feature_dim)

t_start = time()
for i in 1:N_TRAIN
    res_feat = process_sample(train_x[i, :])
    train_features[i, :] = vcat(res_feat, Float32.(train_x[i, :]))
    if i % 200 == 0
        elapsed = round(time() - t_start, digits=1)
        println("  Train: $i / $N_TRAIN  ($(elapsed)s)")
    end
end

for i in 1:N_TEST
    res_feat = process_sample(test_x[i, :])
    test_features[i, :] = vcat(res_feat, Float32.(test_x[i, :]))
    if i % 200 == 0
        elapsed = round(time() - t_start, digits=1)
        println("  Test:  $i / $N_TEST  ($(elapsed)s)")
    end
end
elapsed_total = round(time() - t_start, digits=1)
println("  Reservoir processing complete ($(elapsed_total)s)")

# ── 4. Train readout classifier ─────────────────────────────────────────────
println("\n[4/5] Training readout classifier...")

classes = 0:9
train_y_oh = onehotbatch(train_y, classes)
test_y_oh  = onehotbatch(test_y, classes)

train_X = train_features'
test_X  = test_features'

model = Chain(
    Dense(feature_dim, 256, relu),
    Dense(256, 128, relu),
    Dense(128, 10),
    softmax
)

opt_state = Flux.setup(Adam(LEARNING_RATE), model)
loader = Flux.DataLoader((train_X, train_y_oh), batchsize=BATCH_SIZE, shuffle=true)

for epoch in 1:EPOCHS
    epoch_loss = 0.0
    n_batches = 0
    for (x, y) in loader
        loss_val, grads = Flux.withgradient(model) do m
            Flux.Losses.crossentropy(m(x), y)
        end
        Flux.update!(opt_state, model, grads[1])
        epoch_loss += loss_val
        n_batches += 1
    end
    if epoch % 10 == 0 || epoch == 1
        avg_loss = round(epoch_loss / n_batches, digits=4)
        train_pred = onecold(model(train_X), classes)
        test_pred  = onecold(model(test_X), classes)
        train_acc  = round(100.0 * mean(train_pred .== train_y), digits=1)
        test_acc   = round(100.0 * mean(test_pred .== test_y), digits=1)
        println("  Epoch $epoch/$EPOCHS — Loss: $avg_loss | Train: $(train_acc)% | Test: $(test_acc)%")
    end
end

# Final accuracy
train_pred = onecold(model(train_X), classes)
test_pred  = onecold(model(test_X), classes)
train_acc = round(100.0 * mean(train_pred .== train_y), digits=2)
test_acc  = round(100.0 * mean(test_pred .== test_y), digits=2)
println("\n  Final — Train accuracy: $(train_acc)% | Test accuracy: $(test_acc)%")

# ── 5. Confusion matrix & plots ─────────────────────────────────────────────
println("\n[5/5] Generating confusion matrix...")

n_classes = length(classes)
conf_mat = zeros(Int, n_classes, n_classes)
for (true_label, pred_label) in zip(test_y, test_pred)
    conf_mat[true_label + 1, pred_label + 1] += 1
end

conf_pct = conf_mat ./ max.(sum(conf_mat, dims=2), 1) .* 100

fig = Figure(size=(900, 750))
ax = Axis(fig[1, 1],
    title="Duffing Reservoir — Digit Classification (Test: $(test_acc)%)",
    xlabel="Predicted", ylabel="True",
    xticks=(1:10, string.(0:9)),
    yticks=(1:10, string.(0:9))
)

hm = heatmap!(ax, 1:10, 1:10, conf_pct', colormap=:blues, colorrange=(0, 100))
Colorbar(fig[1, 2], hm, label="Accuracy (%)")

for i in 1:10, j in 1:10
    val = round(conf_pct[j, i], digits=1)
    txt_color = val > 50 ? :white : :black
    text!(ax, i, j; text="$(val)%", align=(:center, :center),
          color=txt_color, fontsize=10.0)
end

out_path = joinpath(CONF_DIR, "duffing_reservoir_confusion_matrix.png")
CairoMakie.save(out_path, fig)
println("  Saved: $out_path")

conf_txt_path = joinpath(CONF_DIR, "duffing_reservoir_confusion_matrix.txt")
open(conf_txt_path, "w") do io
    println(io, "Duffing Reservoir Digit Classification Results")
    println(io, "=" ^ 50)
    println(io, "Reservoir: $N_RESERVOIR Duffing oscillators (Watts-Strogatz small-world)")
    println(io, "Parameters: α=$α_duff, β=$β_duff, δ=$δ_duff, σ=$σ_coup (double-well)")
    println(io, "Features: $(feature_dim) dims (snapshots + stats + raw pixels)")
    println(io, "Train: $N_TRAIN samples | Test: $N_TEST samples")
    println(io, "Train accuracy: $(train_acc)%")
    println(io, "Test accuracy:  $(test_acc)%")
    println(io, "\nConfusion matrix (rows=true, cols=predicted):")
    println(io, "     ", join(lpad.(0:9, 5)))
    for i in 1:10
        println(io, lpad(i - 1, 3), ": ", join(lpad.(conf_mat[i, :], 5)))
    end
end
println("  Saved: $conf_txt_path")

println("\n✓ Duffing classification complete.")
