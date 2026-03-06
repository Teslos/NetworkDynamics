"""
generate_figures.jl

Generates all diagnostic and result figures for the NetworkDynamics project.
Figures are saved to results/figures/ and results/confusion_matrices/.

Usage:
    julia scripts/generate_figures.jl

Individual sections can be run by setting the corresponding flag to true.
"""

const ROOT = dirname(@__DIR__)
const FIG_DIR  = joinpath(ROOT, "results", "figures")
const CONF_DIR = joinpath(ROOT, "results", "confusion_matrices")

mkpath(FIG_DIR)
mkpath(CONF_DIR)

# ── Feature flags ─────────────────────────────────────────────────────────────
const GEN_FHN_DYNAMICS       = true   # FHN phase-space & raster plots
const GEN_NETWORK_TOPOLOGY   = true   # graph topology visualizations
const GEN_FHN_MNIST          = false  # full FHN-MNIST training (slow)
const GEN_DUFFING_MNIST      = false  # full Duffing-MNIST training (slow)
const GEN_KURAMOTO_MNIST     = false  # full Kuramoto-MNIST training (slow)
const GEN_FHN_DRYBEAN        = false  # full FHN-DryBean training (slow)
const GEN_DUFFING_DRYBEAN    = false  # full Duffing-DryBean training (slow)

println("=" ^ 60)
println("NetworkDynamics — figure generation")
println("Output directories:")
println("  figures  → $FIG_DIR")
println("  matrices → $CONF_DIR")
println("=" ^ 60)

# ── 1. FHN dynamics: phase space, raster, chaos trajectory ───────────────────
if GEN_FHN_DYNAMICS
    println("\n[1/2] Generating FHN dynamics figures...")
    using CairoMakie
    using NetworkDynamics
    using Graphs
    using OrdinaryDiffEq
    using LinearAlgebra
    using Distributions
    using SimpleWeightedGraphs

    include(joinpath(ROOT, "src", "models", "chaos_fhn.jl"))
    using .ChaosFHN

    N = 10
    g = complete_graph(N)
    g_directed = SimpleDiGraph(g)

    σ = 0.7; a = 0.5; ϵ = 0.05; R0 = 0.5

    # NetworkDynamics v0.9 API: esum is the pre-aggregated edge sum
    function fhn_vertex!(dv, v, esum, p, t)
        dv[1] = v[1] - v[1]^3 / 3 - v[2] + esum[1]
        dv[2] = ϵ * (v[1] - a)
        nothing
    end

    function fhn_edge_g!(e_dst, v_s, v_d, p, t)
        e_dst[1] = σ * (v_s[1] - v_d[1])
        nothing
    end

    vertex = VertexModel(f=fhn_vertex!, g=1, dim=2, sym=[:u, :v])
    edge   = EdgeModel(g=AntiSymmetric(fhn_edge_g!), outsym=[:c])
    nd     = Network(g_directed, vertex, edge)

    x0 = randn(2 * N)
    prob = ODEProblem(nd, x0, (0.0, 500.0))
    sol  = solve(prob, Tsit5(); saveat=0.1)
    println("  ODE solved: $(length(sol.t)) time steps")

    # Phase-space plot (first 5 oscillators)
    fig_phase = ChaosFHN.analyze_phase_space(sol, 1:5; tsteps=sol.t)
    out = joinpath(FIG_DIR, "phase_space_5_osc_FHN.png")
    CairoMakie.save(out, fig_phase)
    println("  Saved: $out")

    # Trajectory plot
    fig_traj = Figure(size=(900, 400))
    ax = Axis(fig_traj[1, 1], xlabel="Time", ylabel="u / v")
    for i in 1:N
        lines!(ax, sol.t, [sol.u[k][i]     for k in eachindex(sol.t)], color=(:blue, 0.3))
        lines!(ax, sol.t, [sol.u[k][N + i] for k in eachindex(sol.t)], color=(:red,  0.3))
    end
    out2 = joinpath(FIG_DIR, "fhn_chaos_trajectory.png")
    CairoMakie.save(out2, fig_traj)
    println("  Saved: $out2")

    println("  Done.")
end

# ── 2. Network topology figures ───────────────────────────────────────────────
if GEN_NETWORK_TOPOLOGY
    println("\n[2/2] Generating network topology figures...")
    using CairoMakie
    using GraphMakie
    using Graphs
    using NetworkLayout

    include(joinpath(ROOT, "src", "networks", "graph_utils.jl"))
    using .graph_utils

    topologies = [
        ("complete_graph",        create_complete_graph(8)),
        ("barabasi_albert_graph", create_barabasi_albert_graph(30)),
        ("erdos_renyi_graph",     create_erdos_renyi_graph(20, 0.3)),
        ("watts_strogatz_graph",  create_watts_strogatz_graph(20; k=4, prob=0.15)),
    ]

    for (name, (g, _)) in topologies
        fig = Figure(size=(600, 600))
        ax  = Axis(fig[1, 1], title=replace(name, "_" => " "))
        graphplot!(ax, g; layout=Spring())
        out = joinpath(FIG_DIR, "$(name).png")
        CairoMakie.save(out, fig)
        println("  Saved: $out")
    end
    println("  Done.")
end

# ── 3. Full training runs (opt-in) ────────────────────────────────────────────
if GEN_FHN_MNIST
    println("\n[opt] Running FHN-MNIST experiment...")
    include(joinpath(ROOT, "src", "classification", "FitzHug-Nagumo-MNIST.jl"))
end

if GEN_DUFFING_MNIST
    println("\n[opt] Running Duffing-MNIST experiment...")
    include(joinpath(ROOT, "src", "classification", "Duffing-MNIST.jl"))
end

if GEN_KURAMOTO_MNIST
    println("\n[opt] Running Kuramoto-MNIST experiment...")
    include(joinpath(ROOT, "src", "classification", "Kuramoto-MNIST.jl"))
end

if GEN_FHN_DRYBEAN
    println("\n[opt] Running FHN-DryBean experiment...")
    include(joinpath(ROOT, "src", "classification", "FitzHug-Nagumo-DryBean.jl"))
end

if GEN_DUFFING_DRYBEAN
    println("\n[opt] Running Duffing-DryBean experiment...")
    include(joinpath(ROOT, "src", "classification", "Duffing-DryBean.jl"))
end

println("\n✓ Figure generation complete.")
