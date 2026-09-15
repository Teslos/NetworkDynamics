# Figure 1: the four network topologies, as a vector PDF.
#
# The manuscript shipped this as a 600x438 raster (images/image.png) with no
# generating code, so it could not be re-rendered or corrected. This script
# redraws it with the SAME topology parameters the experiments use
# (see build_graph in the reservoir scripts):
#     Barabasi-Albert   barabasi_albert(n, 4)
#     Erdos-Renyi       erdos_renyi(n, 0.1)
#     Watts-Strogatz    watts_strogatz(n, 8, 0.25)
#     complete          complete_graph(n)
# so the figure illustrates the graphs that were actually run rather than
# generic examples.
#
# Usage: julia --project=. scripts/plot_topologies.jl [out.pdf]

using Graphs, Random, Printf
using CairoMakie, GraphMakie, NetworkLayout

out = length(ARGS) >= 1 ? ARGS[1] :
      joinpath(@__DIR__, "..", "results", "figures", "topologies.pdf")
const SEED = 7

# n chosen per panel for legibility: the complete graph saturates visually well
# below the size used in the experiments.
"Largest connected component. At the reduced n used for legibility, Erdos-Renyi at
p = 0.1 is typically disconnected; an isolated vertex forces the spring layout to
expand the axis and crush the rest of the graph. At the n = 256 actually used in the
experiments the same p gives mean degree ~25 and a connected graph."
function giant(g)
    comps = connected_components(g)
    length(comps) == 1 && return g
    keep = comps[argmax(length.(comps))]
    return induced_subgraph(g, keep)[1]
end

panels = [
    ("a)", "Barabási–Albert", () -> giant(barabasi_albert(40, 4; rng = Xoshiro(SEED)))),
    ("b)", "Erdős–Rényi",     () -> giant(erdos_renyi(40, 0.1; rng = Xoshiro(SEED)))),
    ("c)", "Watts–Strogatz",  () -> giant(watts_strogatz(40, 4, 0.25; rng = Xoshiro(SEED)))),
    ("d)", "complete",        () -> complete_graph(20)),
]

fig = Figure(size = (720, 560))
for (k, (tag, name, mk)) in enumerate(panels)
    r, c = fldmod1(k, 2)
    g = mk()
    ax = Axis(fig[r, c]; aspect = DataAspect())
    hidedecorations!(ax); hidespines!(ax)
    graphplot!(ax, g;
               layout = Spring(seed = SEED),
               node_color = (:steelblue, 0.95),
               node_size = 13,
               edge_color = (:gray30, 0.55),
               edge_width = 0.7)
    # panel letter OUTSIDE the axis: placing it inside collided with the graph
    Label(fig[r, c, TopLeft()], tag; fontsize = 15, padding = (0, 6, 4, 0),
          halign = :left)
    @printf("  %s %-18s n=%3d  edges=%4d\n", tag, name, nv(g), ne(g))
end
rowgap!(fig.layout, 4); colgap!(fig.layout, 4)

mkpath(dirname(out))
save(out, fig)
println("wrote ", out)
