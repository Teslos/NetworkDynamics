# Plot a stored confusion matrix in the style of the paper's Figure 4
# (confusion_matrix_mnist_ridge_percent.pdf): row-normalised percentages, Blues
# colormap, 0-100 colourbar, per-cell labels.
#
# Figure 5 was previously a PNG from a different toolchain with different fonts,
# and its underlying numbers existed only as pixels. Both figures can now be drawn
# by this one script, so they are consistent by construction.
#
# Usage:
#   julia --project=. scripts/plot_confusion_matrix.jl \
#         results/confusion_matrices/fhn_drybean64_confusion_matrix.txt \
#         paper/images/confusion_matrix_dry_bean_percent.pdf \
#         "FHN Reservoir, Dry-Bean - Confusion Matrix (%)"

using DelimitedFiles, Printf, Statistics
using CairoMakie

length(ARGS) >= 2 || error("usage: plot_confusion_matrix.jl <matrix.txt> <out.pdf> [title]")
infile, outfile = ARGS[1], ARGS[2]
title_str = length(ARGS) >= 3 ? ARGS[3] : "Confusion Matrix (%)"

# ----- read: '#'-comment header carries the class names, then the integer matrix
# read inside a let-block: a bare assignment to `classes` in a top-level `for`
# would create a new local and silently drop the class names
classes, rows = let cls = String[], rs = Vector{Vector{Int}}()
    for line in eachline(infile)
        s = strip(line)
        isempty(s) && continue
        if startswith(s, "#")
            m = match(r"^#\s*classes:\s*(.+)$", s)
            m !== nothing && (cls = string.(strip.(split(m.captures[1], ","))))
            continue
        end
        push!(rs, parse.(Int, split(s)))
    end
    cls, rs
end
C = reduce(vcat, permutedims.(rows))
K = size(C, 1)
isempty(classes) && (classes = string.(0:(K - 1)))
length(classes) == K || error("class count $(length(classes)) != matrix size $K")

# row-normalise to percentages (each true class sums to 100)
P = 100 .* C ./ max.(sum(C, dims = 2), 1)

# ----- draw.
# Figure 4's convention: the y axis ascends (first class at the bottom) while the
# x axis DESCENDS left-to-right, so the diagonal runs top-left to bottom-right.
# It is an unusual choice, but Figure 5 must match it to be consistent.
xlabs = reverse(classes)      # x: last class on the left
ylabs = classes               # y: first class at the bottom
# value drawn at (x, y) is P[true = y, predicted = K+1-x]
val(x, y) = P[y, K + 1 - x]

long = maximum(length.(classes)) > 3
fig = Figure(size = (1000, 780), fontsize = 13)
ax = Axis(fig[1, 1], title = title_str, titlesize = 15,
          xlabel = "Predicted Class", ylabel = "True Class",
          xticks = (1:K, xlabs), yticks = (1:K, ylabs),
          xticklabelrotation = long ? pi / 4 : 0.0,
          xticklabelsize = long ? 11 : 13, yticklabelsize = long ? 11 : 13,
          aspect = DataAspect())
Z = [val(x, y) for x in 1:K, y in 1:K]          # Z[x, y] -> heatmap at (x, y)
hm = heatmap!(ax, 1:K, 1:K, Z, colormap = :Blues, colorrange = (0, 100))
for x in 1:K, y in 1:K
    v = val(x, y)
    text!(ax, x, y; text = @sprintf("%.1f%%", v), align = (:center, :center),
          color = v > 55 ? :white : :black, fontsize = long ? 9 : 10)
end
Colorbar(fig[1, 2], hm, label = "Percentage (%)", ticks = 0:50:100)
colsize!(fig.layout, 1, Aspect(1, 1.0))
resize_to_layout!(fig)

mkpath(dirname(outfile))
save(outfile, fig)
acc = sum(C[i, i] for i in 1:K) / sum(C)
@printf("wrote %s   (%d classes, %d test samples, accuracy %.4f)\n",
        outfile, K, sum(C), acc)
