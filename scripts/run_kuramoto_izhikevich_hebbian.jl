# Kuramoto-Izhikevich Hebbian pattern recognition -- PyCall-free port.
#
# Reproduces the associative-memory / pattern-recognition experiment of the paper's
# "Pattern recognition using Hebbian learning" section, whose original code
# (src/models/Kuramoto-Izhikevich.jl) depended on PyCall (@sk_import load_digits),
# GLMakie (needs a display), and a specific NetworkDynamics version. This runner
# reuses the corrected, standalone dynamics in src/models/kuramoto_hopfield.jl
# (complex Hermitian Hebbian coupling J_ij = (1/N) sum_k z_i^k conj(z_j^k), phasor
# relaxation dtheta = omega + Im(conj(z) (Jz))) and loads the digit patterns from
# the local data/digits/optdigits.tes (== sklearn load_digits), so it runs with no
# PyCall, no display, and no NetworkDynamics dependency.
#
# Matches the paper: two 8x8 binary digit patterns are stored via the complex
# Hebbian rule; a corrupted cue of one digit is presented and the network relaxes
# to recover the stored pattern (the paper recognizes the digit "1").
#
# Usage: julia --project=. scripts/run_kuramoto_izhikevich_hebbian.jl

using DelimitedFiles, Statistics, Random

KH_SKIP_RUN = true
include(joinpath(@__DIR__, "..", "src", "models", "kuramoto_hopfield.jl"))
using CairoMakie

const STORE_DIGITS = [1, 2]   # patterns memorised (as in the paper)
const CUE_DIGIT    = 1        # digit to recognise from a corrupted cue
const NOISE_FRAC   = 0.15
const THRESH       = 5        # binarise optdigits pixels (0..16): > THRESH -> +1
const SEED         = 1

# Load one representative 8x8 pattern per requested digit from optdigits.tes,
# binarised to {-1, +1} (64-vector, column-major to match reshape(.,8,8)).
function digit_patterns(labels; path=joinpath(@__DIR__, "..", "data", "digits", "optdigits.tes"))
    raw = readdlm(path, ',', Int)
    px, lab = raw[:, 1:64], raw[:, 65]
    out = Dict{Int,Vector{Float64}}()
    for d in labels
        i = findfirst(==(d), lab)
        img = permutedims(reshape(Float64.(px[i, :]), 8, 8))   # 8x8 (row-major source)
        out[d] = vec([v > THRESH ? 1.0 : -1.0 for v in img])
    end
    return out
end

rng = MersenneTwister(SEED)
pats = digit_patterns(STORE_DIGITS)
patterns = [pats[d] for d in STORE_DIGITS]
phases   = [binary_to_phase(p) for p in patterns]

J = hebbian_coupling(phases)                      # complex Hermitian Hebbian rule
target = phases[findfirst(==(CUE_DIGIT), STORE_DIGITS)]
cue = corrupt(target; frac=NOISE_FRAC, rng=rng)   # noisy cue of the '1'
recovered = retrieve(J, cue; T=200.0)             # relax to a stored attractor

println("Kuramoto-Izhikevich Hebbian pattern recognition (PyCall-free)")
println("stored digits: ", STORE_DIGITS, ", cue digit: ", CUE_DIGIT,
        ", noise: ", round(Int, 100NOISE_FRAC), "%")
println("overlap(cue, target)       = ", round(overlap(cue, target), digits=4))
for (d, phi) in zip(STORE_DIGITS, phases)
    println("overlap(recovered, '$d')     = ", round(overlap(recovered, phi), digits=4))
end

# --------------------------------------------------------------- figure
FIGDIR = joinpath(@__DIR__, "..", "results", "figures"); isdir(FIGDIR) || mkpath(FIGDIR)
asimg(p) = rotr90(reshape(p, 8, 8))
ref = angle(mean(cis.(recovered .- target)))
ovr = round(overlap(recovered, target), digits=3)

fig = Figure(size=(900, 320))
Label(fig[1, 1:4], "Kuramoto-Izhikevich Hebbian recall of digit '$CUE_DIGIT' (stored $(STORE_DIGITS))",
      fontsize=17, font=:bold)
for (col, (ttl, m)) in enumerate((
        ("stored '$(STORE_DIGITS[1])'", asimg(phase_to_binary(phases[1]))),
        ("stored '$(STORE_DIGITS[2])'", asimg(phase_to_binary(phases[2]))),
        ("noisy cue ($(round(Int,100NOISE_FRAC))%)", asimg(phase_to_binary(cue))),
        ("recovered (overlap $ovr)", asimg(phase_to_binary(recovered; ref=ref)))))
    ax = Axis(fig[2, col], title=ttl, aspect=DataAspect())
    heatmap!(ax, m, colormap=:grays); hidedecorations!(ax)
end
outfile = joinpath(FIGDIR, "kuramoto_izhikevich_hebbian.png")
save(outfile, fig)
println("\nFigure saved to ", abspath(outfile))
