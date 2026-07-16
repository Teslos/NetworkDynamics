# Kuramoto-Izhikevich Hebbian pattern recognition -- PyCall-free port.
#
# Reproduces the associative-memory / pattern-recognition experiment of the paper's
# "Pattern recognition using Hebbian learning" section, whose original code
# (src/models/Kuramoto-Izhikevich.jl) depended on PyCall (@sk_import load_digits),
# GLMakie (needs a display), and a specific NetworkDynamics version. This runner
# reuses the corrected, standalone dynamics in src/models/kuramoto_hopfield.jl
# (complex outer-product coupling, phasor relaxation dtheta = omega + Im(conj(z)(Jz)))
# and loads the digit patterns from the local data/digits/optdigits.tes (== sklearn
# load_digits), so it runs with no PyCall, no display, and no NetworkDynamics dep.
#
# Outputs:
#   results/figures/all_test.png                 -- the paper's Fig. "all_test":
#       row 1 = noisy test patterns, row 2 = patterns recognised by the network.
#   results/figures/kuramoto_izhikevich_hebbian.png -- a single Hebbian recall.
#
# Usage: julia --project=. scripts/run_kuramoto_izhikevich_hebbian.jl

using DelimitedFiles, Statistics, Random, Printf

KH_SKIP_RUN = true
include(joinpath(@__DIR__, "..", "src", "models", "kuramoto_hopfield.jl"))
using CairoMakie

const STORE_DIGITS = [0, 1, 2, 3, 4]   # digits memorised for the recognition grid
const NOISE_FRAC   = 0.08              # fraction of pixels flipped in each test cue
const THRESH       = 5                 # binarise optdigits pixels (0..16): > THRESH -> +1
const SEED         = 1

# One representative 8x8 pattern per digit from optdigits.tes, binarised to {-1,+1}
# (64-vector, column-major to match reshape(.,8,8)).
function digit_patterns(labels; path=joinpath(@__DIR__, "..", "data", "digits", "optdigits.tes"))
    raw = readdlm(path, ',', Int); px, lab = raw[:, 1:64], raw[:, 65]
    out = Dict{Int,Vector{Float64}}()
    for d in labels
        i = findfirst(==(d), lab)
        img = permutedims(reshape(Float64.(px[i, :]), 8, 8))
        out[d] = vec([v > THRESH ? 1.0 : -1.0 for v in img])
    end
    return out
end

rng = MersenneTwister(SEED)
pats     = digit_patterns(STORE_DIGITS)
patterns = [pats[d] for d in STORE_DIGITS]
phases   = [binary_to_phase(p) for p in patterns]

# Digits are correlated, so the plain Hebbian rule cannot make each an exact fixed
# point; the projection (pseudo-inverse) rule -- the complex Hebbian generalisation
# used here -- stores all of them. (A 2-digit Hebbian recall is shown separately.)
Jproj = projection_coupling(phases)

asimg(p) = rotr90(reshape(p, 8, 8))
FIGDIR = joinpath(@__DIR__, "..", "results", "figures"); isdir(FIGDIR) || mkpath(FIGDIR)

# ---- Fig. all_test: noisy test patterns (row 1) -> recognised patterns (row 2) ----
println("Kuramoto-Izhikevich Hebbian pattern recognition (PyCall-free)")
println("stored digits: ", STORE_DIGITS, ", noise: ", round(Int, 100NOISE_FRAC), "%\n")
cues = Vector{Vector{Float64}}(undef, length(STORE_DIGITS))
recs = Vector{Vector{Float64}}(undef, length(STORE_DIGITS))
for (k, d) in enumerate(STORE_DIGITS)
    cues[k] = corrupt(phases[k]; frac=NOISE_FRAC, rng=rng)
    recs[k] = retrieve(Jproj, cues[k]; T=200.0)
    @printf("  digit '%d': overlap(recovered,'%d') = %.3f\n", d, d, overlap(recs[k], phases[k]))
end

fig = Figure(size=(180*length(STORE_DIGITS), 420))
Label(fig[1, 1:length(STORE_DIGITS)],
      "Pattern recognition with an 8x8 Kuramoto oscillator network", fontsize=17, font=:bold)
for (k, d) in enumerate(STORE_DIGITS)
    ax1 = Axis(fig[2, k], title="test '$d'", aspect=DataAspect())
    heatmap!(ax1, asimg(phase_to_binary(cues[k])), colormap=:grays); hidedecorations!(ax1)
    refk = angle(mean(cis.(recs[k] .- phases[k])))
    ax2 = Axis(fig[3, k], aspect=DataAspect())
    heatmap!(ax2, asimg(phase_to_binary(recs[k]; ref=refk)), colormap=:grays); hidedecorations!(ax2)
end
Label(fig[2, 0], "a) test patterns", rotation=pi/2, font=:bold)
Label(fig[3, 0], "b) recognised", rotation=pi/2, font=:bold)
save(joinpath(FIGDIR, "all_test.png"), fig)
println("\nFigure saved to ", abspath(joinpath(FIGDIR, "all_test.png")))

# ---- single Hebbian recall of digit '1' (the plain complex Hebbian rule) ----
h_store = [0, 1]                                   # two well-separated digits
hp = [binary_to_phase(digit_patterns(h_store)[d]) for d in h_store]
Jheb = hebbian_coupling(hp)
tgt = hp[2]; cue = corrupt(tgt; frac=0.15, rng=rng); rec = retrieve(Jheb, cue; T=200.0)
@printf("\nHebbian rule (store %s): overlap(recovered,'1') = %.3f\n", h_store, overlap(rec, tgt))
fig2 = Figure(size=(720, 240))
for (col, (ttl, m)) in enumerate((("stored '0'", asimg(phase_to_binary(hp[1]))),
        ("stored '1'", asimg(phase_to_binary(hp[2]))),
        ("noisy cue", asimg(phase_to_binary(cue))),
        ("recovered", asimg(phase_to_binary(rec; ref=angle(mean(cis.(rec .- tgt))))))))
    ax = Axis(fig2[1, col], title=ttl, aspect=DataAspect())
    heatmap!(ax, m, colormap=:grays); hidedecorations!(ax)
end
save(joinpath(FIGDIR, "kuramoto_izhikevich_hebbian.png"), fig2)

# ---- Fig. kuramoto_network_diff_phase: forcing-driven recognition (time evolution) ----
# Reproduces the paper's forcing procedure (params omega=1, eps=100, h=0.1, T_f=400):
# for t < T_f the oscillators are injection-locked to a noisy cue of digit '1' with
# the coupling off; at t = T_f the forcing is switched off and the Hebbian coupling
# is restored, and the network relaxes to the stored '1' attractor. We plot the phase
# of each node relative to node 1 over time; nodes on the '1' stroke (resonant) are
# drawn in red, the rest in blue.
const EPS_F = 100.0; const H_F = 0.1; const TF = 400.0; const TEND = 900.0
fp    = [binary_to_phase(pats[0]), binary_to_phase(pats[1])]   # store '0' and '1' (Hebbian)
Jf    = hebbian_coupling(fp)
cue1  = corrupt(fp[2]; frac=0.12, rng=rng)                     # noisy '1'
onpix = pats[1] .> 0                                            # resonant pixels of '1'

function forcing_rhs!(dθ, θ, p, t)
    if t < p.tf
        @inbounds @. dθ = p.k * sin(p.cue - θ)                 # inject-lock to cue, coupling off
    else
        z = cis.(θ); Jz = p.J * z
        @inbounds @. dθ = imag(conj(z) * Jz)                   # Hebbian relaxation, forcing off
    end
    return nothing
end

θ0   = 2pi .* rand(rng, length(cue1)) .- pi
probf = ODEProblem(forcing_rhs!, θ0, (0.0, TEND), (k=EPS_F*H_F, cue=cue1, tf=TF, J=Jf))
solf  = solve(probf, Tsit5(); saveat=0.0:1.0:TEND, reltol=1e-6, abstol=1e-8)
tt    = solf.t
Θ     = reduce(hcat, solf.u)
dphi  = mod2pi.(Θ .- Θ[1:1, :] .+ pi) .- pi                    # phase diff to node 1, in (-pi, pi]

figf = Figure(size=(860, 430))
axf = Axis(figf[1, 1], xlabel="time",
           ylabel="phase relative to node 1  (θᵢ − θ₁)",
           title="Forcing-driven recognition of digit '1'  (ε=100, h=0.1, T_f=400)")
for i in 2:size(Θ, 1)
    lines!(axf, tt, dphi[i, :], color=onpix[i] ? (:crimson, 0.35) : (:steelblue, 0.13))
end
vlines!(axf, [TF], color=:black, linestyle=:dash)
text!(axf, TF + 8, 2.9; text="forcing off", align=(:left, :top))
save(joinpath(FIGDIR, "kuramoto_network_diff_phase.png"), figf)
@printf("Forcing recognition: overlap(final,'1') = %.3f\n", overlap(solf.u[end], fp[2]))
println("Figure saved to ", abspath(joinpath(FIGDIR, "kuramoto_network_diff_phase.png")))
