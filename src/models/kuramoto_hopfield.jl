# Kuramoto / phasor associative memory  -- an oscillatory Hopfield network.
#
# Stores 8x8 binary patterns as PHASE patterns on a network of phase
# oscillators and retrieves a stored pattern from a noisy cue by letting the
# phases relax. This is the corrected/standalone version of the storage-and-
# retrieval logic in Kuramoto-Izhikevich.jl, with:
#
#   * a Hermitian complex Hebbian rule  J_ij = (1/N) sum_mu z_i^mu conj(z_j^mu),
#     z^mu = exp(i*phi^mu), with no self-coupling (J_ii = 0). Hermitian J
#     guarantees the Lyapunov function below and hence convergence.
#   * binary pixels mapped to two-state phasors  (+1 -> phase 0, -1 -> phase pi)
#     instead of normalizing real +-1 values as in the original.
#   * a gauge-invariant overlap (Hopfield order parameter) as the recognition
#     metric, replacing the cosine distance to a forced mixture.
#   * an optional projection (pseudo-inverse) rule that lifts capacity for the
#     correlated digit patterns.
#
# Dynamics (gradient descent on the energy, for Hermitian J, common omega):
#   dtheta_i/dt = omega_i + sum_j |J_ij| sin(theta_j - theta_i + psi_ij)
#               = omega_i + Im( conj(z_i) * (J z)_i ),     z_i = exp(i theta_i)
# Energy:
#   E(theta) = -1/2 sum_ij |J_ij| cos(theta_i - theta_j - psi_ij)
#            = -1/2 Re( z' J z )
# minimized at the stored phase patterns.

using OrdinaryDiffEq
using LinearAlgebra
using Statistics
using Random

# ----------------------------------------------------------------------------
# Patterns
# ----------------------------------------------------------------------------
# 8x8 bitmaps; '#' = +1 ("on"), '.' = -1 ("off"). Kept reasonably distinct so
# the plain Hebbian rule can store all three.

const PATTERN_BITMAPS = Dict(
    '0' => """
    .######.
    ##....##
    ##....##
    ##....##
    ##....##
    ##....##
    ##....##
    .######.
    """,
    '1' => """
    ...##...
    ..###...
    .####...
    ...##...
    ...##...
    ...##...
    ...##...
    .######.
    """,
    '2' => """
    .######.
    ##....##
    .....##.
    ....##..
    ...##...
    ..##....
    .##.....
    ########
    """,
)

# Parse a bitmap string into a length-64 vector in {-1, +1} (column-major,
# matching reshape(.., 8, 8)).
function bitmap_to_pattern(s::AbstractString)
    rows = [strip(r) for r in split(strip(s), '\n')]
    M = reduce(vcat, [permutedims([c == '#' ? 1.0 : -1.0 for c in r]) for r in rows])
    return vec(M)  # 64-vector
end

binary_to_phase(p::AbstractVector) = [x > 0 ? 0.0 : Float64(pi) for x in p]
phase_to_binary(theta::AbstractVector; ref=0.0) = [cos(t - ref) >= 0 ? 1.0 : -1.0 for t in theta]

# ----------------------------------------------------------------------------
# Storage rules (the "Hopfield prescription" and its generalizations)
# ----------------------------------------------------------------------------

# Hebbian outer-product rule, complex/Hermitian. `phase_patterns` is a vector
# of length-N phase vectors.
function hebbian_coupling(phase_patterns)
    N = length(first(phase_patterns))
    J = zeros(ComplexF64, N, N)
    for phi in phase_patterns
        z = cis.(phi)         # exp(i*phi)
        J .+= z * z'          # z_i * conj(z_j)
    end
    J ./= N
    J[diagind(J)] .= 0        # no self-coupling -> well-defined energy
    return J
end

# Projection / pseudo-inverse rule:  J = Z (Z' Z)^{-1} Z'. Makes every stored
# pattern an exact fixed point even when the patterns are correlated (digits
# are), at the cost of needing all patterns up front.
function projection_coupling(phase_patterns)
    N = length(first(phase_patterns))
    Z = reduce(hcat, (cis.(phi) for phi in phase_patterns))   # N x P
    J = Z * pinv(Z' * Z) * Z'
    J[diagind(J)] .= 0
    return Matrix{ComplexF64}(J)
end

# ----------------------------------------------------------------------------
# Dynamics, energy, overlap
# ----------------------------------------------------------------------------

function phasor_rhs!(dtheta, theta, p, t)
    z = cis.(theta)
    Jz = p.J * z
    @. dtheta = p.omega + imag(conj(z) * Jz)
    return nothing
end

energy(J, theta) = -0.5 * real(cis.(theta)' * J * cis.(theta))

# Hopfield order parameter: |<exp(i(theta - phi))>|, in [0,1], gauge invariant.
# 1.0 == perfect retrieval up to a global phase shift.
overlap(theta, phi) = abs(mean(cis.(theta .- phi)))

# Relax a cue to equilibrium under the stored couplings.
function retrieve(J, theta0; omega=zeros(length(theta0)), T=200.0)
    prob = ODEProblem(phasor_rhs!, collect(float(theta0)), (0.0, T), (J=J, omega=omega))
    sol = solve(prob, Tsit5(); reltol=1e-8, abstol=1e-8,
                save_everystep=false, save_start=false)
    return sol.u[end]
end

# Flip a fraction of pixels of a phase pattern (phase noise: 0 <-> pi).
function corrupt(phi; frac=0.2, rng=Random.default_rng())
    out = copy(phi)
    idx = randperm(rng, length(phi))[1:round(Int, frac * length(phi))]
    out[idx] .= mod.(out[idx] .+ pi, 2pi)
    return out
end

# ----------------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------------

# Store `keys`, corrupt one pattern, retrieve, and return everything.
function run_demo_quiet(; keys=['0', '1', '2'], cue_key='2', noise_frac=0.15,
                        rule=:hebbian, seed=1, T=200.0)
    rng = MersenneTwister(seed)

    patterns = [bitmap_to_pattern(PATTERN_BITMAPS[k]) for k in keys]
    phases = [binary_to_phase(p) for p in patterns]

    J = rule == :projection ? projection_coupling(phases) : hebbian_coupling(phases)

    target_idx = findfirst(==(cue_key), keys)
    target = phases[target_idx]
    cue = corrupt(target; frac=noise_frac, rng=rng)
    recovered = retrieve(J, cue; T=T)

    return (; keys, patterns, phases, J, target, cue, recovered, target_idx)
end

function run_demo(; cue_key='2', noise_frac=0.15, rule=:hebbian, kwargs...)
    res = run_demo_quiet(; cue_key=cue_key, noise_frac=noise_frac, rule=rule, kwargs...)
    keys, phases, target, cue, recovered = res.keys, res.phases, res.target, res.cue, res.recovered

    keystr = join(keys, ", ")
    println("Storage rule: $rule, patterns: $keystr, cue: '$cue_key', noise: $(round(Int, noise_frac*100))%")
    println("Overlap of cue with target:       ", round(overlap(cue, target), digits=4))
    for (k, phi) in zip(keys, phases)
        println("Overlap of recovered with '$k':   ", round(overlap(recovered, phi), digits=4))
    end

    return res
end

# ----------------------------------------------------------------------------
# Script entry point (define KH_SKIP_RUN = true before include() to only load
# the definitions).
# ----------------------------------------------------------------------------
if !@isdefined(KH_SKIP_RUN)
    using CairoMakie

    # Same stored digits and same noisy cue ('2', 12% flipped pixels) presented
    # to both storage rules. Hebbian cannot hold '2' (it is correlated with the
    # other stored digits); the projection rule retrieves it.
    cfg = (cue_key='2', noise_frac=0.12, seed=5)
    heb = run_demo(; cfg..., rule=:hebbian)
    prj = run_demo(; cfg..., rule=:projection)

    FIGDIR = joinpath(@__DIR__, "..", "..", "results", "figures")
    isdir(FIGDIR) || mkpath(FIGDIR)

    asimg(p) = rotr90(reshape(p, 8, 8))
    function panel!(gl, row, title, m)
        ax = Axis(gl[row, 1], title=title, aspect=DataAspect())
        heatmap!(ax, m, colormap=:grays)
        hidedecorations!(ax)
    end

    fig = Figure(size=(820, 560))
    Label(fig[1, 1:3], "Oscillator Hopfield: retrieving a 12%-corrupted '2'",
          fontsize=18, font=:bold)
    # column headers via per-panel titles; rows = rules
    for (r, (name, res)) in enumerate((("Hebbian rule", heb), ("Projection rule", prj)))
        ref = angle(mean(cis.(res.recovered .- res.target)))
        ovr = round(overlap(res.recovered, res.target), digits=3)
        panel!(fig[r+1, 1], 1, r == 1 ? "target '2'" : "", asimg(phase_to_binary(res.target)))
        panel!(fig[r+1, 2], 1, r == 1 ? "noisy cue" : "", asimg(phase_to_binary(res.cue)))
        panel!(fig[r+1, 3], 1, r == 1 ? "recovered (overlap $ovr)" : "overlap $ovr",
               asimg(phase_to_binary(res.recovered; ref=ref)))
        Label(fig[r+1, 0], name, rotation=pi/2, font=:bold)
    end
    save(joinpath(FIGDIR, "kuramoto_hopfield_retrieval.png"), fig)
    println("\nFigure saved to ", abspath(joinpath(FIGDIR, "kuramoto_hopfield_retrieval.png")))
end
