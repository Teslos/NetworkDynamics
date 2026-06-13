# Avalanche-criticality diagnostics, following the branching/self-organized-
# criticality framework used by Beattie et al. 2024 (Communications Physics) for
# FitzHugh-Nagumo oscillator ensembles. This is the *correct* notion of
# criticality for that paper -- power-law avalanche statistics and a branching
# ratio ~ 1 -- as opposed to the dynamical-systems edge of chaos (Lyapunov
# exponent) probed in run_reservoir_diagnostics.jl.
#
# An avalanche is a maximal run of consecutive non-empty time bins of population
# spiking activity; its size is the total spike count and its duration the number
# of bins. At criticality:  P(size) ~ size^-tau (tau ~ 1.5 mean-field),
# P(dur) ~ dur^-alpha (alpha ~ 2.0), branching ratio ~ 1.

module AvalancheCriticality

using Statistics

export detect_spikes, population_activity, avalanche_stats, powerlaw_mle, branching_ratio

# Rising-edge threshold crossings of each node's voltage -> Bool (N, T).
function detect_spikes(U; thresh=1.0)
    N, T = size(U)
    sp = falses(N, T)
    @inbounds for i in 1:N, t in 2:T
        if U[i, t-1] < thresh <= U[i, t]
            sp[i, t] = true
        end
    end
    return sp
end

# Total spikes across nodes per time bin (bin = number of integration steps).
function population_activity(sp; bin=1)
    counts = vec(sum(sp, dims=1))
    bin == 1 && return counts
    nb = div(length(counts), bin)
    return [sum(@view counts[(b-1)*bin+1 : b*bin]) for b in 1:nb]
end

# Avalanches = maximal runs of bins with activity > 0. Returns (sizes, durations).
function avalanche_stats(a)
    sizes = Int[]; durs = Int[]
    s = 0; d = 0
    for x in a
        if x > 0
            s += x; d += 1
        elseif d > 0
            push!(sizes, s); push!(durs, d); s = 0; d = 0
        end
    end
    d > 0 && (push!(sizes, s); push!(durs, d))
    return sizes, durs
end

# Discrete power-law MLE exponent for x >= xmin (Clauset-Shalizi-Newman approx).
# Returns NaN if too few samples.
function powerlaw_mle(x; xmin=1)
    xx = filter(>=(xmin), x)
    n = length(xx)
    n < 10 && return NaN
    return 1 + n / sum(log.(xx ./ (xmin - 0.5)))
end

# Naive branching ratio: mean over active bins of a[t+1]/a[t]. ~1 at criticality,
# <1 subcritical (activity dies out), >1 supercritical (runaway/synchronized).
function branching_ratio(a)
    r = Float64[]
    @inbounds for t in 1:length(a)-1
        a[t] > 0 && push!(r, a[t+1] / a[t])
    end
    return isempty(r) ? NaN : mean(r)
end

end # module AvalancheCriticality
