# Diagnose per-edge (pair) UDE XOR models: does the output settle, or keep moving?
using JLD2, OrdinaryDiffEq, Printf, StableRNGs
const N = 5
const PSI_IN = [-π/2 -π/2; -π/2 π/2; π/2 -π/2; π/2 π/2]
const TARGETS = [-π/2, π/2, π/2, -π/2]
function rhs(φ, p)
    K = reshape(p[1:N*N], N, N); h_out, psi_out = p[N*N+1], p[N*N+2]
    dφ = zeros(N)
    for i in 3:N                      # inputs 1,2 clamped
        s = 0.0
        for j in 1:N
            j != i && (s += K[i, j] * sin(φ[j] - φ[i]))
        end
        i == N && (s -= h_out * sin(φ[i] - psi_out))
        dφ[i] = s
    end
    return dφ
end
for seed in parse.(Int, ARGS)
    p = JLD2.load(expanduser("~/CODE/NetworkDynamics/results/models/Wang-Kuramoto-ude-xor-N5_clamp_outfield_pair_seed$seed.jld2"), "nn_params")
    hidden0 = 0.5 .* randn(StableRNGs.StableRNG(seed + 1000), N - 3)
    println("== pair seed $seed")
    for k in 1:4
        u0 = vcat(PSI_IN[k, :], hidden0, 0.0)
        sol = solve(ODEProblem((φ, _, _) -> rhs(φ, p), u0, (0.0, 2000.0)), Tsit5();
                    reltol=1e-10, abstol=1e-10, saveat=[20.0, 200.0, 2000.0])
        w(x) = rem2pi(x - TARGETS[k], RoundNearest) / π
        @printf("  k=%d  out-target at t=20: %+.3fπ  t=200: %+.3fπ  t=2000: %+.3fπ  |dφ_out/dt|@200=%.1e\n",
                k, w(sol.u[1][N]), w(sol.u[2][N]), w(sol.u[3][N]), abs(rhs(sol.u[2], p)[N]))
    end
end
