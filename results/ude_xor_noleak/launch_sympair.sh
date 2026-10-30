#!/bin/bash
# Symmetric per-edge coupling (K_ij = K_ji, as in EP), N=5, clamped inputs + output field
cd ~/CODE/NetworkDynamics
D=results/ude_xor_noleak
seq 1 10 | xargs -P 10 -I{} bash -c "D=results/ude_xor_noleak; tag=N5_coupling_sympair_seed{}; JULIA_NUM_THREADS=1 ~/.juliaup/bin/julia --project=. \$D/Wang-Kuramoto.jl --N=5 --clamp=1 --outfield=1 --coupling=sympair --seed={} > \$D/\$tag.log 2>&1; echo \"\$(grep -h RESULT \$D/\$tag.log || echo FAILED \$tag)\" >> \$D/summary_coupling.txt"
