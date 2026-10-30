#!/bin/bash
# Settling penalty on symmetric per-edge coupling, N=5, clamped inputs + output field.
# Own copy of the script so running jobs are untouched; 5 at a time on the shared box.
cd ~/CODE/NetworkDynamics
D=results/ude_xor_noleak
rm -f $D/summary_settle.txt
{ for s in $(seq 1 10); do echo "10 $s"; echo "100 $s"; done; } |
xargs -P 5 -L 1 bash -c "D=results/ude_xor_noleak; tag=N5_sympair_settle\$0_seed\$1; JULIA_NUM_THREADS=1 nice -n 10 ~/.juliaup/bin/julia --project=. \$D/Wang-Kuramoto-settle.jl --N=5 --clamp=1 --outfield=1 --coupling=sympair --settle=\$0 --seed=\$1 > \$D/\$tag.log 2>&1; echo \"\$(grep -h RESULT \$D/\$tag.log || echo FAILED \$tag)\" >> \$D/summary_settle.txt"
echo DONE >> $D/summary_settle.txt
