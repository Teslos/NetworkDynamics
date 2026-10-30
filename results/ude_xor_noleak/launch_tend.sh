#!/bin/bash
# Long-horizon loss (scored at t = 50, 100, 150, 200) on symmetric and directed
# per-edge coupling, N=5, clamped inputs + output field, 10 seeds each
cd ~/CODE/NetworkDynamics
D=results/ude_xor_noleak
rm -f $D/summary_tend.txt
{ for s in $(seq 1 10); do echo "sympair $s"; echo "pair $s"; done; } |
xargs -P 20 -L 1 bash -c "D=results/ude_xor_noleak; tag=N5_\$0_tend200_seed\$1; JULIA_NUM_THREADS=1 ~/.juliaup/bin/julia +release --project=. \$D/Wang-Kuramoto.jl --N=5 --clamp=1 --outfield=1 --coupling=\$0 --tend=200 --seed=\$1 > \$D/\$tag.log 2>&1; echo \"\$(grep -h RESULT \$D/\$tag.log || echo FAILED \$tag)\" >> \$D/summary_tend.txt"
echo DONE >> $D/summary_tend.txt
