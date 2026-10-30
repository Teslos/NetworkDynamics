#!/bin/bash
# Trajectory backprop + learned fields on every free oscillator (as EP has),
# per-edge coupling symmetric/directed, short window vs long horizon, 10 seeds
cd ~/CODE/NetworkDynamics
D=results/ude_xor_noleak
rm -f $D/summary_hidfields.txt
{ for s in $(seq 1 10); do for c in sympair pair; do for t in 20 200; do echo "$c $t $s"; done; done; done; } |
xargs -P 20 -L 1 bash -c "D=results/ude_xor_noleak; tag=N5_\$0_hidfields_tend\$1_seed\$2; JULIA_NUM_THREADS=1 ~/.juliaup/bin/julia +release --project=. \$D/Wang-Kuramoto.jl --N=5 --clamp=1 --outfield=1 --hidfields=1 --coupling=\$0 --tend=\$1 --seed=\$2 > \$D/\$tag.log 2>&1; echo \"\$(grep -h RESULT \$D/\$tag.log || echo FAILED \$tag)\" >> \$D/summary_hidfields.txt"
echo DONE >> $D/summary_hidfields.txt
