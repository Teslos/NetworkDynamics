#!/bin/bash
# UDE N=5 all-to-all with EP-style setup: clamped inputs, +/- learned output field
cd ~/CODE/NetworkDynamics
D=results/ude_xor_noleak
rm -f $D/summary_epcompare.txt
{ for s in $(seq 1 10); do echo "1 0 $s"; echo "1 1 $s"; done; } |
xargs -P 20 -L 1 bash -c 'D=results/ude_xor_noleak; tag=N5_clamp$0_outfield$1_seed$2; JULIA_NUM_THREADS=1 ~/.juliaup/bin/julia --project=. $D/Wang-Kuramoto.jl --N=5 --clamp=$0 --outfield=$1 --seed=$2 > $D/$tag.log 2>&1; echo "$(grep -h RESULT $D/$tag.log || echo "FAILED $tag")" >> $D/summary_epcompare.txt'
echo DONE >> $D/summary_epcompare.txt
