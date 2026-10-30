#!/bin/bash
# 40 single-threaded UDE XOR jobs, 20 at a time
cd ~/CODE/NetworkDynamics
D=results/ude_xor_noleak
{ for s in $(seq 1 10); do echo "8 ws $s"; done
  for n in 3 4 5; do for s in $(seq 1 10); do echo "$n complete $s"; done; done; } |
xargs -P 20 -L 1 bash -c 'tag=N$0_$1_seed$2; JULIA_NUM_THREADS=1 ~/.juliaup/bin/julia --project=. results/ude_xor_noleak/Wang-Kuramoto.jl --N=$0 --topology=$1 --seed=$2 > results/ude_xor_noleak/$tag.log 2>&1; echo "$(grep -h RESULT results/ude_xor_noleak/$tag.log || echo "FAILED $tag")" >> results/ude_xor_noleak/summary.txt'
echo DONE >> results/ude_xor_noleak/summary.txt
