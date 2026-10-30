#!/bin/bash
# Coupling-form arms, N=5 all-to-all, clamped inputs + learned output field:
# wider shared NN (64), per-edge weights, shared NN with edge identity
cd ~/CODE/NetworkDynamics
D=results/ude_xor_noleak
rm -f $D/summary_coupling.txt
{ for s in $(seq 1 10); do echo "nn 64 $s"; echo "pair 16 $s"; echo "nnid 16 $s"; done; } |
xargs -P 20 -L 1 bash -c 'D=results/ude_xor_noleak; tag=N5_coupling_$0_h$1_seed$2; JULIA_NUM_THREADS=1 ~/.juliaup/bin/julia --project=. $D/Wang-Kuramoto.jl --N=5 --clamp=1 --outfield=1 --coupling=$0 --hidden=$1 --seed=$2 > $D/$tag.log 2>&1; echo "$(grep -h RESULT $D/$tag.log || echo "FAILED $tag: $(grep -m1 -i error $D/$tag.log)")" >> $D/summary_coupling.txt'
echo DONE >> $D/summary_coupling.txt
