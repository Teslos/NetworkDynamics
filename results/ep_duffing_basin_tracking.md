# EP-Duffing per-pattern basin tracking: which XOR corner flips, and when?

Script: `scripts/duffing_basin_tracking.jl`
Run: `julia --project=. scripts/duffing_basin_tracking.jl` (2026-06-30)

## Question

The branch's headline is that **bistable forward inference**, not the EP
gradient, blocks robust Duffing XOR. The anneal experiments support this
indirectly. This script observes the mechanism **directly**: train the round-1
champion (`hidden=2`, linear anneal `frac=0.5`, 2000 epochs), freeze the weights,
then sweep the **evaluation** well depth `s_eval` from near-flat to deep and
measure each XOR pattern's sign-correct fraction over many init draws. With
`a=-s, c=s` the minima stay at ±1 for every `s`, so depth is a free operating
lever — only the barrier (`s/4`) changes.

6 seeds × 40 init draws, test noise 0.1.

## Results — sign-correct fraction per pattern vs evaluation well depth

| s_eval | (-1,-1)→-1 | (-1,+1)→+1 | (+1,-1)→+1 | (+1,+1)→-1 | mean |
|-------:|-----------:|-----------:|-----------:|-----------:|-----:|
| 0.05   | 83%        | 82%        | 81%        | **50%**    | 74%  |
| 0.10   | 83%        | 82%        | 81%        | **50%**    | 74%  |
| 0.20   | 83%        | 82%        | 81%        | **50%**    | 74%  |
| 0.50   | 93%        | 89%        | 77%        | **84%**    | **86%** |
| 1.00   | 68%        | 74%        | 66%        | 76%        | 71%  |

## Conclusion — two competing failure modes, with a depth sweet spot

The prediction (shallow → all correct, deep → some collapse) was too simple. The
data show **two opposing effects** and an intermediate optimum:

1. **Too shallow is quasi-linear and fails the XOR-hard corner.** At `s ≤ 0.2`
   the barrier is negligible, the cells respond near-linearly to the input, and
   the one corner that breaks any linear separator — `(+1,+1)→-1` — sits at
   **exactly 50% (chance)** across all 6 seeds and 40 draws, while the other
   three corners are ~80%. This is a textbook XOR signature: a near-linear
   readout gets three corners and coin-flips the fourth.

2. **Too deep is bistable and degrades every corner.** At `s = 1.0` (the depth
   the net was trained/checkpointed at), the hidden-cell init — not the input —
   starts selecting the output basin, and *all four* patterns fall to 66–76%
   (mean 71%, consistent with the ~73% robust accuracy reported in
   `ep_duffing_anneal.md`).

3. **The sweet spot is intermediate.** At `s = 0.5` the well is nonlinear enough
   to bend the decision boundary and solve the hard corner (84%) but not so deep
   that bistability dominates — **mean peaks at 86%**, the best of any depth.

## Why this matters

- **The mechanism is now observed, not inferred.** The "bistable forward
  inference" story is confirmed *and sharpened*: the deep double-well does not
  merely fail to help, it **overshoots the nonlinearity XOR requires**. XOR needs
  *some* curvature (to separate the `(+1,+1)` corner) but a deep barrier converts
  that curvature into init-determined bistability.
- **Operating depth is a free, untrained lever.** Because the ±1 minima are
  depth-independent, the *same* trained network is markedly more accurate
  evaluated at `s≈0.5` (86%) than at the `s=1.0` it was trained at (71%). The
  training objective is anchored to the wrong depth.
- **Explains the partial-rescue ceiling.** Annealing helps because it passes
  through the favourable shallow/intermediate regime, but it terminates at
  `s=1.0` — past the sweet spot — so the gain is capped at ~73% rather than the
  ~86% available at `s=0.5`.

Still not a robust solve (86% < ~95%): even at the sweet spot, corner
`(+1,-1)→+1` lags at 77%. The smooth XY/Kuramoto net (robust 100%) remains the
only substrate that solves XOR robustly.

## Follow-up this surfaces

Train **and** checkpoint at the sweet-spot depth (`s≈0.5`) rather than annealing
to `s=1.0`, or sweep the operating depth as a hyperparameter — the 86% vs 71% gap
suggests the deep-well training target, not EP, is leaving accuracy on the table.
