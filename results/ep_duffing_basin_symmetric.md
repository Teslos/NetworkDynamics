# EP-Duffing basin-averaging: symmetric vs one-sided gradient

Script: `scripts/duffing_basin_symmetric.jl`
Run: `julia --project=. scripts/duffing_basin_symmetric.jl` (2026-07-01)

## Question

The best Duffing config so far — basin-averaging with Minit=50 + barrier annealing
— reached 81% mean / 3-of-6 seeds solved with a ONE-SIDED EP gradient
(`results/ep_duffing_basin_averaging.md`). Unlike Wang's log cost (which forces
one-sided nudging), our Duffing cost is quadratic `C = ½Σ(x−target)²`, so the ±β
nudge is well-behaved both ways and the symmetric central-difference estimator is
safe and less biased. Does it close the floor to an all-seed solve? Head-to-head,
same config (Minit=50, anneal s:0.3→1.0 over first half, 1500 epochs, 6 seeds,
full-range test), only the estimator changed.

## Results — robust XOR under full-range test init

| gradient              | deep (mean/max/min) | solved | sweet | med final cost |
|-----------------------|---------------------|--------|-------|----------------|
| one-sided (baseline)  | 81 / 98 / 59        | 3/6    | 80    | 0.369          |
| **symmetric ±β**      | **84 / 97 / 68**    | 3/6    | 85    | 0.542          |

(The one-sided row reproduces the recorded strong-config result exactly —
81/98/59, 3/6 — so the comparison is controlled.)

## Conclusion — a real but modest refinement, not a clean solve

The symmetric estimator tightens the distribution: mean 81→84%, **floor 59→68%
(+9)**, sweet-spot 80→85%. But it **converts no additional seeds** over the 90%
threshold — still 3/6 solved. The less-biased gradient helps the marginal seeds a
little without rescuing the genuinely hard ones.

So the "close the floor" quest plateaus here. The 3 unsolved seeds land in hard
training basins that neither annealing nor the symmetric gradient overcomes at
this budget. A clean all-6-seed robust solve is not reached with estimator/schedule
tweaks alone.

## Where the Duffing-bistability investigation lands

**Best result of the arc: symmetric + Minit=50 + anneal → 84% mean, 68% floor,
3/6 seeds fully solved, best seeds ~100%**, under the honest full-range (both-wells)
test where the naive-init control is at chance (51%).

The question "can the bistable Duffing be addressed?" is answered **yes**: from
~chance to 84% mean with best seeds fully basin-invariant, via the standard EP
multistability remedy (full-range init + basin-averaged gradient), optionally
stacked with annealing and a symmetric estimator. The corrected headline stands —
bistable forward inference is the ordinary EP multistability problem, not an
insurmountable wall.

But a **guaranteed all-seed robust solve at this minimal size (N=5, 2 hidden) is
not achieved**. The remaining seed variance points at capacity/architecture, not
the gradient: likely closers are a larger or **layered** network (which helped XY),
more hidden units *with* basin-averaging (untried in combination), or substantially
more epochs. Estimator refinements (symmetric) and schedule tweaks (annealing)
have reached diminishing returns.
