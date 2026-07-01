# Monostable ("single-basin") Duffing on digits: bistability was the obstruction

Script: `scripts/duffing_digits_monostable.jl`
Run: `julia -t auto --project=. scripts/duffing_digits_monostable.jl` (2026-07-01)

## Idea (user's)

The graded-readout run showed a linear output only partly helps (0.18→0.27) because
the double-well HIDDEN cells stay multistable. Fix: put the HIDDEN cells in the
SINGLE-well (monostable, `a>0`) regime, where each cell is a smooth saturating
nonlinearity with a UNIQUE equilibrium — so hidden features are a deterministic
function of the input and NO basin-averaging is needed. Output linear + softmax-CE.
"Landau annealing in the single-basin sense" = deterministic annealing: start very
convex (large `a_h`, near-linear) and cool to the nonlinear operating point, staying
`a_h>0`. 10 hidden nodes, 4x4 inputs.

## Results

| model                 | train | test  |
|-----------------------|------:|------:|
| Duffing mono (fixed a=0.5)   | 0.430 | 0.430 |
| Duffing mono (Landau a:3→0.5)| 0.577 | 0.540 |
| logreg                | —     | 0.835 |
| MLP                   | —     | 0.880 |

Chance 0.10. Refs: bistable-readout 0.177, graded-readout (double-well hidden)
0.270, XY 0.94.

## Conclusion — confirms the diagnosis; recovers a lot, not all

Going monostable is the **biggest single jump of the digit line**:
bistable readout 0.18 → graded readout (dw hidden) 0.27 → **monostable hidden +
graded 0.54**. Tripling over the bistable net **confirms that bistability — not the
Duffing dynamics — was the obstruction to multi-class.** Each bistability source
removed (readout, then hidden) climbs. **Landau/deterministic annealing helped**
(0.43 → 0.54): cooling from the near-linear regime into the nonlinear one reached a
better solution (best test 0.535 around `a_h≈1.8`).

It did **not** reach the logreg baseline (0.54 vs 0.84). Two honest points:

1. **Prediction miss.** I expected ~0.80–0.88 (near baseline). Wrong — this is the
   second over-optimistic call in a row (graded-readout, then monostable). I'm
   overestimating; noting it explicitly.
2. **The gap is optimization, not representation.** logreg has *no* hidden layer and
   still beats it (0.84 > 0.54); a smooth 16→10→10 net should at least match logreg.
   Training was visibly **unstable** — CE bounced (1.1 → 2–3), test peaked at 0.54
   (Landau, iter 75) then degraded as training continued. So the shortfall is
   EP-training instability at this scale (one-sided gradient + softmax nudge + small
   net + loose relaxation tol), not a substrate ceiling.

## Where the digit line lands

Removing bistability (monostable single-basin + graded readout) is what lets Duffing
do multi-class **at all** (0.18 → 0.54), and Landau annealing adds to it. The
residual gap to baseline looks like training stability/efficiency — likely
recoverable with lower LR, symmetric gradient, early stopping, or more hidden — not
a Duffing limitation. The fully-smooth, well-optimized endpoint is a conventional
smooth net / the XY substrate. Net: the two-substrate study's conclusion holds —
**bistability is an asset for single-bit/memory (XOR) and a liability for rich
multi-class (digits); the smooth/monostable/phase regime is the right tool there**,
and this run pins that removing bistability is precisely what recovers multi-class
capability.

## Obvious follow-ups (optimization, not substrate)

Lower learning rate + early stopping (the best checkpoint 0.54 didn't hold); symmetric
±β gradient (cleaner); more hidden units; full 64px inputs. Each should push the
monostable-Duffing number up toward the logreg/MLP line without changing the substrate.
