# Phase-field / general-quartic EP-Duffing: which lever addresses bistability?

Script: `scripts/duffing_phasefield.jl`
Run: `julia --project=. scripts/duffing_phasefield.jl` (2026-07-01)

## Question

Generalize the symmetric double well `V = ¼c·x⁴ + ½a·x²` to a general quartic
(symmetric case special), motivated by phase-field potentials, to test two levers
against the bistable-XOR problem:

1. **Input-controlled tilt (phase-field driving force)** — a dedicated, DIRECTED,
   trainable input→cell projection `B` (field `Σ_k B_ik u_k`). A strong enough tilt
   pushes a symmetric well past its saddle-node bifurcation (|tilt| > 2/(3√3) ≈
   0.385 for the ±1 well) → one input-determined well → input-following inference.
2. **Learnable potential shape** — per-cell trainable `c₃` (cubic), `c₂`
   (quadratic), `c₄` fixed > 0 (boundedness). EP trains them locally
   (`∂E/∂c₃ = x³/3`, `∂E/∂c₂ = x²/2`); `c₃=0, c₂=−1` recovers the ±1 well.

Recurrent free-free `W` symmetric (energy is a gradient); input drive `B` a
separate directed matrix (inputs are clamped external fields). Both trained with
basin-averaging (full-range init + gradient averaged over Minit=20 basins), 5
seeds, 1000 epochs, one-sided gradient, evaluated under the honest full-range test.

## Results — robust XOR under full-range test init

| config                        | mean | max | min | solved | med cost |
|-------------------------------|-----:|----:|----:|--------|----------|
| input-tilt, fixed sym well    | 70   | 84  | 55  | 0/5    | 0.714    |
| input-tilt + learnable shape  | 80   | 88  | 70  | 0/5    | 0.450    |

References (symmetric well): plain basin-averaging Minit=20 → 76% mean / best 100%;
basin-averaging + annealing Minit=50 → 81–84% mean / 3-of-6 solved.

## Conclusion — shape helps, input-tilt doesn't; a regularizer, not a solver

The two levers separate cleanly:

1. **Directed input-tilt (B) on its own does NOT help** — 70% mean, *below* plain
   basin-averaging (76%). This confirms the prior prediction: a linear input tilt
   duplicates the field the symmetric coupling already supplies (identical
   gradient `−u_k x_i`), so replacing symmetric input-coupling with a directed
   projection is neutral-to-slightly-negative. The bifurcation mechanism did not
   materialize as a win at this scale.
2. **Learnable potential shape (asymmetric quartic) DOES help** — adding trainable
   `c₃, c₂` lifts mean 70→**80%**, floor 55→**70%** (best floor at Minit=20), and
   drops cost 0.71→0.45. The user's core idea — generalize the well shape — is the
   part that carries value, consistent with the "decouple nonlinearity from deep
   symmetric bistability" argument.

Honest caveat: **no seed fully solved (0/5 ≥ 90%)**, and the peak (88) is *below*
plain basin-averaging's best seed (100). So learnable shape behaves as a
**regularizer** — it tightens the seed distribution (higher mean and floor, more
*consistent*) rather than enabling occasional perfect solves. This was also the
weaker budget (Minit=20, no annealing) vs the record recipe (Minit=50 + anneal,
84% / 3-of-6).

## Where this leaves the phase-field idea

Partially validated: the **shape** generalization improves consistency and mean;
the **input-tilt** generalization does not (it re-expresses the coupling). The
next test is the obvious combination: **learnable shape + Minit=50 + annealing**,
i.e. stack the shape regularizer on the best-performing recipe, to see whether the
tighter distribution plus the higher-budget peak pushes seeds over the solve
threshold. If even that plateaus below a clean all-seed solve, the evidence points
to N=5/2-hidden capacity, and the faithful "smooth input-following" endpoint
remains the 2-D phase oscillator (i.e. XY).
