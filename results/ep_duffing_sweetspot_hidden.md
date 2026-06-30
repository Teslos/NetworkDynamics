# Does "train deep, operate shallow" generalize across width?

Script: `scripts/duffing_sweetspot_hidden.jl`
Run: `julia --project=. scripts/duffing_sweetspot_hidden.jl` (2026-06-30)

## Question

The sweet-spot study (`results/ep_duffing_sweetspot.md`) showed that for
`hidden=2` the best EP-Duffing XOR config is **train deep** (anneal→1.0, sharp
weights) + **operate shallow** (evaluate at `s=0.5`): 86% / min 82% vs the
champion's 71% / min 61%. Separately, anneal round 2
(`results/ep_duffing_anneal_slow.md`) showed that **evaluated at `s=1.0`**, more
hidden units (4, 6) collapse back to chance (~47–56%).

Was that collapse a *training* failure, or just the wrong operating point? Train
each width deep (anneal→1.0) once per seed, then sweep the operating depth.
`hidden ∈ {2,4,6}`, 6 seeds, 2000 epochs, 40 init draws.

## Results — robust XOR accuracy, mean (min) over seeds

| hidden | eval 0.3 | eval 0.5 | eval 0.7 | eval 1.0 |
|-------:|----------|----------|----------|----------|
| 2      | 77 (69)  | **86 (82)** | 82 (73) | 71 (61) |
| 4      | 52 (46)  | 50 (46)  | 49 (44)  | 49 (43)  |
| 6      | 63 (50)  | 60 (49)  | 54 (47)  | 51 (48)  |

## Conclusion — the trick is operating-point-specific, NOT a width fix

**"Operate shallow" does not generalize to more hidden units.**

- **hidden=2**: the `s=1.0` shortfall is purely an operating-point problem.
  Operating the *same* trained weights at `s=0.5` lifts mean 71%→86% and the floor
  61%→82%. The map is there; only the inference depth was wrong.
- **hidden=4**: sits at chance (~49–52%) at **every** operating depth. No eval
  depth rescues it — the deep-well training never found a usable XOR map, so
  there is nothing for the shallow operating point to recover.
- **hidden=6**: a small lift from operating shallow (51%→63% at `s=0.3`) but it
  never leaves the unreliable regime (min 50% = a seed at chance). Not a solve.

So the width collapse seen earlier is a genuine **training/optimization failure**,
not a wrong operating point. Operating shallow can recover a *good* map that was
operated at the wrong depth (hidden=2); it cannot conjure a map that training
never learned (hidden≥4).

## Why width breaks training here

More coupled bistable hidden cells multiply the competing basins of the free
equilibrium during training. Even with a faithful deep-well EP gradient, the
weight-space loss is harder and the best-checkpoint (lowest free cost on lucky
inits) does not generalize across init draws — the extra capacity adds basin
ambiguity faster than it adds usable representational power. This is consistent
with the static design sweep, where 6 hidden was already worse than 2 at fixed
depth.

## Takeaway for the branch

"Train deep, operate shallow" remains the best EP-Duffing XOR prescription, but it
is a **fix for the operating point at minimal width**, not a remedy for
capacity-induced training failure. Scaling Duffing EP to harder tasks would need
a different lever (e.g. a training scheme that controls hidden-cell basin
degeneracy), not just operating-depth tuning. The smooth XY/Kuramoto net remains
the substrate that trains robustly.
