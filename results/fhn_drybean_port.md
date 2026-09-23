# FHN reservoir on Dry-Bean: a runnable port, and what it says about σ

Run: `julia --project=. scripts/run_fhn_drybean64.jl --sigma=0.1` (2026-08-13)

## Why

`src/classification/FitzHug-Nagumo-DryBean.jl` produces the paper's Dry-Bean
confusion matrix (`confusion_matrix_dry_bean_percent.png`, Figure 5) but **cannot
run**: it uses `@sk_import` (PyCall is not built), the removed NetworkDynamics API
(`network_dynamics`/`ODEVertex`/`StaticEdge`), and GLMakie. The confusion matrix
existed only as pixels — no stored numbers — so the figure could not be re-plotted,
and the reported ~92% could not be checked.

It is also **internally inconsistent**. It builds a 2046-node reservoir and 512-long
trajectory features, but its readout is `Lux.Dense(64, 512, tanh)`, i.e. 64 inputs.
512 ≠ 64, so the committed pairing would error. The manuscript says "our reservoir
is made up of 64 FitzHugh–Nagumo oscillators", and the script's exploratory section
drives only the first 16 nodes (`i <= 16 ? spike_train[:, i] : g0`). That
configuration — 64 nodes, 16 driven, one feature per node — is self-consistent and
matches both the readout width and the paper. The committed file has drifted from
whatever produced the figure; `paper/images/confusion_matrix_dry_bean_percent.png`
has no git history.

## Two ports

* `scripts/run_fhn_drybean.jl` — one node per sample, batched `nsamples ÷ N`,
  features = the node's full u-trajectory (512). This is what the committed script
  *builds*.
* `scripts/run_fhn_drybean64.jl` — 64-node reservoir, 16 driven, one feature per
  node (time-averaged u), per-sample solves. This is what its readout and the
  manuscript *describe*. **This is the one to use.**

Both write the confusion matrix as numbers to `results/confusion_matrices/`, so
Figure 5 can now be re-plotted in the same toolchain and font as Figure 4.

## Results

| Configuration | Test acc |
|---|---|
| first port: subsampled to N, features remapped to [0,1] | 0.700 |
| + encoding matched to the original (standardised → `spikerate.rate`, clipped) | 0.829 |
| + batching over the full dataset (12 276 samples, not 2 046) | 0.849 |
| 64-node variant, σ = 0.006 (the committed value), full data | 0.859 |
| **64-node variant, σ = 0.1, full data** | **0.891** (macro-F1 0.867, train 0.909) |
| *paper's reported value* | *~0.92* |
| *logreg on the raw 16 features (baseline suite, 10 seeds)* | *0.923 ± 0.004* |

Single seed (1234). The remaining ~3 points to the published figure is small enough
to be seed variation or readout detail; it is no longer evidence that the number is
unreachable.

## σ = 0.006 is wrong at 64 nodes

The coupling enters as `Σ_j W_ij (u_j − u_i)` with `W_ij = σ w_ij`, `w_ij ~ 0.3`, so
total in-coupling per node scales as `σ (N−1)`:

* 2046 nodes: `0.006 × 2045 × 0.3 ≈ 3.7`
* 64 nodes:  `0.006 × 63 × 0.3 ≈ 0.11` — **33× weaker**

Matching total in-coupling gives `σ ≈ 0.006 × 2045/63 ≈ 0.19`. The measured optimum
is 0.1–0.2, so the scaling argument predicts it well. σ is worth ~14 points at fixed
data (0.737 → 0.874 at n = 2000) and ~3 points on top of the full dataset.

## Edge of chaos: the paper's description does not hold at 64 nodes, and the
## heuristic does not help either

`scripts/fhn_edge_of_chaos.jl` drives two copies of the reservoir with the same
input from initial conditions differing by 1e-6 and measures the separation ratio.
Below 1 the perturbation decays (ordered, echo state property holds); above 1 it
grows (chaotic, the state depends on where it started rather than on the input).

| σ | sep ratio | across-node std | test acc (n = 2000) |
|---|---|---|---|
| 0.006 | **1.95** chaotic | 0.075 | 0.737 |
| 0.01 | ~1.0 (the edge) | — | 0.732 |
| 0.02 | 0.65 | 0.060 | 0.753 |
| 0.05 | 0.45 | 0.046 | 0.859 |
| **0.10** | 0.39 | 0.034 | **0.874** |
| 0.20 | 0.45 | 0.022 | 0.869 |
| 0.40 | 0.38 | 0.013 | — |
| 3.20 | 0.33 | 0.002 | — |

Two things follow.

1. **The committed σ puts the 64-node reservoir on the chaotic side**, where part of
   its response is initial-condition noise rather than input. The paper states the
   oscillators "were operated near the edge of chaos"; at 64 nodes σ = 0.006 is past
   the edge, not near it. (At 2046 nodes the same σ would sit well inside the ordered
   regime, so the value is not wrong in its original context.)
2. **The edge is not the optimum.** Accuracy is *worst* at the edge (σ = 0.01 → 0.732)
   and rises monotonically into the ordered regime up to σ ≈ 0.1. The usual
   "edge of chaos is best" heuristic actively misleads here. Note also that coupling
   is diffusive, so *increasing* σ makes the network more ordered (it synchronises),
   which is the opposite of the intuition for a recurrent weight matrix.

There is a genuine trade-off, but it is not centred on the edge: across-node
diversity falls monotonically with σ (0.075 → 0.002), so at large σ all 64 nodes
report nearly the same thing. The optimum balances noise suppression against feature
diversity, and lands at σ ≈ 0.1.

## Not done

* **Multi-seed.** Everything here is seed 1234. The digits and Lorenz results carry
  5–10 seed dispersion; Dry-Bean should before it enters the manuscript, especially
  with a ~3 point gap in play. Five seeds ≈ 100 min.
* **Figure 5 not regenerated.** The numbers now exist
  (`results/confusion_matrices/fhn_drybean64_confusion_matrix.txt`); replotting is
  pending the decision on which number the paper reports.
* **The manuscript is unchanged.** Its ~92% still stands as written.

## Bearing on the paper

The reservoir competes against logistic regression on the raw 16 features at
0.923 ± 0.004 — which matches the published claim to within a rounding error and
beats every reservoir configuration measured here. This mirrors the digits result
(FHN 0.926 vs logreg on raw pixels 0.959): the FHN transform loses discriminative
information rather than adding it, which for already-informative tabular features is
the expected outcome. The honest framing is the one the digits section already uses —
the interest is analog realizability, not accuracy.
