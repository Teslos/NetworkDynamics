# Empirical response to the manuscript critique

This directory collects the experiments built to address
[`docs/critique_chaotic_oscillator_networks.md`](../../docs/critique_chaotic_oscillator_networks.md),
the structured review of the manuscript *"Chaotic oscillator networks for
classification tasks."* Each critique item that could be answered with a
simulation has a runnable script and a result file here; the experiment plan and
the item-by-item mapping are in
[`docs/baseline_simulation_plan.md`](../../docs/baseline_simulation_plan.md).

Two distinct papers are involved and should not be conflated:
- the **critiqued manuscript** (digits + XOR + Lorenz + a learned-coupling/UDE
  idea), and
- its main reference, **Al Beattie et al. 2024, *Communications Physics*** —
  "Criticality in FitzHugh-Nagumo oscillator ensembles" (`docs/…Beattie….pdf`),
  which is methodologically careful (10-fold CV, baseline comparison) and uses
  *avalanche* criticality, explicitly **not** the edge of chaos.

All code is pure Julia (no Python/PyCall). `scikit-learn`'s `load_digits` is
reproduced exactly from `data/digits/optdigits.tes`.

---

## TL;DR verdict

1. **The reservoir does not beat one-line baselines.** On digits, logistic
   regression (0.959) and an MLP (0.976) exceed the manuscript's reported 0.88
   and our own faithful FHN-reservoir reproduction (0.936). On dry bean every
   method clusters at ~0.92–0.93. The "surpasses traditional approaches" framing
   is not supported.
2. **Learning the coupling (the signature UDE contribution) gives no
   significant gain** over a fixed random reservoir on dry bean (p≈0.45), and
   both trail raw-feature logistic regression. The contribution is demonstrated
   only on XOR and is statistically redundant on the classification tasks.
3. **"Edge of chaos" is the wrong characterization** for this network — it is
   strongly contracting (no positive Lyapunov exponent), which is exactly what
   Beattie et al. themselves state. The correct notion, **avalanche
   criticality**, *does* hold (branching ratio crosses 1, near-mean-field
   power-law exponents).
4. **Criticality is not the optimum.** With a Watts–Strogatz topology and
   multi-seed averaging, accuracy is *lowest* at criticality (0.853) and
   *highest* in the supercritical/synchronized regime (0.917), which is also the
   most readout-shrinkage-robust. Beattie's "critical ⇒ best accuracy + most
   robust" claims are not reproduced for this network.
5. **No result without dispersion.** Everything is reported as mean ± std over
   ≥8–10 seeds/folds, with a paired Wilcoxon test for model comparisons.

Net: our findings **support the critique** of the manuscript and are largely
**consistent with Beattie et al.** (the careful source paper) — the manuscript's
problems are framing and missing baselines/statistics, not that oscillator
reservoirs are useless.

---

## Reproduce

```
julia --project=. scripts/run_baselines.jl                  # B1–B6
julia --project=. scripts/run_fhn_digits.jl                 # FHN reservoir on digits
julia --project=. scripts/run_reservoir_diagnostics.jl      # B7, B9
julia --project=. scripts/run_avalanche_criticality.jl      # B10
julia --project=. scripts/run_criticality_vs_accuracy.jl    # B11
julia --project=. scripts/run_ude_ablation.jl               # B8
```
Add `--quick` to any for a fast smoke run. Figures are written to
`results/figures/` (gitignored); tables to this directory.

---

## Findings by experiment

| # | Critique item | Experiment | Headline result |
|---|---|---|---|
| B1 | §1.1 baselines | logreg / SVM / MLP / ESN on digits & dry bean | reservoir below baselines (see tables) |
| B2 | §1.3, §6 statistics | ≥10 seeds, mean ± std | all numbers now have dispersion |
| B3 | §1.3 topology claim | paired Wilcoxon protocol | demonstrated (MLP vs logreg, p=0.006) |
| B4 | §6 metrics | macro/weighted F1, per-class | reported for every model |
| B5 | §6 cross-validation | stratified k-fold (dry bean) | done |
| B6 | §2/§6 Lorenz | valid time + NRMSE | 7.0 ± 1.1 Lyapunov times, then diverges |
| B7 | §1.4 resonance | silhouette / Fisher / linear probe | reservoir **less** separable than raw pixels |
| B8 | §1.2 framing | learned-coupling vs fixed reservoir | no significant difference (p≈0.45) |
| B9 | §6 edge of chaos | accuracy + Lyapunov/ESP vs coupling | FHN contracting, **not** at edge of chaos |
| B10 | (Beattie) criticality | avalanche statistics vs coupling | **is** avalanche-critical (branching→1) |
| B11 | capstone | criticality **and** accuracy, same network | weak peak at criticality; shrinkage claim not reproduced |

### Baselines & statistics (B1–B6) — `baseline_results.md`

Digits (== sklearn `load_digits`, 1797×64, 10 classes, 80/20 split, 10 seeds):

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Logistic regression | 0.959 ± 0.012 | 0.959 ± 0.012 |
| Linear SVM | 0.952 ± 0.011 | 0.952 ± 0.011 |
| MLP (1 hidden) | **0.976 ± 0.009** | 0.976 ± 0.009 |
| tanh-ESN (500 units) | 0.833 ± 0.023 | 0.829 ± 0.025 |
| Manuscript FHN (reported) | 0.88 | — |
| FHN reservoir (our reproduction, N=1797) | 0.936 (1 seed) | 0.936 |

Dry bean (13611×16, 7 classes, repeated stratified 5-fold): logreg 0.923, SVM
0.924, MLP 0.933, tanh-ESN 0.930 (all ± ~0.004). Lorenz tanh-ESN: valid
7.0 ± 1.1 Lyapunov times, NRMSE 1.31. A LaTeX summary table for digits is in
`digits_results_table.tex`.

### Continuous-time reservoir baseline (LPCTESN) — `lpctesn_lorenz.md`

A minimal linear-projection continuous-time echo state network (Anantharaman
et al. 2021), added as a continuous-time counterpart to the discrete ESN Lorenz
baseline. It forecasts Lorenz stably (NRMSE ≈ 1.50, comparable to the discrete
ESN's 1.32) but with a **shorter valid horizon — 1.14 ± 0.41 Lyapunov times vs
the discrete ESN's 4.70 ± 1.21** (8 seeds, matched reservoir size). This is
consistent with CTESN being designed for *parametric surrogates of stiff ODEs*,
not autonomous chaotic forecasting — so it is a more faithful continuous-time RC
baseline but not an improvement on this task. Run `scripts/run_lpctesn_lorenz.jl`.

### Mechanism: separability (B7) — `reservoir_diagnostics.md`

On digits, raw pixels are **more** linearly separable than the reservoir states:
silhouette 0.097 (raw) vs 0.013 (FHN) / −0.126 (ESN); linear-probe accuracy
0.956 (raw) vs 0.832 (FHN) / 0.868 (ESN). The reservoir transform does not add
linear separability, so "local chaotic resonance improves classification" is
unsupported.

### Mechanism: edge of chaos vs avalanche criticality (B9, B10)

- **B9** (`reservoir_diagnostics.md`, `edge_of_chaos.png`): the generic ESN shows
  the textbook edge of chaos — accuracy ~0.87 while the Lyapunov exponent is
  negative, collapsing to 0.42 once it crosses 0 near ρ≈1.6. The **FHN reservoir
  has echo-state divergence ≈0 for all σ>0** — strongly contracting, no chaos.
- **B10** (`avalanche_criticality.md`, `avalanche_criticality.png`): with the
  *correct* (Beattie) criticality measure, the FHN reservoir **is** critical —
  the branching ratio rises through 1 near σ≈0.004 with near-mean-field
  exponents (τ≈1.6→1.5, α≈1.8→2.0).
- **Reconciliation:** B9 and B10 are both correct; they measure different things.
  "Not at the edge of chaos" matches Beattie's own statement that the network is
  not chaotic. The manuscript's *chaotic* framing is the issue, not Beattie's
  *avalanche-critical* one.

### Capstone: criticality and accuracy on one network (B11) — `criticality_vs_accuracy.md`

Excitable FHN reservoir (Watts–Strogatz topology, k=10/β=0.3), dry bean,
sweeping coupling and measuring both the branching ratio and accuracy on the same
reservoir, averaged over 3 reservoir realizations. Accuracy is **lowest at
criticality** (0.853 ± 0.015 at σ=0.030, branching≈1.04) and **highest in the
supercritical regime** (0.917 ± 0.004 at σ=3.0), which is also the most
readout-shrinkage-robust (0.916→0.914 from 100→20 nodes, vs critical 0.853→0.811).
So for this network criticality is neither the accuracy optimum nor the most
robust — synchronization provides both, contradicting Beattie's specific claims.
(An earlier complete-graph, single-seed version showed a weak peak at criticality;
the topology- and seed-averaged result does not.)

### Framing: learned coupling vs fixed reservoir (B8) — `ude_ablation.md`

Identical discrete-time FHN reservoir and training; only W trainability differs
(8 seeds, dry bean):

| Model | Test accuracy |
|---|---|
| Fixed reservoir (W frozen) | 0.881 ± 0.033 |
| Learned coupling (W trained, UDE) | 0.889 ± 0.026 |
| Raw-feature logistic regression | 0.904 ± 0.023 |

Paired Wilcoxon (learned vs fixed): **p ≈ 0.45 — not significant.** Learning the
coupling neither beats the fixed reservoir nor a one-line baseline on this task.

---

## Caveats

- The original manuscript scripts (`src/classification/FitzHug-Nagumo-*`) are
  **unrunnable** under the installed stack (NetworkDynamics 0.9.7 removed the old
  API; PyCall not built), so the FHN results use faithful reimplementations of
  the *method*, not the original code.
- The FHN-reservoir digit number (0.936) is a single seed; the FHN classification
  / criticality experiments use one reservoir realization per coupling value, so
  branching-ratio curves are noisy point-to-point (accuracies are seed-averaged).
- B8 uses a discrete-time, linear-W stand-in for the continuous UDE at N=30 on a
  dry-bean subset; a continuous neural-network coupling with more capacity could
  differ. It rules out a *cheap* benefit from learned coupling, not every benefit.
- B10/B11 use a complete-graph coupling; Beattie used Watts–Strogatz. Adding
  topology and multi-realization averaging would make B10/B11 publication-grade.

---

## What remains (not simulations)

The unresolved critique items are manuscript edits, not experiments: the figure
redraws (§2), equation/notation fixes (§3), bibliography cleanup (§4), and the
naming/typo pass (§5). The empty code link (§2) can now point to the real
repository, which contains all of the above.

## File index

| File | Contents |
|---|---|
| `baseline_results.md` / `.csv` | B1–B6 tables, per-seed raw values |
| `digits_results_table.tex` | LaTeX summary table for the digit tests |
| `reservoir_diagnostics.md` | B7 separability, B9 edge-of-chaos sweep |
| `avalanche_criticality.md` | B10 avalanche statistics vs coupling |
| `criticality_vs_accuracy.md` | B11 criticality + accuracy + readout shrinkage |
| `ude_ablation.md` | B8 learned-coupling vs fixed-reservoir ablation |
| `lpctesn_lorenz.md` | continuous-time LPCTESN vs discrete ESN on Lorenz |
