# Simulation & Execution Plan — Resolving the Manuscript Critique

This plan operationalizes [`critique_chaotic_oscillator_networks.md`](critique_chaotic_oscillator_networks.md).
It maps each blocking scientific problem to a concrete, runnable experiment, and
specifies the order in which to run them so the revised paper's claims are
actually supported by data.

The baseline code is implemented and runnable today:

| File | Purpose |
|---|---|
| `src/baselines/baseline_utils.jl` | data loading, stratified split / k-fold, metrics (accuracy, per-class P/R, macro/weighted F1, confusion matrix), paired Wilcoxon signed-rank |
| `src/baselines/baseline_models.jl` | logistic regression, linear SVM, small MLP, tanh-ESN (classification + Lorenz) |
| `scripts/run_baselines.jl` | runs everything over ≥10 seeds and writes tables + figures |

Run them with:

```
julia --project=. scripts/run_baselines.jl            # full: 10 seeds
julia --project=. scripts/run_baselines.jl --quick    # fast check: 3 seeds
```

Outputs: `results/baselines/baseline_results.md` (formatted tables),
`results/baselines/baseline_results.csv` (raw per-seed), and
`results/figures/lorenz_esn_prediction.png`.

> **Data note.** scikit-learn's `load_digits()` is byte-for-byte the UCI
> *optdigits* test partition already in `data/digits/optdigits.tes`
> (1797×64, 10 classes), so the baselines reproduce the paper's dataset without
> any Python/sklearn dependency. Dry Bean is `data/DryBeanDataset.csv`
> (13611×16, 7 classes).

---

## What the baselines already show (full run, mean ± std over 10 seeds)

| Dataset | logreg | linear SVM | MLP | tanh-ESN | paper's number |
|---|---|---|---|---|---|
| Digits (acc) | 0.959 ± 0.012 | 0.952 ± 0.011 | **0.976 ± 0.009** | 0.833 ± 0.023 | **0.88 (FHN)** |
| Dry Bean (acc) | 0.923 ± 0.004 | 0.924 ± 0.004 | **0.933 ± 0.003** | 0.930 ± 0.003 | 0.923 claimed (~0.80 from its matrix) |
| Dry Bean (weighted-F1) | 0.923 | 0.924 | 0.933 | 0.930 | — |

Lorenz tanh-ESN: valid time **7.0 ± 1.1 Lyapunov times** (NRMSE 1.31 ± 0.03 over
the full 60-time-unit horizon, i.e. it diverges after the valid window).
Paired Wilcoxon (MLP vs logreg, digits): p = 0.006 — a real difference, shown
as the template for the topology comparisons the paper currently asserts without
a test.

The conclusion the critique anticipated is confirmed: **on digits the paper's
88% is below a one-line logistic-regression baseline (95.9%)**, and on dry bean
the best reservoir number only ties standard classifiers (~92–93%). The headline
framing ("surpasses traditional approaches") cannot stand without either better
numbers or a reframing.

> **ESN caveat (motivates experiment B9).** The tanh-ESN on digits fell from
> 0.91 (300 units) to 0.83 (500 units) at fixed spectral radius 0.9 / ridge λ —
> i.e. it is sensitive to reservoir size and spectral radius. That is exactly
> why the echo-state-property / spectral-radius sweep (B9) is needed before any
> reservoir number is quoted as a baseline; the ESN here is an *untuned* matched
> baseline, and its hyperparameters (`spectral_radius`, `input_scale`, ridge
> `lambda`, `Nr`) are exposed for the sweep.

---

## Critique item → experiment mapping

### §1.1 Add baselines  *(implemented)*
- **Experiment B1.** logreg, linear SVM, MLP, tanh-ESN on digits and dry bean,
  same splits as the reservoir model. Already produces the comparison table.
- **Resolves:** removes the unsupported "surpass/comparable" claim or forces a
  reframe. This is the single highest-priority fix.

### §1.3 / §6 Seed statistics  *(implemented)*
- **Experiment B2.** Every accuracy is reported as mean ± std over ≥10 seeds;
  digits uses 10× 80/20 stratified splits, dry bean uses repeated stratified
  5-fold. Raw per-seed values are in the CSV.
- **Experiment B3 (topology test).** The paper claims all-to-all and
  Watts–Strogatz beat Barabási–Albert. Use `wilcoxon_signed_rank` on the
  per-seed accuracies of the two topologies (paired by seed). Only claim "A
  beats B" when p < 0.05. The driver demonstrates the test (MLP vs logreg);
  apply the identical call to the reservoir-topology runs.
- **Resolves:** §1.3 (no statistics) and the unsupported topology ranking
  (line 283).

### §6 Metrics beyond accuracy  *(implemented)*
- **Experiment B4.** Macro-F1, weighted-F1 and a per-class precision/recall/F1
  table are reported for every model. Dry Bean is class-imbalanced (DERMASON
  709 vs BOMBAY 104 in a fold), so weighted-F1 and the per-class table are the
  honest numbers.
- **Resolves:** §6 (metrics) and §2's dry-bean 92.3%-vs-matrix contradiction —
  the per-class/macro numbers make the discrepancy explicit and reconcilable.

### §6 Cross-validation  *(implemented)*
- **Experiment B5.** Stratified 5-fold for dry bean (via `stratified_kfold`);
  digits uses the standard split plus the multi-seed estimate as the CV proxy.
- **Resolves:** §6 (cross-validation).

### §6 / §2 Lorenz quantification  *(implemented)*
- **Experiment B6.** tanh-ESN forecast of Lorenz '63; report **valid prediction
  time in Lyapunov times** (λ₁ = 0.9056) and **NRMSE**, with the forecast figure
  showing divergence. Quick-mode: ≈3.7 Lyapunov times before the normalized
  error crosses 0.4.
- **Resolves:** §2 "Lorenz caption oversells" and §6 (Lorenz quantification):
  replaces "very good" with an honest short-horizon number and gives the
  reservoir model a baseline to be compared against.

### §1.4 / §6 Resonance / separability metric  *(planned — needs reservoir states)*
- **Experiment B7.** On the saved reservoir state matrix (the oscillator
  trajectories the readout sees), compute (a) node-energy/amplitude separation
  between the resonant and non-resonant node sets and (b) a linear-probe
  accuracy and silhouette score on the reservoir state vs. class label.
- **How:** export the reservoir state matrix from the existing FHN/Kuramoto/
  Duffing RC scripts (`src/classification/*`), then reuse the metric helpers in
  `baseline_utils.jl`. A linear probe is just `train_logreg` on the reservoir
  state; silhouette is a short addition.
- **Resolves:** §1.4 — turns the "local chaotic resonance" narrative into a
  measured observable.

### §1.2 / §6 Learned-coupling (UDE) vs. fixed reservoir ablation  *(planned)*
- **Experiment B8.** Run the paper's UDE/learned-coupling method (the XOR
  contribution, `notebooks/EP-XY-*` / the UDE formulation) on digits and dry
  bean, and compare head-to-head with the fixed reservoir on the *same* split.
- **Resolves:** §1.2 (contribution/identity conflation) — this is the
  experiment that decides whether the paper is "one method" or "two
  contributions," and is flagged as the top structural fix.

### §6 Reservoir diagnostics (echo-state property, edge of chaos)  *(planned)*
- **Experiment B9.** Spectral-radius sweep of the reservoir: verify the
  echo-state property and plot accuracy vs. spectral radius / vs. σ with error
  bars (≥10 seeds), promoting the supplement's σ-sweep to the main text. The
  ESN code already exposes `spectral_radius`; sweep it and reuse B2's seed loop.
- **Resolves:** §6 (reservoir diagnostics) and substantiates the "edge of
  chaos" claim with statistics.

---

## Execution order (matches the critique's priority list)

1. **B1 + B2 + B4 (this suite).** Baselines with seed statistics and F1. Without
   these the results section cannot support any claim. — *done, full run in
   progress.*
2. **B8 (UDE vs. fixed reservoir).** Settle the framing in §1.2; it shapes the
   whole results section.
3. **Reconcile numbers (§2)** using B4's per-class tables (dry-bean 92.3% vs.
   matrix) and fix the reservoir-size triple (N=64/512–1024/1797) by stating one
   unambiguous system size.
4. **B6 (Lorenz)** to fix the caption and add NRMSE / valid-time.
5. **B3 + B9 (topology test + reservoir diagnostics)** to support the topology
   and edge-of-chaos claims, or drop them.
6. **B7 (resonance metric)** to make the central mechanism measurable.

The remaining critique items (§2 figure inconsistencies, §3 equations, §4
bibliography, §5 typos) are text/figure edits, not simulations, and are tracked
in the critique checklist itself.

---

## Methodological choices (so results are defensible)

- **Same splits across models.** Every model in a given seed sees the identical
  train/test partition, so comparisons are paired (required for B3's Wilcoxon).
- **Standardization fit on train only**, applied to test — no leakage.
- **ESN matched to the reservoir.** `--quick` uses 300 units, full uses 500; set
  `NR` to the reservoir's DOF to honor the "matched size" requirement once the
  paper settles its system size (§2).
- **Dispersion before claims.** No "A beats B" statement without the paired
  Wilcoxon across seeds.
- **Honest Lorenz framing.** Valid time in Lyapunov times + NRMSE, never "very
  good."
