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

### §1.4 / §6 Resonance / separability metric  *(implemented)*
- **Experiment B7.** `scripts/run_reservoir_diagnostics.jl` computes silhouette
  score, Fisher discriminant ratio, and linear-probe accuracy on raw pixels vs
  FHN reservoir states vs ESN states (`silhouette_score`/`fisher_ratio` in
  `baseline_utils.jl`, reservoir from `src/baselines/fhn_reservoir.jl`).
- **Finding:** raw pixels are *more* linearly separable (silhouette 0.11,
  probe 0.96) than either reservoir's states (FHN: silhouette 0.04, probe 0.72;
  ESN: silhouette −0.15, probe 0.84). The reservoir transform does **not**
  improve separability, so the "local chaotic resonance improves
  classification" claim is unsupported as stated.
- **Resolves:** §1.4 — the mechanism is now measured, and the measurement
  contradicts the narrative.

### §1.2 / §6 Learned-coupling (UDE) vs. fixed reservoir ablation  *(planned)*
- **Experiment B8.** Run the paper's UDE/learned-coupling method (the XOR
  contribution, `notebooks/EP-XY-*` / the UDE formulation) on digits and dry
  bean, and compare head-to-head with the fixed reservoir on the *same* split.
- **Resolves:** §1.2 (contribution/identity conflation) — this is the
  experiment that decides whether the paper is "one method" or "two
  contributions," and is flagged as the top structural fix.

### §6 Reservoir diagnostics (echo-state property, edge of chaos)  *(implemented)*
- **Experiment B9.** `scripts/run_reservoir_diagnostics.jl` sweeps the FHN
  coupling strength σ and the ESN spectral radius ρ over seeds, reporting test
  accuracy (with error bars) and an echo-state / edge-of-chaos probe: FHN uses a
  two-initial-condition final-state divergence, ESN uses a Benettin local
  Lyapunov exponent (λ<0 ⇒ ESP holds; λ=0 ⇒ edge of chaos). Figure:
  `results/figures/edge_of_chaos.png`.
- **Finding:** the ESN shows the textbook edge of chaos (accuracy rises as λ→0).
  The **FHN reservoir's ESP divergence is ≈0 for all σ>0** — it is strongly
  contracting/synchronized, *not* operating near an edge of chaos; accuracy
  peaks around σ≈0.72 from richer mixing, not from criticality.
- **Resolves:** §6 (reservoir diagnostics) and tests the "edge of chaos" claim
  with statistics — for the FHN reservoir the claim is not borne out.

### Avalanche criticality (Beattie-style)  *(implemented)*
- **Experiment B10.** After reading the source reference
  (Al Beattie et al. 2024, *Communications Physics*, in `docs/`), it became
  clear that the relevant notion of criticality there is **avalanche/branching
  criticality** (power-law avalanche size/duration, branching ratio ≈ 1),
  *not* the dynamical edge of chaos — the paper explicitly states its network
  has no chaotic regime. `scripts/run_avalanche_criticality.jl` +
  `src/baselines/avalanche.jl` implement this: an excitable FHN network (a>1)
  with input to a subset of nodes, sweeping coupling σ and measuring branching
  ratio, power-law exponents, and mean activity.
- **Finding:** the FHN reservoir *does* show avalanche criticality. The
  branching ratio rises through 1 between σ≈0.003 and 0.01, and near the
  crossing the exponents approach mean-field values (τ≈1.6, α≈1.8). The
  critical coupling (~0.004) is far weaker than the classification operating
  point, which sits in the supercritical/synchronized regime.
- **Reconciliation:** the earlier "not at the edge of chaos" result (B9) and
  this "is at avalanche criticality" result are *both correct and consistent* —
  they measure different things, and Beattie et al. use the latter. The B9
  conclusion therefore does not contradict the source paper; it matches the
  paper's own disclaimer that the network is not chaotic.

### Criticality vs accuracy (capstone)  *(implemented)*
- **Experiment B11.** `scripts/run_criticality_vs_accuracy.jl` tests Beattie's
  core claim on ONE reservoir: an excitable FHN network (100 nodes, dry-bean
  features as input currents to 16 nodes), sweeping coupling σ and measuring on
  the same reservoir both the **classification accuracy** (full-state readout)
  and the **avalanche branching ratio** (a separate Poisson probe), plus a
  readout-shrinkage curve using random node subsets.
- **Finding:** accuracy is 0.86–0.905 across the whole coupling range, best
  (0.905) at criticality (σ=0.030, branching≈1.09) — a weak peak, matching
  Beattie's ~90.75% level and their robustness-without-tuning claim. However,
  the readout-shrinkage robustness is greatest for the **supercritical**
  (synchronized) reservoir, not the critical one, so Beattie's specific
  "critical ⇒ most shrinkage-robust" claim is not cleanly reproduced here.
- **Caveats:** one reservoir realization per σ, complete-graph (not
  Watts-Strogatz) coupling, 420 samples, naive branching estimator — suggestive
  same-network evidence, not a definitive replication.

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
