# EP-XY digits scale-up — Stage 1: harder classes vs a logreg baseline

Scripts: `scripts/xy_digits_stage1.jl`, `scripts/xy_digits_stage1_baseline.jl`
Run: `julia -t auto --project=. scripts/xy_digits_stage1.jl` (2026-06-30/07-01, 20 threads)

## Goal

Stage 0 proved the pipeline trains and the N≈100 XY net relaxes, but on an easy
3-class subset (0,1,2) where a linear model also hits 100% — uninformative. Stage
1 raises difficulty and adds an interpretable baseline:

- **Task A**: confusable digits {3, 5, 8}
- **Task B**: 5-class {0, 1, 2, 3, 4}

For each, the XY net is EP-trained and a **softmax logistic regression** is fit on
the *same* train/test split (raw pixel features), so XY accuracy is read against a
known linear bar. Operating point from Stage 0: `N_ev=1000` (T=100), steady-state
tolerance relaxed to 1e-3. 60 train / 30 test per class, 30 hidden, 120 epochs,
β=0.01, Adam.

## Results

| task | chance | XY test | logreg test | XY train |
|------|-------:|--------:|------------:|---------:|
| {3,5,8}        | 0.333 | **0.978** | 0.978 | 0.994 |
| {0,1,2,3,4}    | 0.200 | 0.980 | 0.993 | 1.000 |

(cost {3,5,8}: 1.193→0.023; cost {0..4}: 1.544→0.037 — both descend smoothly.)

> Note: the logreg numbers were recomputed with `xy_digits_stage1_baseline.jl`
> after an `argmax`-on-a-row-matrix bug in the in-script baseline reported 0.000.
> The fix is in `xy_digits_stage1.jl`; the baseline script reproduces the exact
> split (same seed/selection order) so the numbers pair directly with the XY run.

## Conclusion — EP-XY is a genuinely competitive multi-class classifier

The EP-trained XY net **matches** the linear baseline on the confusable 3-class
task (0.978 = 0.978) and lands **just below** it on 5-class (0.980 vs 0.993). It
does not beat logreg, but operating at linear-baseline level is a real positive:
the substrate that solves XOR robustly **scales to multi-class digit
classification** — a clean contrast with Duffing, which couldn't even train XOR
robustly. EP genuinely trains a recurrent oscillator network as a classifier.

The small Task-B gap (XY 0.980, train 1.000 vs logreg 0.993) is mild overfitting:
the XY net fits the train set perfectly while logreg, with explicit L2,
generalizes slightly better. Regularization / more data would likely close it.

## The compute reality

EP at this scale is **expensive**: Task A took **3317 s (~55 min)** and Task B
**4094 s (~68 min)** — ~2 h total on 20 threads. The cost is intrinsic: each
gradient step is 3 ODE-relaxations-to-equilibrium *per sample* (free, ±β), the
relaxation horizon is long (T=100, from the slow gradient-flow tail), and the
all-to-all coupling is O(N²) per solver step. Roughly 65k–90k ODE solves per
task.

## Stage 2 plan

- **Full 10-class** is the goal, but at ~2 h for 5 classes it needs the cost cut
  first: 4×4 input downsampling (N≈100→~50, O(N²) ~4× cheaper), one-sided
  gradient (`symmetric=false`, 2 relaxations not 3), fewer epochs (cost is flat
  by ~60), and/or a shorter horizon with the 1e-3 tolerance.
- Add **L2 / weight decay** to the XY weights to close the generalization gap.
- Compare against the FHN reservoir and MLP baselines (see the baseline suite),
  not just logreg, on the full task.
