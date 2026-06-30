# EP-XY digits scale-up — Stage 2: full 10-class vs logreg + MLP

Script: `scripts/xy_digits_stage2.jl`
Run: `julia -t auto --project=. scripts/xy_digits_stage2.jl` (2026-07-01, 20 threads)

## Goal

Full 10-class digit classification with the EP-trained XY net, with the compute
cuts Stage 1 flagged as prerequisites (Stage 1 cost ~2 h for 5 classes). Cuts:
4×4 input downsampling (64→16 input cells, N≈46), one-sided EP gradient
(`symmetric=false`, 2 relaxations/sample not 3), 80 epochs, L2 weight decay, 50
train / 30 test per class. Baselines (logreg, 1-hidden-layer tanh MLP) are fit on
the **same 4×4 features** for a fair bar at the XY net's input resolution.

## Results

| model    | train acc | test acc |
|----------|----------:|---------:|
| XY (EP)  | 0.792     | 0.767    |
| logreg   | 0.852     | 0.843    |
| MLP      | 0.962     | 0.883    |

Chance = 0.100. Trained 80 epochs in **165 s** (cost 2.577 → 0.811).
(For context, the repo's full-8×8 baselines are logreg ~95.9%, MLP ~97.6%, FHN
reservoir ~93.6% — higher because they use the full resolution, not 4×4.)

## Conclusion — scales to 10-class, but under-trained here (not a ceiling)

EP-XY is a **genuine 10-class classifier** (76.7% test, far above 10% chance),
but under this cut-down budget it **trails logreg (84.3%) and MLP (88.3%)** — it
no longer matches the linear baseline as it did on the easier 3/5-class subsets.

The important diagnostic is the *direction* of the failure:

- **It is underfitting, not overfitting.** XY train accuracy (0.792) is barely
  above its test (0.767) and **below logreg's train (0.852)** and MLP's (0.962).
  The XY net did not even fit the training set, so the test gap is not a
  generalization problem.
- **The cost did not converge.** It fell 2.58 → 0.81 but was still high and
  *bouncing* (epoch 50: 1.18, epoch 60: 1.28, epoch 70: 0.78), the signature of
  a noisy/under-run optimizer, not a settled minimum.

Both point to **under-training**, induced by the very cuts that made the run
cheap: the one-sided gradient is noisier, 80 epochs is short for 10 classes, 20
hidden cells is little capacity, and 50 samples/class is sparse. So the
logreg/MLP gap here is most likely a budget artifact, not evidence that EP-XY
can't reach the baseline at 10 classes.

## The compute cuts worked

165 s vs Stage 1's ~2 h (for fewer classes). 4×4 downsampling (N≈46, O(N²) ~4×
cheaper) plus the one-sided gradient bought ~50× speedup, which leaves ample
budget to *re-spend* on a better-resourced run.

## Follow-up: test the under-training hypothesis (Stage 2b)

Re-run with the budget the 165 s baseline frees up: symmetric (±β) gradient,
~200 epochs, more hidden (~40), more samples/class, longer horizon. If XY then
fits the train set and closes on logreg/MLP, the gap was under-training; if it
plateaus below despite fitting train, that's a genuine capacity/conditioning
ceiling for EP-XY at 10 classes. Either outcome is a clean result.
