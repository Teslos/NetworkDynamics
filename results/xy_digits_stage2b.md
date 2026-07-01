# EP-XY digits scale-up — Stage 2b: under-training or a real ceiling?

Script: `scripts/xy_digits_stage2b.jl`
Run: `julia -t auto --project=. scripts/xy_digits_stage2b.jl` (2026-07-01, 20 threads)

## Goal

Stage 2 got XY 0.767 test on full 10-class (vs logreg 0.843, MLP 0.883) but was
underfitting with an unconverged cost — consistent with under-training from the
compute cuts. Stage 2b re-spends the budget (Stage 2 ran in 165 s) to decide:
under-training or a genuine ceiling? Removed the cheap-but-lossy knobs, kept the
rest controlled (same 4×4 downsampling, same baselines):
symmetric (±β) gradient, 150 epochs (was 80), 40 hidden (was 20), 70 train / 35
test per class (was 50/30). N=66.

## Results

| model    | train acc | test acc | Stage 2 (train/test) |
|----------|----------:|---------:|----------------------|
| XY (EP)  | 0.816     | 0.797    | 0.792 / 0.767        |
| logreg   | 0.846     | 0.837    | 0.852 / 0.843        |
| MLP      | 0.951     | 0.900    | 0.962 / 0.883        |

Chance = 0.100. Trained 150 epochs in **2743 s (~46 min)**; cost 2.375 → 0.801
(min 0.634, still bouncing: epoch 105 = 0.69, 120 = 0.80, 135 = 0.83).

## Conclusion — optimization/conditioning ceiling, NOT pure under-training

The re-spent budget helped only marginally, and the failure signature persists —
so the Stage 2 gap is **not** simple under-training:

1. **Diminishing returns.** ~16.6× more compute (165 s → 2743 s: symmetric
   gradient, ~2× epochs, 2× hidden, 40% more data) moved test just **+3.0 pts**
   (0.767 → 0.797) and train **+2.4 pts** (0.792 → 0.816).
2. **Still cannot fit the train set.** XY train is 0.816 — *below* logreg's train
   (0.846) and far below MLP's (0.951) — despite ample capacity (40 hidden). Pure
   under-training would have been relieved by the added capacity + epochs; it was
   not.
3. **Cost won't converge.** It floors near ~0.63 and keeps bouncing regardless of
   capacity or epoch count.

Together these indicate a **genuine optimization/conditioning limit** for EP-XY at
10 classes, not a budget artifact. This coheres with the branch's gradient-fidelity
finding (`results/ep_duffing_*` / the EP-XY fidelity study): the XY **weight
gradient is basin-sensitive** — it degrades whenever a nudge rearranges phase
locking. At XOR the clean bias gradients dominated and training was robust; at
10-class scale (10 outputs, 40 hidden, richer phase locking) the weight-gradient
conditioning becomes the binding bottleneck, capping how well EP can drive the
free cost — and hence the train fit — down.

## Where the EP-XY scale-up lands

- Trains XORrobustly (10/10 seeds) and **matches logreg** on easy 3-class and
  confusable {3,5,8} (Stage 1: 0.978 = 0.978).
- Scales to full 10-class as a real classifier (~0.80 test) but hits an
  optimization ceiling, staying **~4 pts below logreg and ~10 below MLP** at 4×4
  resolution — and, tellingly, **cannot fit the training set** there.
- The boundary is set by weight-gradient conditioning, the same phenomenon the
  fidelity study identified — a satisfying closure: **gradient conditioning
  governs EP trainability, and it degrades with scale.**

## If pushed further (not expected to break the ceiling)

Lower learning rate + schedule, larger β, layered (feedforward-symmetric) rather
than fully-connected coupling to tame basin rearrangements, or full-8×8 inputs.
The diminishing return from Stage 2→2b suggests these would yield small gains, not
close the MLP gap.
