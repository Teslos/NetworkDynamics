# Layered EP-Duffing on digits: the multi-class bistable readout is the wall

Script: `scripts/duffing_digits_layered.jl`
Run: `julia -t auto --project=. scripts/duffing_digits_layered.jl` (2026-07-01)

## Question

Layered structure + capacity solved the bistable-Duffing *XOR* (95%,
`results/ep_duffing_layered.md`), and the same fix scaled XY to ~94% on 10-class
digits. Does layered Duffing also scale to multi-class digits? A-priori worry
(flagged before the run): Duffing's readout is 10 *independent double wells* — a
one-hot needs one cell in the `+1` well and nine in `−1`, i.e. **2¹⁰ output basins
per input**, vs XY's smooth `argmax sin(φ)`.

Setup: 4×4-pooled inputs (16 cells), layered input(16)→hidden(40)→output(10),
one-hot on ±1 wells, basin-averaging (full-range hidden/output init) + annealing,
threaded batch relaxation, vs logreg/MLP on the same 4×4 features.

## Results

Corrected protocol, 5 seeds, validation-selected checkpoint (see below):

| model                      | train         | test          |
|----------------------------|--------------:|--------------:|
| Duffing (layered, val-sel) | 0.165 ± 0.053 | **0.164 ± 0.035** |
| &nbsp;&nbsp;(final iterate)| —             | 0.167 ± 0.030 |
| logreg                     | —             | 0.835 ± 0.008 |
| MLP                        | —             | 0.877 ± 0.004 |

Per-seed test: 0.182, 0.165, 0.182, 0.102, 0.188. The originally reported
single-seed, test-selected value was 0.177, within the seed spread: at this level
the selection bias has almost nothing to select from, which is itself consistent
with the conclusion below. Selected checkpoint and final iterate agree, i.e. the
network is stuck rather than unstable.

Chance = 0.100. Cost: 18 → 3 (by iter 25) → **flat ~3** thereafter; test 0.10–0.18.

## Conclusion — layered Duffing does NOT scale to 10-class; readout is the cause

Test accuracy barely clears chance. The diagnostic is the cost: it drops 18→3
(the solves *respond* to the dynamics — not init noise) then **goes flat** — the
network **cannot fit the one-hot map**. Contrast the layered *XOR* case, where the
cost collapsed to 0.04. So this is a structural inability to fit, not (only)
under-training.

This confirms the predicted bottleneck — the **per-cell bistable readout**:

- XY reads a smooth, effectively competitive `argmax sin(φ)` over graded phases.
- Duffing has **10 independent double-well output cells with no competition**. A
  correct one-hot requires a specific assignment across 2¹⁰ output basins per
  input, which basin-averaging over a batch of 100 cannot cover.

Layered capacity + basin-averaging solved the *single*-output bistable XOR; it does
not carry to the *multi*-output bistable readout.

## The substrate asymmetry, now pinned exactly

- **Single bistable output (XOR)**: layered Duffing solves robustly (95%) — matches XY.
- **Multi-class bistable readout (digits)**: Duffing hits a wall XY does not,
  because XY's phase readout is smooth/competitive while Duffing's is 10
  independent bistable wells.

This is the clean boundary of the whole two-substrate study: EP trains both
substrates once multistability (basin-averaging) and capacity (layered) are
handled — *for tasks with a smooth or single-bistable output*. The multi-class
one-hot on independent double wells is where the position-encoded (Duffing) and
phase-encoded (XY) substrates genuinely diverge.

## Caveats and the fix that would likely work (but changes the substrate)

Caveats: minimal setup (4×4, 200 iters, one-sided, no output competition) and a
very short 15 s runtime (worth a relaxation-convergence check). But the flat cost
points to a structural, not budget, limit.

The principled fix is a **graded / competitive readout** — operate the output
cells at a shallow (sub-barrier) depth so their positions are continuous, or read a
continuous observable and apply softmax, restoring XY-like competition. That would
very likely recover digit accuracy — but it removes the output bistability, i.e. it
moves Duffing toward the phase-oscillator readout. Which is the honest endpoint:
for multi-class classification, the smooth (XY/phase) readout is the right tool;
the deep-double-well readout is a liability, exactly as the substrate analysis
predicted.

## Evaluation protocol corrected (2026-09-13)

The number first recorded here came from a single seed whose checkpoint was
selected by repeatedly scoring the **test** set and keeping the maximum. The
script now carves a stratified 20% validation split out of the training partition
to select the checkpoint, evaluates the test partition once per seed on
checkpoints fixed in advance, and averages over 5 seeds that resample the split,
the initialisation and the batch order (shared helper
`src/utils/eval_protocol.jl`). Per-seed records:
`results/ep_duffing_digits_layered_seeds.json`.
