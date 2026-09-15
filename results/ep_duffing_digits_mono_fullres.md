# Monostable Duffing digits at full 64px: 0.940 ± 0.012, just under logreg

Script: `scripts/duffing_digits_mono_fullres.jl`
Run: `julia -t auto --project=. scripts/duffing_digits_mono_fullres.jl`
(2026-09-12, 20 threads, 5 seeds, ~4.5 min/seed)
Record: `results/ep_duffing_digits_mono_fullres_seeds.json`

## What changed

The identical stable monostable setup from mono v2
(`results/ep_duffing_digits_mono_v2.md`, 0.84 on 4x4-pooled inputs) run on the
**full 64 pixels** (no pooling), Wang's 100/70-per-class split — equal footing with
XY Stage 3. Same substrate/training: single-well hidden (a>0), linear/softmax
readout, symmetric ±β gradient, Landau annealing. Only the input resolution changed.

**Evaluation protocol corrected (2026-09-12).** The first version of this
experiment reported **0.959**, which was the maximum over 17 evaluations *of the
test set*, from a single seed. That is a selection-biased number, and a single
seed contradicts the manuscript's statement that every accuracy is a mean over
seeds. The script now follows `src/hybrid/readout_ablation.py`:

* a stratified **20% validation** split is carved out of the 100-image/class
  training partition (training sees 80/class), and the checkpoint is selected on
  validation accuracy;
* the test partition is evaluated **once per seed**, at the end, for two
  checkpoints fixed in advance — the validation-selected one and the final
  iterate;
* 5 seeds resample the split, the initialization and the batch order;
* logreg/MLP are refit per seed on the **same reduced training split**, so the
  baseline comparison is paired and like-for-like (this is why logreg reads
  0.950 here and 0.959 in `results/baselines/`, which uses the full 80/20 split
  of all 1797 images).

## Results (5 seeds, mean ± std)

| model                       | train         | test          |
|-----------------------------|--------------:|--------------:|
| Duffing mono 64px (val-sel) | 0.983 ± 0.012 | **0.940 ± 0.012** |
| Duffing mono 64px (final iterate) | —       | 0.943 ± 0.008 |
| logreg 64px                 | —             | 0.950 ± 0.005 |
| MLP 64px                    | —             | 0.964 ± 0.006 |

Per seed (test, validation-selected): 0.937, 0.921, 0.939, 0.954, 0.947.
Paired difference Duffing − logreg: **−1.03 ± 1.60 pp**, Duffing ahead on 1/5
seeds. Chance 0.10. Ref: Duffing mono 4x4 = 0.84.

Two things are worth noting in the numbers:

* **The old 0.959 was selection, not substrate.** Honest selection on a
  validation split gives 0.940 ± 0.012 — about 1.9 pp lower, and the previous
  figure sits above the best of five honest seeds (0.954).
* **The validation selection buys nothing here.** The final iterate scores
  0.943 ± 0.008, statistically indistinguishable from the selected checkpoint
  (and nominally higher). Training is stable: CE falls monotonically to ~0.01 and
  validation accuracy plateaus in the 0.94–0.97 band after the anneal, so there
  is no informative peak to select. The selection step matters as a *protocol*
  guarantee, not as a performance gain.

## Conclusion — competitive with, but not matching, the linear baseline

At full resolution the monostable Duffing reaches **0.940 ± 0.012**, one
percentage point below a logistic regression fit on the same data
(0.950 ± 0.005) and about two below a small MLP (0.964 ± 0.006), while fitting the
training set (0.983 ± 0.012). The earlier claim of "equal to logreg" does not
survive an unbiased protocol; the honest statement is that it comes within about
a percentage point, with a seed-to-seed spread comparable to that gap.

What does survive is the resolution and regime story, which is much larger than
this correction: 4x4 pooling caps the same network at 0.84, and every bistable
variant fails outright.

Complete digit line, one bistability source removed at a time, then resolution:

| configuration | test |
|---|---|
| bistable readout (double-well output) | 0.18 |
| graded readout, double-well hidden | 0.27 |
| monostable hidden, unstable training (v1) | 0.54 |
| monostable, stable training, 4x4 (v2) | 0.84 |
| **monostable, stable training, full 64px** | **0.940 ± 0.012** |
| logreg / MLP (64px, same split) | 0.950 / 0.964 |

Caveat on that table: only the bottom row uses the corrected protocol. The rows
above it are single-seed numbers whose checkpoint was selected on the test set,
so they are upper bounds in the same way 0.959 was. The near-chance rows (0.18,
0.27) cannot move much, but 0.54 and 0.84 would likely come down somewhat if
rerun the same way.

## Two-substrate study

Both substrates classify well in their smooth regime, both at roughly the level
of a linear baseline rather than above it (XY: see
`results/xy_digits_stage3.md`). And both use the bistable/deep-double-well regime
for single-bit/memory tasks (XOR: layered Duffing 95% 6/6, XY 10/10). The regime
split — bistable = memory, monostable/smooth = multi-class — holds for both
position-encoded (Duffing) and phase-encoded (XY) oscillator networks. EP trains
oscillator networks to near-baseline multi-class accuracy provided the substrate
operates in its smooth regime.

The Duffing "can't do digits" wall was entirely about bistability (readout +
hidden features) and training stability; removed, the monostable Duffing lands
within about a point of the best conventional baselines on this split.
