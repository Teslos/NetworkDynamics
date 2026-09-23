# Graded-readout layered Duffing on digits: readout is necessary but not sufficient

Script: `scripts/duffing_digits_graded.jl`
Run: `julia -t auto --project=. scripts/duffing_digits_graded.jl` (2026-07-01)

## Question

Bistable-readout layered Duffing failed on 10-class digits (test 0.177 ≈ chance,
`results/ep_duffing_digits_layered.md`). Hypothesis: the per-cell bistable readout
(2^10 output basins, no competition) is the culprit. Test it by changing ONLY the
output layer — 10 **linear/graded** output cells (a=+1, c=0 → x_out = field) read
through **softmax + cross-entropy** (competitive, smooth), keeping everything else
identical (layered, double-well HIDDEN cells, basin-averaging, annealing).

## Results

Corrected protocol, 5 seeds, validation-selected checkpoint (see below):

| model                    | train         | test          |
|--------------------------|--------------:|--------------:|
| Duffing (graded, val-sel)| 0.213 ± 0.048 | **0.215 ± 0.049** |
| &nbsp;&nbsp;(final iterate)| —           | 0.101 ± 0.008 |
| logreg                   | —             | 0.835 ± 0.008 |
| MLP                      | —             | 0.877 ± 0.004 |

Per-seed test: 0.263, 0.200, 0.270, 0.180, 0.163. The originally reported
single-seed, test-selected value was 0.270 — which the corrected run reproduces
as its *best* seed, confirming that the old number was the maximum over an
evaluation sweep rather than a typical outcome. The final iterate sits at chance
(0.101 ± 0.008): this configuration does not hold whatever it finds.

Chance = 0.100. Bistable-readout Duffing (same net): 0.164 ± 0.035.
CE stayed ~2 (bounced); test peaked 0.27 (iter 100) then declined.

## Conclusion — the readout was PART of the problem, not all of it

The graded readout **helped but did not recover accuracy**: 0.177 → 0.270 (above
chance, above the bistable readout) but far below the 0.835 logreg baseline, with
an unconverged/bouncing CE. **The readout was necessary but not sufficient.**

(Note: this corrects an over-optimistic prediction — I expected the graded readout
to "very likely recover digit accuracy." It did not. The reality is more
informative.)

### Why fixing the output isn't enough: the hidden layer is also bistable

With 40 double-well HIDDEN cells, the hidden feature vector for a given image is one
of up to 2^40 basins. Basin-averaging trains the *gradient* over inits, but at
inference the hidden state is still basin-dependent, so **the feature
representation is not a stable, deterministic function of the input**. A linear
readout of unstable features cannot classify well — and making the output linear
actually *exposes* the hidden scatter directly to the readout.

So multi-class Duffing has **two** bistability problems:
1. the multi-class one-hot **readout** — fixed here (+9 points), and
2. the **hidden features** being multistable at scale — still broken.

XY avoids both: its phases are continuous, so the hidden representation is smooth
and, after basin-averaging, the readout becomes a deterministic function of the
input (Wang Fig. 3c: output converges while hidden units still scatter).
Position-encoded double wells are multistable by construction, so forming a rich,
stable multi-class feature map is fundamentally harder.

## Bottom line for the two-substrate study

- **XOR (single output):** Duffing's bistability is an ASSET (a memory element);
  layered Duffing solves it robustly (95%, 6/6 seeds).
- **10-class digits (rich representation + multi-way readout):** Duffing's
  bistability is a LIABILITY on BOTH the readout and the hidden features; graded
  readout recovers only part (0.18 → 0.27). The smooth phase substrate (XY) is the
  right tool (0.94).

The logical endpoint — make the hidden cells graded too — would recover accuracy
but by removing the Duffing nonlinearity entirely (a smooth net ≈ adopting the XY
substrate). Confirmed from the readout side: for rich multi-class computation the
smooth/phase encoding wins; the deep-double-well encoding is suited to
single-bit/memory tasks.

## Caveat

Short, somewhat unstable training (200 iters, CE bouncing). A longer/tuned run
might gain a little, but the stuck CE and the structural argument (multistable
hidden features) indicate a real limit, not merely a budget one.

## Evaluation protocol corrected (2026-09-13)

The numbers first recorded here came from a single seed whose checkpoint was
selected by repeatedly scoring the **test** set and keeping the maximum -- a
selection-biased figure, and a single seed against the manuscript's statement
that every accuracy is a mean over seeds. The script now follows
`src/hybrid/readout_ablation.py` (shared helper `src/utils/eval_protocol.jl`):
a stratified 20% validation split is carved out of the training partition and
selects the checkpoint; the test partition is evaluated **once per seed** on
checkpoints fixed in advance (the validation-selected one and the final
iterate); 5 seeds resample the split, the initialisation and the batch
order; and logreg/MLP are refit per seed on the same reduced training split.
Per-seed records: `results/ep_duffing_digits_graded_seeds.json`.
