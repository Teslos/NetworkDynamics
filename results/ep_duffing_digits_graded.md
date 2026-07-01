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

| model            | train | test  |
|------------------|------:|------:|
| Duffing (graded) | 0.268 | 0.270 |
| logreg           | —     | 0.835 |
| MLP              | —     | 0.880 |

Chance = 0.100. Bistable-readout Duffing (same net): 0.177. XY: 0.94.
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
