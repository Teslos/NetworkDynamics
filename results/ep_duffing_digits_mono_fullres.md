# Monostable Duffing digits at full 64px: matches baselines (0.96)

Script: `scripts/duffing_digits_mono_fullres.jl`
Run: `julia -t auto --project=. scripts/duffing_digits_mono_fullres.jl` (2026-07-15)

## What changed

The identical stable monostable setup from mono v2
(`results/ep_duffing_digits_mono_v2.md`, 0.84 on 4x4-pooled inputs) run on the
**full 64 pixels** (no pooling), Wang's 100/70-per-class split — equal footing with
XY Stage 3 (full 64px, 0.94). Same substrate/training: single-well hidden (a>0),
linear/softmax readout, symmetric ±β gradient, Landau annealing, best-checkpoint.
Only the input resolution changed.

## Results

| model             | train | test  |
|-------------------|------:|------:|
| Duffing mono 64px | 0.993 | **0.959** |
| logreg 64px       | —     | 0.959 |
| MLP 64px          | —     | 0.964 |

Chance 0.10. Refs: Duffing mono 4x4 = 0.84; XY full-64px = 0.94. Stable training:
CE 2.3 → 0.03 monotone, test climbed to 0.95+ as `a_h` annealed (0.26@a_h=3 →
0.92@a_h=1.8 → 0.96@a_h=0.5), 400 iters in 45 s.

## Conclusion — fully competitive multi-class classifier

At full resolution, monostable Duffing reaches **0.959 — equal to logreg (0.959),
essentially matching MLP (0.964), and above XY's 0.94** — and fits the train set
(0.993). The 4x4 result (0.84) was purely resolution-limited; nothing about the
substrate caps it. The monostable single-basin Duffing is a fully competitive
10-class classifier.

Complete digit line, one bistability source removed at a time, then resolution:

| configuration | test |
|---|---|
| bistable readout (double-well output) | 0.18 |
| graded readout, double-well hidden | 0.27 |
| monostable hidden, unstable training (v1) | 0.54 |
| monostable, stable training, 4x4 (v2) | 0.84 |
| **monostable, stable training, full 64px** | **0.96** |
| logreg / MLP (64px) | 0.959 / 0.964 |
| XY (full 64px) | 0.94 |

## Two-substrate study: final state

Both substrates are strong multi-class classifiers in their smooth regime:
**XY phase 0.94, monostable Duffing 0.96** (both full 64px, both ≈ logreg/MLP). And
both use the bistable/deep-double-well regime for single-bit/memory (XOR: layered
Duffing 95% 6/6, XY 10/10). The regime split — bistable = memory, monostable/smooth
= multi-class — holds for both position-encoded (Duffing) and phase-encoded (XY)
oscillator networks. EP trains oscillator networks to baseline-competitive
multi-class accuracy provided the substrate operates in its smooth regime.

The Duffing "can't do digits" wall was entirely about bistability (readout + hidden
features) and training stability; removed, the monostable Duffing matches the best
baselines. Study complete.
