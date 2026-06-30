# EP-Duffing sweet-spot training: train deep, operate shallow

Script: `scripts/duffing_sweetspot.jl`
Run: `julia --project=. scripts/duffing_sweetspot.jl` (2026-06-30)

## Question

Basin tracking (`results/ep_duffing_basin_tracking.md`) found that the round-1
champion — annealed to `s=1.0` — is most accurate when **evaluated** at `s=0.5`
(mean 86% vs 71% at `s=1.0`), and speculated that **training/checkpointing at the
sweet-spot depth** `s≈0.5` would be an easy further win. This script tests that
by crossing training depth against operating (evaluation) depth. With `a=-s, c=s`
the ±1 minima are depth-independent, so operating depth is a free lever.

`hidden=2`, 6 seeds, 2000 epochs, 40 init draws (matches basin tracking).

## Results

| config                       | mean acc | max acc | min acc | median best-cost |
|------------------------------|---------:|--------:|--------:|------------------|
| fixed  s=0.5 → eval 0.5      | 52%      | 62%     | 46%     | 0.0005           |
| anneal→0.5  → eval 0.5       | 60%      | 71%     | 49%     | 0.0010           |
| anneal→1.0  → eval 1.0 (champ)| 71%     | 95%     | 61%     | 0.0024           |
| **anneal→1.0 → eval 0.5**    | **86%**  | 94%     | **82%** | 0.0024           |

## Conclusion — the hypothesis was wrong; the real lever is decoupling

**Training at the sweet-spot depth does NOT work.** Fixed `s=0.5` (52%) and
annealing only to `s=0.5` (60%) both underperform — directly refuting the
basin-tracking note's guess. The sweet spot is **not** a training-depth property.

The win comes from **decoupling training depth from operating depth**:

> **Train deep (`s=1.0`), operate shallow (`s=0.5`).**

This config gives the best mean (86%) and — more importantly — the best
**robustness**: the accuracy floor jumps from min 61% (champion, eval at 1.0) to
**min 82%**, with a tight 82–94% spread. The high-variance champion (max 95% but
min 61%) becomes a reliable ~86% once operated at the shallower depth.

## Mechanism

The two depths play different roles:

- **Deep well during training** provides strong restoring force, pinning every
  pattern hard onto ±1 and giving a large, clean EP gradient signal → it learns
  **sharp, well-separated weights**. Training in a shallow well (52–60%) gives
  weak pinning and correspondingly weak weights.
- **Shallow well at inference** lowers the barrier so the clamped inputs, not the
  hidden-cell noise, select the output basin → **input-determined** forward
  inference, exactly the smooth regime where the XY net succeeds.

So the deep well is good for *learning* and bad for *inference*; the shallow well
is the reverse. Using each where it helps is what closes most of the gap.

## Status

`train@1.0 / operate@0.5` (mean 86%, min 82%) is the **best and most robust
EP-Duffing XOR result to date**, up from the champion's 71%/61%. It is still
short of the smooth XY/Kuramoto net's robust 100% — even here, corner
`(+1,-1)→+1` lagged at 77% in the basin-tracking breakdown — so EP still does not
*robustly solve* Duffing XOR. But "train deep, operate shallow" is a clean,
transferable prescription for double-well EP substrates and a concrete
improvement over training and operating at a single depth.
