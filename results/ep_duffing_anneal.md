# EP-Duffing barrier annealing: does growing the well during training rescue XOR?

Script: `scripts/duffing_anneal.jl`
Run: `julia --project=. scripts/duffing_anneal.jl` (2026-06-30)

## Question

The static design sweep (`results/ep_duffing_design_sweep.md`) showed that
reshaping the double-well at a **fixed** depth leaves robust XOR accuracy at
chance: the forward equilibrium is bistable from epoch 1, so the output basin is
picked by the hidden-cell init, not the input, before training can steer it.

Barrier annealing attacks that directly. With `a = -s, c = s` the minima stay
pinned at `x = ±1` (encoding unchanged) while barrier depth `= s/4`:

- **Start near-flat** (`s = 0.05`): the potential barely constrains the cells, so
  the free equilibrium is **input-determined** and smooth — the regime where the
  XY/Kuramoto net solves XOR robustly.
- **Ramp `s → 1.0`** (linearly over the first `anneal_frac` of training, then
  hold): the wells deepen and lock in the ±1 levels around an XOR map ideally
  already learned in the easy regime.

Best free-phase cost is checkpointed **only once `s` reaches `1.0`**, so restored
weights are valid at the target deep-well substrate; robust accuracy is scored
there. Config: N = 5 (2 hidden), 6 seeds × 15 init draws, test noise 0.1, 2000
epochs. Chance for sign-correct XOR is ~50%.

## Results

| anneal frac | mean acc | max acc | min acc | median best/final cost |
|-------------|---------:|--------:|--------:|------------------------|
| 0.0 (ctrl)  | 60%      | 78%     | 47%     | 0.0013 / 0.2404        |
| 0.50        | 73%      | 93%     | 65%     | 0.0024 / 0.4850        |
| 0.90        | 71%      | 85%     | 55%     | 0.0029 / 0.2949        |

The control row reproduces the static sweep's `s=1.0, hidden=2` row exactly
(60/78/47, 0.0013/0.2404) — a cross-script sanity check.

## Conclusion

**Annealing helps — but only partially.** Ramping the barrier up during training
lifts mean robust accuracy from 60% (chance band) to **71–73%**, with the best
seed reaching **93%** (near-solve) and the min rising from 47% to 65% (frac 0.5).
The improvement is robust to the ramp fraction (0.5 ≈ 0.9).

This is a **directional confirmation of the bistability diagnosis**: starting in
the smooth, input-determined regime and deepening the wells afterwards does carry
some of the learned XOR map into the deep-well substrate — exactly what the
static sweep could not achieve. So forward-inference conditioning is indeed the
lever, not the gradient.

But it is **not a full rescue**. Mean robust accuracy stays ~71–73%, well below
the XY/Kuramoto net's robust 100% (10/10 seeds, `scripts/multiseed_convergence.jl`).
As the wells deepen, the network must keep every pattern in its input-determined
basin; some patterns still get recaptured by the wrong basin during the ramp.

Net effect on the branch's claim: EP **can** be coaxed toward XOR on the Duffing
substrate via a forward-inference curriculum (best seed 93%), which strengthens
"forward-inference conditioning governs trainability" — but EP still does not
solve Duffing XOR *robustly*, and the smooth XY net remains the only substrate
that does.

## Possible follow-ups

- Slower / nonlinear ramp (e.g. cosine, or longer hold at intermediate depths).
- Anneal + more hidden units (the static sweep's two levers combined).
- Per-pattern basin tracking to see *which* patterns flip during the ramp.
