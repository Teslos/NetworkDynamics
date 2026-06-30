# EP-Duffing design sweep: can shallower wells / more hidden units rescue XOR?

Script: `scripts/duffing_design_sweep.jl`
Run: `julia --project=. scripts/duffing_design_sweep.jl` (2026-06-30)

## Question

The deep-well / 2-hidden EP-Duffing baseline sits at chance on XOR
(`scripts/multiseed_convergence.jl`) because the bistable forward inference lets
the hidden-cell **initialization**, not the input, pick the output basin. This
sweep tests two levers meant to break that lock-in:

- **Well depth** `s` (`a = -s`, `c = s`): keeps minima fixed at `x = ±1` so the
  ±1 encoding is unchanged, while barrier depth `= s/4`. Shallower wells should
  let the input coupling/bias drive a cell into the input-determined basin
  (closer to the smooth XY inference that does solve XOR).
- **Hidden units**: `N = 2 inputs + n_hidden + 1 output`.

Scored by **robust accuracy** = fraction of (pattern × init-draw) that are
sign-correct under fresh free-cell noise. Chance for sign-correct XOR is ~50%.

Config: 6 seeds × 15 init draws, test noise 0.1, 2000 epochs.

## Results

| depth s | hidden | mean acc | max acc | min acc | median best/final cost |
|--------:|-------:|---------:|--------:|--------:|------------------------|
| 1.00    | 2      | 60%      | 78%     | 47%     | 0.0013 / 0.2404        |
| 1.00    | 6      | 49%      | 58%     | 43%     | 0.0001 / 1.3283        |
| 0.50    | 2      | 57%      | 63%     | 53%     | 0.0005 / 0.5822        |
| 0.50    | 6      | 51%      | 63%     | 38%     | 0.0003 / 0.5466        |
| 0.30    | 2      | 61%      | 70%     | 40%     | 0.0009 / 0.7668        |
| 0.30    | 6      | 56%      | 75%     | 38%     | 0.0012 / 0.5504        |

## Conclusion

**Neither lever rescues XOR.** Mean robust accuracy stays at chance (~49–61%)
across all six configurations; the best single seed only reaches 70–78%.

- **Well depth barely matters.** Going from `s = 1.0` to `s = 0.3` (barrier
  depth 0.25 → 0.075) does not move mean accuracy out of the chance band.
- **More hidden units do not help** — if anything `n_hidden = 6` is slightly
  worse than `n_hidden = 2`, and its final cost is higher (training is harder
  with more bistable cells, not easier).
- The **best free-phase cost** during training is driven very low (~1e-4) in
  every config, yet **final cost stays high** and robust accuracy is at chance.
  The network can momentarily fit but cannot *robustly infer*: the basin is
  still selected by initialization rather than by the input.

This reinforces the branch's headline finding: **gradient fidelity ≠
trainability**. EP gives faithful gradients on the Duffing network
(`scripts/check_ep_gradient_fidelity.jl`), but the bistability of the double-well
makes the forward equilibrium ill-posed as a function of the input, and
reshaping the wells within the ±1 encoding does not fix it. The smooth XY /
Kuramoto network — which has no such basin ambiguity — solves XOR robustly.
