# EP-Duffing annealing round 2: slower ramp × more hidden units

Script: `scripts/duffing_anneal_slow.jl`
Run: `julia --project=. scripts/duffing_anneal_slow.jl` (2026-06-30)

## Question

Round 1 (`results/ep_duffing_anneal.md`) showed barrier annealing partially
rescues XOR: `hidden=2, linear ramp, frac=0.5, 2000 epochs` reached mean robust
accuracy 73% / best seed 93%, vs the 60% no-anneal control. This round tests
whether the two natural levers push past that:

- **Slower ramp** — 3000 epochs (vs 2000) and a **cosine ease-in** schedule that
  keeps the wells shallow (input-determined inference) longer before deepening,
  alongside the plain linear ramp.
- **More hidden units** — `N_HIDDEN ∈ {2, 4, 6}` (2 included as the round-1
  reference). This was the static sweep's other lever, which did nothing at fixed
  depth; the question is whether it helps once the curriculum has put cells in
  input-determined basins.

Same setup otherwise: `a=-s, c=s` (minima fixed at ±1, barrier `s/4`), start
`s=0.05` → `s=1.0`, checkpoint best free cost only at the deep well, score robust
accuracy there. 6 seeds × 15 init draws, test noise 0.1.

## Results

| hidden | schedule    | mean acc | max acc | min acc | median best/final cost |
|-------:|-------------|---------:|--------:|--------:|------------------------|
| 2      | linear 0.5  | 65%      | 80%     | 60%     | 0.0007 / 0.5327        |
| 2      | linear 0.9  | 64%      | 65%     | 60%     | 0.0011 / 0.5012        |
| 2      | cosine 0.9  | 62%      | 73%     | 50%     | 0.0010 / 0.4880        |
| 4      | linear 0.5  | 51%      | 57%     | 43%     | 0.0002 / 1.0029        |
| 4      | linear 0.9  | 48%      | 55%     | 43%     | 0.0003 / 0.9886        |
| 4      | cosine 0.9  | 49%      | 53%     | 45%     | 0.0005 / 1.0280        |
| 6      | linear 0.5  | 47%      | 53%     | 42%     | 0.0000 / 0.5850        |
| 6      | linear 0.9  | 56%      | 80%     | 43%     | 0.0004 / 1.1154        |
| 6      | cosine 0.9  | 55%      | 72%     | 42%     | 0.0005 / 1.0178        |

## Conclusion

**Neither lever beats round 1.** The best config here (`hidden=2, linear 0.5`,
65%) is *below* round 1's `hidden=2, linear 0.5, 2000 epochs` (73%), and no
configuration approaches it.

- **More hidden units HURT.** 4 and 6 hidden collapse back to the chance band
  (~47–56% mean) even with the curriculum, and their final cost is markedly
  higher (≈1.0 vs ≈0.5). More bistable cells = more basin ambiguity that all
  patterns must satisfy simultaneously; the smooth-start curriculum does not
  overcome it. This matches the static sweep, where 6 hidden was already worse
  than 2 at fixed depth.
- **A slower ramp does not help.** At `hidden=2`, 3000 epochs gives 65% (linear
  0.5) vs round 1's 73% at 2000 — slightly *worse*. The cosine ease-in is no
  better than linear. The extra hold epochs at the deep well appear to give
  patterns more opportunities to drift into the wrong basin rather than locking
  in the easy-regime solution.

**Best EP-Duffing XOR config to date remains round 1:** `hidden=2, linear ramp,
frac=0.5, 2000 epochs` → mean 73%, best seed 93%. The partial rescue is real but
does not scale with either ramp slowness or width, and EP still does not solve
Duffing XOR robustly. The smooth XY/Kuramoto net (robust 100%) remains the only
substrate that does.

## Takeaway for the branch

The annealing curriculum's benefit is fragile and substrate-specific: it works
only at minimal width and a moderate ramp, and adding capacity *removes* the
gain. This reinforces — rather than weakens — the headline that the deep
double-well's bistable forward inference, not the gradient, is the binding
constraint on EP trainability here.
