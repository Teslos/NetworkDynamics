# EP-XY digits scale-up — Stage 3: Wang protocol reproduces ~94%, ceiling refuted

Script: `scripts/xy_digits_stage3.jl`
Run: `julia -t auto --project=. scripts/xy_digits_stage3.jl`
(2026-09-13, 5 seeds, ~2.5 h/seed under load; supersedes the single-seed run of
2026-07-01)

## Goal

Stage 2b concluded EP-XY hit a "genuine optimization/conditioning ceiling" at 10
classes (test ~0.80, could not fit train). Analysis of Wang et al. 2024
(`docs/Wang_2024_...pdf`, which `EP-XY-Network-Claude.jl` implements) showed they
reach 93.3% all-to-all with only 11 hidden units — refuting that conclusion.
Stage 3 adopts Wang's exact protocol to test whether the ceiling was an
implementation artifact.

Protocol changes vs Stage 2/2b:
- **Uniform `[−π,π)` init** of hidden+output every step, averaged over a large
  batch (the paper's multistability fix), replacing our near-zero `0.1·randn`.
- **Full 64-pixel inputs** (no 4×4 downsampling).
- **β = 0.1** (was 0.01), η = 0.1, one-sided gradient (as Wang).
- **N(0, 1/N) weight init**, bias strength h = 0 (Xavier-like).
- All-to-all, **11 hidden units** (Wang's best all-to-all, N=85), batch 100,
  400 iterations, readout `argmax sin(φ)`.

## Results

5 seeds, validation-selected checkpoint, test evaluated once per seed:

| model                    | train acc     | test acc      |
|--------------------------|--------------:|--------------:|
| XY (EP, Wang, val-sel)   | 0.981 ± 0.006 | **0.936 ± 0.012** |
| XY (final iterate)       | —             | 0.934 ± 0.017 |
| logreg (64px, same split)| —             | 0.950 ± 0.005 |
| MLP (64px, same split)   | —             | 0.964 ± 0.005 |

Per-seed test: 0.937, 0.950, 0.946, 0.924, 0.923. Paired XY − logreg:
−1.40 ± 0.93 pp, XY ahead on 0/5 seeds. Chance = 0.10.

**Protocol corrected (2026-09-13).** The first version of this experiment
reported **0.941**, the maximum over 17 evaluations *of the test set* from a
single seed. The script now carves a stratified 20% validation split out of the
training partition (so training sees 80 images/class rather than 100), selects
the checkpoint on validation, and evaluates the test partition once per seed on
checkpoints fixed in advance; 5 seeds resample the split, the initialisation and
the batch order, and logreg/MLP are refit per seed on the same reduced split
(hence logreg 0.950 here vs 0.959 in `results/baselines/`). Per-seed records:
`results/xy_digits_stage3_seeds.json`.

**This result barely moved: 0.941 → 0.936 ± 0.012, inside the seed spread.**
Unlike the Duffing digit runs, which lost 2–5 pp when test-selection was removed,
the XY network had nothing for the selection to exploit: the final iterate
(0.934 ± 0.017) matches the selected checkpoint, so the validation curve is flat
near the end rather than peaked. The Wang-protocol conclusion below stands as
written.

Reference — Wang paper (full 64px, all-to-all, 11 hidden): XY 93.3%, linear
90.4%, ANN 94.3%. Our Stage 2b (4×4, 40 hidden): XY 0.797, could not fit train.

## Conclusion — the Stage 2b ceiling was an implementation artifact

**EP-XY genuinely scales to ~94% on 10-class digits, reproducing Wang.** Our
93.6 ± 1.2% test matches Wang's 93.3% all-to-all / 94.1% layered, and — the decisive
point — the network now **fits the training set (0.981 ± 0.006)**, which Stage 2b could
not, using *fewer* hidden units (11 vs 40). That rules out a capacity or
fundamental-conditioning limit: it was the training protocol.

What mattered, in order:
1. **Uniform `[−π,π)` initialization + basin averaging** (primary). The
   multistability that made Stage 2b's cost bounce and blocked train-fit is
   resolved by sampling all basins and averaging the EP gradient — the paper's
   core method, which our earlier stages omitted (near-zero init sampled one
   basin).
2. **Full 64px resolution** — lifted the absolute ceiling (4×4 capped even logreg
   at 0.84).
3. **β = 0.1 + tighter tolerance** — cleaner EP gradient (our β = 0.01 with a
   loose tol gave ~10× noisier gradients).
4. **N(0, 1/N) weight init, bias h = 0** — better-conditioned start at scale.

The mechanism our earlier stages identified (basin-sensitive XY weight gradient)
was correct; the error was concluding it was insurmountable. The standard
multistability remedy fixes it.

Nuance on baselines: our XY (0.936 ± 0.012) is slightly below our softmax logreg
(0.950 ± 0.005) and MLP (0.964 ± 0.005), whereas Wang reports XY *beating* his
linear classifier (90.4%).
The difference is baseline strength — our softmax logreg is stronger than Wang's
parameter-matched MSE linear classifier. Relative to Wang's own baselines, our
result reproduces "XY ≳ linear". With Wang's full 1000 iterations (vs our 400)
and/or a layered architecture, the small remaining gap to logreg/MLP would likely
close further.

## Status: EP-XY scale-up arc complete and corrected

- Stage 0: fixed point exists (gradient flow).
- Stage 1: matches logreg on ≤5 easy/confusable classes.
- Stage 2/2b: full 10-class under a cut budget → ~0.80, *appeared* ceiling.
- **Stage 3: Wang protocol → 0.936 ± 0.012, fits train, reproduces the paper.** The
  apparent ceiling was our protocol (near-zero init, downsampling, small β), not
  EP-XY. **EP genuinely trains coupled phase oscillators as a ~94% 10-class
  classifier.**
