# Critique & Revision Checklist
## "Chaotic oscillator networks for classification tasks"

Prepared as a structured review of the uploaded manuscript (`main.tex`, `supplement.tex`,
figures, and both `.bib` files). Items are grouped by severity. Line numbers refer to
`main.tex` unless noted. Tick the boxes as you address each point.

---

## 1. Most serious scientific problems
*(These are the items most likely to trigger rejection. Address these first.)*

- [ ] **Add baselines.** The abstract claims the framework can "surpass traditional
  approaches" and the dry-bean section says performance is "comparable to other methods,"
  but there is no baseline number anywhere. This is worse than neutral: on scikit-learn
  digits, plain logistic regression / linear SVM reaches ~95–97%, so the reported **88%
  (FHN) is below a one-line baseline**. On dry bean, standard classifiers reach ~92–93%,
  so the best number only matches the trivial baseline. Add a comparison table (logistic
  regression, linear SVM, small MLP, and a standard tanh-ESN of matched reservoir size) or
  remove the "surpasses/comparable" framing.

- [ ] **Resolve the contribution/identity conflation.** The abstract and Methods sell *one*
  idea — a neural network that **approximates the coupling terms** (UDE formulation,
  Eq. 5/6; XOR experiment). But the two main quantitative results (digits, dry bean) are
  ordinary **reservoir computing** with a *fixed/random* reservoir and a trained readout —
  the coupling is **not** learned there. The signature contribution is therefore
  demonstrated only on XOR, while the headline numbers use a different method. Either
  (a) reframe to present these as two clearly-separated contributions, or (b) run the
  learned-coupling (UDE) approach on digits/dry bean and report *those* numbers.

- [ ] **Add statistics — no result currently has any.** Every number is a single run.
  Reservoir construction, ER/Watts–Strogatz wiring, weight init, and spike encoding are all
  stochastic. Report **mean ± std over ≥10 seeds**. Without this, the claim (line 283) that
  all-to-all and Watts–Strogatz "outperform" Barabási–Albert has zero statistical support.

- [ ] **Quantify the "local chaotic resonance" mechanism.** The entire premise is that input
  induces a localized resonance/echo the readout exploits, but no observable is defined and
  nothing is measured. Add a resonance/separability metric (e.g. node-energy or amplitude
  separation between resonant vs. non-resonant sets, or a linear-probe/silhouette score on
  the reservoir state). Right now it is narrative, not result.

---

## 2. Internal inconsistencies (figures, numbers, captions)

- [ ] **Dry-bean accuracy contradicts its own figure.** Text says **92.3%** (line 276), but
  the confusion-matrix diagonals (SEKER 69.9, BARBUNYA 92.8, BOMBAY 88.8, CALI 80.6,
  HOROZ 73.1, SIRA 67.7, DERMASON 86.2) average ~**80%** (macro), and support-weighting
  pulls it lower (the large classes are among the worst). Reconcile the number with the
  matrix.

- [ ] **Best digits accuracy: main vs. supplement.** Main text says best = **88%**
  (line 260); supplement reports up to **95%** at σ=0.72 (supplement Table, line 234;
  text line 221). Report the same best number in both.

- [ ] **Reservoir size given three different values.** "N = 64 FHN / 128 DOF" (line 375)
  vs. supplement table "512–1024" (supplement line 112) vs. figure caption
  "1797 nodes and 3227412 edges" (line 248). Make the system size unambiguous.

- [ ] **"Complete graph … density of 0.25"** (lines 375 + 382) is a contradiction — a
  complete graph has density 1.0.

- [ ] **XOR figure is triply inconsistent.** Text: "5-oscillator fully connected"
  (line 287) and "5-oscillator … all-to-all" (line 305); caption: "8 node network …
  Watts-Strogatz" (line 302); label: `fig:xor_erdos_renyi` (Erdős–Rényi). Node count,
  topology, and label must agree.

- [ ] **Dynamics figure doesn't match its description.** Lines 253–255 reference a "red
  vertical marker" for the distortion and "red arrows" marking higher-amplitude resonant
  oscillators. The figure used (`FitzHug-Nagumo_MNIST.pdf`, `fig:net_dynamics`) shows a
  uniform synchronized band with **no markers, no arrows, no distinguishable
  resonant/non-resonant traces**. The *same single figure* is also cited for both the
  no-distortion and with-distortion cases. Provide separate free vs. driven panels with the
  annotations actually drawn.

- [ ] **Lorenz caption oversells.** Caption (line 383) says "The performance for the
  attractor is very good," but the figure shows divergence by t≈10 — which the body
  (line 387) correctly admits. Align the caption with the honest short-horizon framing and
  add a quantitative valid-time / NRMSE.

- [ ] **Batch vs. sample-count confusion.** "iteration over 1797 batches" with a "training
  set of 1437 samples" (line 260) doesn't parse — 1797 is the full dataset size. State
  epochs, batch size, and the 1437/360 split cleanly.

- [ ] **Abstract has a missing number** (line 61): "demonstrates an accuracy in machine
  learning classification task" — insert the value.

- [ ] **Unused results.** `duffing_reservoir_confusion_matrix.png`,
  `fhn_reservoir_confusion_matrix.png`, `kuramoto_reservoir_confusion_matrix.png` are in
  `images/` but never referenced; Duffing appears in the supplement parameter table only.
  Either include and discuss, or remove.

- [ ] **Empty code link.** `[github\url{}]` (line 393) is an unfilled placeholder. For a
  reproducibility claim this must be a real, working URL.

---

## 3. Equation / math issues

- [ ] **Complex Hebbian coupling missing the imaginary unit.** Line 347 reads
  `c_ij = k_ij e^{ψ_ij}`; the Hoppensteadt–Izhikevich form is `c_ij = k_ij e^{iψ_ij}`.
  As written it is not a phase factor.

- [ ] **Redundant stiffness term in Eq. 5/6.** The template ODE
  `ẍ = αẋ + βx + γx + Φ` (lines 139–147) has `βx + γx`, which collapses to one
  coefficient. One term was presumably meant to be nonlinear (e.g. cubic `x³` for Duffing).
  Fix or explain.

- [ ] **Kuramoto sign / normalization consistency.** Eq. (line 129) uses
  `sin(φ_i − φ_j)` (conventional attractive coupling is `sin(φ_j − φ_i)`); with
  `E = −½ Σ k cos(φ_i + ψ − φ_j)` and `dφ/dt = −∂E/∂φ`, check signs end-to-end. The
  dynamics has a `1/N` factor that is absent from the energy — reconcile.

- [ ] **LaTeX rendering glitches.** `d_r >> d_x` (line 182) has a stray `\>` artifact — use
  `\gg`. `C^n` (line 336) → `\mathbb{C}^n`.

- [ ] **EP gradient form.** The `d/dβ(∂F/∂θ)` leading equality (line 213) is nonstandard;
  consider citing the Scellier–Bengio form directly so it's unambiguous.

---

## 4. Bibliography problems

- [ ] **Lorenz 1963 duplicated and mis-keyed.** `norton_deterministic_1963` and
  `lorenz_deterministic_1963` are the *same* paper; the `norton_…` entry has the **wrong
  author ("Norton")** and a corrupted journal name ("Atmospheric Scienc-es"). The
  introduction (line 71) cites the wrong key, so Lorenz's foundational work is currently
  attributed to "Norton." Fix the citation and delete the bad entry.

- [ ] **~30 duplicate keys across the two `.bib` files** (e.g. `garcia_machine_2022` /
  `…-1`, `wang_brain-inspired_2024` / `…-1`, `wang_interpretable_2024` / `…-1`). These are
  Zotero export artifacts; with biber they cause warnings and can silently select the wrong
  variant. Merge into a single clean `.bib`.

- [ ] **Dangling reference.** Line 150 ends mid-sentence: "as reported by" — complete it or
  remove it.

---

## 5. Naming / typo inconsistencies

- [ ] `Izhikievich` (lines 81, 340) → **Izhikevich**
- [ ] `Watt-Strogatz` (line 150) → **Watts–Strogatz**
- [ ] `Albert-Barnabasi` (line 156) → **Barabási–Albert**
- [ ] `Beattie` (line 161) vs. `Bettie` (line 243) → pick the correct spelling consistently
- [ ] `MINST` (line 260) → **MNIST**
- [ ] `Hopefield` → **Hopfield**; `Grosseberg` → **Grossberg** (line 357)
- [ ] "osculation amplitudes" (line 253) → "oscillation"
- [ ] "facilities richer mixing" (line 283) → "facilitates"
- [ ] "makes it possible to and analyze" (line 79) → missing verb
- [ ] "ML can infer chaotic dynamics,directly" (line 79) → comma/spacing
- [ ] **Stray second `\documentclass[11pt]{article}` mid-supplement** (supplement line 145)
  — this breaks compilation; remove it.

---

## 6. Statistics & methods to add (necessary conditions)

- [ ] **Baseline comparison table** — logistic regression, linear SVM, small MLP, and a
  standard tanh-ESN of matched size, on each dataset, same splits.
- [ ] **Repeated runs with dispersion** — mean ± std (ideally 95% CIs) over ≥10 seeds for
  every accuracy and for the Lorenz error. Use a **paired Wilcoxon signed-rank test** across
  seeds before claiming any topology beats another.
- [ ] **Metrics beyond accuracy** — macro/weighted **F1** and per-class precision/recall,
  since dry bean is class-imbalanced (accuracy alone is misleading there).
- [ ] **Cross-validation** — stratified k-fold for dry bean; standard split + a CV estimate
  for digits.
- [ ] **Quantitative resonance/separability metric** — node-energy/amplitude separation
  between resonant and non-resonant sets, or a linear-probe / silhouette score on the
  reservoir state, so the central mechanism is measured.
- [ ] **Reservoir diagnostics** — verify the echo-state property (spectral-radius sweep),
  and show accuracy-vs-σ / accuracy-vs-spectral-radius curves with error bars to
  substantiate the "edge of chaos" claim. (Promote the supplement's σ-sweep to the main
  text, with statistics.)
- [ ] **Lorenz quantification** — report **valid prediction time** (in Lyapunov times) and
  **NRMSE** vs. the tanh-ESN baseline, instead of "very good."
- [ ] **Ablation** — learned-coupling (UDE) vs. fixed reservoir on the *same* task; this is
  the experiment that actually justifies the paper's framing.

---

## Suggested priority order

1. Fix the framing/contribution mismatch (§1, item 2) — it shapes everything else.
2. Add baselines + seed statistics (§1 and §6) — without these the results cannot support
   the claims.
3. Reconcile every number/figure inconsistency in §2 (especially dry-bean 92.3% vs. its
   matrix, and the reservoir-size triple).
4. Redraw the dynamics figure to show distortion + resonant/non-resonant traces, or soften
   the text to match.
5. Clean the bibliography (Lorenz mis-key, duplicates) and fix the supplement's stray
   `\documentclass`.
6. Sweep the typos/naming (§5).
