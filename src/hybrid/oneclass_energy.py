"""Digits "one at a time": ten one-class energy experts + energy recognition.

Extreme class-incremental limit of the isolation recipe.  Each digit gets its
own small oscillator subsystem (an energy model), trained *generatively* by
minimising its free energy (equilibrium energy) on that digit's data only --
so in-class inputs relax into a deep well.  Experts are trained sequentially
and frozen, so adding a digit never touches earlier ones: **zero forgetting by
construction**.  There is no discriminative readout; classification is pure
recognition -- pick the expert whose (calibrated) free energy is lowest.

This is where discrimination moves entirely to inference, and the per-expert
energy *calibration* (z-scoring each expert's energy by its own in-class
statistics) is what makes the energy-routing signal usable -- the weak link
found in the two-task experiment.
"""
from __future__ import annotations

import argparse
import functools

import jax
import jax.numpy as jnp
import numpy as np
import optax
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import jax_energy as E
import jax_reservoir as R


def load_features(reservoir_size, seed):
    d = load_digits()
    x_tr, x_te, y_tr, y_te = train_test_split(
        d.data, d.target, test_size=0.2, random_state=seed, stratify=d.target)
    sc = StandardScaler().fit(x_tr)
    x_tr, x_te = sc.transform(x_tr).astype(np.float32), sc.transform(x_te).astype(np.float32)
    w = R.build_weights(64, reservoir_size, seed=seed)
    ztr, zte = np.asarray(R.reservoir_features(x_tr, w)), np.asarray(R.reservoir_features(x_te, w))
    mean, std = ztr.mean(0, keepdims=True), ztr.std(0, ddof=1, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (ztr - mean) / std, y_tr.astype(np.int64), (zte - mean) / std, y_te.astype(np.int64)


def free_energy(p, cfg, z):
    """Equilibrium (free) energy of the expert on inputs z; envelope-theorem grad."""
    phi = jax.lax.stop_gradient(E.relax_free(p, cfg, z))
    return E.energy_per_sample(p, phi, z)             # (B,)


def train_expert(z_c, feat, cfg, steps, lr, wd, seed):
    p = E.init_params(feat, cfg.n_classes, jax.random.PRNGKey(seed))  # n_classes = H oscillators
    opt = optax.adamw(lr, weight_decay=wd)             # weight decay prevents energy collapse
    state = opt.init(p)
    zc = jnp.asarray(z_c)

    @jax.jit
    def step(p, state):
        loss, g = jax.value_and_grad(lambda pp: free_energy(pp, cfg, zc).mean())(p)
        updates, state = opt.update(g, state, p)
        return optax.apply_updates(p, updates), state, loss

    for _ in range(steps):
        p, state, _ = step(p, state)
    return p


def train_energy_classifier(ztr, ytr, feat, cfg, epochs, lr, wd, seed):
    """Contrastive fix: 10 experts trained jointly as an energy classifier
    (logit_c = -free_energy_c).  The shared softmax supplies the missing
    'push out-of-class energy up' signal that pure one-class learning lacks.
    Experts stay structurally independent (each its own oscillators)."""
    params = [E.init_params(feat, cfg.n_classes, jax.random.PRNGKey(seed + c)) for c in range(10)]
    opt = optax.adamw(lr, weight_decay=wd)
    state = opt.init(params)

    def loss_fn(params, zb, yb):
        Es = jnp.stack([free_energy(p, cfg, zb) for p in params], axis=1)   # (B, 10)
        return optax.softmax_cross_entropy_with_integer_labels(-Es, yb).mean()

    @jax.jit
    def step(params, state, zb, yb):
        loss, g = jax.value_and_grad(loss_fn)(params, zb, yb)
        updates, state = opt.update(g, state, params)
        return optax.apply_updates(params, updates), state, loss

    rng = np.random.default_rng(seed)
    ztr_j, ytr_j = jnp.asarray(ztr), jnp.asarray(ytr)
    batch = 64
    for _ in range(epochs):
        idx = rng.permutation(len(ytr))
        for i in range(0, len(ytr) - batch + 1, batch):
            b = idx[i:i + batch]
            params, state, _ = step(params, state, ztr_j[b], ytr_j[b])
    return params


def train_incremental(ztr, ytr, feat, cfg, epochs, lr, wd, seed, replay_per_class):
    """Strict continual: add one digit at a time, FREEZE old experts (zero
    forgetting), train each new expert contrastively via a softmax over the
    seen experts' -energy.  On class-c positives -> new expert wins (target c);
    on replayed old samples -> the frozen old expert wins (target j), which
    teaches the new expert NOT to claim old classes.  Only the new expert is
    updated; old experts are stop_gradient'd."""
    experts, rz, ry = [], [], []
    for c in range(10):
        z_pos = ztr[ytr == c]
        y_pos = np.full(len(z_pos), c)
        if replay_per_class > 0 and rz:
            z_all = np.concatenate([z_pos] + rz)
            y_all = np.concatenate([y_pos] + ry)
        else:
            z_all, y_all = z_pos, y_pos

        p = E.init_params(feat, cfg.n_classes, jax.random.PRNGKey(seed + c))
        opt = optax.adamw(lr, weight_decay=wd)
        state = opt.init(p)
        frozen = list(experts)
        first = not frozen

        def loss_fn(p, zb, yb):
            e_new = free_energy(p, cfg, zb)
            if first:                      # nothing to contrast the very first expert against
                return e_new.mean()
            efz = [jax.lax.stop_gradient(free_energy(pf, cfg, zb)) for pf in frozen]
            logits = -jnp.stack(efz + [e_new], axis=1)   # index j = expert j; index c = new
            return optax.softmax_cross_entropy_with_integer_labels(logits, yb).mean()

        @jax.jit
        def step(p, state, zb, yb):
            loss, g = jax.value_and_grad(loss_fn)(p, zb, yb)
            updates, state = opt.update(g, state, p)
            return optax.apply_updates(p, updates), state, loss

        rng = np.random.default_rng(seed + 100 + c)
        zj, yj = jnp.asarray(z_all), jnp.asarray(y_all)
        batch = min(64, len(y_all))
        for _ in range(epochs):
            idx = rng.permutation(len(y_all))
            for i in range(0, len(y_all) - batch + 1, batch):
                b = idx[i:i + batch]
                p, state, _ = step(p, state, zj[b], yj[b])
        experts.append(p)                  # freeze
        if replay_per_class > 0:
            k = min(replay_per_class, len(z_pos))
            rz.append(z_pos[:k]); ry.append(np.full(k, c))
    return experts


# --------------------------------------------------------------------------- #
# Density-RATIO experts against a fixed background.
# --------------------------------------------------------------------------- #
#
# What actually broke `train_incremental` was NOT forgetting -- old experts are
# frozen, so they cannot forget.  It is that raw free energies are incomparable
# across independently trained experts.  F is positively homogeneous of degree 1
# in the parameters: scaling theta scales both E and the phase gradient by the
# same factor, so phi* is unchanged and F(lam*theta) = lam*F(theta).  Minimising
# F is therefore unbounded below along a ray, and only weight decay sets the
# scale -- at stationarity (decoupled wd) Euler's theorem gives theta.grad = F,
# hence ||theta||^2 = |F|/wd.  Every expert lands on its own arbitrary energy
# scale, and argmin_c F_c compares incommensurable numbers.  Expert 0 is worse
# still: with no negatives its objective (`if first: return e_new.mean()`) is
# pure unconstrained energy minimisation, i.e. the collapse ray itself.
#
# Fix: each expert estimates a density RATIO against a fixed, class- and
# task-independent background p_bg by logistic (NCE-style) regression,
#
#     s_c(z) = b_c - F_c(z)   ~   log[ p_c(z) / p_bg(z) ]
#     L_c    = softplus(-s_c(z+)) + softplus(s_c(z-)),   z+ ~ D_c,  z- ~ p_bg
#
# Since every expert is anchored to the SAME p_bg, the scores share a scale by
# construction and argmax_c s_c(z) is Bayes-optimal (up to log-priors, absorbed
# by b_c) without any expert ever having seen another.  Consequences:
#   * expert 0 is no longer special -- it has negatives from the start, so its
#     objective is literally identical to every other expert's;
#   * training is order-independent and embarrassingly parallel; the "task
#     sequence" stops mattering, which is zero forgetting in the strongest sense;
#   * the collapse ray stops being a descent direction (inflating ||theta||
#     drives the negative term up), so weight decay becomes optional;
#   * the update is still two relaxations and a local contrast,
#       dL/dtheta = sig(-s+) dE/dtheta|_{phi*(z+)} - sig(s-) dE/dtheta|_{phi*(z-)},
#     i.e. positive-phase minus negative-phase, exactly the contrastive-Hebbian
#     form -- nothing here needs backprop through the solver.
#
# One more thing has to be fixed for any of that to be meaningful, and it is the
# deeper reason the original one-class experts collapsed to chance.  F(z) is a
# pointwise MINIMUM over phi of a function that is affine in z, so F is CONCAVE
# in z -- verified numerically: over 512 random midpoints at each of two field
# scales, F(mid) >= mean F(ends) holds 1024/1024 times (with the relaxation
# converged, residual <= 3e-3), and F falls linearly to -infinity along every ray
# (F at ||z|| = 1,2,4,...,64 is -3.97, -7.76, -15.4, ..., -188).  Therefore:
#   * -F is CONVEX, so {z : s_c(z) <= 0} is a convex set and {s_c > 0} is the
#     complement of one.  An expert can only ever carve out a cone-like region,
#     never a bounded blob around its class.  One-vs-background is then literally
#     unrealisable when the background surrounds the class -- no optimiser, no
#     learning rate, and no amount of data can fix it;
#   * exp(-F) is not integrable, so the "one-class energy expert" was never a
#     normalisable density in the first place.  There is no p_c, hence no
#     log-ratio for the model to represent.
# The fix is a BASE MEASURE: score with F~_c(z) = F_c(z) + 1/2 gamma_c ||z-m_c||^2,
# i.e. p_c ∝ q_c(z) exp(-F_c(z)) with q_c Gaussian.  The quadratic outgrows F's
# linear descent, so exp(-F~) is proper and the high-score region can be bounded.
# m_c is the class mean and needs only class c's own data, so this costs nothing
# in the continual setting.  Run with --base-only to ablate the oscillator tilt
# and see what the substrate contributes over the bare Gaussian.


def make_background(ztr, mode, jitter=1e-2):
    """Fixed negative distribution over reservoir features (no labels, no replay).

    "gauss"     : N(0, I) -- features are already standardized, so this needs no
                  stored data at all, but it ignores feature correlations and so
                  places few negatives where classes actually compete.
    "gaussfull" : N(mu, Sigma) with Sigma the (ridged) global feature covariance.
                  Fixed O(feat^2) cost, estimated once from the unlabeled stream;
                  negatives land on the data manifold, which is what makes the
                  ratio informative in the contested region.
    """
    feat = ztr.shape[1]
    mean = ztr.mean(0).astype(np.float32)
    if mode == "gauss":
        chol = np.eye(feat, dtype=np.float32)
    elif mode == "gaussfull":
        cov = np.cov(ztr, rowvar=False).astype(np.float64)
        cov += (jitter * np.trace(cov) / feat) * np.eye(feat)
        chol = np.linalg.cholesky(cov).astype(np.float32)
    else:
        raise ValueError(f"unknown background mode: {mode}")

    def sample(rng, n):
        return mean + rng.standard_normal((n, feat)).astype(np.float32) @ chol.T

    return sample


def tilted_free_energy(expert, cfg, z, use_osc=True):
    """F~_c(z) = F_c(z) + 1/2 gamma_c ||z - m_c||^2   (base measure + oscillator tilt).

    The quadratic is the base measure that makes exp(-F~) a proper density; the
    oscillator free energy F_c is the tilt.  Dropping the tilt (use_osc=False)
    leaves an isotropic Gaussian, which is the ablation that says how much the
    oscillator substrate is actually contributing.
    """
    gamma = jax.nn.softplus(expert["rho"])
    base = 0.5 * gamma * jnp.sum((z - expert["m"]) ** 2, axis=1)
    if not use_osc:
        return base
    return free_energy(expert["p"], cfg, z) + base


def ratio_scores(expert, cfg, z, use_osc=True):
    """s_c(z) = b_c - F~_c(z)  ~  log[p_c(z)/p_bg(z)]; comparable across experts."""
    return expert["bias"] - tilted_free_energy(expert, cfg, z, use_osc)


@functools.partial(jax.jit, static_argnames=("cfg", "steps", "use_osc"))
def langevin_negatives(expert, cfg, z0, key, steps, eta, temp, clip, use_osc=True):
    """Hard negatives: Langevin sampling from a frozen expert's own model.

    dz = -grad_z F~_c(z) dt + sqrt(2T) dW.  Proper only because of the base
    measure -- grad_z F_c alone is bounded away from zero along every ray
    (F_c is concave and 1-homogeneous-ish in z), so the untilted model has no
    invariant distribution to sample from.
    """
    def body(carry, _):
        z, key = carry
        key, sub = jax.random.split(key)
        gz = jax.grad(lambda zz: tilted_free_energy(expert, cfg, zz, use_osc).sum())(z)
        norm = jnp.linalg.norm(gz, axis=1, keepdims=True)
        gz = gz * jnp.minimum(1.0, clip / (norm + 1e-8))
        noise = jax.random.normal(sub, z.shape, dtype=z.dtype)
        z = z - eta * gz + jnp.sqrt(2.0 * eta * temp) * noise
        return (z, key), None

    (z, _), _ = jax.lax.scan(body, (z0, key), None, length=steps)
    return z


def init_ratio_expert(z_pos, feat, cfg, seed, gamma_init, use_osc):
    """Base measure from the class's own data; bias so that s_pos starts near 0."""
    expert = {
        "p": E.init_params(feat, cfg.n_classes, jax.random.PRNGKey(seed)),
        "m": jnp.asarray(z_pos.mean(0)),
        "rho": jnp.asarray(np.log(np.expm1(gamma_init)), dtype=jnp.float32),
        "bias": jnp.zeros((), dtype=jnp.float32),
    }
    f0 = tilted_free_energy(expert, cfg, jnp.asarray(z_pos[:256]), use_osc)
    expert["bias"] = jnp.asarray(f0.mean())
    return expert


def train_expert_ratio(z_pos, bg_sample, feat, cfg, epochs, lr, wd, seed,
                       neg_mult=1.0, batch=64, frozen=(), langevin_frac=0.0,
                       lang_steps=20, lang_eta=0.02, lang_temp=1.0, lang_clip=10.0,
                       gamma_init=0.05, use_osc=True):
    """Train ONE expert as a log-density-ratio.  Sees only its own class + p_bg."""
    expert = init_ratio_expert(z_pos, feat, cfg, seed, gamma_init, use_osc)
    n_pos = len(z_pos)
    bs = min(batch, n_pos)
    total_steps = max(1, epochs * (n_pos // bs))
    # Cosine decay is not cosmetic here.  The logistic loss saturates once the
    # class is separated from the background, and Adam's normalised update keeps
    # moving parameters at ~lr/step on the residual gradient noise -- which walks
    # the score scale off along the (now flat) collapse ray and silently breaks
    # cross-expert comparability again.  Fixed-lr runs of the SAME config landed
    # at 0.744 / 0.850 / 0.914; decaying the step size to zero removes that drift.
    sched = optax.cosine_decay_schedule(lr, total_steps)
    tx = optax.adamw(sched, weight_decay=wd) if wd > 0 else optax.adam(sched)
    state = tx.init(expert)

    def loss_fn(expert, zp, zn):
        s_pos = ratio_scores(expert, cfg, zp, use_osc)
        s_neg = ratio_scores(expert, cfg, zn, use_osc)
        return jax.nn.softplus(-s_pos).mean() + jax.nn.softplus(s_neg).mean()

    @jax.jit
    def step(expert, state, zp, zn):
        loss, g = jax.value_and_grad(loss_fn)(expert, zp, zn)
        upd, state = tx.update(g, state, expert)
        return optax.apply_updates(expert, upd), state, loss

    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed + 7919)
    zp_all = jnp.asarray(z_pos)
    n_neg = max(1, int(round(bs * neg_mult)))
    n_lang = int(round(n_neg * langevin_frac)) if frozen else 0
    last = 0.0

    for _ in range(epochs):
        idx = rng.permutation(n_pos)
        for i in range(0, n_pos - bs + 1, bs):
            zn = jnp.asarray(bg_sample(rng, n_neg - n_lang))
            if n_lang:
                key, k0 = jax.random.split(key)
                ef = frozen[int(rng.integers(len(frozen)))]
                z0 = jnp.asarray(bg_sample(rng, n_lang))
                zl = langevin_negatives(ef, cfg, z0, k0, lang_steps,
                                        lang_eta, lang_temp, lang_clip, use_osc)
                zn = jnp.concatenate([zn, zl], axis=0)
            expert, state, last = step(expert, state, zp_all[idx[i:i + bs]], zn)
    return expert, float(last)


def train_incremental_ratio(ztr, ytr, feat, cfg, epochs, lr, wd, seed, bg_sample,
                            neg_mult=1.0, langevin_frac=0.0, gamma_init=0.05,
                            use_osc=True, n_classes=10, verbose=True):
    """Class-incremental training with NO cross-expert coupling in the objective.

    Each expert is trained once, against the fixed background only, and frozen.
    Nothing an expert learns can be affected by a class that arrives later, and
    nothing about the arrival ORDER can change the final model.
    """
    experts = []
    for c in range(n_classes):
        z_c = ztr[ytr == c]
        # Langevin hard negatives (if enabled) come from already-frozen experts,
        # so the background is no longer strictly identical across experts --
        # that is the price of harder negatives.  Off by default.
        expert, loss = train_expert_ratio(
            z_c, bg_sample, feat, cfg, epochs, lr, wd, seed + c,
            neg_mult=neg_mult, frozen=list(experts), langevin_frac=langevin_frac,
            gamma_init=gamma_init, use_osc=use_osc)
        experts.append(expert)
        if verbose:
            tn = float(jnp.sqrt(sum(jnp.sum(x ** 2)
                                    for x in jax.tree_util.tree_leaves(expert["p"]))))
            s_in = np.asarray(ratio_scores(expert, cfg, jnp.asarray(z_c), use_osc))
            print(f"    expert {c}: loss={loss:.4f}  bias={float(expert['bias']):+8.2f}  "
                  f"gamma={float(jax.nn.softplus(expert['rho'])):.4f}  "
                  f"||theta||={tn:7.2f}  in-class s mean={s_in.mean():+7.2f}")
    return experts


def ratio_score_matrix(experts, cfg, z, use_osc=True):
    return np.stack([np.asarray(ratio_scores(e, cfg, jnp.asarray(z), use_osc))
                     for e in experts], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reservoir-size", type=int, default=256)
    ap.add_argument("--osc-per-expert", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-baselines", action="store_true",
                    help="jump straight to the density-ratio experts")
    ap.add_argument("--ratio-epochs", type=int, default=150)
    ap.add_argument("--ratio-lr", type=float, default=3e-3)
    ap.add_argument("--ratio-wd", type=float, default=0.0,
                    help="0 = rely on the ratio objective alone to bound ||theta||")
    ap.add_argument("--neg-mult", type=float, default=1.0,
                    help="negatives per positive in each minibatch")
    ap.add_argument("--bg-mode", nargs="+", default=["gauss", "gaussfull"],
                    choices=["gauss", "gaussfull"])
    ap.add_argument("--langevin-frac", type=float, default=0.0,
                    help="fraction of negatives drawn by Langevin from frozen experts")
    ap.add_argument("--gamma-init", type=float, default=0.05,
                    help="initial base-measure precision (per feature)")
    ap.add_argument("--base-only", action="store_true",
                    help="ablation: drop the oscillator tilt, keep only the Gaussian base")
    args = ap.parse_args()

    print("JAX devices:", jax.devices())
    ztr, ytr, zte, yte = load_features(args.reservoir_size, args.seed)
    feat = ztr.shape[1]
    cfg = E.EnergyConfig(n_classes=args.osc_per_expert)   # H oscillators per expert (no readout)

    if not args.skip_baselines:
        run_baselines(args, ztr, ytr, zte, yte, feat, cfg)

    # ---- Density-ratio experts: incremental, order-free, calibrated ----
    print("\nDensity-RATIO experts (fixed background; each expert trained ALONE):")
    for bgm in args.bg_mode:
        bg = make_background(ztr, bgm)
        tag = bgm + (f"+langevin{args.langevin_frac:g}" if args.langevin_frac else "")
        tag += " [BASE-ONLY ablation: no oscillators]" if args.base_only else ""
        print(f"  background = {tag}")
        exp = train_incremental_ratio(
            ztr, ytr, feat, cfg, epochs=args.ratio_epochs, lr=args.ratio_lr,
            wd=args.ratio_wd, seed=args.seed, bg_sample=bg,
            neg_mult=args.neg_mult, langevin_frac=args.langevin_frac,
            gamma_init=args.gamma_init, use_osc=not args.base_only)
        S = ratio_score_matrix(exp, cfg, zte, use_osc=not args.base_only)
        acc = float((S.argmax(1) == yte).mean())
        print(f"    all-10 (argmax_c s_c)   : {acc:.4f}")
        per = [float((S[yte == c].argmax(1) == c).mean()) for c in range(10)]
        print("    per-digit recall        : " + " ".join(f"{c}:{per[c]:.2f}" for c in range(10)))
        curve = []
        for c in range(10):
            sel = np.isin(yte, np.arange(c + 1))
            curve.append(float((S[sel][:, :c + 1].argmax(1) == yte[sel]).mean()))
        print("    acc on classes seen far : " + " ".join(f"{c}:{curve[c]:.2f}" for c in range(10)))


def run_baselines(args, ztr, ytr, zte, yte, feat, cfg) -> None:
    """The three prior recipes, kept verbatim as reference points."""
    experts, mu, sig = [], [], []
    print(f"\nTraining 10 one-class energy experts ({args.osc_per_expert} oscillators each), "
          f"incrementally + frozen:")
    for c in range(10):
        zc = ztr[ytr == c]
        p = train_expert(zc, feat, cfg, args.steps, args.lr, args.weight_decay, args.seed + c)
        fe_c = np.asarray(free_energy(p, cfg, jnp.asarray(zc)))   # in-class energy stats
        experts.append(p)
        mu.append(fe_c.mean())
        sig.append(fe_c.std() + 1e-6)
        print(f"  expert {c}: in-class free energy mean={fe_c.mean():.3f} std={fe_c.std():.3f}")

    mu, sig = np.asarray(mu), np.asarray(sig)
    # free energy of every test sample under every expert
    E_all = np.stack([np.asarray(free_energy(p, cfg, jnp.asarray(zte))) for p in experts], axis=1)

    pred_raw = E_all.argmin(1)
    pred_cal = ((E_all - mu[None]) / sig[None]).argmin(1)
    acc_raw = float((pred_raw == yte).mean())
    acc_cal = float((pred_cal == yte).mean())

    print(f"\nRecognition by lowest energy (no readout, no task oracle):")
    print(f"  raw argmin energy       : {acc_raw:.4f}")
    print(f"  calibrated (z-scored)   : {acc_cal:.4f}")
    # per-digit recall under calibrated recognition
    per = [float((pred_cal[yte == c] == c).mean()) for c in range(10)]
    print("  per-digit recall (cal)  : " + " ".join(f"{c}:{per[c]:.2f}" for c in range(10)))

    # ---- Contrastive fix: energy classifier (cross-class negatives) ----
    print("\nContrastive energy classifier (10 experts, shared softmax over -energy):")
    cparams = train_energy_classifier(ztr, ytr, feat, cfg, epochs=30, lr=1e-3,
                                      wd=args.weight_decay, seed=args.seed)
    Ec = np.stack([np.asarray(free_energy(p, cfg, jnp.asarray(zte))) for p in cparams], axis=1)
    acc_c = float((Ec.argmin(1) == yte).mean())
    print(f"  recognition by lowest energy (argmin): {acc_c:.4f}  (joint upper bound)")

    # ---- Incremental-contrastive: one digit at a time, old experts frozen ----
    print("\nIncremental-contrastive (one digit at a time; old experts FROZEN = zero forgetting):")
    for rp in (0, 20):
        exp = train_incremental(ztr, ytr, feat, cfg, epochs=40, lr=1e-3,
                                wd=args.weight_decay, seed=args.seed, replay_per_class=rp)
        Ei = np.stack([np.asarray(free_energy(p, cfg, jnp.asarray(zte))) for p in exp], axis=1)
        acc = float((Ei.argmin(1) == yte).mean())
        tag = "contrast-vs-frozen only" if rp == 0 else f"+ replay {rp}/class"
        print(f"  {tag:24}: all-10 = {acc:.4f}")
        if rp == 20:
            curve = []
            for c in range(10):
                sel = np.isin(yte, np.arange(c + 1))
                pr = Ei[sel][:, :c + 1].argmin(1)
                curve.append(float((pr == yte[sel]).mean()))
            print("    acc on classes seen so far: " + " ".join(f"{c}:{curve[c]:.2f}" for c in range(10)))


if __name__ == "__main__":
    main()
