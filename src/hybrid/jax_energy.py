"""JAX port of the energy-based phase-oscillator readout (equilibrium prop).

Faithful to ``energy_head.py`` (torch): the XY energy

    E(phi, z) = -1/2 sum_{c!=d} K_cd cos(phi_c - phi_d)
                - sum_c [a_c(z) cos(phi_c) + b_c(z) sin(phi_c)]

relaxed by fixed-step gradient descent (the ``--energy-solver euler`` path),
here run entirely on the GPU inside a jitted ``lax.scan`` and batched across
samples.  Training uses the one-sided EP surrogate loss: the free/nudged
equilibria are stop-gradient'd, and the parameter gradient flows only through
the explicit energy(phi_fixed, z; params) dependence -- exactly as in torch.
"""
from __future__ import annotations

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp


class EnergyParams(NamedTuple):
    w_cos: jax.Array   # (C, feat)
    b_cos: jax.Array   # (C,)
    w_sin: jax.Array   # (C, feat)
    b_sin: jax.Array   # (C,)
    raw_coupling: jax.Array  # (C, C)


class EnergyConfig(NamedTuple):
    n_classes: int
    n_hidden: int = 0          # extra oscillators not read out (total = n_classes + n_hidden)
    phase_dt: float = 0.08
    free_steps: int = 200
    nudged_steps: int = 80
    beta: float = 0.2
    logit_scale: float = 5.0
    coupling_scale: float = 0.02
    # Relaxation solver: "euler" (fixed-step lax.scan) or a diffrax solver
    # ("tsit5" adaptive explicit RK, "kvaerno5"/"implicit_euler" implicit).
    # For the diffrax paths, free_steps/nudged_steps set the *time budget*
    # T = steps * phase_dt; the solver adapts within it and stops early at
    # steady state, so ||grad E|| is small by construction (no drift).
    solver: str = "euler"
    rtol: float = 1e-3
    atol: float = 1e-5
    ss_rtol: float = 1e-4
    ss_atol: float = 1e-5
    max_steps: int = 2048


def init_params(feature_dim: int, n_classes: int, key, coupling_scale: float = 0.02,
                n_hidden: int = 0) -> EnergyParams:
    n_total = n_classes + n_hidden   # hidden oscillators participate in dynamics, not readout
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    # Match nn.Linear default init scale (Kaiming-uniform-ish, bound 1/sqrt(fan_in)).
    bound = 1.0 / jnp.sqrt(feature_dim)
    w_cos = jax.random.uniform(k1, (n_total, feature_dim), minval=-bound, maxval=bound)
    b_cos = jax.random.uniform(k2, (n_total,), minval=-bound, maxval=bound)
    w_sin = jax.random.uniform(k3, (n_total, feature_dim), minval=-bound, maxval=bound)
    b_sin = jax.random.uniform(k4, (n_total,), minval=-bound, maxval=bound)
    raw_coupling = coupling_scale * jax.random.normal(k5, (n_total, n_total))
    return EnergyParams(w_cos, b_cos, w_sin, b_sin, raw_coupling)


def symmetric_coupling(raw: jax.Array) -> jax.Array:
    s = 0.5 * (raw + raw.T)
    return s - jnp.diag(jnp.diag(s))


def energy_per_sample(p: EnergyParams, phi: jax.Array, z: jax.Array) -> jax.Array:
    a = z @ p.w_cos.T + p.b_cos                     # (B, C)
    b = z @ p.w_sin.T + p.b_sin
    coupling = symmetric_coupling(p.raw_coupling)   # (C, C)
    phase_diff = phi[:, :, None] - phi[:, None, :]  # (B, C, C)
    pair = -0.5 * (coupling[None] * jnp.cos(phase_diff)).sum(axis=(1, 2))
    field = -(a * jnp.cos(phi) + b * jnp.sin(phi)).sum(axis=1)
    return pair + field                             # (B,)


def logits(cfg: EnergyConfig, phi: jax.Array) -> jax.Array:
    # Read out only the first n_classes oscillators (hidden ones are internal).
    return cfg.logit_scale * jnp.cos(phi[..., :cfg.n_classes])


# --- single-sample energy/gradient (for the vmapped diffrax relaxation) ----- #
def _energy_single(p: EnergyParams, phi: jax.Array, z: jax.Array) -> jax.Array:
    a = p.w_cos @ z + p.b_cos                       # (C,)
    b = p.w_sin @ z + p.b_sin
    coupling = symmetric_coupling(p.raw_coupling)
    pd = phi[:, None] - phi[None, :]                # (C, C)
    pair = -0.5 * (coupling * jnp.cos(pd)).sum()
    field = -(a * jnp.cos(phi) + b * jnp.sin(phi)).sum()
    return pair + field


def _grad_single(p, cfg, phi, z, label, beta):
    def f(ph):
        e = _energy_single(p, ph, z)
        if label is not None:   # static branch (Python None vs traced scalar)
            ce = -jax.nn.log_softmax(cfg.logit_scale * jnp.cos(ph[:cfg.n_classes]))[label]
            e = e + beta * ce
        return e
    return jax.grad(f)(phi)


def _cost_sum(p, cfg, phi, z, labels, beta):
    total = energy_per_sample(p, phi, z).sum()
    # ``labels is None`` is a static (Python-level) branch; ``beta`` may be a
    # traced value, so never compare it to 0.0 here.
    if labels is not None:
        ce = optax_softmax_ce(logits(cfg, phi), labels).sum()
        total = total + beta * ce
    return total


def optax_softmax_ce(logit: jax.Array, labels: jax.Array) -> jax.Array:
    logp = jax.nn.log_softmax(logit, axis=-1)
    return -jnp.take_along_axis(logp, labels[:, None], axis=-1)[:, 0]


def phase_gradient(p, cfg, phi, z, labels=None, beta=0.0):
    return jax.grad(lambda ph: _cost_sum(p, cfg, ph, z, labels, beta))(phi)


@functools.partial(jax.jit, static_argnames=("cfg", "steps", "with_nudge"))
def _relax(p, cfg, z, phi0, steps, labels, beta, with_nudge):
    lab = labels if with_nudge else None
    bet = beta if with_nudge else 0.0

    def step(phi, _):
        g = phase_gradient(p, cfg, phi, z, lab, bet)
        phi = phi - cfg.phase_dt * g
        phi = jnp.arctan2(jnp.sin(phi), jnp.cos(phi))
        return phi, None

    phi, _ = jax.lax.scan(step, phi0, None, length=steps)
    return phi


_DIFFRAX_SOLVERS = ("tsit5", "kvaerno5", "implicit_euler")


def _relax_diffrax(p, cfg, z, phi0, T, labels, beta):
    """Adaptive/implicit gradient-flow relaxation via diffrax, vmapped.

    Integrates dphi/dt = -grad(E + beta*C) and stops at steady state, so the
    equilibrium residual is small by construction regardless of coupling.
    """
    import diffrax

    solver = {"tsit5": diffrax.Tsit5,
              "kvaerno5": diffrax.Kvaerno5,
              "implicit_euler": diffrax.ImplicitEuler}[cfg.solver]()
    controller = diffrax.PIDController(rtol=cfg.rtol, atol=cfg.atol)
    event = diffrax.Event(diffrax.steady_state_event(rtol=cfg.ss_rtol, atol=cfg.ss_atol))

    def relax_one(phi0_i, z_i, label_i):
        term = diffrax.ODETerm(
            lambda t, phi, args: -_grad_single(p, cfg, phi, z_i, label_i, beta)
        )
        sol = diffrax.diffeqsolve(
            term, solver, t0=0.0, t1=T, dt0=cfg.phase_dt, y0=phi0_i,
            stepsize_controller=controller, event=event, max_steps=cfg.max_steps,
            saveat=diffrax.SaveAt(t1=True), throw=False,
        )
        phi = sol.ys[-1]
        return jnp.arctan2(jnp.sin(phi), jnp.cos(phi))

    if labels is None:
        return jax.vmap(lambda ph, zz: relax_one(ph, zz, None))(phi0, z)
    return jax.vmap(relax_one)(phi0, z, labels)


def relax_free(p, cfg, z):
    phi0 = jnp.zeros((z.shape[0], cfg.n_classes + cfg.n_hidden), dtype=z.dtype)
    if cfg.solver == "euler":
        return _relax(p, cfg, z, phi0, cfg.free_steps, None, 0.0, False)
    return _relax_diffrax(p, cfg, z, phi0, cfg.free_steps * cfg.phase_dt, None, 0.0)


def relax_nudged(p, cfg, z, free_phi, labels):
    if cfg.solver == "euler":
        return _relax(p, cfg, z, free_phi, cfg.nudged_steps, labels, cfg.beta, True)
    return _relax_diffrax(p, cfg, z, free_phi, cfg.nudged_steps * cfg.phase_dt, labels, cfg.beta)


def ep_loss(p, cfg, z, labels, coupling_reg):
    """One-sided EP surrogate (+ coupling penalty).  Equilibria detached."""
    free = jax.lax.stop_gradient(relax_free(p, cfg, z))
    nudged = jax.lax.stop_gradient(relax_nudged(p, cfg, z, free, labels))
    zc = jax.lax.stop_gradient(z)
    free_e = energy_per_sample(p, free, zc).mean()
    nudged_e = energy_per_sample(p, nudged, zc).mean()
    surrogate = (nudged_e - free_e) / cfg.beta
    penalty = coupling_reg * (symmetric_coupling(p.raw_coupling) ** 2).mean()
    return surrogate + penalty


@functools.partial(jax.jit, static_argnames=("cfg",))
def predict(p, cfg, z):
    free = relax_free(p, cfg, z)
    lg = logits(cfg, free)
    residual = jnp.linalg.norm(phase_gradient(p, cfg, free, z), axis=1).mean()
    return lg.argmax(axis=1), residual
