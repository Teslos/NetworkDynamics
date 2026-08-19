"""JAX/GPU port of the frozen FitzHugh-Nagumo reservoir.

Mirrors ``reservoir.py`` (torch) exactly: same Euler dynamics, same 5*N
feature extraction (v mean / v last / v std / v max / w mean).  The whole
trajectory is integrated with a jitted ``lax.scan`` so it runs on the GPU in
one batched pass, and the reservoir weights are frozen (no gradients).

The weights can be built from a numpy seed (a fresh random frozen reservoir)
or injected from an ``.npz`` produced by ``dump_torch_reservoir_weights.py``
(to reproduce a specific torch reservoir bit-for-bit for validation).
"""
from __future__ import annotations

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class ReservoirWeights(NamedTuple):
    w_in: jax.Array       # (n_reservoir, input_dim)
    laplacian: jax.Array  # (n_reservoir, n_reservoir)


class ReservoirConfig(NamedTuple):
    coupling_strength: float = 0.3
    dt: float = 0.05
    n_steps: int = 120
    washout: int = 40
    a: float = 0.7
    b: float = 0.8
    epsilon: float = 0.08


def build_weights(
    input_dim: int,
    n_reservoir: int,
    seed: int = 42,
    input_scale: float = 0.5,
    edge_probability: float = 0.12,
) -> ReservoirWeights:
    """Random frozen reservoir (numpy RNG; not bit-identical to torch)."""
    rng = np.random.default_rng(seed)
    w_in = rng.standard_normal((n_reservoir, input_dim)).astype(np.float32)
    w_in *= input_scale
    adjacency = (rng.random((n_reservoir, n_reservoir)) < edge_probability).astype(np.float32)
    adjacency = np.triu(adjacency, 1)
    adjacency = adjacency + adjacency.T
    degree = adjacency.sum(axis=1)
    laplacian = np.diag(degree) - adjacency
    return ReservoirWeights(jnp.asarray(w_in), jnp.asarray(laplacian.astype(np.float32)))


def load_weights(path: str) -> ReservoirWeights:
    data = np.load(path)
    return ReservoirWeights(jnp.asarray(data["w_in"]), jnp.asarray(data["laplacian"]))


@functools.partial(jax.jit, static_argnames=("cfg",))
def _features(x: jax.Array, w: ReservoirWeights, cfg: ReservoirConfig) -> jax.Array:
    currents = x @ w.w_in.T                       # (B, n) static drive
    batch, n = currents.shape
    v0 = jnp.zeros((batch, n), dtype=x.dtype)
    w0 = jnp.zeros((batch, n), dtype=x.dtype)

    def step(carry, _):
        v, ww = carry
        coupling = -(v @ w.laplacian.T)
        dv = v - v ** 3 / 3.0 - ww + currents + cfg.coupling_strength * coupling
        dw = cfg.epsilon * (v + cfg.a - cfg.b * ww)
        v = v + cfg.dt * dv
        ww = ww + cfg.dt * dw
        return (v, ww), (v, ww)

    (_, _), (vs, ws) = jax.lax.scan(step, (v0, w0), None, length=cfg.n_steps)
    vs = vs[cfg.washout:]   # (T, B, n)
    ws = ws[cfg.washout:]
    return jnp.concatenate(
        [
            vs.mean(axis=0),
            vs[-1],
            vs.std(axis=0, ddof=1),   # torch .std() is unbiased (ddof=1)
            vs.max(axis=0),
            ws.mean(axis=0),
        ],
        axis=1,
    )


def reservoir_features(
    x: np.ndarray | jax.Array,
    weights: ReservoirWeights,
    cfg: ReservoirConfig = ReservoirConfig(),
    chunk: int = 4096,
) -> jax.Array:
    """Extract 5*N features for ``x`` (chunked so full MNIST fits in memory)."""
    x = jnp.asarray(x, dtype=jnp.float32)
    if x.shape[0] <= chunk:
        return _features(x, weights, cfg)
    outputs = [_features(x[i:i + chunk], weights, cfg) for i in range(0, x.shape[0], chunk)]
    return jnp.concatenate(outputs, axis=0)


@functools.partial(jax.jit, static_argnames=("cfg", "steps_per_row", "washout_steps"))
def _features_temporal(x_seq, w, cfg, steps_per_row, washout_steps):
    """Row-streamed features: x_seq is (B, T_rows, d_step); each row is projected
    to a drive current and held for ``steps_per_row`` Euler sub-steps, so the
    reservoir integrates the sequence over T_rows * steps_per_row steps.
    """
    currents_rows = jnp.einsum("btd,nd->btn", x_seq, w.w_in)     # (B, T_rows, N)
    currents = jnp.repeat(currents_rows, steps_per_row, axis=1)  # (B, total, N)
    currents = jnp.swapaxes(currents, 0, 1)                      # (total, B, N)
    batch, n = x_seq.shape[0], w.w_in.shape[0]
    v0 = jnp.zeros((batch, n), dtype=x_seq.dtype)
    w0 = jnp.zeros((batch, n), dtype=x_seq.dtype)

    def step(carry, current):
        v, ww = carry
        coupling = -(v @ w.laplacian.T)
        dv = v - v ** 3 / 3.0 - ww + current + cfg.coupling_strength * coupling
        dw = cfg.epsilon * (v + cfg.a - cfg.b * ww)
        v = v + cfg.dt * dv
        ww = ww + cfg.dt * dw
        return (v, ww), (v, ww)

    (_, _), (vs, ws) = jax.lax.scan(step, (v0, w0), currents)
    vs = vs[washout_steps:]
    ws = ws[washout_steps:]
    return jnp.concatenate(
        [vs.mean(axis=0), vs[-1], vs.std(axis=0, ddof=1), vs.max(axis=0), ws.mean(axis=0)],
        axis=1,
    )


def reservoir_features_temporal(
    x_seq: np.ndarray | jax.Array,
    weights: ReservoirWeights,
    cfg: ReservoirConfig = ReservoirConfig(),
    steps_per_row: int = 5,
    washout_steps: int = 0,
    chunk: int = 2048,
) -> jax.Array:
    """Row-streamed 5*N features (chunked). x_seq: (B, T_rows, d_step)."""
    x_seq = jnp.asarray(x_seq, dtype=jnp.float32)
    if x_seq.shape[0] <= chunk:
        return _features_temporal(x_seq, weights, cfg, steps_per_row, washout_steps)
    outputs = [
        _features_temporal(x_seq[i:i + chunk], weights, cfg, steps_per_row, washout_steps)
        for i in range(0, x_seq.shape[0], chunk)
    ]
    return jnp.concatenate(outputs, axis=0)
