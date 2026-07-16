from __future__ import annotations

import torch
from torch import Tensor, nn


class FrozenFHNReservoir(nn.Module):
    """Fixed FitzHugh-Nagumo reservoir with Euler integration."""

    def __init__(
        self,
        input_dim: int,
        n_reservoir: int = 64,
        coupling_strength: float = 0.3,
        input_scale: float = 0.5,
        edge_probability: float = 0.12,
        dt: float = 0.05,
        n_steps: int = 120,
        washout: int = 40,
        seed: int = 42,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if n_reservoir <= 0:
            raise ValueError("n_reservoir must be positive")
        if washout >= n_steps:
            raise ValueError("washout must be smaller than n_steps")
        if not 0.0 <= edge_probability <= 1.0:
            raise ValueError("edge_probability must be between 0 and 1")

        generator = torch.Generator().manual_seed(seed)

        w_in = (
            torch.randn(n_reservoir, input_dim, generator=generator)
            * input_scale
        )

        adjacency = (
            torch.rand(
                n_reservoir,
                n_reservoir,
                generator=generator,
            )
            < edge_probability
        ).float()

        adjacency = torch.triu(adjacency, diagonal=1)
        adjacency = adjacency + adjacency.T

        degree = adjacency.sum(dim=1)
        laplacian = torch.diag(degree) - adjacency

        self.register_buffer("w_in", w_in)
        self.register_buffer("laplacian", laplacian)

        self.n_reservoir = n_reservoir
        self.coupling_strength = coupling_strength
        self.dt = dt
        self.n_steps = n_steps
        self.washout = washout

        self.a = 0.7
        self.b = 0.8
        self.epsilon = 0.08

    @property
    def output_dim(self) -> int:
        """Dimension of the extracted reservoir feature vector."""
        return 5 * self.n_reservoir

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """
        Simulate the reservoir and extract trajectory features.

        Two input modes are supported:

        - Static, shape [batch, input_dim]: the projected input is held as a
          constant drive current for ``n_steps`` integration steps. This is the
          original behaviour used for tabular/image datasets.
        - Temporal, shape [batch, T, input_dim]: the input is a sequence (for
          example binned spike trains) and the projected input drives the
          reservoir step by step for ``T`` steps. This is the appropriate mode
          for continuous spiking datasets such as SHD.

        Returns:
            Feature tensor with shape [batch, 5 * n_reservoir].
        """
        if x.ndim == 2:
            temporal = False
        elif x.ndim == 3:
            temporal = True
        else:
            raise ValueError(
                "x must have shape [batch, input_dim] or "
                "[batch, T, input_dim]"
            )

        if x.shape[-1] != self.w_in.shape[1]:
            raise ValueError(
                f"Expected input dimension {self.w_in.shape[1]}, "
                f"received {x.shape[-1]}"
            )

        # Projected input drive. Static mode: [batch, n_reservoir].
        # Temporal mode: [batch, T, n_reservoir].
        currents = x @ self.w_in.T

        if temporal:
            n_steps = x.shape[1]
            if self.washout >= n_steps:
                raise ValueError(
                    f"washout ({self.washout}) must be smaller than the "
                    f"input sequence length ({n_steps})"
                )
        else:
            n_steps = self.n_steps

        batch_size = x.shape[0]
        v = torch.zeros(
            batch_size,
            self.n_reservoir,
            device=x.device,
            dtype=x.dtype,
        )
        w = torch.zeros_like(v)

        recorded_v: list[Tensor] = []
        recorded_w: list[Tensor] = []

        for step in range(n_steps):
            current = currents[:, step] if temporal else currents
            coupling = -(v @ self.laplacian.T)

            dv = (
                v
                - v.pow(3) / 3.0
                - w
                + current
                + self.coupling_strength * coupling
            )
            dw = self.epsilon * (v + self.a - self.b * w)

            v = v + self.dt * dv
            w = w + self.dt * dw

            if step >= self.washout:
                recorded_v.append(v.clone())
                recorded_w.append(w.clone())

        v_traj = torch.stack(recorded_v, dim=1)
        w_traj = torch.stack(recorded_w, dim=1)

        return torch.cat(
            [
                v_traj.mean(dim=1),
                v_traj[:, -1],
                v_traj.std(dim=1),
                v_traj.amax(dim=1),
                w_traj.mean(dim=1),
            ],
            dim=1,
        )
