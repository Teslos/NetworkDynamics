from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class EnergyOscillatorHead(nn.Module):
    """Energy-based phase oscillator output layer.

    The relaxation dynamics follow gradient descent on the XY energy:

        E(phi, z) = -½ Σ_{c≠d} K_cd cos(phi_c - phi_d)
                    - Σ_c [a_c(z) cos(phi_c) + b_c(z) sin(phi_c)]

    By default the relaxation uses an adaptive RK45 integrator
    (scipy.integrate.solve_ivp) so the step size shrinks automatically when
    the energy landscape steepens during training — preventing the instability
    that occurs with fixed-step Euler when coupling weights grow.

    Set use_ode=False to revert to the original fixed-step Euler loop.
    """

    def __init__(
        self,
        feature_dim: int,
        n_classes: int,
        phase_dt: float = 0.08,
        free_steps: int = 80,
        nudged_steps: int = 30,
        beta: float = 0.2,
        logit_scale: float = 5.0,
        coupling_scale: float = 0.02,
        relax_tol: float | None = 1e-3,
        use_ode: bool = True,
        ode_rtol: float = 1e-3,
        ode_atol: float = 1e-5,
    ) -> None:
        super().__init__()

        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if n_classes <= 1:
            raise ValueError("n_classes must be at least 2")
        if beta == 0:
            raise ValueError("beta must be nonzero")

        self.n_classes = n_classes
        self.phase_dt = phase_dt
        self.free_steps = free_steps
        self.nudged_steps = nudged_steps
        self.beta = beta
        self.logit_scale = logit_scale
        self.relax_tol = relax_tol
        # Adaptive ODE solver (fix #3): RK45 via scipy.integrate.solve_ivp.
        # Replaces fixed-step Euler to prevent divergence as the landscape steepens.
        self.use_ode = use_ode
        self.ode_rtol = ode_rtol
        self.ode_atol = ode_atol

        self.field_cos = nn.Linear(feature_dim, n_classes)
        self.field_sin = nn.Linear(feature_dim, n_classes)

        self.raw_coupling = nn.Parameter(
            coupling_scale * torch.randn(n_classes, n_classes)
        )

    def symmetric_coupling(self) -> Tensor:
        """Return a symmetric coupling matrix with zero diagonal."""
        coupling = 0.5 * (self.raw_coupling + self.raw_coupling.T)
        return coupling - torch.diag_embed(torch.diagonal(coupling))

    def energy_per_sample(self, phi: Tensor, z: Tensor) -> Tensor:
        """Compute the physical energy for each sample.

        Args:
            phi: Phase tensor [batch, n_classes].
            z:   Reservoir feature tensor [batch, feature_dim].

        Returns:
            Energy values [batch].
        """
        a = self.field_cos(z)
        b = self.field_sin(z)
        coupling = self.symmetric_coupling()

        phase_difference = phi.unsqueeze(2) - phi.unsqueeze(1)

        pair_energy = -0.5 * (
            coupling.unsqueeze(0) * torch.cos(phase_difference)
        ).sum(dim=(1, 2))

        field_energy = -(a * torch.cos(phi) + b * torch.sin(phi)).sum(dim=1)

        return pair_energy + field_energy

    def logits(self, phi: Tensor) -> Tensor:
        """Convert oscillator phases into class logits."""
        return self.logit_scale * torch.cos(phi)

    def cost_per_sample(self, phi: Tensor, labels: Tensor) -> Tensor:
        return F.cross_entropy(self.logits(phi), labels, reduction="none")

    def phase_gradient(
        self,
        phi: Tensor,
        z: Tensor,
        labels: Tensor | None = None,
        beta: float = 0.0,
    ) -> Tensor:
        """Compute d(E + beta*C)/dphi."""
        with torch.enable_grad():
            phi_variable = phi.detach().requires_grad_(True)

            total = self.energy_per_sample(phi_variable, z).sum()

            if labels is not None and beta != 0.0:
                total = total + beta * self.cost_per_sample(
                    phi_variable, labels
                ).sum()

            gradient = torch.autograd.grad(total, phi_variable, create_graph=False)[0]

        return gradient

    # ------------------------------------------------------------------
    # Adaptive ODE relaxation (fix #3)
    # ------------------------------------------------------------------

    def _relax_ode(
        self,
        z: Tensor,
        initial_phi: Tensor,
        steps: int,
        labels: Tensor | None,
        beta: float,
    ) -> Tensor:
        """Adaptive RK45 relaxation via scipy.integrate.solve_ivp.

        Uses the same time budget (T = steps * phase_dt) as the Euler loop
        but lets the solver choose step sizes automatically.  When relax_tol
        is set, integration stops early once the mean gradient norm drops
        below the tolerance.
        """
        import scipy.integrate
        import numpy as np

        device = initial_phi.device
        dtype = initial_phi.dtype
        batch_size, C = initial_phi.shape

        # scipy runs on CPU; move tensors and model parameters there for the
        # duration of the solve, then restore the original device.
        original_device = next(self.parameters()).device
        if original_device.type != "cpu":
            self.cpu()

        z_cpu = z.detach().cpu()
        labels_cpu = labels.cpu() if labels is not None else None

        phi0_np = initial_phi.detach().cpu().numpy().ravel().astype(np.float64)

        def dyn(t: float, phi_flat: np.ndarray) -> np.ndarray:
            phi_t = torch.tensor(
                phi_flat.reshape(batch_size, C), dtype=dtype, device="cpu"
            )
            grad = self.phase_gradient(
                phi=phi_t, z=z_cpu, labels=labels_cpu, beta=beta
            )
            return (-grad).numpy().ravel().astype(np.float64)

        T_max = float(steps * self.phase_dt)
        events = []

        if self.relax_tol is not None:
            # Terminal event: fire (and stop) when mean residual < relax_tol.
            def converged(t: float, phi_flat: np.ndarray) -> float:
                phi_t = torch.tensor(
                    phi_flat.reshape(batch_size, C), dtype=dtype, device="cpu"
                )
                grad = self.phase_gradient(
                    phi=phi_t, z=z_cpu, labels=labels_cpu, beta=beta
                )
                return grad.norm(dim=1).mean().item() - self.relax_tol

            converged.terminal = True   # stop integration when event fires
            converged.direction = -1    # only trigger on decreasing crossings
            events.append(converged)

        sol = scipy.integrate.solve_ivp(
            dyn,
            [0.0, T_max],
            phi0_np,
            method="RK45",
            rtol=self.ode_rtol,
            atol=self.ode_atol,
            events=events or None,
            # Cap individual steps at 2× the original Euler step so the solver
            # stays responsive to the landscape rather than taking huge leaps.
            max_step=float(self.phase_dt) * 2.0,
        )

        # Restore model to original device before returning.
        if original_device.type != "cpu":
            self.to(original_device)

        phi_final = torch.tensor(
            sol.y[:, -1].reshape(batch_size, C), dtype=dtype, device=device
        )
        # Wrap phases back to [-pi, pi] for stable downstream computation.
        return torch.atan2(torch.sin(phi_final), torch.cos(phi_final)).detach()

    # ------------------------------------------------------------------
    # Euler relaxation (original, kept as fallback)
    # ------------------------------------------------------------------

    def _relax_euler(
        self,
        z: Tensor,
        phi: Tensor,
        steps: int,
        labels: Tensor | None,
        beta: float,
    ) -> Tensor:
        for _ in range(steps):
            gradient = self.phase_gradient(phi=phi, z=z, labels=labels, beta=beta)
            if (
                self.relax_tol is not None
                and gradient.norm(dim=1).mean() < self.relax_tol
            ):
                break
            phi = phi - self.phase_dt * gradient
            phi = torch.atan2(torch.sin(phi), torch.cos(phi))
        return phi.detach()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def relax(
        self,
        z: Tensor,
        initial_phi: Tensor,
        steps: int,
        labels: Tensor | None = None,
        beta: float = 0.0,
    ) -> Tensor:
        """Relax phases to (approximate) equilibrium.

        Dispatches to the adaptive ODE solver (use_ode=True, default) or the
        original fixed-step Euler loop (use_ode=False).
        """
        phi = initial_phi.detach()
        if self.use_ode:
            return self._relax_ode(z, phi, steps, labels, beta)
        return self._relax_euler(z, phi, steps, labels, beta)

    def equilibria(self, z: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
        """Compute free and nudged equilibria."""
        initial_phi = torch.zeros(
            z.shape[0], self.n_classes, device=z.device, dtype=z.dtype
        )
        free_phi = self.relax(z=z, initial_phi=initial_phi, steps=self.free_steps)
        nudged_phi = self.relax(
            z=z,
            initial_phi=free_phi,
            steps=self.nudged_steps,
            labels=labels,
            beta=self.beta,
        )
        return free_phi, nudged_phi

    def ep_surrogate_loss(
        self,
        z: Tensor,
        free_phi: Tensor,
        nudged_phi: Tensor,
    ) -> Tensor:
        """One-sided EP surrogate loss (equilibria detached)."""
        free_energy = self.energy_per_sample(free_phi.detach(), z.detach()).mean()
        nudged_energy = self.energy_per_sample(nudged_phi.detach(), z.detach()).mean()
        return (nudged_energy - free_energy) / self.beta

    def equilibrium_residual(self, z: Tensor, phi: Tensor) -> Tensor:
        """Mean L2 norm of the phase-energy gradient (diagnostic)."""
        gradient = self.phase_gradient(phi=phi, z=z)
        return gradient.norm(dim=1).mean()

    @torch.no_grad()
    def predict(self, z: Tensor) -> Tensor:
        initial_phi = torch.zeros(
            z.shape[0], self.n_classes, device=z.device, dtype=z.dtype
        )
        free_phi = self.relax(z=z, initial_phi=initial_phi, steps=self.free_steps)
        return self.logits(free_phi).argmax(dim=1)
