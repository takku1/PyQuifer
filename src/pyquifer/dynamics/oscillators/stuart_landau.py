"""Stuart-Landau oscillator — amplitude + phase dynamics with Hopf bifurcation.

Unlike Kuramoto (phase-only), Stuart-Landau tracks both amplitude and phase
via complex state z. The bifurcation parameter mu controls the distance from
the Hopf bifurcation point, making it a natural fit for criticality control.
"""
import torch
import torch.nn as nn
import math
from typing import Dict, Literal, Optional

from pyquifer.dynamics.oscillators.kuramoto import _rk4_step


class StuartLandauOscillator(nn.Module):
    """
    Stuart-Landau oscillator: amplitude + phase dynamics with Hopf bifurcation.

    Unlike Kuramoto (phase-only), Stuart-Landau tracks both amplitude and phase
    via complex state z:

        dz/dt = (mu + i*omega) * z - |z|^2 * z + coupling * (z_mean - z)

    Where:
    - mu > 0: limit cycle (oscillating)
    - mu < 0: fixed point (damped)
    - mu = 0: Hopf bifurcation (criticality!)

    The amplitude dynamics make this a natural fit for criticality control:
    |mu| = distance from bifurcation.

    Args:
        num_oscillators: Number of oscillators
        mu: Bifurcation parameter (>0 oscillating, <0 damped, =0 critical)
        omega_range: Range for natural frequencies
        coupling: Global coupling strength
        dt: Integration timestep
    """

    def __init__(self,
                 num_oscillators: int,
                 mu: float = 0.1,
                 omega_range: tuple = (0.5, 1.5),
                 coupling: float = 1.0,
                 dt: float = 0.01,
                 integration_method: Literal['euler', 'rk4'] = 'euler'):
        super().__init__()
        self.num_oscillators = num_oscillators
        self.dt = dt
        self.integration_method = integration_method

        # Bifurcation parameter (adapted by dynamics, not backprop)
        self.register_buffer('mu', torch.tensor(mu))

        # Natural frequencies
        self.omega = nn.Parameter(
            torch.rand(num_oscillators) * (omega_range[1] - omega_range[0]) + omega_range[0]
        )

        # Coupling strength
        self.coupling = nn.Parameter(torch.tensor(coupling))

        # Complex state z = r * exp(i * theta)
        # Initialize with small random amplitudes
        r_init = torch.rand(num_oscillators) * 0.3 + 0.1
        theta_init = torch.rand(num_oscillators) * 2 * math.pi
        self.register_buffer('z_real', r_init * torch.cos(theta_init))
        self.register_buffer('z_imag', r_init * torch.sin(theta_init))

        self.register_buffer('step_count', torch.tensor(0))

    @property
    def z(self) -> torch.Tensor:
        """Complex oscillator states."""
        return torch.complex(self.z_real, self.z_imag)

    @property
    def amplitudes(self) -> torch.Tensor:
        """Current oscillator amplitudes."""
        return torch.abs(self.z)

    @property
    def phases(self) -> torch.Tensor:
        """Current oscillator phases."""
        return torch.angle(self.z)

    def forward(self, steps: int = 1,
                external_input: Optional[torch.Tensor] = None) -> Dict:
        """
        Evolve Stuart-Landau dynamics.

        Args:
            steps: Number of integration steps
            external_input: Optional complex driving input (num_oscillators,)

        Returns:
            Dict with amplitudes, phases, order_parameter, z
        """
        z = self.z

        for _ in range(steps):
            if self.integration_method == 'rk4':
                _ext = external_input

                def _sl_rhs(zz):
                    zz_abs_sq = zz.real ** 2 + zz.imag ** 2
                    mu_iw = torch.complex(
                        self.mu.expand(self.num_oscillators),
                        self.omega
                    )
                    nl = zz_abs_sq * zz
                    zm = zz.mean()
                    ct = self.coupling * (zm - zz)
                    rhs = mu_iw * zz - nl + ct
                    if _ext is not None:
                        rhs = rhs + _ext
                    return rhs

                z = _rk4_step(_sl_rhs, z, self.dt)
            else:
                z_abs_sq = z.real ** 2 + z.imag ** 2
                mu_plus_iw = torch.complex(
                    self.mu.expand(self.num_oscillators),
                    self.omega
                )
                nonlinear = z_abs_sq * z
                z_mean = z.mean()
                coupling_term = self.coupling * (z_mean - z)
                dz = mu_plus_iw * z - nonlinear + coupling_term
                if external_input is not None:
                    dz = dz + external_input
                z = z + dz * self.dt

        # Update stored state
        with torch.no_grad():
            self.z_real.copy_(z.real.detach())
            self.z_imag.copy_(z.imag.detach())
            self.step_count.add_(steps)

        amplitudes = torch.abs(z)
        phases = torch.angle(z)
        order_param = torch.abs(z.mean())

        return {
            'amplitudes': amplitudes,
            'phases': phases,
            'order_parameter': order_param,
            'z': z,
            'mean_amplitude': amplitudes.mean(),
        }

    def get_order_parameter(self) -> torch.Tensor:
        """Global synchronization level."""
        return torch.abs(self.z.mean())

    def get_critical_coupling(self) -> torch.Tensor:
        """Estimate critical coupling Kc from frequency spread.

        For globally coupled Stuart-Landau oscillators, synchronization
        onset requires coupling to exceed the natural frequency spread.
        Kc ~ max(omega) - min(omega) (Chadwick et al. 2025).

        Returns:
            Estimated critical coupling strength
        """
        return self.omega.max() - self.omega.min()

    def get_criticality_distance(self) -> torch.Tensor:
        """Distance from Hopf bifurcation (|mu|). 0 = critical."""
        return torch.abs(self.mu)

    def set_mu(self, mu: float):
        """Set bifurcation parameter (for criticality control)."""
        with torch.no_grad():
            self.mu.fill_(mu)

    def reset(self):
        """Reset oscillators to random small-amplitude state."""
        r = torch.rand(self.num_oscillators) * 0.3 + 0.1
        theta = torch.rand(self.num_oscillators) * 2 * math.pi
        self.z_real.copy_(r * torch.cos(theta))
        self.z_imag.copy_(r * torch.sin(theta))
        self.step_count.zero_()
