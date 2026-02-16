"""
Short-Term Plasticity Module for PyQuifer

Implements Tsodyks-Markram synapse model — the #1 missing synapse mechanism.
Short-term facilitation and depression allow synapses to act as dynamic filters:
- Facilitation (u): repeated spikes increase release probability (working memory)
- Depression (x): repeated spikes deplete vesicle pool (adaptation)

The interplay of u and x creates frequency-dependent filtering:
low-pass (depression dominant) or band-pass (facilitation + depression).

References:
- Tsodyks & Markram (1997). The neural code between neocortical pyramidal neurons.
- Tsodyks, Pawelzik & Markram (1998). Neural networks with dynamic synapses.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Dict


class TsodyksMarkramSynapse(nn.Module):
    """
    Tsodyks-Markram short-term plasticity model.

    State variables:
    - u: utilization parameter (facilitation). Increases with each spike.
      du/dt = -u/tau_f + U*(1-u)*spike
    - x: available resources (depression). Decreases with each spike.
      dx/dt = (1-x)/tau_d - u*x*spike
    - PSP = weight * u * x  (postsynaptic potential)

    Args:
        num_synapses: Number of synapses
        U: Baseline release probability (0.05=facilitating, 0.5=depressing)
        tau_f: Facilitation time constant (ms-equivalent steps)
        tau_d: Depression time constant (ms-equivalent steps)
        dt: Integration timestep
    """

    def __init__(self,
                 num_synapses: int,
                 U: float = 0.2,
                 tau_f: float = 200.0,
                 tau_d: float = 800.0,
                 dt: float = 1.0):
        super().__init__()
        self.num_synapses = num_synapses
        self.U = U
        self.tau_f = tau_f
        self.tau_d = tau_d
        self.dt = dt

        # Synaptic weights (learnable)
        self.weight = nn.Parameter(torch.randn(num_synapses) * 0.1)

        # STP state (managed by dynamics, not backprop)
        self.register_buffer('u', torch.full((num_synapses,), U))  # utilization
        self.register_buffer('x', torch.ones(num_synapses))         # available resources

    def forward(self, pre_spikes: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process presynaptic spikes through STP dynamics.

        Args:
            pre_spikes: Binary spike tensor (..., num_synapses)

        Returns:
            Dict with:
            - psp: Postsynaptic potential (..., num_synapses)
            - u: Current utilization state
            - x: Current resource state
            - efficacy: u * x (instantaneous synaptic efficacy)
        """
        with torch.no_grad():
            # Continuous decay between spikes
            self.u.add_((-self.u / self.tau_f) * self.dt)
            self.x.add_(((1.0 - self.x) / self.tau_d) * self.dt)

            # Spike-triggered updates
            spike_mask = pre_spikes > 0.5
            if spike_mask.dim() > 1:
                # Batch case: use mean spike pattern for state update
                spike_mean = pre_spikes.float().mean(dim=0) if pre_spikes.dim() > 1 else pre_spikes.float()
                spike_mask_1d = spike_mean > 0.1
            else:
                spike_mask_1d = spike_mask

            # Facilitation: u increases on spike
            du_spike = self.U * (1.0 - self.u)
            self.u[spike_mask_1d] = self.u[spike_mask_1d] + du_spike[spike_mask_1d]

            # Depression: x decreases on spike
            dx_spike = -self.u * self.x
            self.x[spike_mask_1d] = self.x[spike_mask_1d] + dx_spike[spike_mask_1d]

            # Clamp to valid ranges
            self.u.clamp_(0.0, 1.0)
            self.x.clamp_(0.0, 1.0)

        # PSP = weight * u * x * spike
        efficacy = self.u * self.x
        psp = self.weight * efficacy * pre_spikes.float()

        return {
            'psp': psp,
            'u': self.u.clone(),
            'x': self.x.clone(),
            'efficacy': efficacy,
        }

    @classmethod
    def facilitating(cls, num_synapses: int, **kwargs) -> 'TsodyksMarkramSynapse':
        """Preset for facilitating synapses (Tsodyks & Markram 1997).

        Low baseline release probability, long facilitation time constant,
        short depression recovery. Repeated spikes progressively increase
        synaptic efficacy — useful for working memory.

        Args:
            num_synapses: Number of synapses
            **kwargs: Override any default parameter
        """
        defaults = dict(U=0.05, tau_f=750.0, tau_d=50.0)
        defaults.update(kwargs)
        return cls(num_synapses=num_synapses, **defaults)

    @classmethod
    def depressing(cls, num_synapses: int, **kwargs) -> 'TsodyksMarkramSynapse':
        """Preset for depressing synapses (Tsodyks & Markram 1997).

        High baseline release probability, short facilitation time constant,
        long depression recovery. First spike is strong, subsequent ones
        weaken — useful for change detection / adaptation.

        Args:
            num_synapses: Number of synapses
            **kwargs: Override any default parameter
        """
        defaults = dict(U=0.5, tau_f=20.0, tau_d=800.0)
        defaults.update(kwargs)
        return cls(num_synapses=num_synapses, **defaults)

    def reset(self):
        """Reset STP state to baseline."""
        self.u.fill_(self.U)
        self.x.fill_(1.0)


class STPLayer(nn.Module):
    """
    Layer wrapping Tsodyks-Markram synapses with input/output projections.

    Combines STP dynamics with a linear projection, suitable for
    integration into spiking networks.

    Args:
        input_dim: Presynaptic dimension
        output_dim: Postsynaptic dimension
        U: Baseline release probability
        tau_f: Facilitation time constant
        tau_d: Depression time constant
        dt: Integration timestep
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 U: float = 0.2,
                 tau_f: float = 200.0,
                 tau_d: float = 800.0,
                 dt: float = 1.0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Input projection to synapse space
        self.proj = nn.Linear(input_dim, output_dim, bias=False)

        # STP dynamics on projected synapses
        self.stp = TsodyksMarkramSynapse(
            num_synapses=output_dim, U=U, tau_f=tau_f, tau_d=tau_d, dt=dt)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process input through projection + STP.

        Args:
            x: Input tensor (..., input_dim)

        Returns:
            Dict with:
            - output: STP-modulated output (..., output_dim)
            - efficacy: Current synaptic efficacy (output_dim,)
        """
        projected = self.proj(x)
        stp_result = self.stp(projected)

        return {
            'output': stp_result['psp'],
            'efficacy': stp_result['efficacy'],
            'u': stp_result['u'],
            'x': stp_result['x'],
        }

    def reset(self):
        """Reset STP state."""
        self.stp.reset()
