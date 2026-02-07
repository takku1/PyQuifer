"""
Wilson-Cowan Neural Mass Model for PyQuifer

Alternative oscillator engine based on excitatory/inhibitory population dynamics.
Produces oscillations from E/I interaction — no explicit phase needed.

The Wilson-Cowan model describes the mean-field dynamics of coupled
excitatory (E) and inhibitory (I) populations:

    tau_E * dE/dt = -E + S(w_EE * E - w_EI * I + I_ext_E)
    tau_I * dI/dt = -I + S(w_IE * E - w_II * I + I_ext_I)

Where S(x) = 1 / (1 + exp(-gain * (x - threshold))) is a sigmoid.

Key properties:
- E/I balance creates natural oscillations
- Frequency depends on tau_E, tau_I ratio
- Bounded output [0, 1] — no explosion
- Can model alpha, beta, gamma rhythms

References:
- Wilson & Cowan (1972). Excitatory and inhibitory interactions
  in localized populations of model neurons.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Dict, Tuple, Literal


class WilsonCowanPopulation(nn.Module):
    """
    Single Wilson-Cowan E/I population pair.

    Produces oscillatory dynamics from excitatory-inhibitory interaction.
    Output is bounded [0, 1] by sigmoid activation.

    Args:
        tau_E: Excitatory time constant (larger = slower oscillation)
        tau_I: Inhibitory time constant (usually faster than E)
        w_EE: E→E connection weight (recurrent excitation)
        w_EI: I→E connection weight (inhibition to excitatory)
        w_IE: E→I connection weight (excitation to inhibitory)
        w_II: I→I connection weight (recurrent inhibition)
        gain: Sigmoid gain (steepness)
        threshold: Sigmoid threshold (midpoint)
        dt: Integration timestep
    """

    def __init__(self,
                 tau_E: float = 10.0,
                 tau_I: float = 5.0,
                 w_EE: float = 12.0,
                 w_EI: float = 4.0,
                 w_IE: float = 13.0,
                 w_II: float = 11.0,
                 gain: float = 1.0,
                 threshold: float = 4.0,
                 dt: float = 0.1,
                 ei_balance: float = 1.0,
                 integration_method: Literal['euler', 'rk4'] = 'euler'):
        super().__init__()
        self.tau_E = tau_E
        self.tau_I = tau_I
        self.w_EE = w_EE
        self.w_EI = w_EI
        self.w_IE = w_IE
        self.w_II = w_II
        self.gain = gain
        self.threshold = threshold
        self.dt = dt
        self.ei_balance = ei_balance  # G-09: scales w_EI to tune E/I ratio
        self.integration_method = integration_method

        # E/I state (evolved by dynamics, not backprop)
        self.register_buffer('E', torch.tensor(0.1))
        self.register_buffer('I', torch.tensor(0.1))

        # History for oscillation analysis
        self.register_buffer('E_history', torch.zeros(500))
        self.register_buffer('hist_ptr', torch.tensor(0))

    @classmethod
    def from_ei_ratio(cls, target_ratio: float = 0.8, **kwargs) -> 'WilsonCowanPopulation':
        """Construct population with specified E/(E+I) balance.

        Adjusts ei_balance so that at steady state, E/(E+I) ~ target_ratio.
        Higher target_ratio = more excitation-dominant.

        Args:
            target_ratio: Target E/(E+I) ratio in [0, 1]. Default 0.8 (excitation-dominant).
            **kwargs: Additional keyword arguments passed to __init__

        Returns:
            WilsonCowanPopulation with tuned ei_balance
        """
        # At steady state with default params, E/(E+I) ~ 0.5 when ei_balance=1.0
        # Scaling w_EI inversely shifts the balance toward excitation
        # ei_balance = (1 - target) / (1 - 0.5) maps linearly
        default_ratio = 0.5
        if target_ratio >= 1.0:
            ei_balance = 0.01
        elif target_ratio <= 0.0:
            ei_balance = 10.0
        else:
            ei_balance = (1.0 - target_ratio) / (1.0 - default_ratio)
        return cls(ei_balance=ei_balance, **kwargs)

    def _sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        """Wilson-Cowan sigmoid activation function."""
        return 1.0 / (1.0 + torch.exp(-self.gain * (x - self.threshold)))

    def forward(self,
                steps: int = 1,
                I_ext_E: float = 0.0,
                I_ext_I: float = 0.0) -> Dict[str, torch.Tensor]:
        """
        Evolve E/I dynamics.

        Args:
            steps: Number of integration steps
            I_ext_E: External input to excitatory population
            I_ext_I: External input to inhibitory population

        Returns:
            Dict with E, I, oscillation_power
        """
        E = self.E
        I = self.I
        # G-09: apply ei_balance scaling to inhibitory weight
        effective_w_EI = self.w_EI * self.ei_balance

        for _ in range(steps):
            if self.integration_method == 'rk4':
                _iee = I_ext_E
                _iei = I_ext_I

                def _wc_rhs(state):
                    e, i = state
                    inp_e = self.w_EE * e - effective_w_EI * i + _iee
                    inp_i = self.w_IE * e - self.w_II * i + _iei
                    de = (-e + self._sigmoid(inp_e)) / self.tau_E
                    di = (-i + self._sigmoid(inp_i)) / self.tau_I
                    return (de, di)

                k1 = _wc_rhs((E, I))
                k2 = _wc_rhs((E + 0.5 * self.dt * k1[0], I + 0.5 * self.dt * k1[1]))
                k3 = _wc_rhs((E + 0.5 * self.dt * k2[0], I + 0.5 * self.dt * k2[1]))
                k4 = _wc_rhs((E + self.dt * k3[0], I + self.dt * k3[1]))
                E = E + (self.dt / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
                I = I + (self.dt / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            else:
                input_E = self.w_EE * E - effective_w_EI * I + I_ext_E
                input_I = self.w_IE * E - self.w_II * I + I_ext_I
                dE = (-E + self._sigmoid(input_E)) / self.tau_E
                dI = (-I + self._sigmoid(input_I)) / self.tau_I
                E = E + dE * self.dt
                I = I + dI * self.dt

            # Clamp to [0, 1]
            E = E.clamp(0.0, 1.0)
            I = I.clamp(0.0, 1.0)

        # Update state
        with torch.no_grad():
            self.E.copy_(E.detach())
            self.I.copy_(I.detach())

            # Record history
            idx = self.hist_ptr % 500
            self.E_history[idx] = E.item()
            self.hist_ptr.add_(1)

        # Oscillation power: variance of recent E activity
        n_valid = min(self.hist_ptr.item(), 500)
        if n_valid > 10:
            recent = self.E_history[:n_valid]
            oscillation_power = recent.var()
        else:
            oscillation_power = torch.tensor(0.0, device=self.E_history.device)

        return {
            'E': E,
            'I': I,
            'oscillation_power': oscillation_power,
        }

    def get_oscillation_frequency(self, dt_ms: float = None) -> torch.Tensor:
        """
        Estimate dominant oscillation frequency from E activity history via FFT.

        Args:
            dt_ms: Timestep in milliseconds (defaults to self.dt)

        Returns:
            Dominant frequency in Hz
        """
        if dt_ms is None:
            dt_ms = self.dt

        n_valid = min(self.hist_ptr.item(), 500)
        if n_valid < 20:
            return torch.tensor(0.0, device=self.E_history.device)

        signal = self.E_history[:n_valid]
        signal = signal - signal.mean()  # Remove DC

        fft = torch.fft.rfft(signal)
        power = torch.abs(fft) ** 2
        power[0] = 0  # Ignore DC

        peak_idx = torch.argmax(power)
        freqs = torch.fft.rfftfreq(n_valid, d=dt_ms / 1000.0)
        return freqs[peak_idx]

    def reset(self):
        """Reset to initial state."""
        self.E.fill_(0.1)
        self.I.fill_(0.1)
        self.E_history.zero_()
        self.hist_ptr.zero_()


class WilsonCowanNetwork(nn.Module):
    """
    Network of coupled Wilson-Cowan populations.

    Multiple E/I populations coupled together, producing
    complex oscillatory patterns. Can serve as an alternative
    to Kuramoto banks for generating rhythmic dynamics.

    Args:
        num_populations: Number of coupled E/I populations
        coupling_strength: Inter-population coupling
        tau_E: Excitatory time constant
        tau_I: Inhibitory time constant
        dt: Integration timestep
    """

    def __init__(self,
                 num_populations: int = 4,
                 coupling_strength: float = 0.5,
                 tau_E: float = 10.0,
                 tau_I: float = 5.0,
                 dt: float = 0.1,
                 integration_method: Literal['euler', 'rk4'] = 'euler'):
        super().__init__()
        self.num_populations = num_populations
        self.coupling_strength = coupling_strength

        # Create populations with slightly different parameters for diversity
        self.populations = nn.ModuleList()
        for i in range(num_populations):
            pop = WilsonCowanPopulation(
                tau_E=tau_E * (1.0 + 0.1 * (i - num_populations / 2)),
                tau_I=tau_I * (1.0 + 0.1 * (i - num_populations / 2)),
                dt=dt,
                integration_method=integration_method,
            )
            self.populations.append(pop)

    def forward(self,
                steps: int = 1,
                external_input: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Evolve coupled population dynamics.

        Args:
            steps: Number of integration steps
            external_input: Optional input per population (num_populations,)

        Returns:
            Dict with E_states, I_states, mean_E, synchronization
        """
        for _ in range(steps):
            # Get current E states for coupling
            E_states = torch.stack([p.E.detach() for p in self.populations])
            mean_E = E_states.mean()

            for i, pop in enumerate(self.populations):
                # Coupling: drive toward mean field
                coupling_input = self.coupling_strength * (mean_E - E_states[i])

                ext = 0.0
                if external_input is not None:
                    ext = external_input[i].item()

                pop(steps=1, I_ext_E=coupling_input.item() + ext)

        E_final = torch.stack([p.E.detach() for p in self.populations])
        I_final = torch.stack([p.I.detach() for p in self.populations])

        # Synchronization: std of E activities (low std = synchronized)
        sync = 1.0 - E_final.std().clamp(max=1.0)

        return {
            'E_states': E_final,
            'I_states': I_final,
            'mean_E': E_final.mean(),
            'mean_I': I_final.mean(),
            'synchronization': sync,
        }

    def reset(self):
        """Reset all populations."""
        for pop in self.populations:
            pop.reset()
