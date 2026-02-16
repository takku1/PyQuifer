"""
Spiking Neural Network Module for PyQuifer

Implements bio-inspired spiking neurons with surrogate gradient training.
Based on Leaky Integrate-and-Fire (LIF) dynamics with learnable parameters.

Key features:
- Differentiable spiking via surrogate gradients (fast sigmoid)
- Learnable membrane time constants and thresholds
- STDP-inspired Hebbian learning option
- Oscillatory SNN layers that can replace/complement Kuramoto banks
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Literal


class SurrogateSpike(torch.autograd.Function):
    """
    Surrogate gradient for spiking neurons.
    Forward: Heaviside step function (threshold crossing)
    Backward: Fast sigmoid gradient for differentiability
    """
    scale = 25.0  # Steepness of surrogate gradient

    @staticmethod
    def forward(ctx, membrane_potential: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(membrane_potential, threshold)
        return (membrane_potential >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        membrane_potential, threshold = ctx.saved_tensors
        # Fast sigmoid surrogate gradient
        grad_input = grad_output * SurrogateSpike.scale / (
            1 + (SurrogateSpike.scale * torch.abs(membrane_potential - threshold)) ** 2
        )
        return grad_input, None


def surrogate_spike(membrane_potential: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
    """Apply surrogate spike function."""
    return SurrogateSpike.apply(membrane_potential, threshold)


class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire Neuron with learnable parameters.

    Membrane dynamics: dV/dt = -(V - V_rest)/tau + I(t)
    Spike: when V >= threshold, emit spike and reset to V_reset

    Parameters are learnable for task-specific adaptation.
    """

    def __init__(self,
                 tau: float = 10.0,
                 threshold: float = 1.0,
                 v_reset: float = 0.0,
                 v_rest: float = 0.0,
                 dt: float = 1.0,
                 learnable: bool = True,
                 refractory_period: int = 0,
                 spike_mode: Literal['binary', 'ternary', 'graded'] = 'binary',
                 max_grade: float = 3.0):
        """
        Args:
            tau: Membrane time constant (higher = slower decay)
            threshold: Spike threshold voltage
            v_reset: Voltage after spike
            v_rest: Resting membrane potential
            dt: Simulation timestep
            learnable: Whether parameters are learnable
            refractory_period: Number of timesteps after spike where neuron cannot fire (G-13)
            spike_mode: 'binary' (0/1), 'ternary' (-1/0/1), or 'graded' (multi-bit) (G-14)
            max_grade: Maximum graded spike amplitude (only for spike_mode='graded')
        """
        super().__init__()
        self.dt = dt
        self.v_reset = v_reset
        self.v_rest = v_rest
        self.refractory_period = refractory_period
        self.spike_mode = spike_mode
        self.max_grade = max_grade

        # Learnable parameters (use log for positivity constraint)
        if learnable:
            self.log_tau = nn.Parameter(torch.tensor(math.log(tau)))
            self.threshold = nn.Parameter(torch.tensor(threshold))
        else:
            self.register_buffer('log_tau', torch.tensor(math.log(tau)))
            self.register_buffer('threshold', torch.tensor(threshold))

    @property
    def tau(self) -> torch.Tensor:
        return torch.exp(self.log_tau)

    @property
    def decay(self) -> torch.Tensor:
        """Membrane decay factor per timestep."""
        return torch.exp(-self.dt / self.tau)

    def forward(self, current: torch.Tensor, membrane: torch.Tensor,
                refractory_counter: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, ...]:
        """
        Single timestep update.

        Args:
            current: Input current (batch, neurons)
            membrane: Current membrane potential (batch, neurons)
            refractory_counter: Optional refractory counter tensor. If refractory_period > 0
                               and this is None, a zero counter is created.

        Returns:
            spikes: Spike tensor (batch, neurons)
            membrane: Updated membrane potential (batch, neurons)
            refractory_counter: (only if refractory_period > 0) Updated counter
        """
        # Handle refractory counter initialization
        if self.refractory_period > 0 and refractory_counter is None:
            refractory_counter = torch.zeros_like(membrane)

        # Leaky integration
        membrane = self.decay * (membrane - self.v_rest) + self.v_rest + current * self.dt / self.tau

        # G-13: Clamp membrane during refractory period
        if self.refractory_period > 0:
            in_refractory = (refractory_counter > 0).float()
            membrane = membrane * (1 - in_refractory) + self.v_reset * in_refractory

        # G-14: Spike generation based on mode
        if self.spike_mode == 'binary':
            spikes = surrogate_spike(membrane, self.threshold)
        elif self.spike_mode == 'ternary':
            # Positive spike when V >= threshold, negative when V <= -threshold + 2*v_rest
            pos_spikes = surrogate_spike(membrane, self.threshold)
            neg_spikes = surrogate_spike(-membrane + 2 * self.v_rest, self.threshold)
            spikes = pos_spikes - neg_spikes
        elif self.spike_mode == 'graded':
            # Multi-bit: amplitude encodes how much threshold was exceeded
            over_threshold = surrogate_spike(membrane, self.threshold)
            grade = ((membrane - self.threshold) / (self.threshold + 1e-8)).clamp(0, self.max_grade)
            spikes = over_threshold * grade
        else:
            spikes = surrogate_spike(membrane, self.threshold)

        # G-13: Force zero spikes during refractory
        if self.refractory_period > 0:
            spikes = spikes * (1 - in_refractory)

        # Reset after spike (use abs for ternary/graded)
        spike_mask = (spikes.abs() > 0).float()
        membrane = membrane * (1 - spike_mask) + self.v_reset * spike_mask

        # G-13: Update refractory counter
        if self.refractory_period > 0:
            refractory_counter = (refractory_counter - 1).clamp(min=0)
            refractory_counter = refractory_counter + spike_mask * self.refractory_period
            return spikes, membrane, refractory_counter

        return spikes, membrane


class SpikingLayer(nn.Module):
    """
    A layer of LIF neurons with input projection and recurrent connections.
    Suitable for sequence processing with temporal credit assignment.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 tau: float = 10.0,
                 threshold: float = 1.0,
                 dt: float = 1.0,
                 recurrent: bool = False,
                 dropout: float = 0.0):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Number of LIF neurons
            tau: Membrane time constant
            threshold: Spike threshold
            dt: Simulation timestep
            recurrent: Enable recurrent connections
            dropout: Dropout on input (applied before integration)
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.recurrent = recurrent

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim, bias=False)

        # Recurrent connections (optional)
        if recurrent:
            self.recurrent_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
            # Initialize recurrent weights small to prevent explosion
            nn.init.normal_(self.recurrent_proj.weight, std=0.01)

        # LIF neuron dynamics
        self.lif = LIFNeuron(tau=tau, threshold=threshold, dt=dt)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def prune(self, fraction: float):
        """Magnitude-prune input projection weights (G-05).

        Zeros out the smallest `fraction` of weights by absolute value
        and applies a permanent mask so they stay zero in forward passes.

        Args:
            fraction: Fraction of weights to prune, in [0, 1]
        """
        with torch.no_grad():
            w = self.input_proj.weight
            magnitudes = w.abs().flatten()
            k = max(1, int(fraction * magnitudes.numel()))
            threshold_val = magnitudes.kthvalue(k).values

            mask = (w.abs() >= threshold_val).float()
            if not hasattr(self, 'weight_mask'):
                self.register_buffer('weight_mask', mask)
            else:
                self.weight_mask.copy_(mask)
            w.mul_(mask)

    def forward(self, x: torch.Tensor, return_membrane: bool = False) -> torch.Tensor:
        """
        Process sequence through spiking layer.

        Args:
            x: Input tensor (batch, seq_len, input_dim)
            return_membrane: Also return membrane trace

        Returns:
            spikes: Spike tensor (batch, seq_len, hidden_dim)
            membrane: (optional) Membrane trace (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Initialize membrane potential
        membrane = torch.zeros(batch_size, self.hidden_dim, device=device)

        all_spikes = []
        all_membrane = [] if return_membrane else None

        # Apply pruning mask if it exists
        if hasattr(self, 'weight_mask') and self.weight_mask is not None:
            with torch.no_grad():
                self.input_proj.weight.mul_(self.weight_mask)

        for t in range(seq_len):
            # Input current from external input
            current = self.input_proj(self.dropout(x[:, t, :]))

            # Add recurrent current from previous spikes
            if self.recurrent and t > 0:
                current = current + self.recurrent_proj(all_spikes[-1])

            # LIF dynamics
            spikes, membrane = self.lif(current, membrane)

            all_spikes.append(spikes)
            if return_membrane:
                all_membrane.append(membrane.clone())

        spikes_out = torch.stack(all_spikes, dim=1)

        if return_membrane:
            return spikes_out, torch.stack(all_membrane, dim=1)
        return spikes_out


class OscillatorySNN(nn.Module):
    """
    Spiking Neural Network with emergent oscillatory dynamics.

    Uses inhibitory interneurons to create natural oscillations,
    providing an alternative to explicit Kuramoto oscillators.
    The network can exhibit gamma, beta, alpha-like rhythms
    depending on connectivity and time constants.
    """

    def __init__(self,
                 input_dim: int,
                 num_excitatory: int = 64,
                 num_inhibitory: int = 16,
                 tau_exc: float = 10.0,
                 tau_inh: float = 5.0,
                 dt: float = 1.0):
        """
        Args:
            input_dim: Input feature dimension
            num_excitatory: Number of excitatory neurons
            num_inhibitory: Number of inhibitory neurons
            tau_exc: Time constant for excitatory neurons
            tau_inh: Time constant for inhibitory neurons (usually faster)
            dt: Simulation timestep
        """
        super().__init__()
        self.num_excitatory = num_excitatory
        self.num_inhibitory = num_inhibitory
        total_neurons = num_excitatory + num_inhibitory

        # Input projection (to excitatory neurons only)
        self.input_proj = nn.Linear(input_dim, num_excitatory, bias=False)

        # Excitatory population
        self.exc_lif = LIFNeuron(tau=tau_exc, threshold=1.0, dt=dt)

        # Inhibitory population (faster dynamics)
        self.inh_lif = LIFNeuron(tau=tau_inh, threshold=0.8, dt=dt)

        # Connectivity matrices
        # E -> E (excitatory to excitatory, sparse)
        self.W_ee = nn.Parameter(torch.randn(num_excitatory, num_excitatory) * 0.1)
        # E -> I (excitatory to inhibitory)
        self.W_ei = nn.Parameter(torch.randn(num_inhibitory, num_excitatory) * 0.2)
        # I -> E (inhibitory to excitatory, negative)
        self.W_ie = nn.Parameter(torch.randn(num_excitatory, num_inhibitory) * 0.3)

        # Output projection
        self.output_proj = nn.Linear(num_excitatory, input_dim)

    def forward(self, x: torch.Tensor, steps_per_input: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process input through oscillatory SNN.

        Args:
            x: Input tensor (batch, seq_len, input_dim)
            steps_per_input: Simulation steps per input timestep

        Returns:
            output: Processed output (batch, seq_len, input_dim)
            firing_rate: Mean firing rate of excitatory population (batch, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Initialize membrane potentials
        v_exc = torch.zeros(batch_size, self.num_excitatory, device=device)
        v_inh = torch.zeros(batch_size, self.num_inhibitory, device=device)

        outputs = []
        firing_rates = []

        for t in range(seq_len):
            # External input current
            ext_current = self.input_proj(x[:, t, :])

            # Run multiple simulation steps per input
            exc_spikes_sum = torch.zeros(batch_size, self.num_excitatory, device=device)
            inh_spikes = torch.zeros(batch_size, self.num_inhibitory, device=device)

            for _ in range(steps_per_input):
                # Recurrent currents
                i_ee = torch.matmul(exc_spikes_sum / steps_per_input, self.W_ee.T)
                i_ie = -torch.abs(torch.matmul(inh_spikes, self.W_ie.T))  # Inhibitory current (negative)

                # Update excitatory neurons
                exc_current = ext_current + i_ee + i_ie
                exc_spikes, v_exc = self.exc_lif(exc_current, v_exc)
                exc_spikes_sum = exc_spikes_sum + exc_spikes

                # Update inhibitory neurons (driven by excitatory)
                i_ei = torch.matmul(exc_spikes, self.W_ei.T)
                inh_spikes, v_inh = self.inh_lif(i_ei, v_inh)

            # Output based on excitatory firing rate
            rate = exc_spikes_sum / steps_per_input
            output = self.output_proj(rate)
            outputs.append(output)
            firing_rates.append(rate.mean(dim=1))

        return torch.stack(outputs, dim=1), torch.stack(firing_rates, dim=1)

    def get_oscillation_frequency(self, firing_rates: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """
        Estimate dominant oscillation frequency from firing rate trace using FFT.

        Args:
            firing_rates: Firing rate trace (batch, seq_len)
            dt: Time step in ms

        Returns:
            Dominant frequency in Hz for each batch element
        """
        # Remove DC component
        rates_centered = firing_rates - firing_rates.mean(dim=1, keepdim=True)

        # FFT
        fft_result = torch.fft.rfft(rates_centered, dim=1)
        power = torch.abs(fft_result) ** 2

        # Find peak frequency (excluding DC)
        power[:, 0] = 0
        peak_idx = torch.argmax(power, dim=1)

        # Convert to frequency
        seq_len = firing_rates.shape[1]
        freqs = torch.fft.rfftfreq(seq_len, d=dt / 1000.0)  # Convert dt from ms to s
        dominant_freq = freqs[peak_idx]

        return dominant_freq


class STDPLayer(nn.Module):
    """
    Spike-Timing Dependent Plasticity (STDP) learning rule.

    Implements online, local Hebbian learning based on precise spike timing.
    Potentiation when pre-spike precedes post-spike, depression otherwise.
    """

    def __init__(self,
                 pre_size: int,
                 post_size: int,
                 tau_plus: float = 20.0,
                 tau_minus: float = 20.0,
                 a_plus: float = 0.01,
                 a_minus: float = 0.01,
                 w_max: float = 1.0,
                 w_min: float = 0.0,
                 target_rate: float = 0.1,
                 homeostatic_strength: float = 0.001,
                 auto_scale: bool = False):
        """
        Args:
            pre_size: Number of presynaptic neurons
            post_size: Number of postsynaptic neurons
            tau_plus: Time constant for LTP trace
            tau_minus: Time constant for LTD trace
            a_plus: LTP amplitude
            a_minus: LTD amplitude
            w_max: Maximum weight
            w_min: Minimum weight
            target_rate: Target post-synaptic firing rate for homeostatic regulation
            homeostatic_strength: Strength of Zenke-style homeostatic correction
            auto_scale: If True, scale a_plus/a_minus inversely by running rate (G-06)
        """
        super().__init__()
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.w_max = w_max
        self.w_min = w_min
        self.target_rate = target_rate
        self.homeostatic_strength = homeostatic_strength
        self.auto_scale = auto_scale

        # Synaptic weights
        self.weights = nn.Parameter(torch.rand(post_size, pre_size) * 0.5)

        # Eligibility traces: per-batch (batch, size) to avoid cross-sample contamination
        self.register_buffer('pre_trace', torch.zeros(1, pre_size))
        self.register_buffer('post_trace', torch.zeros(1, post_size))

        # Homeostatic: running firing rate of post-synaptic neurons (EMA)
        self.register_buffer('running_rate', torch.full((post_size,), target_rate))

    def _ensure_trace_batch(self, batch_size: int, device: torch.device):
        """Expand traces to match batch size if needed."""
        if self.pre_trace.shape[0] != batch_size:
            with torch.no_grad():
                self.pre_trace = torch.zeros(batch_size, self.pre_trace.shape[1], device=device)
                self.post_trace = torch.zeros(batch_size, self.post_trace.shape[1], device=device)

    def forward(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor,
                dt: float = 1.0, learn: bool = True) -> torch.Tensor:
        """
        Apply STDP update and compute postsynaptic current.

        Args:
            pre_spikes: Presynaptic spikes (batch, pre_size)
            post_spikes: Postsynaptic spikes (batch, post_size)
            dt: Timestep
            learn: Whether to update weights

        Returns:
            current: Postsynaptic current (batch, post_size)
        """
        batch_size = pre_spikes.shape[0]
        self._ensure_trace_batch(batch_size, pre_spikes.device)

        # Decay traces
        with torch.no_grad():
            self.pre_trace.mul_(math.exp(-dt / self.tau_plus))
            self.post_trace.mul_(math.exp(-dt / self.tau_minus))

        if learn:
            # G-06: Auto-scale LTP/LTD by inverse running rate
            if self.auto_scale:
                rate_scale = 1.0 / self.running_rate.mean().clamp(min=0.01)
                effective_a_plus = self.a_plus * rate_scale
                effective_a_minus = self.a_minus * rate_scale
            else:
                effective_a_plus = self.a_plus
                effective_a_minus = self.a_minus

            # Compute weight updates BEFORE updating traces (M-03 fix)
            # Per-batch STDP, averaged over batch for shared weight update
            # LTP: post spike after pre spike (use old pre_trace)
            # dw_plus[b] = outer(post_spikes[b], pre_trace[b]), then average
            dw_plus = effective_a_plus * torch.einsum('bp,bq->pq', post_spikes, self.pre_trace) / batch_size

            # LTD: pre spike after post spike (use old post_trace)
            dw_minus = -effective_a_minus * torch.einsum('bp,bq->pq', self.post_trace, pre_spikes) / batch_size

            # Homeostatic correction: push rates toward target
            # dw_homeo = -strength * (rate - target).unsqueeze(1)
            # This penalizes synapses of neurons firing too fast, boosts slow ones
            rate_error = self.running_rate - self.target_rate
            dw_homeo = -self.homeostatic_strength * rate_error.unsqueeze(1)

            # Apply weight updates with bounds (STDP + homeostatic)
            with torch.no_grad():
                self.weights.add_(dw_plus + dw_minus + dw_homeo).clamp_(
                    self.w_min, self.w_max
                )

                # Update running firing rate (EMA with tau ~100 steps)
                rate_alpha = 0.01
                batch_rate = post_spikes.mean(dim=0)
                self.running_rate.mul_(1 - rate_alpha).add_(rate_alpha * batch_rate)

        # Update traces with new spikes AFTER weight computation (per-batch)
        with torch.no_grad():
            self.pre_trace.add_(pre_spikes)
            self.post_trace.add_(post_spikes)

        # Compute current
        current = torch.matmul(pre_spikes, self.weights.T)
        return current

    def reset_traces(self):
        """Reset eligibility traces (call at sequence boundaries)."""
        with torch.no_grad():
            self.pre_trace.zero_()
            self.post_trace.zero_()


class AdExNeuron(nn.Module):
    """
    Adaptive Exponential Integrate-and-Fire (AdEx) neuron.

    Extends LIF with:
    1. Exponential spike initiation: voltage accelerates near threshold
    2. Adaptation current w: accumulates with each spike, slows firing

    Dynamics:
        C * dV/dt = -gL*(V - EL) + gL*DT*exp((V - VT)/DT) - w + I
        tau_w * dw/dt = a*(V - EL) - w
        On spike: V → V_reset, w → w + b

    By varying (a, b), produces 8+ firing patterns:
    - (a=0, b=0): Regular spiking
    - (a=0, b>0): Adapting
    - (a>0, b=0): Subthreshold oscillation
    - (a>0, b>0): Adapting + oscillation
    - (a<0, b=0): Fast spiking / delayed
    - etc.

    Args:
        C: Membrane capacitance
        gL: Leak conductance
        EL: Leak reversal potential
        VT: Threshold voltage (for exponential term)
        DT: Sharpness of spike initiation (slope factor)
        V_reset: Reset voltage after spike
        V_cutoff: Voltage cutoff for spike detection
        tau_w: Adaptation time constant
        a: Subthreshold adaptation coupling
        b: Spike-triggered adaptation increment
        dt: Integration timestep
    """

    def __init__(self,
                 C: float = 1.0,
                 gL: float = 0.1,
                 EL: float = -0.7,
                 VT: float = -0.5,
                 DT: float = 0.1,
                 V_reset: float = -0.7,
                 V_cutoff: float = 0.5,
                 tau_w: float = 100.0,
                 a: float = 0.0,
                 b: float = 0.05,
                 dt: float = 1.0):
        super().__init__()
        self.C = C
        self.gL = gL
        self.EL = EL
        self.VT = VT
        self.DT = DT
        self.V_reset = V_reset
        self.V_cutoff = V_cutoff
        self.tau_w = tau_w
        self.a = a
        self.b = b
        self.dt = dt

    def forward(self,
                current: torch.Tensor,
                V: torch.Tensor,
                w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single timestep update.

        Args:
            current: Input current (..., neurons)
            V: Membrane potential (..., neurons)
            w: Adaptation current (..., neurons)

        Returns:
            spikes: Binary spike tensor
            V: Updated membrane potential
            w: Updated adaptation current
        """
        # Exponential term (clamped to prevent overflow)
        exp_term = self.gL * self.DT * torch.exp(
            torch.clamp((V - self.VT) / (self.DT + 1e-8), max=10.0)
        )

        # Voltage dynamics: C * dV/dt = -gL*(V-EL) + gL*DT*exp(...) - w + I
        dV = (-self.gL * (V - self.EL) + exp_term - w + current) / self.C
        V_new = V + dV * self.dt

        # Adaptation dynamics: tau_w * dw/dt = a*(V-EL) - w
        dw = (self.a * (V - self.EL) - w) / self.tau_w
        w_new = w + dw * self.dt

        # Spike detection
        spikes = (V_new >= self.V_cutoff).float()

        # Surrogate gradient for differentiability
        if self.training:
            v_scaled = V_new - self.V_cutoff
            grad = 25.0 / (1 + (25.0 * torch.abs(v_scaled)) ** 2)
            spikes = spikes + (grad - spikes).detach()

        # Reset on spike
        V_new = V_new * (1 - spikes) + self.V_reset * spikes
        w_new = w_new + self.b * spikes  # Spike-triggered adaptation

        return spikes, V_new, w_new

    def init_state(self, shape: Tuple[int, ...],
                   device: torch.device = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize V and w to resting state."""
        V = torch.full(shape, self.EL, device=device)
        w = torch.zeros(shape, device=device)
        return V, w


class SpikeEncoder(nn.Module):
    """Encode continuous values into spike trains (G-04).

    Two encoding modes:
    - 'rate': Poisson rate coding. Each neuron fires with probability proportional
      to the input value. Based on Heeger (2000) Poisson spike generation.
    - 'population': Gaussian population coding. N neurons with tuning curves
      centered at different values fire probabilistically based on proximity to input.

    Args:
        num_neurons: Number of output spiking neurons
        mode: 'rate' or 'population'
        gain: Gain factor for rate coding (scales input to firing probability)
        sigma: Width of Gaussian tuning curves (population mode)
    """

    def __init__(self,
                 num_neurons: int,
                 mode: Literal['rate', 'population'] = 'rate',
                 gain: float = 1.0,
                 sigma: float = 0.3):
        super().__init__()
        self.num_neurons = num_neurons
        self.mode = mode
        self.gain = gain
        self.sigma = sigma

        if mode == 'population':
            # Tuning curve centers uniformly spaced in [0, 1]
            centers = torch.linspace(0, 1, num_neurons)
            self.register_buffer('centers', centers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input values as spike trains.

        Args:
            x: Input tensor (..., features). Values should be roughly in [0, 1] for rate mode.

        Returns:
            Spike tensor (..., num_neurons)
        """
        if self.mode == 'rate':
            # Expand to num_neurons if needed
            if x.shape[-1] != self.num_neurons:
                x = x.unsqueeze(-1).expand(*x.shape, self.num_neurons)
                # Average over original feature dim
                if x.dim() > 2:
                    x = x.mean(dim=-2)
            rates = (self.gain * x).clamp(0, 1)
            spikes = torch.bernoulli(rates)
            return spikes

        else:  # population
            # Gaussian tuning curves: r_i = exp(-(x - mu_i)^2 / (2*sigma^2))
            # x shape: (..., features), centers: (num_neurons,)
            # Average features to scalar for tuning
            x_scalar = x.mean(dim=-1, keepdim=True)  # (..., 1)
            # Broadcasting: (..., 1) - (num_neurons,) -> (..., num_neurons)
            rates = torch.exp(-((x_scalar - self.centers) ** 2) / (2 * self.sigma ** 2))
            spikes = torch.bernoulli(rates)
            return spikes


class SpikeDecoder(nn.Module):
    """Decode spike trains back to continuous values (G-08).

    Three decoding modes:
    - 'rate': Mean firing rate over time dimension
    - 'first_spike': Inverse latency coding (1 / (first_spike_time + 1))
    - 'population_vector': Weighted average by neuron index

    Args:
        num_neurons: Number of input spiking neurons
        mode: Decoding mode
        time_dim: Which dimension is the time/sequence dimension
    """

    def __init__(self,
                 num_neurons: int,
                 mode: Literal['rate', 'first_spike', 'population_vector'] = 'rate',
                 time_dim: int = 1):
        super().__init__()
        self.num_neurons = num_neurons
        self.mode = mode
        self.time_dim = time_dim

        if mode == 'population_vector':
            # Neuron indices as weights for population vector decoding
            weights = torch.linspace(0, 1, num_neurons)
            self.register_buffer('weights', weights)

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """Decode spike tensor to continuous output.

        Args:
            spikes: Spike tensor. For 'rate'/'first_spike': (batch, time, neurons).
                    For 'population_vector': (..., neurons).

        Returns:
            Decoded continuous values
        """
        if self.mode == 'rate':
            return spikes.mean(dim=self.time_dim)

        elif self.mode == 'first_spike':
            # Find first spike time per neuron
            # Shape: (batch, time, neurons)
            has_spike = spikes.abs() > 0.5
            # Argmax returns first True along time dim
            first_time = has_spike.float().argmax(dim=self.time_dim).float()
            # If no spike: set to max time
            no_spike = ~has_spike.any(dim=self.time_dim)
            first_time[no_spike] = spikes.shape[self.time_dim]
            # Inverse latency: early spike = high value
            return 1.0 / (first_time + 1.0)

        elif self.mode == 'population_vector':
            # Weighted average: sum(spike_i * w_i) / sum(spike_i + eps)
            weighted = spikes * self.weights
            return weighted.sum(dim=-1) / (spikes.sum(dim=-1) + 1e-8)

        else:
            return spikes.mean(dim=self.time_dim)


class SynapticDelay(nn.Module):
    """Circular buffer implementing axonal/synaptic transmission delays (G-15).

    Stores a history of spike patterns and delivers delayed versions.
    Delays are per-synapse (not per-neuron pair) for efficiency.

    Args:
        num_synapses: Number of synapses (typically num_neurons)
        max_delay: Maximum delay in timesteps
    """

    def __init__(self, num_synapses: int, max_delay: int = 10):
        super().__init__()
        self.num_synapses = num_synapses
        self.max_delay = max_delay

        # Circular buffer: (max_delay, num_synapses)
        self.register_buffer('buffer', torch.zeros(max_delay, num_synapses))
        self.register_buffer('ptr', torch.tensor(0))

    def forward(self, spikes: torch.Tensor,
                delays: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Store current spikes and retrieve delayed spikes.

        Args:
            spikes: Current spike pattern (num_synapses,) or (batch, num_synapses)
            delays: Per-synapse delay in timesteps (num_synapses,).
                   Integer values in [0, max_delay-1]. Defaults to max_delay-1.

        Returns:
            Delayed spikes, same shape as input
        """
        was_batched = spikes.dim() > 1
        if was_batched:
            # Use mean over batch for buffer storage
            spikes_1d = spikes.mean(dim=0)
        else:
            spikes_1d = spikes

        # Store current spikes
        with torch.no_grad():
            idx = self.ptr % self.max_delay
            self.buffer[idx] = spikes_1d.detach()
            self.ptr.add_(1)

        # Retrieve delayed spikes
        if delays is None:
            delays = torch.full((self.num_synapses,), self.max_delay - 1,
                                device=spikes.device, dtype=torch.long)
        else:
            delays = delays.long().clamp(0, self.max_delay - 1)

        # Index into circular buffer: (ptr - 1 - delay) % max_delay
        retrieve_idx = (self.ptr - 1 - delays) % self.max_delay
        delayed = self.buffer[retrieve_idx, torch.arange(self.num_synapses, device=spikes.device)]

        if was_batched:
            delayed = delayed.unsqueeze(0).expand(spikes.shape[0], -1)

        return delayed

    def reset(self):
        """Clear delay buffer."""
        self.buffer.zero_()
        self.ptr.zero_()


if __name__ == '__main__':
    print("--- Spiking Neural Network Examples ---")

    # Example 1: Basic SpikingLayer
    print("\n1. SpikingLayer")
    layer = SpikingLayer(input_dim=8, hidden_dim=32, tau=10.0, recurrent=True)
    x = torch.randn(4, 50, 8)  # batch=4, seq=50, dim=8
    spikes, membrane = layer(x, return_membrane=True)
    print(f"   Input: {x.shape}")
    print(f"   Spikes: {spikes.shape}, mean rate: {spikes.mean().item():.3f}")
    print(f"   Membrane: {membrane.shape}")

    # Example 2: OscillatorySNN
    print("\n2. OscillatorySNN")
    osc_snn = OscillatorySNN(input_dim=8, num_excitatory=64, num_inhibitory=16)
    output, rates = osc_snn(x, steps_per_input=5)
    print(f"   Output: {output.shape}")
    print(f"   Firing rates: {rates.shape}, mean: {rates.mean().item():.3f}")

    # Estimate oscillation frequency
    freq = osc_snn.get_oscillation_frequency(rates, dt=1.0)
    print(f"   Dominant freq: {freq.mean().item():.1f} Hz")

    # Example 3: Training with surrogate gradients
    print("\n3. Training SpikingLayer")
    layer = SpikingLayer(input_dim=4, hidden_dim=16, tau=10.0)
    readout = nn.Linear(16, 2)
    optimizer = torch.optim.Adam(list(layer.parameters()) + list(readout.parameters()), lr=0.01)

    for epoch in range(10):
        x = torch.randn(8, 20, 4)
        target = torch.randint(0, 2, (8,))

        optimizer.zero_grad()
        spikes = layer(x)
        # Rate-based readout
        rates = spikes.mean(dim=1)  # (batch, hidden)
        logits = readout(rates)
        loss = nn.functional.cross_entropy(logits, target)
        loss.backward()
        optimizer.step()

        if epoch % 3 == 0:
            acc = (logits.argmax(dim=1) == target).float().mean()
            print(f"   Epoch {epoch}: loss={loss.item():.4f}, acc={acc.item():.2f}")

    # Example 4: STDP learning
    print("\n4. STDP Layer")
    stdp = STDPLayer(pre_size=10, post_size=5)
    print(f"   Initial weights mean: {stdp.weights.mean().item():.4f}")

    for t in range(100):
        pre = (torch.rand(1, 10) > 0.8).float()
        post = (torch.rand(1, 5) > 0.8).float()
        current = stdp(pre, post, learn=True)

    print(f"   Final weights mean: {stdp.weights.mean().item():.4f}")
    print(f"   Weight range: [{stdp.weights.min().item():.4f}, {stdp.weights.max().item():.4f}]")
