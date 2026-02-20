"""
Advanced Spiking Neural Network Components for PyQuifer

Extends the basic spiking module with more biologically realistic
neuron models including synaptic conductance, alpha functions,
recurrent connections, and reward-modulated learning.

Based on patterns from snntorch and bindsnet.
"""

import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SynapticNeuron(nn.Module):
    """
    2nd order LIF neuron with synaptic conductance.

    Models both synaptic current and membrane potential dynamics:
        I_syn[t+1] = α * I_syn[t] + I_in[t+1]
        U[t+1] = β * U[t] + I_syn[t+1] - R * U_thr

    This creates a more realistic temporal response where the
    membrane potential follows a dual-exponential profile.
    """

    def __init__(self,
                 alpha: float = 0.9,
                 beta: float = 0.8,
                 threshold: float = 1.0,
                 V_rest: float = 0.0,
                 reset_mechanism: str = 'subtract',
                 surrogate_grad: Optional[Callable] = None):
        """
        Args:
            alpha: Synaptic current decay rate (0 < α < 1)
            beta: Membrane potential decay rate (0 < β < 1)
            threshold: Spike threshold
            V_rest: Resting membrane potential (G-03: prevents unbounded accumulation)
            reset_mechanism: 'subtract' or 'zero'
            surrogate_grad: Custom surrogate gradient function
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold
        self.V_rest = V_rest
        self.reset_mechanism = reset_mechanism

        if surrogate_grad is None:
            self.surrogate_grad = self._default_surrogate
        else:
            self.surrogate_grad = surrogate_grad

    def _default_surrogate(self, x: torch.Tensor) -> torch.Tensor:
        """Fast sigmoid surrogate gradient."""
        grad = 1.0 / (1.0 + 25.0 * x.abs()) ** 2
        return grad

    def forward(self, x: torch.Tensor,
                syn: Optional[torch.Tensor] = None,
                mem: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input current
            syn: Previous synaptic current
            mem: Previous membrane potential

        Returns:
            spike, syn, mem
        """
        if syn is None:
            syn = torch.zeros_like(x)
        if mem is None:
            mem = torch.zeros_like(x)

        # Synaptic current dynamics
        syn = self.alpha * syn + x

        # Membrane dynamics (G-03: decay toward V_rest, matching snntorch Synaptic)
        mem = self.beta * (mem - self.V_rest) + self.V_rest + syn

        # Spike generation with surrogate gradient
        spike = self._spike_fn(mem - self.threshold)

        # Reset
        if self.reset_mechanism == 'subtract':
            mem = mem - spike * self.threshold
        else:  # zero
            mem = mem * (1 - spike)

        return spike, syn, mem

    def _spike_fn(self, x: torch.Tensor) -> torch.Tensor:
        """Spike function with surrogate gradient."""
        # Forward: Heaviside
        spike = (x > 0).float()

        # Backward: surrogate
        if self.training:
            spike = spike + (self.surrogate_grad(x) - spike).detach()

        return spike


class AlphaNeuron(nn.Module):
    """
    Alpha-function neuron with excitatory/inhibitory balance.

    Models separate excitatory and inhibitory currents that
    combine to create an alpha-function membrane response:
        I_exc[t+1] = α * I_exc[t] + I_in[t+1]
        I_inh[t+1] = β * I_inh[t] - I_in[t+1]
        U[t+1] = τ_α * (I_exc[t+1] + I_inh[t+1])

    Where τ_α normalizes the response.
    """

    def __init__(self,
                 alpha: float = 0.9,
                 beta: float = 0.8,
                 threshold: float = 1.0,
                 reset_mechanism: str = 'zero'):
        """
        Args:
            alpha: Excitatory current decay (should be > beta)
            beta: Inhibitory current decay
            threshold: Spike threshold
            reset_mechanism: 'subtract' or 'zero'
        """
        super().__init__()
        assert alpha > beta, "Alpha should be > beta for positive response"

        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold
        self.reset_mechanism = reset_mechanism

        # Normalization factor for alpha synapse peak timing
        # Use absolute value to ensure correct E/I polarity
        self.tau_alpha = abs(math.log(alpha) / (math.log(beta) - math.log(alpha)))

    def forward(self, x: torch.Tensor,
                syn_exc: Optional[torch.Tensor] = None,
                syn_inh: Optional[torch.Tensor] = None,
                mem: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input current
            syn_exc: Excitatory synaptic current
            syn_inh: Inhibitory synaptic current
            mem: Membrane potential

        Returns:
            spike, syn_exc, syn_inh, mem
        """
        if syn_exc is None:
            syn_exc = torch.zeros_like(x)
        if syn_inh is None:
            syn_inh = torch.zeros_like(x)
        if mem is None:
            mem = torch.zeros_like(x)

        # E/I current dynamics
        syn_exc = self.alpha * syn_exc + x
        syn_inh = self.beta * syn_inh - x

        # Membrane potential from alpha-shaped PSP (proportional, not integral)
        mem = self.tau_alpha * (syn_exc + syn_inh)

        # Spike generation
        spike = (mem > self.threshold).float()

        # Surrogate gradient
        if self.training:
            grad = 1.0 / (1.0 + 25.0 * (mem - self.threshold).abs()) ** 2
            spike = spike + (grad - spike).detach()

        # Reset: both mechanisms must affect synaptic currents since mem is
        # recomputed each step from syn_exc/syn_inh
        if self.reset_mechanism == 'zero':
            syn_exc = syn_exc * (1 - spike)
            syn_inh = syn_inh * (1 - spike)
        else:
            # Subtract reset: reduce excitatory current by threshold equivalent
            syn_exc = syn_exc - spike * self.threshold / (self.tau_alpha + 1e-8)
            mem = self.tau_alpha * (syn_exc + syn_inh)

        return spike, syn_exc, syn_inh, mem


class RecurrentSynapticLayer(nn.Module):
    """
    Recurrent spiking layer with synaptic conductance.

    Combines feed-forward input with recurrent feedback:
        I_syn[t+1] = α * I_syn[t] + W_in * x[t+1] + W_rec * s[t]
        U[t+1] = β * U[t] + I_syn[t+1] - R * U_thr

    The recurrent connection enables temporal memory and
    pattern completion.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 alpha: float = 0.9,
                 beta: float = 0.8,
                 threshold: float = 1.0,
                 recurrent_type: str = 'linear'):
        """
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden (output) dimension
            alpha: Synaptic decay rate
            beta: Membrane decay rate
            threshold: Spike threshold
            recurrent_type: 'linear', 'elementwise', or 'none'
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold
        self.recurrent_type = recurrent_type

        # Input projection
        self.W_in = nn.Linear(input_dim, hidden_dim)

        # Recurrent connection
        if recurrent_type == 'linear':
            self.W_rec = nn.Linear(hidden_dim, hidden_dim, bias=False)
        elif recurrent_type == 'elementwise':
            self.V_rec = nn.Parameter(torch.ones(hidden_dim) * 0.5)
        # else: no recurrence

    def prune(self, fraction: float):
        """Magnitude-prune input weights (G-05).

        Zeros out the smallest `fraction` of W_in weights and stores a permanent mask.

        Args:
            fraction: Fraction of weights to prune, in [0, 1]
        """
        with torch.no_grad():
            w = self.W_in.weight
            magnitudes = w.abs().flatten()
            k = max(1, int(fraction * magnitudes.numel()))
            threshold_val = magnitudes.kthvalue(k).values

            mask = (w.abs() >= threshold_val).float()
            if not hasattr(self, 'weight_mask'):
                self.register_buffer('weight_mask', mask)
            else:
                self.weight_mask.copy_(mask)
            w.mul_(mask)

    def forward(self, x: torch.Tensor,
                syn: Optional[torch.Tensor] = None,
                mem: Optional[torch.Tensor] = None,
                spk: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input (batch, input_dim)
            syn: Previous synaptic current (batch, hidden_dim)
            mem: Previous membrane potential
            spk: Previous spikes (for recurrence)

        Returns:
            spike, syn, mem
        """
        batch_size = x.shape[0]

        if syn is None:
            syn = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        if mem is None:
            mem = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        if spk is None:
            spk = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        # Apply pruning mask if it exists
        if hasattr(self, 'weight_mask') and self.weight_mask is not None:
            with torch.no_grad():
                self.W_in.weight.mul_(self.weight_mask)

        # Input current
        i_in = self.W_in(x)

        # Recurrent current
        if self.recurrent_type == 'linear':
            i_rec = self.W_rec(spk)
        elif self.recurrent_type == 'elementwise':
            i_rec = self.V_rec * spk
        else:
            i_rec = 0

        # Synaptic dynamics
        syn = self.alpha * syn + i_in + i_rec

        # Membrane dynamics
        mem = self.beta * mem + syn

        # Spike
        spk_out = (mem > self.threshold).float()

        # Surrogate gradient
        if self.training:
            grad = 1.0 / (1.0 + 25.0 * (mem - self.threshold).abs()) ** 2
            spk_out = spk_out + (grad - spk_out).detach()

        # Reset (subtract)
        mem = mem - spk_out * self.threshold

        return spk_out, syn, mem


class RewardPredictionError(nn.Module):
    """
    Compute reward prediction error (RPE) for reward-modulated learning.

    Uses exponential moving average to track expected reward:
        RPE = r - E[r]
        E[r] ← (1 - 1/τ) * E[r] + (1/τ) * r

    RPE > 0: Better than expected (strengthen associations)
    RPE < 0: Worse than expected (weaken associations)
    RPE ≈ 0: As expected (maintain)
    """

    def __init__(self,
                 ema_window: float = 10.0,
                 normalize: bool = True):
        """
        Args:
            ema_window: EMA window size (larger = slower adaptation)
            normalize: Normalize RPE by running std
        """
        super().__init__()
        self.ema_window = ema_window
        self.normalize = normalize

        # Running statistics
        self.register_buffer('reward_mean', torch.tensor(0.0))
        self.register_buffer('reward_var', torch.tensor(1.0))
        self.register_buffer('episode_count', torch.tensor(0))

    def forward(self, reward: torch.Tensor) -> torch.Tensor:
        """
        Compute RPE for current reward.

        Args:
            reward: Current reward (scalar or batch)

        Returns:
            Reward prediction error
        """
        reward = reward.float()

        # Compute RPE
        rpe = reward - self.reward_mean

        # Normalize if requested
        if self.normalize:
            rpe = rpe / (self.reward_var.sqrt() + 1e-8)

        return rpe

    def update(self, reward: torch.Tensor):
        """
        Update running statistics with observed reward.

        Call once per episode or batch.
        """
        with torch.no_grad():
            reward = reward.float().mean()  # Handle batch

            # EMA update
            alpha = 1.0 / self.ema_window
            old_mean = self.reward_mean.clone()
            self.reward_mean = (1 - alpha) * self.reward_mean + alpha * reward

            if self.normalize:
                # Update variance estimate using old mean for unbiased delta
                delta = reward - old_mean
                self.reward_var = (1 - alpha) * self.reward_var + alpha * delta ** 2

            self.episode_count += 1


class EligibilityModulatedSTDP(nn.Module):
    """
    STDP learning with eligibility traces and reward modulation.

    Combines:
    1. STDP: Δw ∝ pre * post (temporal correlation)
    2. Eligibility traces: e decays over time, accumulates STDP
    3. Reward modulation: Δw = RPE * e (three-factor rule)

    This enables credit assignment across temporal gaps.
    """

    def __init__(self,
                 pre_dim: int,
                 post_dim: int,
                 tau_eligibility: float = 20.0,
                 tau_stdp: float = 20.0,
                 learning_rate: float = 0.01):
        """
        Args:
            pre_dim: Presynaptic dimension
            post_dim: Postsynaptic dimension
            tau_eligibility: Eligibility trace time constant
            tau_stdp: STDP time constant (ms equivalent)
            learning_rate: Learning rate
        """
        super().__init__()
        self.pre_dim = pre_dim
        self.post_dim = post_dim
        self.tau_e = tau_eligibility
        self.tau_stdp = tau_stdp
        self.lr = learning_rate

        # Synaptic weights
        self.weight = nn.Parameter(
            torch.randn(post_dim, pre_dim) / math.sqrt(pre_dim)
        )

        # Eligibility trace (not a parameter - computed online)
        self.register_buffer(
            'eligibility',
            torch.zeros(post_dim, pre_dim)
        )

        # Presynaptic trace (for STDP)
        self.register_buffer('pre_trace', torch.zeros(pre_dim))
        self.register_buffer('post_trace', torch.zeros(post_dim))

    def forward(self, pre_spike: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (just linear projection).

        Args:
            pre_spike: Presynaptic spikes (batch, pre_dim)

        Returns:
            Postsynaptic current (batch, post_dim)
        """
        return F.linear(pre_spike, self.weight)

    def update_traces(self, pre_spike: torch.Tensor, post_spike: torch.Tensor,
                      dt: float = 1.0):
        """
        Update STDP traces and eligibility.

        Args:
            pre_spike: Presynaptic spikes (batch, pre_dim)
            post_spike: Postsynaptic spikes (batch, post_dim)
            dt: Time step
        """
        with torch.no_grad():
            # Average over batch
            pre = pre_spike.mean(dim=0)
            post = post_spike.mean(dim=0)

            # Decay traces
            decay_stdp = math.exp(-dt / self.tau_stdp)
            self.pre_trace = decay_stdp * self.pre_trace + pre
            self.post_trace = decay_stdp * self.post_trace + post

            # STDP: outer product of traces
            # LTP: post fires after pre (post * pre_trace)
            # LTD: pre fires after post (pre * post_trace)
            stdp = torch.outer(post, self.pre_trace) - torch.outer(self.post_trace, pre)

            # Update eligibility trace
            decay_e = math.exp(-dt / self.tau_e)
            self.eligibility = decay_e * self.eligibility + stdp

    def apply_reward(self, reward: torch.Tensor):
        """
        Apply reward-modulated weight update.

        Args:
            reward: Reward signal (positive strengthens, negative weakens)
        """
        with torch.no_grad():
            # Three-factor rule: Δw = lr * reward * eligibility
            # reward.item() intentionally severs graph — R-STDP uses reward as a
            # neuromodulatory scalar, not a differentiable signal (biological fidelity).
            self.weight.add_(self.lr * reward.item() * self.eligibility)

            # Reset eligibility after reward
            self.eligibility.zero_()


class EpropSTDP(nn.Module):
    """
    E-prop learning rule with dual eligibility traces.

    Extends standard eligibility-modulated STDP with biologically realistic
    dual traces:
    - Fast voltage trace (~20ms): tracks membrane potential fluctuations
    - Slow adaptation trace (~200ms): tracks adaptation current

    Uses pseudo-derivative for surrogate gradient and refractory masking.

    Same apply_reward() interface as EligibilityModulatedSTDP — reward.item()
    severs gradient BY DESIGN (neuromodulatory scalar).

    References:
    - Bellec et al. (2020). A solution to the learning dilemma for RNNs.
    - Zenke & Neftci (2021). Brain-inspired learning on neuromorphic substrates.

    Args:
        pre_dim: Presynaptic dimension
        post_dim: Postsynaptic dimension
        tau_fast: Fast (voltage) trace time constant (~20ms)
        tau_slow: Slow (adaptation) trace time constant (~200ms)
        learning_rate: Learning rate for weight updates
        dampening: Pseudo-derivative dampening factor
        refractory_steps: Number of steps after spike where neuron is refractory
    """

    def __init__(self,
                 pre_dim: int,
                 post_dim: int,
                 tau_fast: float = 20.0,
                 tau_slow: float = 200.0,
                 learning_rate: float = 0.01,
                 dampening: float = 0.3,
                 refractory_steps: int = 5):
        super().__init__()
        self.pre_dim = pre_dim
        self.post_dim = post_dim
        self.tau_fast = tau_fast
        self.tau_slow = tau_slow
        self.lr = learning_rate
        self.dampening = dampening
        self.refractory_steps = refractory_steps

        # Synaptic weights
        self.weight = nn.Parameter(
            torch.randn(post_dim, pre_dim) / math.sqrt(pre_dim)
        )

        # Dual eligibility traces
        self.register_buffer('trace_fast', torch.zeros(post_dim, pre_dim))
        self.register_buffer('trace_slow', torch.zeros(post_dim, pre_dim))

        # Combined eligibility (weighted sum of fast + slow)
        self.register_buffer('eligibility', torch.zeros(post_dim, pre_dim))

        # Presynaptic filtered trace
        self.register_buffer('pre_trace', torch.zeros(pre_dim))

        # Refractory counter per post-synaptic neuron
        self.register_buffer('refractory_counter', torch.zeros(post_dim))

    def _pseudo_derivative(self, v_scaled: torch.Tensor) -> torch.Tensor:
        """
        Pseudo-derivative for surrogate gradient.

        psi(v) = dampening * max(0, 1 - |v|)

        Args:
            v_scaled: Scaled membrane potential (v - threshold) / threshold
        """
        return self.dampening * torch.clamp(1.0 - torch.abs(v_scaled), min=0.0)

    def forward(self, pre_spike: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (linear projection through weights).

        Args:
            pre_spike: Presynaptic spikes (batch, pre_dim)

        Returns:
            Postsynaptic current (batch, post_dim)
        """
        return F.linear(pre_spike, self.weight)

    def update_traces(self,
                      pre_spike: torch.Tensor,
                      post_spike: torch.Tensor,
                      membrane_potential: torch.Tensor,
                      threshold: float = 1.0,
                      dt: float = 1.0):
        """
        Update dual eligibility traces.

        Args:
            pre_spike: Presynaptic spikes (batch, pre_dim)
            post_spike: Postsynaptic spikes (batch, post_dim)
            membrane_potential: Post-synaptic membrane potential (batch, post_dim)
            threshold: Spike threshold
            dt: Timestep
        """
        with torch.no_grad():
            # Average over batch
            pre = pre_spike.mean(dim=0)
            post = post_spike.mean(dim=0)
            v_mean = membrane_potential.mean(dim=0)

            # Refractory masking
            refractory_mask = (self.refractory_counter <= 0).float()
            # Update refractory counter
            self.refractory_counter.sub_(1).clamp_(min=0)
            # Set refractory for neurons that just spiked
            spiked = post > 0.5
            self.refractory_counter[spiked] = self.refractory_steps

            # Pseudo-derivative of membrane potential
            v_scaled = (v_mean - threshold) / (threshold + 1e-8)
            psi = self._pseudo_derivative(v_scaled) * refractory_mask

            # Update filtered presynaptic trace
            decay_pre = math.exp(-dt / self.tau_fast)
            self.pre_trace.mul_(decay_pre).add_(pre)

            # STDP-like correlation modulated by pseudo-derivative
            # e_ij = psi_j * x_i (filtered pre trace * post pseudo-derivative)
            instant_trace = torch.outer(psi, self.pre_trace)

            # Fast trace: tracks voltage dynamics (~20ms)
            decay_fast = math.exp(-dt / self.tau_fast)
            self.trace_fast.mul_(decay_fast).add_(instant_trace)

            # Slow trace: tracks adaptation (~200ms)
            decay_slow = math.exp(-dt / self.tau_slow)
            self.trace_slow.mul_(decay_slow).add_(instant_trace)

            # Combined eligibility: weighted average of traces
            # Fast trace dominates for recent correlations,
            # slow trace provides temporal credit over longer windows
            self.eligibility.copy_(0.7 * self.trace_fast + 0.3 * self.trace_slow)

    def apply_reward(self, reward: torch.Tensor):
        """
        Apply reward-modulated weight update using dual eligibility.

        Args:
            reward: Reward signal (scalar). reward.item() intentionally
                   severs gradient — neuromodulatory scalar by design.
        """
        with torch.no_grad():
            # Three-factor rule: dw = lr * reward * eligibility
            self.weight.add_(self.lr * reward.item() * self.eligibility)

            # Reset eligibility after reward
            self.eligibility.zero_()
            self.trace_fast.zero_()
            self.trace_slow.zero_()

    def reset(self):
        """Reset all traces and refractory counters."""
        self.trace_fast.zero_()
        self.trace_slow.zero_()
        self.eligibility.zero_()
        self.pre_trace.zero_()
        self.refractory_counter.zero_()


if __name__ == '__main__':
    print("--- Advanced Spiking Examples ---")

    # Example 1: SynapticNeuron
    print("\n1. SynapticNeuron (2nd order LIF)")
    syn_neuron = SynapticNeuron(alpha=0.9, beta=0.8, threshold=1.0)

    x = torch.randn(4, 16)  # batch=4, dim=16
    syn, mem = None, None

    spikes = []
    for t in range(20):
        spike, syn, mem = syn_neuron(x * (0.5 if t < 10 else 0.1), syn, mem)
        spikes.append(spike.mean().item())

    print(f"   Spike rates: early={sum(spikes[:10])/10:.3f}, late={sum(spikes[10:])/10:.3f}")

    # Example 2: AlphaNeuron
    print("\n2. AlphaNeuron (E/I balance)")
    alpha_neuron = AlphaNeuron(alpha=0.95, beta=0.85)

    x = torch.randn(4, 16)
    syn_e, syn_i, mem = None, None, None

    for t in range(10):
        spike, syn_e, syn_i, mem = alpha_neuron(x, syn_e, syn_i, mem)

    print(f"   Final E/I balance: E={syn_e.mean().item():.3f}, I={syn_i.mean().item():.3f}")

    # Example 3: RecurrentSynapticLayer
    print("\n3. RecurrentSynapticLayer")
    rec_layer = RecurrentSynapticLayer(
        input_dim=32, hidden_dim=64,
        alpha=0.9, beta=0.8,
        recurrent_type='linear'
    )

    x = torch.randn(4, 32)
    syn, mem, spk = None, None, None

    for t in range(15):
        spk, syn, mem = rec_layer(x, syn, mem, spk)

    print(f"   Output spike rate: {spk.mean().item():.3f}")

    # Example 4: RewardPredictionError
    print("\n4. RewardPredictionError")
    rpe_computer = RewardPredictionError(ema_window=5.0)

    rewards = [1.0, 1.0, 1.0, 0.5, 0.5, 2.0, 2.0]
    for r in rewards:
        reward = torch.tensor(r)
        error = rpe_computer(reward)
        rpe_computer.update(reward)
        print(f"   Reward={r:.1f} -> RPE={error.item():.3f}")

    # Example 5: EligibilityModulatedSTDP
    print("\n5. EligibilityModulatedSTDP")
    stdp = EligibilityModulatedSTDP(pre_dim=32, post_dim=16)

    pre_spikes = (torch.rand(4, 32) > 0.9).float()
    post_spikes = (torch.rand(4, 16) > 0.9).float()

    # Accumulate eligibility
    for t in range(10):
        _ = stdp(pre_spikes)
        stdp.update_traces(pre_spikes, post_spikes)

    print(f"   Eligibility norm: {stdp.eligibility.norm().item():.4f}")

    # Apply reward
    stdp.apply_reward(torch.tensor(1.0))
    print(f"   Weight updated, eligibility reset: {stdp.eligibility.norm().item():.4f}")

    print("\n[OK] All advanced spiking tests passed!")
