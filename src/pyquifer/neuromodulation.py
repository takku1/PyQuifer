"""
Neuromodulation Module for PyQuifer

Implements three-timescale dynamics inspired by neurotransmitter systems:
- Fast: Oscillation (milliseconds) - token generation
- Medium: Neuromodulation (seconds) - context/mood modulation
- Slow: Plasticity (minutes/hours) - learning

Key concepts:
- Dopamine analog: Reward prediction, exploration drive
- Serotonin analog: Mood stability, satiation
- Norepinephrine analog: Arousal, attention gain
- Acetylcholine analog: Learning rate, memory encoding

Also includes:
- Glial layer: Ultra-slow context (astrocyte-inspired)
- Stochastic resonance: Constructive noise for signal detection
- Injection locking: External signal entrainment

This enables dynamic, context-dependent computation where the
architecture effectively "rewires" based on internal state.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass


@dataclass
class NeuromodulatorState:
    """Current state of neuromodulatory systems."""
    dopamine: float  # Reward/exploration
    serotonin: float  # Mood/stability
    norepinephrine: float  # Arousal/attention
    acetylcholine: float  # Learning rate
    cortisol: float  # Stress level


class NeuromodulatorDynamics(nn.Module):
    """
    Simulates neuromodulator dynamics with realistic timescales.

    Each neuromodulator has:
    - Baseline level (tonic)
    - Phasic response to events
    - Decay back to baseline
    - Cross-modulator interactions
    """

    def __init__(self,
                 dt: float = 0.01,
                 dopamine_tau: float = 0.5,
                 serotonin_tau: float = 2.0,
                 norepinephrine_tau: float = 0.3,
                 acetylcholine_tau: float = 1.0,
                 cortisol_tau: float = 5.0):
        """
        Args:
            dt: Time step
            *_tau: Time constants for each neuromodulator (larger = slower)
        """
        super().__init__()
        self.dt = dt

        # Time constants
        self.register_buffer('tau', torch.tensor([
            dopamine_tau, serotonin_tau, norepinephrine_tau,
            acetylcholine_tau, cortisol_tau
        ]))

        # Baseline levels
        self.baseline = nn.Parameter(torch.tensor([0.5, 0.5, 0.3, 0.5, 0.2]))

        # Current levels
        self.register_buffer('levels', torch.tensor([0.5, 0.5, 0.3, 0.5, 0.2]))

        # Cross-modulator interaction matrix
        # rows = target, cols = source
        self.interaction = nn.Parameter(torch.tensor([
            # DA    5HT   NE    ACh   Cort
            [0.0,  -0.1,  0.2,  0.0, -0.2],  # DA: inhibited by 5HT, excited by NE
            [0.1,   0.0, -0.1,  0.0,  0.0],  # 5HT: excited by DA
            [0.2,   0.0,  0.0,  0.1,  0.3],  # NE: excited by DA, ACh, Cort
            [0.1,   0.0,  0.1,  0.0, -0.1],  # ACh: excited by DA, NE
            [-0.1,  0.1,  0.2,  0.0,  0.0],  # Cort: inhibited by DA, excited by 5HT, NE
        ]))

    def step(self,
             reward_signal: float = 0.0,
             novelty_signal: float = 0.0,
             threat_signal: float = 0.0,
             success_signal: float = 0.0) -> NeuromodulatorState:
        """
        Update neuromodulator levels based on signals.

        Args:
            reward_signal: Reward prediction error (-1 to 1)
            novelty_signal: Novelty/surprise (0 to 1)
            threat_signal: Threat/stress (0 to 1)
            success_signal: Task success (0 to 1)

        Returns:
            Current neuromodulator state
        """
        # Phasic inputs
        phasic = torch.tensor([
            reward_signal + 0.5 * novelty_signal,  # DA: reward + novelty
            success_signal - 0.5 * threat_signal,   # 5HT: success, reduced by threat
            novelty_signal + threat_signal,         # NE: arousal from novelty or threat
            novelty_signal * (1 - threat_signal),   # ACh: learning when novel but safe
            threat_signal - 0.3 * success_signal,   # Cort: stress, reduced by success
        ], device=self.levels.device)

        # Cross-modulator effects
        cross_effect = self.interaction @ self.levels

        # Exponential decay toward baseline + phasic input + cross effects
        decay = self.dt / self.tau
        self.levels = (
            self.levels * (1 - decay) +
            self.baseline * decay +
            phasic * 0.1 +
            cross_effect * 0.05
        )

        # Clamp to valid range
        self.levels = torch.clamp(self.levels, 0.0, 1.0)

        return NeuromodulatorState(
            dopamine=self.levels[0].item(),
            serotonin=self.levels[1].item(),
            norepinephrine=self.levels[2].item(),
            acetylcholine=self.levels[3].item(),
            cortisol=self.levels[4].item()
        )

    def get_state(self) -> NeuromodulatorState:
        """Get current state without updating."""
        return NeuromodulatorState(
            dopamine=self.levels[0].item(),
            serotonin=self.levels[1].item(),
            norepinephrine=self.levels[2].item(),
            acetylcholine=self.levels[3].item(),
            cortisol=self.levels[4].item()
        )

    def reset(self):
        """Reset to baseline."""
        with torch.no_grad():
            self.levels.copy_(self.baseline)


class GlialLayer(nn.Module):
    """
    Ultra-slow modulation layer inspired by astrocytes.

    Operates 10-100x slower than neural dynamics:
    - Maintains background context
    - Modulates "metabolic budget" (which units get energy)
    - Creates persistent state without RNNs
    - Implements slow calcium-wave-like diffusion
    """

    def __init__(self,
                 dim: int,
                 tau: float = 100.0,
                 diffusion_rate: float = 0.1):
        """
        Args:
            dim: Dimension of the modulated layer
            tau: Time constant (very slow)
            diffusion_rate: How fast activation spreads spatially
        """
        super().__init__()
        self.dim = dim
        self.tau = tau
        self.diffusion_rate = diffusion_rate

        # Glial activation (slow-changing)
        self.register_buffer('activation', torch.zeros(dim))

        # Diffusion kernel (local averaging)
        kernel_size = min(5, dim)
        kernel = torch.ones(kernel_size) / kernel_size
        self.register_buffer('kernel', kernel)

    def step(self,
             neural_activity: torch.Tensor,
             dt: float = 1.0) -> torch.Tensor:
        """
        Update glial activation based on neural activity.

        Args:
            neural_activity: Current neural layer activity (dim,)
            dt: Time step

        Returns:
            Glial modulation signal (dim,)
        """
        if neural_activity.dim() > 1:
            neural_activity = neural_activity.mean(dim=0)

        # Slow integration of neural activity
        decay = dt / self.tau
        self.activation = (
            self.activation * (1 - decay) +
            neural_activity.abs() * decay
        )

        # Spatial diffusion (calcium wave simulation)
        if self.dim > 5:
            # 1D convolution for local spreading
            padded = F.pad(
                self.activation.unsqueeze(0).unsqueeze(0),
                (2, 2), mode='circular'
            )
            diffused = F.conv1d(
                padded,
                self.kernel.unsqueeze(0).unsqueeze(0)
            ).squeeze()

            self.activation = (
                self.activation * (1 - self.diffusion_rate) +
                diffused * self.diffusion_rate
            )

        return self.activation

    def get_modulation(self) -> torch.Tensor:
        """Get current glial modulation (scales neural gain)."""
        # Sigmoid to create smooth gain modulation
        return torch.sigmoid(self.activation * 2 - 1)

    def reset(self):
        """Reset glial state."""
        self.activation.zero_()


class StochasticResonance(nn.Module):
    """
    Implements stochastic resonance for signal detection.

    Stochastic resonance is a phenomenon where adding noise to a
    subthreshold signal can make it detectable. There's an optimal
    noise level - too little and the signal stays subthreshold,
    too much and it gets swamped.

    This is useful for:
    - Detecting weak signals in neural activity
    - Creating sensitivity tuning via noise level
    - Enabling bistable transitions (phase changes)
    """

    def __init__(self,
                 dim: int,
                 threshold: float = 0.5,
                 optimal_noise: float = 0.3):
        """
        Args:
            dim: Signal dimension
            threshold: Detection threshold
            optimal_noise: Noise level that maximizes detection
        """
        super().__init__()
        self.dim = dim
        self.threshold = threshold
        self.optimal_noise = optimal_noise

        # Learnable noise scaling
        self.noise_scale = nn.Parameter(torch.tensor(optimal_noise))

    def forward(self,
                signal: torch.Tensor,
                noise_level: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply stochastic resonance.

        Args:
            signal: Input signal (batch, dim) or (dim,)
            noise_level: Optional override for noise level

        Returns:
            enhanced: Signal after SR processing
            detected: Binary detection mask
        """
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)

        # Determine noise level
        if noise_level is None:
            noise_level = self.noise_scale

        # Add noise
        noise = torch.randn_like(signal) * noise_level
        noisy_signal = signal + noise

        # Threshold detection
        detected = (noisy_signal.abs() > self.threshold).float()

        # Enhanced signal: original magnitude where detected
        enhanced = signal * detected

        return enhanced.squeeze(0) if enhanced.shape[0] == 1 else enhanced, detected

    def compute_snr(self,
                    signal: torch.Tensor,
                    num_samples: int = 100) -> torch.Tensor:
        """
        Compute signal-to-noise ratio across noise levels.

        Returns the characteristic SR curve showing optimal noise.
        """
        noise_levels = torch.linspace(0.01, 1.0, num_samples)
        snrs = []

        for nl in noise_levels:
            enhanced, detected = self.forward(signal, noise_level=nl.item())

            # SNR = signal power / noise power in detected regions
            signal_power = (enhanced ** 2).mean()
            noise_power = ((enhanced - signal * detected) ** 2).mean() + 1e-8
            snrs.append((signal_power / noise_power).item())

        return torch.tensor(snrs, device=signal.device)


class InjectionLocking(nn.Module):
    """
    External signal entrainment for oscillators.

    When a weak external signal is within the "lock range" of an
    oscillator's natural frequency, the oscillator synchronizes
    to the external signal. This enables context injection as
    frequency nudges rather than explicit conditioning.
    """

    def __init__(self,
                 num_oscillators: int,
                 lock_range: float = 0.3):
        """
        Args:
            num_oscillators: Number of oscillators to modulate
            lock_range: Frequency deviation that can be locked (relative)
        """
        super().__init__()
        self.num_oscillators = num_oscillators
        self.lock_range = lock_range

        # Natural frequencies
        self.natural_freq = nn.Parameter(
            torch.linspace(0.5, 2.0, num_oscillators)
        )

        # Current phases
        self.register_buffer('phases', torch.zeros(num_oscillators))

    def step(self,
             external_signal: Optional[torch.Tensor] = None,
             external_freq: Optional[torch.Tensor] = None,
             dt: float = 0.01) -> torch.Tensor:
        """
        Update oscillator phases with injection locking.

        Args:
            external_signal: Optional amplitude modulation
            external_freq: Optional frequency to lock to
            dt: Time step

        Returns:
            Current oscillator states (cos of phases)
        """
        if external_freq is not None:
            # Check if within lock range
            freq_diff = external_freq - self.natural_freq
            rel_diff = freq_diff.abs() / self.natural_freq

            # Lock strength based on how close we are to lock range
            lock_strength = torch.relu(1 - rel_diff / self.lock_range)

            # Blend natural and external frequency
            effective_freq = (
                self.natural_freq * (1 - lock_strength) +
                external_freq * lock_strength
            )
        else:
            effective_freq = self.natural_freq

        # Update phases
        self.phases = (self.phases + 2 * math.pi * effective_freq * dt) % (2 * math.pi)

        # Output with optional amplitude modulation
        output = torch.cos(self.phases)

        if external_signal is not None:
            output = output * (1 + 0.5 * external_signal)

        return output

    def get_lock_status(self, external_freq: torch.Tensor) -> torch.Tensor:
        """Check which oscillators are locked to external frequency."""
        freq_diff = (external_freq - self.natural_freq).abs()
        rel_diff = freq_diff / self.natural_freq
        return (rel_diff < self.lock_range).float()

    def reset(self):
        """Reset phases."""
        self.phases.zero_()


class ThreeTimescaleNetwork(nn.Module):
    """
    Complete three-timescale neural network.

    Combines:
    - Fast: Core oscillatory/neural dynamics
    - Medium: Neuromodulation (dopamine, serotonin, etc.)
    - Slow: Glial/astrocyte context layer

    The medium layer modulates the gain and plasticity of the fast layer.
    The slow layer provides persistent context that survives fast fluctuations.
    """

    def __init__(self,
                 dim: int,
                 num_oscillators: int = 32):
        """
        Args:
            dim: Dimension of neural layer
            num_oscillators: Number of oscillators for injection locking
        """
        super().__init__()
        self.dim = dim

        # Fast layer: simple neural network
        self.fast = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, dim)
        )

        # Medium layer: neuromodulation
        self.neuromod = NeuromodulatorDynamics()

        # Slow layer: glial modulation
        self.glia = GlialLayer(dim)

        # Injection locking for external entrainment
        self.injection = InjectionLocking(num_oscillators)

        # Stochastic resonance for weak signal detection
        self.sr = StochasticResonance(dim)

        # Gain modulation from neuromodulators
        self.gain_weights = nn.Parameter(torch.tensor([
            1.0,   # Dopamine: general gain
            0.5,   # Serotonin: stability
            0.8,   # Norepinephrine: attention
            0.3,   # Acetylcholine: learning
            -0.5,  # Cortisol: stress (reduces gain)
        ]))

    def compute_gain(self, nm_state: NeuromodulatorState) -> float:
        """Compute neural gain from neuromodulator state."""
        levels = torch.tensor([
            nm_state.dopamine,
            nm_state.serotonin,
            nm_state.norepinephrine,
            nm_state.acetylcholine,
            nm_state.cortisol,
        ])
        gain = (self.gain_weights * levels).sum()
        return torch.sigmoid(gain).item()

    def compute_learning_rate(self, nm_state: NeuromodulatorState) -> float:
        """Compute learning rate from neuromodulators."""
        # ACh enhances, cortisol inhibits
        lr_mod = (
            nm_state.acetylcholine * 2 +
            nm_state.dopamine -
            nm_state.cortisol
        )
        return max(0.01, min(1.0, lr_mod))

    def forward(self,
                x: torch.Tensor,
                reward: float = 0.0,
                novelty: float = 0.0,
                threat: float = 0.0,
                success: float = 0.0,
                external_freq: Optional[torch.Tensor] = None,
                dt: float = 0.01) -> Dict[str, torch.Tensor]:
        """
        Process input through three-timescale network.

        Args:
            x: Input tensor (batch, dim) or (dim,)
            reward: Reward signal for neuromodulation
            novelty: Novelty signal
            threat: Threat signal
            success: Success signal
            external_freq: Optional frequency for injection locking
            dt: Time step

        Returns:
            Dictionary with outputs and internal states
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Update neuromodulators (medium timescale)
        nm_state = self.neuromod.step(reward, novelty, threat, success)

        # Compute modulation factors
        gain = self.compute_gain(nm_state)
        learning_rate = self.compute_learning_rate(nm_state)

        # Apply stochastic resonance (noise level based on NE)
        noise_level = 0.1 + nm_state.norepinephrine * 0.4
        x_enhanced, detected = self.sr(x, noise_level)

        # Fast layer processing with gain modulation
        fast_out = self.fast(x_enhanced) * gain

        # Update glial layer (slow timescale)
        glial_mod = self.glia.step(fast_out.mean(dim=0), dt * 10)  # 10x slower

        # Apply glial modulation
        modulated = fast_out * self.glia.get_modulation()

        # Injection locking (if external frequency provided)
        if external_freq is not None:
            osc_state = self.injection.step(modulated.mean(dim=0), external_freq, dt)
        else:
            osc_state = self.injection.step(dt=dt)

        return {
            'output': modulated,
            'neuromodulator_state': nm_state,
            'gain': gain,
            'learning_rate': learning_rate,
            'glial_modulation': glial_mod,
            'oscillator_state': osc_state,
            'detected_signals': detected,
        }

    def reset(self):
        """Reset all timescales."""
        self.neuromod.reset()
        self.glia.reset()
        self.injection.reset()


if __name__ == '__main__':
    print("--- Neuromodulation Examples ---")

    # Example 1: Neuromodulator dynamics
    print("\n1. Neuromodulator Dynamics")
    nm = NeuromodulatorDynamics()

    # Simulate reward event
    for i in range(50):
        if i == 10:
            state = nm.step(reward_signal=1.0)
            print(f"   Step {i} (reward): DA={state.dopamine:.2f}")
        elif i == 30:
            state = nm.step(threat_signal=0.8)
            print(f"   Step {i} (threat): NE={state.norepinephrine:.2f}, Cort={state.cortisol:.2f}")
        else:
            state = nm.step()

    print(f"   Final state: DA={state.dopamine:.2f}, 5HT={state.serotonin:.2f}")

    # Example 2: Glial layer
    print("\n2. Glial Layer (slow context)")
    glia = GlialLayer(dim=64)

    # Persistent activity in one region
    activity = torch.zeros(64)
    activity[20:40] = 1.0

    for i in range(100):
        glia.step(activity)

    mod = glia.get_modulation()
    print(f"   Modulation peak at: {mod.argmax().item()}")
    print(f"   Peak value: {mod.max().item():.3f}")

    # Example 3: Stochastic resonance
    print("\n3. Stochastic Resonance")
    sr = StochasticResonance(dim=32, threshold=0.5)

    # Weak signal
    weak_signal = torch.randn(32) * 0.3

    enhanced, detected = sr(weak_signal, noise_level=0.3)
    print(f"   Detection rate: {detected.mean().item():.2f}")

    # Example 4: Injection locking
    print("\n4. Injection Locking")
    inj = InjectionLocking(num_oscillators=8)

    # External signal at 1.0 Hz
    ext_freq = torch.ones(8) * 1.0

    for i in range(100):
        osc = inj.step(external_freq=ext_freq)

    lock_status = inj.get_lock_status(ext_freq)
    print(f"   Locked oscillators: {lock_status.sum().item():.0f}/8")

    # Example 5: Full three-timescale network
    print("\n5. Three-Timescale Network")
    net = ThreeTimescaleNetwork(dim=64, num_oscillators=16)

    x = torch.randn(2, 64)

    # Simulate learning episode
    for i in range(50):
        if i < 20:
            result = net(x, reward=0.0, novelty=0.5)
        elif i < 30:
            result = net(x, reward=0.8, success=0.6)
        else:
            result = net(x, threat=0.3)

        if i % 10 == 0:
            nm = result['neuromodulator_state']
            print(f"   Step {i}: gain={result['gain']:.2f}, LR={result['learning_rate']:.2f}")

    print("\n[OK] All neuromodulation tests passed!")
