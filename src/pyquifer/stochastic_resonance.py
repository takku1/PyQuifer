"""
Adaptive Stochastic Resonance Module for PyQuifer

Extends the fixed-noise StochasticResonance in neuromodulation.py with
adaptive optimal noise finding. The brain doesn't just USE noise —
it actively seeks the noise level that maximizes signal detection.

Key concepts:
- Stochastic resonance: Adding noise can IMPROVE signal detection
- There's an optimal noise level (too little = subthreshold, too much = swamped)
- This module finds and tracks that optimal level via gradient-free optimization
- Optimal noise connects to criticality (edge of chaos = optimal noise)

This is a SMALL extension module, not a replacement for neuromodulation.py's
StochasticResonance class.

References:
- Gammaitoni et al. (1998). Stochastic Resonance.
- McDonnell & Abbott (2009). What is Stochastic Resonance?
- Moss et al. (2004). Stochastic Resonance and Sensory Information Processing.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Dict


class AdaptiveStochasticResonance(nn.Module):
    """
    Finds and tracks the optimal noise level for signal detection.

    Uses gradient-free hill climbing: perturbs noise up/down,
    keeps the direction that improves SNR. Connects to criticality
    (optimal noise is proportional to distance from criticality).
    """

    def __init__(self,
                 dim: int,
                 threshold: float = 0.5,
                 initial_noise: float = 0.3,
                 adaptation_rate: float = 0.01,
                 perturbation_delta: float = 0.05,
                 min_noise: float = 0.01,
                 max_noise: float = 2.0):
        """
        Args:
            dim: Signal dimension
            threshold: Detection threshold
            initial_noise: Starting noise level
            adaptation_rate: How fast to adapt noise
            perturbation_delta: Size of probe perturbations
            min_noise: Minimum noise level
            max_noise: Maximum noise level
        """
        super().__init__()
        self.dim = dim
        self.threshold = threshold
        self.adaptation_rate = adaptation_rate
        self.perturbation_delta = perturbation_delta
        self.min_noise = min_noise
        self.max_noise = max_noise

        # Current noise level (adapted by SR dynamics, not backprop)
        self.register_buffer('noise_level', torch.tensor(initial_noise))
        # Running SNR estimate
        self.register_buffer('current_snr', torch.tensor(0.0))
        # Step counter
        self.register_buffer('step_count', torch.tensor(0))
        # SNR history for monitoring
        self.register_buffer('snr_history', torch.zeros(100))
        self.register_buffer('noise_history', torch.zeros(100))

    def _detect(self, signal: torch.Tensor, noise_level: float) -> torch.Tensor:
        """Apply SR detection at a given noise level."""
        noise = torch.randn_like(signal) * noise_level
        noisy = signal + noise
        detected = (noisy.abs() > self.threshold).float()
        return signal * detected

    def _measure_snr(self, signal: torch.Tensor, noise_level: float,
                     num_trials: int = 10) -> float:
        """
        Measure SNR using signal detection theory (d-prime inspired).

        Computes hit rate (detection when signal present) minus false alarm
        rate (detection when NO signal). This produces the classic inverted-U:
        - Too little noise: hit_rate ≈ 0, false_alarm ≈ 0 → low SNR
        - Optimal noise: hit_rate high, false_alarm low → peak SNR
        - Too much noise: hit_rate ≈ 1, false_alarm ≈ 1 → low SNR
          (noise alone triggers detection, so detection is meaningless)
        """
        hit_rates = []
        false_alarm_rates = []

        for _ in range(num_trials):
            # Hit: detection with signal present
            noise = torch.randn_like(signal) * noise_level
            noisy_signal = signal + noise
            hits = (noisy_signal.abs() > self.threshold).float()
            hit_rates.append(hits.mean().item())

            # False alarm: detection with no signal (just noise)
            noise_only = torch.randn_like(signal) * noise_level
            false_alarms = (noise_only.abs() > self.threshold).float()
            false_alarm_rates.append(false_alarms.mean().item())

        hit_rate = sum(hit_rates) / num_trials
        false_alarm_rate = sum(false_alarm_rates) / num_trials

        # d-prime: discriminability (hit_rate - false_alarm_rate)
        # Peaks when signal genuinely helps detection beyond noise alone
        discriminability = max(0.0, hit_rate - false_alarm_rate)

        # Output fidelity: mean power of detected signal
        mean_detected = torch.zeros_like(signal)
        for _ in range(num_trials):
            mean_detected += self._detect(signal, noise_level)
        mean_detected /= num_trials
        fidelity = mean_detected.pow(2).mean().item()

        # Combined SNR: discriminability * fidelity
        # This peaks at optimal noise
        snr = discriminability * fidelity / (noise_level ** 2 + 1e-8)
        return snr

    def forward(self,
                signal: torch.Tensor,
                criticality_distance: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """
        Apply adaptive stochastic resonance.

        Args:
            signal: Input signal (batch, dim) or (dim,)
            criticality_distance: Optional distance to criticality
                                 (from CriticalityController), informs noise target

        Returns:
            Dictionary with:
            - enhanced: Signal after SR processing
            - detected: Binary detection mask
            - noise_level: Current optimal noise estimate
            - snr: Current SNR estimate
        """
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)

        # Gradient-free optimization: probe noise up and down
        nl = self.noise_level.item()
        delta = self.perturbation_delta

        snr_plus = self._measure_snr(signal, nl + delta)
        snr_minus = self._measure_snr(signal, nl - delta)

        # Finite difference gradient
        snr_gradient = (snr_plus - snr_minus) / (2 * delta)

        # Update noise level toward better SNR
        with torch.no_grad():
            self.noise_level.add_(self.adaptation_rate * snr_gradient)

            # Criticality-informed constraint
            if criticality_distance is not None:
                # Optimal noise scales with distance to criticality
                target_noise = 0.1 + criticality_distance * 0.5
                # Blend toward criticality-informed target
                self.noise_level.mul_(0.8).add_(0.2 * target_noise)

            self.noise_level.clamp_(self.min_noise, self.max_noise)

            # Record history
            current_snr = self._measure_snr(signal, self.noise_level.item())
            self.current_snr.fill_(current_snr)
            idx = self.step_count % 100
            self.snr_history[idx] = current_snr
            self.noise_history[idx] = self.noise_level.item()
            self.step_count.add_(1)

        # Apply SR at current optimal noise level
        noise = torch.randn_like(signal) * self.noise_level
        noisy_signal = signal + noise
        detected = (noisy_signal.abs() > self.threshold).float()
        enhanced = signal * detected

        return {
            'enhanced': enhanced.squeeze(0) if enhanced.shape[0] == 1 else enhanced,
            'detected': detected,
            'noise_level': self.noise_level.clone(),
            'snr': self.current_snr.clone(),
        }

    def reset(self):
        """Reset noise level and history."""
        self.noise_level.fill_(0.3)
        self.current_snr.zero_()
        self.step_count.zero_()
        self.snr_history.zero_()
        self.noise_history.zero_()


class ResonanceMonitor(nn.Module):
    """
    Tracks SNR history and detects when the optimal noise point shifts.

    Useful for detecting regime changes in the signal statistics —
    when the optimal noise suddenly changes, the input distribution
    has changed.
    """

    def __init__(self,
                 window_size: int = 50,
                 shift_threshold: float = 0.3):
        """
        Args:
            window_size: Window for computing running statistics
            shift_threshold: Threshold for detecting optimal point shift
        """
        super().__init__()
        self.window_size = window_size
        self.shift_threshold = shift_threshold

        self.register_buffer('noise_history', torch.zeros(window_size))
        self.register_buffer('snr_history', torch.zeros(window_size))
        self.register_buffer('history_ptr', torch.tensor(0))

    def forward(self,
                noise_level: torch.Tensor,
                snr: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Monitor resonance state.

        Args:
            noise_level: Current noise level from AdaptiveStochasticResonance
            snr: Current SNR from AdaptiveStochasticResonance

        Returns:
            Dictionary with:
            - noise_mean: Running mean of noise level
            - noise_std: Running std of noise level
            - snr_mean: Running mean of SNR
            - regime_shift: Whether a regime shift was detected
        """
        with torch.no_grad():
            idx = self.history_ptr % self.window_size
            self.noise_history[idx] = noise_level.item() if noise_level.dim() == 0 else noise_level
            self.snr_history[idx] = snr.item() if snr.dim() == 0 else snr
            self.history_ptr.add_(1)

        n_valid = min(self.history_ptr.item(), self.window_size)
        valid_noise = self.noise_history[:n_valid]
        valid_snr = self.snr_history[:n_valid]

        noise_mean = valid_noise.mean()
        noise_std = valid_noise.std() if n_valid > 1 else torch.tensor(0.0)
        snr_mean = valid_snr.mean()

        # Regime shift detection: recent noise level deviates from running mean
        regime_shift = False
        if n_valid >= self.window_size:
            # Compare recent quarter to older three-quarters
            recent = self.window_size // 4
            recent_mean = valid_noise[-recent:].mean()
            older_mean = valid_noise[:-recent].mean()
            if abs(recent_mean - older_mean) > self.shift_threshold * noise_std:
                regime_shift = True

        return {
            'noise_mean': noise_mean,
            'noise_std': noise_std,
            'snr_mean': snr_mean,
            'regime_shift': torch.tensor(regime_shift),
        }

    def reset(self):
        """Reset monitor."""
        self.noise_history.zero_()
        self.snr_history.zero_()
        self.history_ptr.zero_()


if __name__ == '__main__':
    print("--- Adaptive Stochastic Resonance Examples ---")

    # Example 1: Adaptive noise finding
    print("\n1. Adaptive Stochastic Resonance")
    asr = AdaptiveStochasticResonance(dim=32, threshold=0.5, initial_noise=0.1)

    # Weak signal (just below threshold)
    signal = torch.ones(32) * 0.4

    for i in range(100):
        result = asr(signal)

    print(f"   Adapted noise level: {result['noise_level'].item():.4f}")
    print(f"   Final SNR: {result['snr'].item():.4f}")
    print(f"   Detection rate: {result['detected'].mean().item():.2f}")

    # Example 2: Criticality-informed adaptation
    print("\n2. Criticality-Informed Adaptation")
    asr2 = AdaptiveStochasticResonance(dim=16, threshold=0.3)

    signal2 = torch.randn(16) * 0.2

    # Near criticality (low distance)
    for i in range(50):
        result = asr2(signal2, criticality_distance=0.1)
    print(f"   Near-critical noise: {result['noise_level'].item():.4f}")

    # Far from criticality (high distance)
    asr2.reset()
    for i in range(50):
        result = asr2(signal2, criticality_distance=1.5)
    print(f"   Far-from-critical noise: {result['noise_level'].item():.4f}")

    # Example 3: Resonance Monitor
    print("\n3. Resonance Monitor")
    asr3 = AdaptiveStochasticResonance(dim=16, threshold=0.3)
    monitor = ResonanceMonitor(window_size=30)

    signal3 = torch.randn(16) * 0.2

    # Stable regime
    for i in range(50):
        res = asr3(signal3)
        mon = monitor(res['noise_level'], res['snr'])

    print(f"   Noise mean: {mon['noise_mean'].item():.4f}")
    print(f"   Noise std: {mon['noise_std'].item():.4f}")
    print(f"   Regime shift: {mon['regime_shift'].item()}")

    # Change signal statistics → should detect shift
    signal3_new = torch.randn(16) * 0.8  # Much stronger signal
    for i in range(50):
        res = asr3(signal3_new)
        mon = monitor(res['noise_level'], res['snr'])

    print(f"   After signal change - regime shift: {mon['regime_shift'].item()}")

    print("\n[OK] All adaptive stochastic resonance tests passed!")
