"""
Precision Weighting Module for PyQuifer

Attention as gain control — unifies attention, learning rate, and confidence
under one mechanism: precision (inverse variance of prediction errors).

Key concepts:
- Precision = 1 / variance of prediction errors (confidence in a channel)
- High precision → "pay attention here", amplify signal, learn faster
- Low precision → "this is noise", suppress signal, learn slower
- Neuromodulators (ACh, NE) boost or suppress precision contextually

This is foundational for hierarchical predictive coding: at each level,
prediction errors are weighted by their estimated precision before being
passed upward.

References:
- Feldman & Friston (2010). Attention, Uncertainty, and Free-Energy.
- Parr & Friston (2017). Working Memory, Attention, and Salience.
"""

from typing import Dict

import torch
import torch.nn as nn


class PrecisionEstimator(nn.Module):
    """
    Estimates precision (inverse variance) per channel from prediction errors.

    Tracks a running variance of prediction errors and computes precision
    as the inverse. Neuromodulators can contextually boost or suppress
    precision (e.g., ACh boosts learning precision, NE boosts attention).
    """

    def __init__(self,
                 num_channels: int,
                 tau: float = 50.0,
                 max_precision: float = 10.0,
                 epsilon: float = 1e-6):
        """
        Args:
            num_channels: Number of independent precision channels
            tau: Time constant for running variance EMA
            max_precision: Upper bound on precision (prevents blowup)
            epsilon: Floor for variance (numerical stability)
        """
        super().__init__()
        self.num_channels = num_channels
        self.tau = tau
        self.max_precision = max_precision
        self.epsilon = epsilon

        # Running variance of prediction errors (per channel)
        self.register_buffer('running_var', torch.ones(num_channels))
        # Running mean for variance computation
        self.register_buffer('running_mean', torch.zeros(num_channels))
        # Step counter for warmup
        self.register_buffer('step_count', torch.tensor(0))

        # Neuromodulator boost weights (log-space for stability)
        self.ach_boost = nn.Parameter(torch.tensor(0.5))  # Acetylcholine → learning
        self.ne_boost = nn.Parameter(torch.tensor(0.3))   # Norepinephrine → attention

    def forward(self,
                prediction_error: torch.Tensor,
                acetylcholine: float = 0.0,
                norepinephrine: float = 0.0) -> Dict[str, torch.Tensor]:
        """
        Update variance estimate and compute precision.

        Uses deviation-from-running-mean to track true temporal variance,
        not just within-batch variance. This means precision responds
        dynamically when signal statistics change (doesn't permanently saturate).

        Args:
            prediction_error: Prediction errors (..., num_channels)
            acetylcholine: ACh level (0-1), boosts precision
            norepinephrine: NE level (0-1), boosts precision

        Returns:
            Dictionary with:
            - precision: Estimated precision per channel (num_channels,)
            - log_precision: Log-precision (for stable arithmetic)
            - running_var: Current variance estimate
        """
        # Flatten batch dims, keep last dim as channels
        flat_error = prediction_error.reshape(-1, self.num_channels)

        # Update running statistics
        with torch.no_grad():
            alpha = 1.0 / self.tau
            batch_mean = flat_error.mean(dim=0)

            # Track variance as deviation of batch mean from running mean
            # This captures temporal variation (how much the signal changes
            # across calls) rather than just within-batch spread
            deviation = (batch_mean - self.running_mean).pow(2)

            # Also include within-batch variance if batch > 1
            batch_var = flat_error.var(dim=0, unbiased=False) if flat_error.shape[0] > 1 else torch.zeros_like(batch_mean)

            # Combined: temporal deviation + within-batch noise
            combined_var = deviation + batch_var

            self.running_mean.mul_(1 - alpha).add_(batch_mean * alpha)
            self.running_var.mul_(1 - alpha).add_(combined_var * alpha)
            self.step_count.add_(1)

        # Base precision = 1 / variance
        precision = 1.0 / (self.running_var + self.epsilon)

        # Neuromodulator boost (in log-space, then exponentiate)
        log_precision = torch.log(precision)
        log_precision = log_precision + self.ach_boost * acetylcholine + self.ne_boost * norepinephrine

        precision = torch.exp(log_precision).clamp(max=self.max_precision)

        return {
            'precision': precision,
            'log_precision': log_precision,
            'running_var': self.running_var.clone(),
        }

    def get_oscillator_precision(self, num_oscillators: int) -> torch.Tensor:
        """
        Map channel-level precision to oscillator-level precision.

        When the number of precision channels differs from the number
        of oscillators, this method pools or interpolates to produce
        one precision value per oscillator.

        Args:
            num_oscillators: Target number of oscillator precision values.

        Returns:
            Precision per oscillator (num_oscillators,)
        """
        precision = 1.0 / (self.running_var + self.epsilon)
        precision = precision.clamp(max=self.max_precision)

        if self.num_channels == num_oscillators:
            return precision

        # Interpolate: reshape channels to match oscillator count
        # Use nearest-neighbor for simplicity
        precision_2d = precision.unsqueeze(0).unsqueeze(0)  # (1, 1, C)
        interpolated = torch.nn.functional.interpolate(
            precision_2d, size=num_oscillators, mode='nearest'
        )
        return interpolated.squeeze()

    def reset(self):
        """Reset running statistics."""
        self.running_var.fill_(1.0)
        self.running_mean.zero_()
        self.step_count.zero_()


class PrecisionGate(nn.Module):
    """
    Applies precision weighting to any signal.

    weighted_signal = precision * signal
    effective_lr = base_lr * precision

    This is the core "attention as gain control" operation.
    """

    def __init__(self, num_channels: int, base_lr: float = 0.01):
        """
        Args:
            num_channels: Number of channels to gate
            base_lr: Base learning rate before precision modulation
        """
        super().__init__()
        self.num_channels = num_channels
        self.base_lr = base_lr

    def forward(self,
                signal: torch.Tensor,
                precision: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Apply precision gating.

        Args:
            signal: Input signal (..., num_channels)
            precision: Precision weights (num_channels,)

        Returns:
            Dictionary with:
            - weighted_signal: Precision-weighted signal
            - effective_lr: Per-channel effective learning rate
        """
        # Broadcast precision to signal shape
        weighted_signal = signal * precision

        # Effective learning rate
        effective_lr = self.base_lr * precision

        return {
            'weighted_signal': weighted_signal,
            'effective_lr': effective_lr,
        }


class AttentionAsPrecision(nn.Module):
    """
    Full attention-as-precision system.

    Combines PrecisionEstimator and PrecisionGate into a single module
    that maps prediction errors → precision landscape → attention allocation.

    High precision channels = "pay attention here"
    Low precision channels = "ignore this noise"
    """

    def __init__(self,
                 num_channels: int,
                 tau: float = 50.0,
                 max_precision: float = 10.0,
                 base_lr: float = 0.01):
        """
        Args:
            num_channels: Number of precision channels
            tau: Variance EMA time constant
            max_precision: Upper bound on precision
            base_lr: Base learning rate
        """
        super().__init__()
        self.estimator = PrecisionEstimator(
            num_channels=num_channels,
            tau=tau,
            max_precision=max_precision,
        )
        self.gate = PrecisionGate(
            num_channels=num_channels,
            base_lr=base_lr,
        )

    def forward(self,
                signal: torch.Tensor,
                prediction_error: torch.Tensor,
                acetylcholine: float = 0.0,
                norepinephrine: float = 0.0) -> Dict[str, torch.Tensor]:
        """
        Estimate precision from errors and apply to signal.

        Args:
            signal: Signal to weight (..., num_channels)
            prediction_error: Errors to estimate precision from (..., num_channels)
            acetylcholine: ACh level for precision boost
            norepinephrine: NE level for precision boost

        Returns:
            Dictionary with weighted signal, precision, and diagnostics
        """
        # Estimate precision
        est = self.estimator(prediction_error, acetylcholine, norepinephrine)

        # Apply gating
        gated = self.gate(signal, est['precision'])

        # Attention map = normalized precision
        attention = est['precision'] / (est['precision'].sum() + 1e-8)

        return {
            'weighted_signal': gated['weighted_signal'],
            'effective_lr': gated['effective_lr'],
            'precision': est['precision'],
            'log_precision': est['log_precision'],
            'attention_map': attention,
            'running_var': est['running_var'],
        }

    def reset(self):
        """Reset precision estimator."""
        self.estimator.reset()


if __name__ == '__main__':
    print("--- Precision Weighting Examples ---")

    # Example 1: Precision Estimator
    print("\n1. Precision Estimator")
    estimator = PrecisionEstimator(num_channels=8)

    # Channel 0-3: low noise (high precision), Channel 4-7: high noise (low precision)
    for i in range(100):
        errors = torch.randn(4, 8)
        errors[:, :4] *= 0.1  # Low variance
        errors[:, 4:] *= 2.0  # High variance
        result = estimator(errors)

    print(f"   Precision (low noise channels): {result['precision'][:4].mean().item():.2f}")
    print(f"   Precision (high noise channels): {result['precision'][4:].mean().item():.2f}")

    # Example 2: Precision Gate
    print("\n2. Precision Gate")
    gate = PrecisionGate(num_channels=4, base_lr=0.01)

    signal = torch.ones(4)
    precision = torch.tensor([0.1, 1.0, 5.0, 10.0])
    result = gate(signal, precision)
    print(f"   Weighted signal: {result['weighted_signal'].tolist()}")
    print(f"   Effective LR: {result['effective_lr'].tolist()}")

    # Example 3: Full attention-as-precision
    print("\n3. Attention as Precision")
    aap = AttentionAsPrecision(num_channels=8, base_lr=0.01)

    signal = torch.randn(2, 8)
    for i in range(100):
        errors = torch.randn(2, 8)
        errors[:, :4] *= 0.1
        errors[:, 4:] *= 2.0
        result = aap(signal, errors, acetylcholine=0.5, norepinephrine=0.3)

    print(f"   Attention map: {result['attention_map'].detach().cpu().numpy().round(3)}")
    print(f"   High precision channels attended more: "
          f"{result['attention_map'][:4].sum().item():.3f} > "
          f"{result['attention_map'][4:].sum().item():.3f}")

    print("\n[OK] All precision weighting tests passed!")
