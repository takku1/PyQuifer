"""
Self-Organized Criticality (SOC) Module for PyQuifer

Implements mechanisms to maintain the system at the edge of chaos -
the critical state where information processing is optimal.

Key concepts:
- Avalanche detection: Power-law distributed bursts of activity
- Criticality metrics: Branching ratio, susceptibility, correlation length
- Adaptive control: Automatically tune parameters to maintain criticality
- Homeostatic regulation: Keep activity in healthy dynamic range

Based on work by Per Bak, John Beggs, Dietmar Plenz.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Dict, List
from collections import deque


def phase_activity_to_spikes(phases: torch.Tensor,
                             threshold: float = 1.6) -> torch.Tensor:
    """
    Convert oscillator phases to spike-like activity for criticality measurement.

    Uses y = 1 + sin(θ) with threshold detection, following the
    excitatory/inhibitory Kuramoto model (Ferrara et al. 2024,
    arXiv:2512.17317) where spikes are detected when y > threshold.

    This produces binary avalanche-compatible events rather than
    continuous sinusoidal values, which is necessary for meaningful
    branching ratio and avalanche size measurements.

    Args:
        phases: Oscillator phases (any shape)
        threshold: Spike threshold on 1+sin(θ). Default 1.6 matches
                  the experimental convention (range of y is [0, 2]).

    Returns:
        Spike counts: number of oscillators that fired (scalar tensor)
    """
    y = 1.0 + torch.sin(phases)
    spikes = (y > threshold).float()
    return spikes.sum()


class AvalancheDetector(nn.Module):
    """
    Detects and analyzes avalanches of activity in neural dynamics.

    An avalanche is a cascade of activity triggered by a perturbation.
    In critical systems, avalanche sizes follow a power law distribution.

    Monitors activity and identifies avalanche events for analysis.
    """

    def __init__(self,
                 activity_threshold: float = 0.5,
                 min_avalanche_size: int = 2,
                 max_history: int = 1000):
        """
        Args:
            activity_threshold: Threshold for detecting active units
            min_avalanche_size: Minimum size to count as avalanche
            max_history: Number of avalanche sizes to remember
        """
        super().__init__()
        self.activity_threshold = activity_threshold
        self.min_avalanche_size = min_avalanche_size
        self.max_history = max_history

        # Avalanche tracking
        self.register_buffer('in_avalanche', torch.tensor(False))
        self.register_buffer('current_size', torch.tensor(0))
        self.register_buffer('avalanche_sizes', torch.zeros(max_history))
        self.register_buffer('size_ptr', torch.tensor(0))
        self.register_buffer('num_avalanches', torch.tensor(0))

    def forward(self, activity: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process activity and detect avalanches.

        Args:
            activity: Activity tensor (any shape, will be flattened)

        Returns:
            Dictionary with:
            - num_active: Number of active units
            - in_avalanche: Whether currently in avalanche
            - avalanche_ended: Whether an avalanche just ended
            - last_size: Size of last completed avalanche
        """
        # Count active units
        active = (activity.abs() > self.activity_threshold).sum()

        avalanche_ended = False
        last_size = torch.tensor(0, device=activity.device)

        with torch.no_grad():
            if active > 0:
                # Activity present
                if not self.in_avalanche:
                    # Start new avalanche
                    self.in_avalanche.fill_(True)
                    self.current_size.zero_()

                self.current_size.add_(active)
            else:
                # No activity
                if self.in_avalanche:
                    # Avalanche ended
                    if self.current_size >= self.min_avalanche_size:
                        # Record avalanche
                        self.avalanche_sizes[self.size_ptr % self.max_history] = self.current_size
                        self.size_ptr.add_(1)
                        self.num_avalanches.add_(1)
                        last_size = self.current_size.clone()
                        avalanche_ended = True

                    self.in_avalanche.fill_(False)
                    self.current_size.zero_()

        return {
            'num_active': active,
            'in_avalanche': self.in_avalanche.clone(),
            'avalanche_ended': torch.tensor(avalanche_ended, device=activity.device),
            'last_size': last_size
        }

    def get_size_distribution(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get avalanche size distribution for power-law analysis.

        Returns:
            sizes: Unique avalanche sizes
            counts: Count of each size
        """
        valid = self.avalanche_sizes[:min(self.size_ptr.item(), self.max_history)]
        if len(valid) == 0:
            dev = self.avalanche_sizes.device
            return torch.tensor([], device=dev), torch.tensor([], device=dev)

        sizes = valid.unique(sorted=True)
        # Vectorized: compare all valid entries against all unique sizes at once
        counts = (valid.unsqueeze(-1) == sizes).sum(0)

        return sizes, counts

    def compute_power_law_exponent(self) -> torch.Tensor:
        """
        Estimate power-law exponent from avalanche size distribution.

        For critical systems, this should be approximately -1.5 (tau ~ 1.5).

        Returns:
            Estimated exponent (negative value expected)
        """
        sizes, counts = self.get_size_distribution()
        if len(sizes) < 3:
            return torch.tensor(0.0, device=self.avalanche_sizes.device)

        # Log-log linear regression (no +1 smoothing: sizes/counts are already ≥1)
        log_sizes = torch.log(sizes.float())
        log_counts = torch.log(counts.float())

        # Simple linear regression
        n = len(log_sizes)
        sum_x = log_sizes.sum()
        sum_y = log_counts.sum()
        sum_xy = (log_sizes * log_counts).sum()
        sum_xx = (log_sizes * log_sizes).sum()

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x + 1e-6)

        return slope

    def reset(self):
        """Reset avalanche tracking."""
        self.in_avalanche.fill_(False)
        self.current_size.zero_()
        self.avalanche_sizes.zero_()
        self.size_ptr.zero_()
        self.num_avalanches.zero_()


class BranchingRatio(nn.Module):
    """
    Computes the branching ratio - a key criticality metric.

    Branching ratio sigma = average descendants per ancestor
    - sigma < 1: subcritical (activity dies out)
    - sigma = 1: critical (sustained activity)
    - sigma > 1: supercritical (activity explodes)

    Critical systems have sigma very close to 1.

    Uses the ratio-of-means estimator (Harris 1963; Beggs & Plenz 2003):
    sigma = mean(descendants) / mean(ancestors), which is bounded by
    the data range and robust to near-zero ancestor counts that cause
    the mean-of-ratios estimator to explode.

    Args:
        window_size: Number of time steps to average over
        variance_threshold: If activity variance falls below this, return
                           sigma=1.0 with converged=True (quiescent regime)
        estimator: 'ratio_of_means' (default, robust) or 'mean_of_ratios'
                   (legacy, can blow up on bursty subcritical data)
    """

    def __init__(self, window_size: int = 50, variance_threshold: float = 1e-10,
                 estimator: str = 'ratio_of_means'):
        """
        Args:
            window_size: Number of time steps to average over
            variance_threshold: If activity variance falls below this, return
                               sigma=1.0 with converged=True (truly quiescent).
                               Set very low (1e-10) so that only genuinely
                               zero-variance signals trigger the shortcut.
            estimator: 'ratio_of_means' (default) or 'mean_of_ratios' (legacy)
        """
        super().__init__()
        self.window_size = window_size
        self.variance_threshold = variance_threshold
        self.estimator = estimator

        self.register_buffer('activity_history', torch.zeros(window_size))
        self.register_buffer('history_ptr', torch.tensor(0))

    def forward(self, activity: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Update history and compute branching ratio.

        Args:
            activity: Current activity level (scalar or will be summed)

        Returns:
            Dictionary with:
            - branching_ratio: Current estimate of sigma
            - criticality_distance: How far from critical (|sigma - 1|)
        """
        if activity.numel() > 1:
            activity = activity.sum()

        with torch.no_grad():
            self.activity_history[self.history_ptr % self.window_size] = activity
            self.history_ptr.add_(1)

        # Need at least 2 points
        n_valid = min(self.history_ptr.item(), self.window_size)
        dev = activity.device
        if n_valid < 2:
            return {
                'branching_ratio': torch.tensor(1.0, device=dev),
                'criticality_distance': torch.tensor(0.0, device=dev)
            }

        history = self.activity_history[:n_valid]

        # G-19: Variance check — if activity is near-constant, ratio is unstable
        activity_var = history.var()
        if activity_var.item() < self.variance_threshold:
            return {
                'branching_ratio': torch.tensor(1.0, device=dev),
                'criticality_distance': torch.tensor(0.0, device=dev),
                'converged': True,
            }

        ancestors = history[:-1]
        descendants = history[1:]

        if self.estimator == 'mean_of_ratios':
            # Legacy mean-of-ratios: can blow up on bursty subcritical data
            sigma = (descendants / (ancestors + 1e-6)).mean()
        else:
            # Ratio-of-means (Harris 1963; Beggs & Plenz 2003):
            # bounded by data range, no blowup from near-zero ancestors
            sigma = descendants.mean() / (ancestors.mean() + 1e-6)

        return {
            'branching_ratio': sigma,
            'criticality_distance': torch.abs(sigma - 1.0),
            'converged': False,
        }

    def reset(self):
        """Reset history."""
        self.activity_history.zero_()
        self.history_ptr.zero_()


class NoProgressDetector(nn.Module):
    """
    Detects stagnation: no monotonic improvement over a window.

    Fortress signature: activity exists but goes nowhere.
    sigma ~= 1 (critical) + zero progress = locked position.

    Uses the same circular-buffer pattern as BranchingRatio.

    Args:
        window_size: Number of steps to track
        progress_threshold: Slope below this magnitude counts as stalled
    """

    def __init__(self, window_size: int = 30, progress_threshold: float = 0.01):
        super().__init__()
        self.window_size = window_size
        self.progress_threshold = progress_threshold

        self.register_buffer('history', torch.zeros(window_size))
        self.register_buffer('hist_ptr', torch.tensor(0))
        self.register_buffer('stagnation_count', torch.tensor(0))

    def forward(self, evaluation: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Record an evaluation and detect stagnation.

        Args:
            evaluation: Scalar evaluation (or tensor that will be meaned)

        Returns:
            progress_stalled: bool — no improvement for window_size steps
            stagnation_duration: int — how many steps with no progress
            trend: float — slope of evaluation over window (-/0/+)
            entropy_change: float — change in evaluation entropy
        """
        if evaluation.numel() > 1:
            evaluation = evaluation.mean()

        dev = evaluation.device

        with torch.no_grad():
            self.history[self.hist_ptr % self.window_size] = evaluation.detach()
            self.hist_ptr.add_(1)

        n_valid = min(self.hist_ptr.item(), self.window_size)

        if n_valid < 3:
            return {
                'progress_stalled': torch.tensor(False, device=dev),
                'stagnation_duration': torch.tensor(0, device=dev),
                'trend': torch.tensor(0.0, device=dev),
                'entropy_change': torch.tensor(0.0, device=dev),
            }

        # Get valid history in chronological order
        if self.hist_ptr.item() <= self.window_size:
            valid = self.history[:n_valid]
        else:
            # Circular buffer: reorder chronologically
            ptr = self.hist_ptr.item() % self.window_size
            valid = torch.cat([self.history[ptr:], self.history[:ptr]])[-n_valid:]

        # Linear regression slope over window
        x = torch.arange(n_valid, dtype=torch.float32, device=dev)
        x_mean = x.mean()
        y_mean = valid.mean()
        slope = ((x - x_mean) * (valid - y_mean)).sum() / ((x - x_mean).pow(2).sum() + 1e-8)

        # Entropy change: std of recent half vs older half
        half = n_valid // 2
        if half >= 2:
            older_std = valid[:half].std()
            recent_std = valid[half:].std()
            entropy_change = recent_std - older_std
        else:
            entropy_change = torch.tensor(0.0, device=dev)

        stalled = torch.abs(slope) < self.progress_threshold

        with torch.no_grad():
            if stalled:
                self.stagnation_count.add_(1)
            else:
                self.stagnation_count.zero_()

        return {
            'progress_stalled': stalled,
            'stagnation_duration': self.stagnation_count.clone(),
            'trend': slope,
            'entropy_change': entropy_change,
        }

    def reset(self):
        """Reset history."""
        self.history.zero_()
        self.hist_ptr.zero_()
        self.stagnation_count.zero_()


class KuramotoCriticalityMonitor(nn.Module):
    """
    Criticality monitor designed for Kuramoto oscillator networks.

    Unlike BranchingRatio (designed for discrete spiking avalanches),
    this measures criticality through the lens of synchronization
    phase transitions, which is the correct framework for continuous
    oscillator dynamics.

    Key metrics:
    - **Synchronization susceptibility** χ = N · var(R):
      Peaks at the critical coupling K_c. This is the oscillator-network
      analog of the diverging susceptibility at a phase transition
      (Acebrón et al. 2005 Rev Mod Phys).

    - **Order parameter regime**: R ∈ [0.3, 0.7] with high variance
      indicates the critical band between incoherent (R≈0) and
      fully synchronized (R≈1) regimes.

    - **Criticality sigma**: Normalized to match BranchingRatio interface.
      sigma < 1: subcritical (too incoherent, low R)
      sigma ≈ 1: critical (medium R, high susceptibility)
      sigma > 1: supercritical (too synchronized, R→1)

    References:
    - Acebrón et al. (2005) "The Kuramoto model" Rev Mod Phys
    - Breakspear et al. (2010) "Generative Models of Cortical Oscillations"
    - Ferrara et al. (2024) arXiv:2512.17317 (E/I Kuramoto avalanches)

    Args:
        window_size: Number of R samples to track
        critical_R_low: Lower bound of critical R band (default 0.3)
        critical_R_high: Upper bound of critical R band (default 0.7)
    """

    def __init__(self, window_size: int = 50,
                 critical_R_low: float = 0.3,
                 critical_R_high: float = 0.7):
        super().__init__()
        self.window_size = window_size
        self.critical_R_low = critical_R_low
        self.critical_R_high = critical_R_high

        self.register_buffer('R_history', torch.zeros(window_size))
        self.register_buffer('hist_ptr', torch.tensor(0))

    def forward(self, R: torch.Tensor, num_oscillators: int = 1
                ) -> Dict[str, torch.Tensor]:
        """
        Update with current order parameter R and compute criticality.

        Args:
            R: Current Kuramoto order parameter (scalar tensor)
            num_oscillators: N, for susceptibility normalization

        Returns:
            Dict with:
            - branching_ratio: sigma ∈ (0, 2) where 1.0 = critical
            - criticality_distance: |sigma - 1.0|
            - susceptibility: χ = N · var(R)
            - R_mean: Mean R over window
            - R_var: Variance of R over window
        """
        if R.numel() > 1:
            R = R.mean()

        dev = R.device

        with torch.no_grad():
            self.R_history[self.hist_ptr % self.window_size] = R.detach()
            self.hist_ptr.add_(1)

        n_valid = min(self.hist_ptr.item(), self.window_size)

        if n_valid < 5:
            return {
                'branching_ratio': torch.tensor(1.0, device=dev),
                'criticality_distance': torch.tensor(0.0, device=dev),
                'susceptibility': torch.tensor(0.0, device=dev),
                'R_mean': R.detach(),
                'R_var': torch.tensor(0.0, device=dev),
            }

        history = self.R_history[:n_valid]
        R_mean = history.mean()
        R_var = history.var()
        susceptibility = num_oscillators * R_var

        # Compute sigma: map (R_mean, R_var) to a branching-ratio-like metric
        # In the critical band [0.3, 0.7]: sigma ≈ 1.0
        # Below 0.3 (subcritical/incoherent): sigma < 1.0
        # Above 0.7 (supercritical/synchronized): sigma > 1.0
        R_mid = (self.critical_R_low + self.critical_R_high) / 2.0
        R_range = (self.critical_R_high - self.critical_R_low) / 2.0

        # Deviation from critical band center, normalized
        deviation = (R_mean - R_mid) / R_range  # -1 at low edge, +1 at high edge
        # Map to sigma: center of band → 1.0. The 0.2 coefficient gives
        # sigma ∈ [0.8, 1.2] at extremes, with band edges at ~0.85/1.15.
        # This is deliberately flatter than branching-ratio (which uses 0.5)
        # because the fast inhibitory path handles R overshoots directly —
        # the sigma just needs to guide the slow homeostatic coupling.
        sigma = 1.0 + 0.2 * torch.tanh(deviation)

        # Bonus: high variance (susceptibility) pushes sigma toward 1.0
        # because high var(R) is the hallmark of criticality
        var_bonus = torch.clamp(R_var * 10.0, 0.0, 0.3)
        sigma = sigma + var_bonus * (1.0 - sigma)  # pull toward 1.0

        criticality_distance = torch.abs(sigma - 1.0)

        return {
            'branching_ratio': sigma,
            'criticality_distance': criticality_distance,
            'susceptibility': susceptibility,
            'R_mean': R_mean,
            'R_var': R_var,
        }

    def reset(self):
        self.R_history.zero_()
        self.hist_ptr.zero_()


class CriticalityController(nn.Module):
    """
    Adaptive controller that maintains the system at criticality.

    Monitors criticality metrics and adjusts global parameters
    (coupling strength, noise level, etc.) to keep the system
    at the edge of chaos where information processing is optimal.

    This is a meta-level controller for PyQuifer's dynamics.
    """

    def __init__(self,
                 target_branching_ratio: float = 1.0,
                 adaptation_rate: float = 0.01,
                 min_coupling: float = 0.1,
                 max_coupling: float = 2.0,
                 min_noise: float = 0.01,
                 max_noise: float = 1.0,
                 kp: float = 1.0,
                 ki: float = 0.1,
                 integral_windup_limit: float = 10.0):
        """
        Args:
            target_branching_ratio: Target sigma (1.0 for critical)
            adaptation_rate: How fast to adapt parameters
            min_coupling: Minimum allowed coupling strength
            max_coupling: Maximum allowed coupling strength
            min_noise: Minimum noise level
            max_noise: Maximum noise level
            kp: Proportional gain for PI controller
            ki: Integral gain for PI controller
            integral_windup_limit: Anti-windup clamp for integral term
        """
        super().__init__()
        self.target_sigma = target_branching_ratio
        self.adaptation_rate = adaptation_rate
        self.min_coupling = min_coupling
        self.max_coupling = max_coupling
        self.min_noise = min_noise
        self.max_noise = max_noise
        self.kp = kp
        self.ki = ki
        self.integral_windup_limit = integral_windup_limit

        # Criticality monitors
        self.avalanche_detector = AvalancheDetector()
        self.branching_monitor = BranchingRatio()

        # Controlled state (adapted by controller dynamics, not backprop)
        self.register_buffer('coupling_adjustment', torch.tensor(1.0))
        self.register_buffer('noise_adjustment', torch.tensor(1.0))

        # PI controller integral error accumulator
        self.register_buffer('integral_error', torch.tensor(0.0))

        # History for stability analysis
        self.register_buffer('sigma_history', torch.zeros(100))
        self.register_buffer('sigma_ptr', torch.tensor(0))

    def forward(self, activity: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Monitor activity and compute control signals.

        Args:
            activity: Current system activity

        Returns:
            Dictionary with criticality metrics and control adjustments
        """
        # Update monitors
        avalanche_info = self.avalanche_detector(activity)
        branching_info = self.branching_monitor(activity)

        sigma = branching_info['branching_ratio']

        # Record sigma history
        with torch.no_grad():
            self.sigma_history[self.sigma_ptr % 100] = sigma
            self.sigma_ptr.add_(1)

        # PI controller (Shew et al. 2015 criticality homeostasis)
        sigma_error = sigma - self.target_sigma

        # Accumulate integral error with anti-windup clamp
        with torch.no_grad():
            self.integral_error.add_(sigma_error)
            self.integral_error.clamp_(-self.integral_windup_limit,
                                        self.integral_windup_limit)

        # PI control law: delta = -(Kp * error + Ki * integral) * rate
        pi_signal = self.kp * sigma_error + self.ki * self.integral_error
        coupling_delta = -pi_signal * self.adaptation_rate
        noise_delta = pi_signal * self.adaptation_rate * 0.5  # Noise has smaller effect

        # Apply adjustments (with clamping)
        with torch.no_grad():
            self.coupling_adjustment.copy_(torch.clamp(
                self.coupling_adjustment + coupling_delta,
                self.min_coupling, self.max_coupling
            ))
            self.noise_adjustment.copy_(torch.clamp(
                self.noise_adjustment + noise_delta,
                self.min_noise, self.max_noise
            ))

        # Power-law exponent (for logging/analysis)
        power_law_exp = self.avalanche_detector.compute_power_law_exponent()

        return {
            'branching_ratio': sigma,
            'criticality_distance': branching_info['criticality_distance'],
            'power_law_exponent': power_law_exp,
            'coupling_adjustment': self.coupling_adjustment.clone(),
            'noise_adjustment': self.noise_adjustment.clone(),
            'num_avalanches': self.avalanche_detector.num_avalanches.clone(),
            **avalanche_info
        }

    def get_adjusted_coupling(self, base_coupling: torch.Tensor) -> torch.Tensor:
        """Apply coupling adjustment to a base coupling value."""
        return base_coupling * self.coupling_adjustment

    def get_adjusted_noise(self, base_noise: torch.Tensor) -> torch.Tensor:
        """Apply noise adjustment to a base noise level."""
        return base_noise * self.noise_adjustment

    def is_critical(self, tolerance: float = 0.1) -> bool:
        """Check if system is currently in critical regime."""
        if self.sigma_ptr < 10:
            return False
        # Handle circular buffer: gather last 10 entries by index
        ptr = self.sigma_ptr.item()
        indices = [(ptr - 1 - i) % 100 for i in range(10)]
        recent_sigma = self.sigma_history[indices].mean()
        return abs(recent_sigma - self.target_sigma) < tolerance

    def reset(self):
        """Reset all monitors and adjustments."""
        self.avalanche_detector.reset()
        self.branching_monitor.reset()
        with torch.no_grad():
            self.coupling_adjustment.fill_(1.0)
            self.noise_adjustment.fill_(1.0)
        self.integral_error.zero_()
        self.sigma_history.zero_()
        self.sigma_ptr.zero_()


class HomeostaticRegulator(nn.Module):
    """
    Maintains activity levels within a healthy dynamic range.

    Implements homeostatic plasticity that prevents runaway excitation
    or silencing, keeping the system in an operational regime.

    Works alongside criticality control to ensure stability.
    """

    def __init__(self,
                 target_activity: float = 0.5,
                 adaptation_rate: float = 0.001,
                 time_constant: float = 100.0):
        """
        Args:
            target_activity: Target mean activity level (0-1)
            adaptation_rate: How fast to adapt thresholds
            time_constant: Time constant for activity averaging
        """
        super().__init__()
        self.target_activity = target_activity
        self.adaptation_rate = adaptation_rate
        self.tau = time_constant

        # Running average of activity
        self.register_buffer('running_activity', torch.tensor(target_activity))

        # Homeostatic scaling factor (adapted by homeostatic rule, not backprop)
        self.register_buffer('scaling', torch.tensor(1.0))

        # Threshold adjustment (adapted by homeostatic rule, not backprop)
        self.register_buffer('threshold_offset', torch.tensor(0.0))

    def forward(self, activity: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Update homeostatic state and compute regulation signals.

        Args:
            activity: Current activity (will be averaged)

        Returns:
            Dictionary with homeostatic state and adjustments
        """
        # Update running average
        mean_activity = activity.mean()
        alpha = 1.0 / self.tau

        with torch.no_grad():
            self.running_activity = (1 - alpha) * self.running_activity + alpha * mean_activity

        # Compute error
        activity_error = self.running_activity - self.target_activity

        # Adjust scaling and threshold
        # If activity too high: decrease scaling, increase threshold
        # If activity too low: increase scaling, decrease threshold
        with torch.no_grad():
            self.scaling.copy_(torch.clamp(
                self.scaling - activity_error * self.adaptation_rate,
                0.1, 10.0
            ))
            self.threshold_offset.copy_(torch.clamp(
                self.threshold_offset + activity_error * self.adaptation_rate * 0.1,
                -1.0, 1.0
            ))

        return {
            'running_activity': self.running_activity.clone(),
            'activity_error': activity_error,
            'scaling': self.scaling.clone(),
            'threshold_offset': self.threshold_offset.clone()
        }

    def apply_scaling(self, x: torch.Tensor) -> torch.Tensor:
        """Apply homeostatic scaling to input."""
        return x * self.scaling

    def apply_threshold(self, threshold: torch.Tensor) -> torch.Tensor:
        """Apply threshold offset to base threshold."""
        return threshold + self.threshold_offset

    def reset(self):
        """Reset homeostatic state."""
        self.running_activity.fill_(self.target_activity)
        with torch.no_grad():
            self.scaling.fill_(1.0)
            self.threshold_offset.fill_(0.0)


class KoopmanBifurcationDetector(nn.Module):
    """
    Bifurcation detection via Dynamic Mode Decomposition (DMD) eigenvalues.

    Uses time-delay (Hankel) embedding of state history, then performs
    DMD via SVD to extract dynamic modes. Eigenvalue magnitudes track
    stability: |lambda| approaching 1.0 indicates approaching bifurcation.

    This is a spectral complement to the branching ratio — works on
    continuous dynamics rather than discrete avalanches.

    Args:
        state_dim: Dimension of the state being monitored
        buffer_size: Number of time steps to store
        delay_dim: Number of delay embeddings (Hankel matrix rows)
        rank: SVD truncation rank for DMD
        compute_every: Only recompute eigenvalues every N steps
    """

    def __init__(self,
                 state_dim: int,
                 buffer_size: int = 200,
                 delay_dim: int = 10,
                 rank: int = 5,
                 compute_every: int = 10,
                 bootstrap_n: int = 20,
                 min_confidence: int = 1):
        """
        Args:
            state_dim: Dimension of the state being monitored
            buffer_size: Number of time steps to store
            delay_dim: Number of delay embeddings (Hankel matrix rows)
            rank: SVD truncation rank for DMD
            compute_every: Only recompute eigenvalues every N steps
            bootstrap_n: Number of bootstrap subsamples for confidence (BOP-DMD, Sashidhar & Kutz 2022)
            min_confidence: Require N consecutive triggers before reporting bifurcation
        """
        super().__init__()
        self.state_dim = state_dim
        self.buffer_size = buffer_size
        self.delay_dim = delay_dim
        self.rank = rank
        self.compute_every = compute_every
        self.bootstrap_n = bootstrap_n
        self.min_confidence = min_confidence

        # State history
        self.register_buffer('history', torch.zeros(buffer_size, state_dim))
        self.register_buffer('hist_ptr', torch.tensor(0))

        # Cached results
        self.register_buffer('stability_margin', torch.tensor(1.0))
        self.register_buffer('stability_margin_std', torch.tensor(0.0))
        self.register_buffer('max_eigenvalue_mag', torch.tensor(0.0))
        self.register_buffer('approaching_bifurcation', torch.tensor(False))
        self.register_buffer('consecutive_triggers', torch.tensor(0))

    def _build_hankel(self) -> Optional[torch.Tensor]:
        """
        Build time-delay (Hankel) embedding matrix.

        Returns:
            Hankel matrix (delay_dim * state_dim, num_windows) or None if insufficient data
        """
        n_valid = min(self.hist_ptr.item(), self.buffer_size)
        n_windows = n_valid - self.delay_dim

        if n_windows < self.delay_dim:
            return None

        # Build Hankel matrix: each column is a delay-embedded snapshot
        rows = []
        for d in range(self.delay_dim):
            rows.append(self.history[d:d + n_windows])

        # (delay_dim, n_windows, state_dim) -> (delay_dim * state_dim, n_windows)
        H = torch.cat(rows, dim=1).T  # (delay_dim * state_dim, n_windows)
        return H

    def _dmd(self, H: torch.Tensor) -> torch.Tensor:
        """
        Dynamic Mode Decomposition via SVD.

        Returns eigenvalue magnitudes of the linear dynamics operator.
        """
        # Split into X (t) and Y (t+1)
        X = H[:, :-1]
        Y = H[:, 1:]

        # SVD of X
        try:
            U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        except RuntimeError:
            return torch.tensor([0.0], device=X.device)

        # Truncate to rank
        r = min(self.rank, len(S), U.shape[1])
        if r == 0:
            return torch.tensor([0.0], device=X.device)

        U_r = U[:, :r]
        S_r = S[:r]
        Vh_r = Vh[:r, :]

        # DMD operator: A_tilde = U_r^T Y V_r S_r^{-1}
        S_inv = 1.0 / (S_r + 1e-8)
        A_tilde = U_r.T @ Y @ Vh_r.T @ torch.diag(S_inv)

        # Eigenvalues of A_tilde
        try:
            eigenvalues = torch.linalg.eigvals(A_tilde)
        except RuntimeError:
            return torch.tensor([0.0], device=X.device)

        return torch.abs(eigenvalues)

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Record state and detect approaching bifurcation.

        Args:
            state: Current system state (state_dim,) or flattened

        Returns:
            Dict with:
            - stability_margin: 1 - max(|eigenvalue|). Approaching 0 = nearing bifurcation
            - max_eigenvalue_mag: Maximum eigenvalue magnitude
            - approaching_bifurcation: Whether margin is below threshold
        """
        if state.dim() > 1:
            state = state.flatten()[:self.state_dim]

        # Store in history
        with torch.no_grad():
            idx = self.hist_ptr % self.buffer_size
            self.history[idx] = state.detach()
            self.hist_ptr.add_(1)

        # Only recompute periodically (DMD is expensive)
        if self.hist_ptr % self.compute_every == 0:
            H = self._build_hankel()
            if H is not None:
                # Bootstrap confidence (BOP-DMD, Sashidhar & Kutz 2022)
                max_mags = []
                n_cols = H.shape[1]
                for _ in range(self.bootstrap_n):
                    # Subsample 80% of columns
                    n_sub = max(3, int(0.8 * n_cols))
                    idx = torch.randperm(n_cols)[:n_sub]
                    H_sub = H[:, idx.sort().values]
                    eig_mags = self._dmd(H_sub)
                    if len(eig_mags) > 0:
                        max_mags.append(eig_mags.max().item())

                if len(max_mags) >= 3:
                    max_mags_t = torch.tensor(max_mags)
                    mean_mag = max_mags_t.mean()
                    std_mag = max_mags_t.std()

                    with torch.no_grad():
                        self.max_eigenvalue_mag.copy_(mean_mag.clamp(max=5.0))
                        self.stability_margin.copy_((1.0 - mean_mag).clamp(min=-1.0))
                        self.stability_margin_std.copy_(std_mag)

                        # Trigger when margin < 0.1 AND bootstrap std is not too large
                        # (high std means unreliable estimate — don't trigger)
                        margin = (1.0 - mean_mag).item()
                        reliable = std_mag.item() < 0.3  # reject high-variance estimates
                        raw_trigger = margin < 0.1 and reliable
                        if raw_trigger:
                            self.consecutive_triggers.add_(1)
                        else:
                            self.consecutive_triggers.zero_()

                        self.approaching_bifurcation.fill_(
                            self.consecutive_triggers.item() >= self.min_confidence
                        )

        return {
            'stability_margin': self.stability_margin.clone(),
            'stability_margin_std': self.stability_margin_std.clone(),
            'max_eigenvalue_mag': self.max_eigenvalue_mag.clone(),
            'approaching_bifurcation': self.approaching_bifurcation.clone(),
        }

    def reset(self):
        """Reset history and cached results."""
        self.history.zero_()
        self.hist_ptr.zero_()
        self.stability_margin.fill_(1.0)
        self.stability_margin_std.fill_(0.0)
        self.max_eigenvalue_mag.fill_(0.0)
        self.approaching_bifurcation.fill_(False)
        self.consecutive_triggers.zero_()


if __name__ == '__main__':
    print("--- Self-Organized Criticality Examples ---")

    # Example 1: Avalanche Detection
    print("\n1. Avalanche Detection")
    detector = AvalancheDetector(activity_threshold=0.3)

    # Simulate some activity with avalanches
    for i in range(100):
        # Random activity with occasional bursts
        if i % 20 < 5:  # Burst period
            activity = torch.rand(10) * 2
        else:  # Quiet period
            activity = torch.rand(10) * 0.1

        result = detector(activity)
        if result['avalanche_ended']:
            print(f"   Step {i}: Avalanche ended, size={result['last_size'].item():.0f}")

    sizes, counts = detector.get_size_distribution()
    print(f"   Total avalanches: {detector.num_avalanches.item()}")
    print(f"   Power-law exponent: {detector.compute_power_law_exponent().item():.2f}")

    # Example 2: Branching Ratio
    print("\n2. Branching Ratio")
    branching = BranchingRatio(window_size=30)

    # Subcritical dynamics (decaying)
    activity = 10.0
    for i in range(30):
        activity = activity * 0.9 + torch.rand(1).item() * 0.5
        result = branching(torch.tensor(activity))
    print(f"   Subcritical sigma: {result['branching_ratio'].item():.3f}")

    branching.reset()

    # Critical dynamics (sustained)
    activity = 5.0
    for i in range(30):
        activity = activity * 0.95 + torch.rand(1).item() * 0.5
        result = branching(torch.tensor(activity))
    print(f"   Near-critical sigma: {result['branching_ratio'].item():.3f}")

    # Example 3: Criticality Controller
    print("\n3. Criticality Controller")
    controller = CriticalityController(adaptation_rate=0.05)

    # Simulate dynamics with controller
    coupling = 1.0
    activity = 5.0

    for i in range(100):
        # Simple dynamics
        activity = coupling * activity * 0.8 + torch.rand(1).item() * 2

        # Get control signals
        result = controller(torch.tensor([activity]))

        # Apply adjustments
        coupling = controller.get_adjusted_coupling(torch.tensor(1.0)).item()

        if i % 25 == 0:
            print(f"   Step {i}: sigma={result['branching_ratio'].item():.3f}, "
                  f"coupling_adj={result['coupling_adjustment'].item():.3f}")

    print(f"   Final: is_critical={controller.is_critical()}")

    # Example 4: Homeostatic Regulator
    print("\n4. Homeostatic Regulator")
    regulator = HomeostaticRegulator(target_activity=0.3, adaptation_rate=0.01)

    # Start with high activity
    for i in range(100):
        activity = torch.rand(10) * regulator.scaling

        result = regulator(activity)

        if i % 25 == 0:
            print(f"   Step {i}: running_act={result['running_activity'].item():.3f}, "
                  f"scaling={result['scaling'].item():.3f}")
