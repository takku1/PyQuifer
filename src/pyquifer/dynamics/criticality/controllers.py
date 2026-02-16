"""
Criticality control — active regulation of critical dynamics.

Contains controllers that actively adjust system parameters to maintain
criticality or homeostatic balance:

- NoProgressDetector: Detect stagnation / fortress signatures
- CriticalityController: PI controller for branching ratio → coupling/noise
- HomeostaticRegulator: Keep activity in healthy dynamic range
"""

import torch
import torch.nn as nn
from typing import Dict

from pyquifer.dynamics.criticality.monitors import AvalancheDetector, BranchingRatio


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
