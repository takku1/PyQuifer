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
        last_size = torch.tensor(0)

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
            'avalanche_ended': torch.tensor(avalanche_ended),
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
            return torch.tensor([]), torch.tensor([])

        sizes = valid.unique(sorted=True)
        counts = torch.tensor([(valid == s).sum() for s in sizes])

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
            return torch.tensor(0.0)

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
    """

    def __init__(self, window_size: int = 50):
        """
        Args:
            window_size: Number of time steps to average over
        """
        super().__init__()
        self.window_size = window_size

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
        if n_valid < 2:
            return {
                'branching_ratio': torch.tensor(1.0),
                'criticality_distance': torch.tensor(0.0)
            }

        history = self.activity_history[:n_valid]

        # Branching ratio = mean of per-step ratios (unbiased by Jensen's inequality)
        ancestors = history[:-1]
        descendants = history[1:]

        # Mean-of-ratios: unbiased estimator
        sigma = (descendants / (ancestors + 1e-6)).mean()

        return {
            'branching_ratio': sigma,
            'criticality_distance': torch.abs(sigma - 1.0)
        }

    def reset(self):
        """Reset history."""
        self.activity_history.zero_()
        self.history_ptr.zero_()


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
                 max_noise: float = 1.0):
        """
        Args:
            target_branching_ratio: Target sigma (1.0 for critical)
            adaptation_rate: How fast to adapt parameters
            min_coupling: Minimum allowed coupling strength
            max_coupling: Maximum allowed coupling strength
            min_noise: Minimum noise level
            max_noise: Maximum noise level
        """
        super().__init__()
        self.target_sigma = target_branching_ratio
        self.adaptation_rate = adaptation_rate
        self.min_coupling = min_coupling
        self.max_coupling = max_coupling
        self.min_noise = min_noise
        self.max_noise = max_noise

        # Criticality monitors
        self.avalanche_detector = AvalancheDetector()
        self.branching_monitor = BranchingRatio()

        # Controlled state (adapted by controller dynamics, not backprop)
        self.register_buffer('coupling_adjustment', torch.tensor(1.0))
        self.register_buffer('noise_adjustment', torch.tensor(1.0))

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

        # Compute control adjustments
        sigma_error = sigma - self.target_sigma

        # If sigma < 1 (subcritical): increase coupling, decrease noise
        # If sigma > 1 (supercritical): decrease coupling, increase noise
        coupling_delta = -sigma_error * self.adaptation_rate
        noise_delta = sigma_error * self.adaptation_rate * 0.5  # Noise has smaller effect

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
        self.coupling_adjustment.data.fill_(1.0)
        self.noise_adjustment.data.fill_(1.0)
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
        self.scaling.data.fill_(1.0)
        self.threshold_offset.data.fill_(0.0)


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
