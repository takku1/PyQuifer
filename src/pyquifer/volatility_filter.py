"""
Volatility Filter Module for PyQuifer

Implements Hierarchical Gaussian Filter (HGF) concepts for adaptive learning.
Instead of fixed learning rates, the system learns HOW FAST to learn based
on environmental volatility.

Core insight: When things are changing fast (high volatility), learn faster.
When things are stable (low volatility), consolidate and learn slower.

Three levels of belief:
  Level 0: Value beliefs (what is the signal?)
  Level 1: Volatility beliefs (how stable is the signal?)
  Level 2: Meta-volatility (how stable is the stability?) [optional]

Key equations (Weber et al., 2023):
  Value PE:      delta = observation - predicted_mean
  Volatility PE: Delta = (predicted_precision / posterior_precision)
                         + predicted_precision * delta^2 - 1
  Effective LR:  lr = base_lr * sigmoid(volatility_PE)
  Precision:     pi_hat = 1 / (1/pi_prev + exp(tonic_volatility + kappa * mu_parent))

References:
- Mathys et al. (2011). A Bayesian foundation for individual learning.
- Weber et al. (2023). The generalized Hierarchical Gaussian Filter.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Dict, List


class VolatilityNode(nn.Module):
    """
    Single node in a Hierarchical Gaussian Filter.

    Tracks beliefs about a signal: mean (expected value), precision (confidence),
    and updates via prediction errors weighted by precision.

    Can be used standalone for simple adaptive learning, or stacked
    in a HierarchicalVolatilityFilter for multi-timescale adaptation.
    """

    def __init__(self,
                 dim: int,
                 tonic_volatility: float = -4.0,
                 autoconnection: float = 1.0,
                 tonic_drift: float = 0.0,
                 initial_precision: float = 1.0,
                 min_observation_precision: float = 0.01):
        """
        Args:
            dim: Dimensionality of the signal being tracked.
            tonic_volatility: Log-space baseline volatility (omega).
                             More negative = more stable. Default -4.0 ~ variance of 0.018.
            autoconnection: How much the mean persists from previous step (lambda).
                           1.0 = random walk, 0.0 = fully driven by parents.
            tonic_drift: Systematic drift in the mean per step.
            initial_precision: Starting precision (inverse variance).
            min_observation_precision: Floor on observation precision. Prevents
                                     mean explosions when predicted precision is tiny.
        """
        super().__init__()
        self.dim = dim
        self.tonic_volatility = tonic_volatility
        self.autoconnection = autoconnection
        self.tonic_drift = tonic_drift
        self.min_observation_precision = min_observation_precision

        # Beliefs: mean and precision
        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('precision', torch.full((dim,), initial_precision))

        # Predicted values (from prediction step)
        self.register_buffer('predicted_mean', torch.zeros(dim))
        self.register_buffer('predicted_precision', torch.full((dim,), initial_precision))

        # Prediction errors
        self.register_buffer('value_pe', torch.zeros(dim))
        self.register_buffer('volatility_pe', torch.zeros(dim))

        # Effective precision (gamma = omega * pi_hat, used by parent updates)
        self.register_buffer('effective_precision', torch.zeros(dim))

        # Step counter
        self.register_buffer('step_count', torch.tensor(0))

    def predict(self, volatility_parent_mean: Optional[torch.Tensor] = None,
                kappa: float = 1.0) -> None:
        """
        Prediction step: predict next mean and precision.

        Args:
            volatility_parent_mean: Mean of volatility parent node (if any).
            kappa: Volatility coupling strength.
        """
        with torch.no_grad():
            # Mean prediction: lambda * previous_mean + drift
            self.predicted_mean.copy_(
                self.autoconnection * self.mean + self.tonic_drift
            )

            # Precision prediction: 1 / (1/pi_prev + total_volatility)
            # Total volatility = exp(omega + kappa * parent_mean)
            if volatility_parent_mean is not None:
                log_vol = self.tonic_volatility + kappa * volatility_parent_mean
            else:
                log_vol = torch.full((self.dim,), self.tonic_volatility,
                                     device=self.mean.device)

            # Numerical stability: clamp log-volatility
            log_vol = log_vol.clamp(-80.0, 80.0)
            total_volatility = torch.exp(log_vol)

            # Predicted precision
            pred_prec = 1.0 / (1.0 / self.precision.clamp(min=1e-8) + total_volatility)
            self.predicted_precision.copy_(pred_prec.clamp(min=1e-8))

            # Effective precision (gamma): used by parent volatility updates
            self.effective_precision.copy_(total_volatility * self.predicted_precision)

    def update(self, observation: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Update step: compute prediction errors and update beliefs.

        Args:
            observation: Observed signal (dim,).

        Returns:
            Dict with value_pe, volatility_pe, mean, precision, effective_lr.
        """
        with torch.no_grad():
            device = observation.device

            # Value prediction error
            delta = observation - self.predicted_mean.to(device)
            self.value_pe.copy_(delta)

            # Posterior precision (from value observation)
            # pi_new = pi_hat + observation_precision
            # Use max(pi_hat, min_obs_prec) as observation precision
            # to prevent mean explosion when predicted precision is tiny
            obs_prec = self.predicted_precision.clamp(min=self.min_observation_precision)
            posterior_precision = (self.predicted_precision + obs_prec).clamp(min=1e-8)

            # Volatility prediction error
            # Delta = (pi_hat / pi_posterior) + pi_hat * delta^2 - 1
            vol_pe = (self.predicted_precision / posterior_precision
                      + self.predicted_precision * delta.pow(2).clamp(max=100.0) - 1.0)
            self.volatility_pe.copy_(vol_pe)

            # Posterior mean update (precision-weighted)
            # mu_new = mu_hat + (1/pi_new) * delta
            mean_update = (delta / posterior_precision).clamp(-10.0, 10.0)
            self.mean.copy_(self.predicted_mean + mean_update)

            # Update precision
            self.precision.copy_(posterior_precision)

            # Effective learning rate: sigmoid of volatility PE
            # High volatility PE -> high lr (things are changing, learn fast)
            # Low/negative volatility PE -> low lr (stable, consolidate)
            # Clamp vol_pe to prevent sigmoid saturation to exactly 0 or 1
            effective_lr = torch.sigmoid(vol_pe.clamp(-10.0, 10.0))

            self.step_count.add_(1)

        return {
            'value_pe': delta,
            'volatility_pe': vol_pe,
            'mean': self.mean.clone(),
            'precision': self.precision.clone(),
            'effective_lr': effective_lr,
            'effective_precision': self.effective_precision.clone(),
        }

    def forward(self, observation: torch.Tensor,
                volatility_parent_mean: Optional[torch.Tensor] = None,
                kappa: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Full predict-update cycle.

        Args:
            observation: Observed signal (dim,).
            volatility_parent_mean: Mean from volatility parent (optional).
            kappa: Volatility coupling strength.

        Returns:
            Dict with value_pe, volatility_pe, mean, precision, effective_lr.
        """
        self.predict(volatility_parent_mean=volatility_parent_mean, kappa=kappa)
        return self.update(observation)

    def reset(self):
        """Reset all beliefs to initial state."""
        self.mean.zero_()
        self.precision.fill_(1.0)
        self.predicted_mean.zero_()
        self.predicted_precision.fill_(1.0)
        self.value_pe.zero_()
        self.volatility_pe.zero_()
        self.effective_precision.zero_()
        self.step_count.zero_()


class HierarchicalVolatilityFilter(nn.Module):
    """
    Multi-level Hierarchical Gaussian Filter.

    Stacks VolatilityNodes in a hierarchy where higher levels track the
    volatility (stability) of lower levels. This creates multi-timescale
    adaptation: fast changes at the bottom, slow meta-learning at the top.

    2-level: value + volatility
    3-level: value + volatility + meta-volatility

    The key insight: learning rates EMERGE from volatility prediction errors,
    rather than being manually set. When the environment becomes unpredictable,
    the filter automatically speeds up learning.
    """

    def __init__(self,
                 dim: int,
                 num_levels: int = 2,
                 tonic_volatilities: Optional[List[float]] = None,
                 kappas: Optional[List[float]] = None):
        """
        Args:
            dim: Signal dimensionality at the bottom level.
            num_levels: Number of hierarchy levels (2 or 3).
            tonic_volatilities: Per-level tonic volatility (log-space).
                               Default: [-4.0, -6.0, -8.0] (more stable at higher levels).
            kappas: Volatility coupling strengths between levels.
                   Default: [1.0, 1.0].
        """
        super().__init__()
        assert 2 <= num_levels <= 3, "HGF supports 2 or 3 levels"

        self.dim = dim
        self.num_levels = num_levels

        # Default volatilities: more stable at higher levels
        if tonic_volatilities is None:
            tonic_volatilities = [-4.0, -6.0, -8.0][:num_levels]
        if kappas is None:
            kappas = [1.0] * (num_levels - 1)

        self.kappas = kappas

        # Create nodes: level 0 = value, level 1 = volatility, level 2 = meta-volatility
        self.levels = nn.ModuleList()
        for i in range(num_levels):
            self.levels.append(VolatilityNode(
                dim=dim,
                tonic_volatility=tonic_volatilities[i],
                autoconnection=1.0,
                initial_precision=1.0 if i == 0 else 0.5,
            ))

    def forward(self, observation: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Run one full HGF update cycle.

        Order: predict top-down, then update bottom-up.

        Args:
            observation: Raw signal (dim,).

        Returns:
            Dict with per-level results and overall effective_lr.
        """
        # --- Prediction phase (top-down) ---
        # Highest level predicts first (no parent)
        self.levels[-1].predict(volatility_parent_mean=None)

        # Lower levels use parent means
        for i in range(self.num_levels - 2, -1, -1):
            parent_mean = self.levels[i + 1].mean
            kappa = self.kappas[i] if i < len(self.kappas) else 1.0
            self.levels[i].predict(
                volatility_parent_mean=parent_mean,
                kappa=kappa,
            )

        # --- Update phase (bottom-up) ---
        # Level 0: update from observation
        result_0 = self.levels[0].update(observation)

        # Level 1+: update from child's volatility PE
        level_results = [result_0]
        for i in range(1, self.num_levels):
            child = self.levels[i - 1]
            kappa = self.kappas[i - 1] if i - 1 < len(self.kappas) else 1.0

            # Parent mean update from child volatility PE
            with torch.no_grad():
                vol_pe = child.volatility_pe.clamp(-10.0, 10.0)
                gamma = child.effective_precision.clamp(0.0, 10.0)
                parent = self.levels[i]

                # Mean update: mu += (kappa * gamma * Delta) / (2 * pi)
                mean_delta = (kappa * gamma * vol_pe) / (2 * parent.precision.clamp(min=1e-8))
                mean_delta = mean_delta.clamp(-2.0, 2.0)
                parent.mean.add_(mean_delta)
                # Clamp parent mean: it represents log-volatility offset,
                # so keep it in a range where exp(tonic + kappa*mean) is sensible
                parent.mean.clamp_(-20.0, 20.0)

                # Precision update: pi += 0.5 * (kappa * gamma)^2
                prec_delta = 0.5 * (kappa * gamma).pow(2)
                prec_delta = prec_delta.clamp(0.0, 5.0)
                parent.precision.add_(prec_delta)
                parent.precision.clamp_(min=1e-8, max=1e4)

                # Compute volatility PE for this level
                parent_vol_pe = (
                    parent.predicted_precision / parent.precision
                    + parent.predicted_precision * mean_delta.pow(2) - 1.0
                ).clamp(-10.0, 10.0)
                parent.volatility_pe.copy_(parent_vol_pe)

            level_results.append({
                'value_pe': mean_delta,
                'volatility_pe': parent.volatility_pe.clone(),
                'mean': parent.mean.clone(),
                'precision': parent.precision.clone(),
                'effective_lr': torch.sigmoid(parent.volatility_pe),
            })

        # Effective learning rate from level 0 (what downstream modules should use)
        effective_lr = result_0['effective_lr']

        # Volatility summary: mean volatility PE across levels
        all_vol_pe = torch.stack([r['volatility_pe'] for r in level_results])
        mean_volatility = all_vol_pe.mean()

        return {
            'effective_lr': effective_lr,
            'value_pe': result_0['value_pe'],
            'volatility_pe': result_0['volatility_pe'],
            'mean_volatility': mean_volatility,
            'precision': result_0['precision'],
            'level_results': level_results,
        }

    def get_volatility_summary(self) -> Dict[str, torch.Tensor]:
        """Get a compact summary of volatility state across levels."""
        return {
            f'level_{i}_mean': self.levels[i].mean.clone()
            for i in range(self.num_levels)
        } | {
            f'level_{i}_precision': self.levels[i].precision.clone()
            for i in range(self.num_levels)
        } | {
            f'level_{i}_vol_pe': self.levels[i].volatility_pe.clone()
            for i in range(self.num_levels)
        }

    def reset(self):
        """Reset all levels."""
        for level in self.levels:
            level.reset()


class VolatilityGatedLearning(nn.Module):
    """
    Wraps any learning rule with volatility-adaptive learning rates.

    Instead of a fixed learning rate, the effective rate is modulated by
    a VolatilityNode or HierarchicalVolatilityFilter. When the tracked
    signal is volatile, learning speeds up; when stable, it slows down.

    Usage:
        gate = VolatilityGatedLearning(dim=32, base_lr=0.01)
        for observation in stream:
            lr = gate(observation)  # returns per-dimension effective lr
            # Use lr in your learning rule
    """

    def __init__(self,
                 dim: int,
                 base_lr: float = 0.01,
                 min_lr: float = 0.001,
                 max_lr: float = 0.1,
                 num_levels: int = 2,
                 tonic_volatility: float = -4.0):
        """
        Args:
            dim: Signal dimensionality.
            base_lr: Base learning rate (modulated by volatility).
            min_lr: Floor on effective learning rate.
            max_lr: Ceiling on effective learning rate.
            num_levels: Hierarchy depth (2 or 3).
            tonic_volatility: Baseline volatility for the filter.
        """
        super().__init__()
        self.dim = dim
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.max_lr = max_lr

        self.filter = HierarchicalVolatilityFilter(
            dim=dim,
            num_levels=num_levels,
            tonic_volatilities=[tonic_volatility, tonic_volatility - 2.0, tonic_volatility - 4.0][:num_levels],
        )

    def forward(self, observation: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute volatility-gated effective learning rate.

        Args:
            observation: Signal to track (dim,).

        Returns:
            Dict with 'effective_lr' (dim,), 'volatility_pe', 'mean_volatility'.
        """
        result = self.filter(observation)

        # Scale sigmoid output to [min_lr, max_lr] range
        raw_lr = result['effective_lr']  # sigmoid output in [0, 1]
        scaled_lr = self.min_lr + (self.max_lr - self.min_lr) * raw_lr

        # Also provide the base_lr-relative version
        lr_multiplier = scaled_lr / self.base_lr

        return {
            'effective_lr': scaled_lr,
            'lr_multiplier': lr_multiplier,
            'volatility_pe': result['volatility_pe'],
            'mean_volatility': result['mean_volatility'],
            'precision': result['precision'],
        }

    def get_mean_lr(self) -> float:
        """Get the current mean effective learning rate."""
        lr = self.min_lr + (self.max_lr - self.min_lr) * torch.sigmoid(
            self.filter.levels[0].volatility_pe
        )
        return lr.mean().item()

    def reset(self):
        """Reset the volatility filter."""
        self.filter.reset()


if __name__ == '__main__':
    print("=== Volatility Filter Demo ===\n")

    # --- Demo 1: Single VolatilityNode ---
    print("--- Single VolatilityNode ---")
    node = VolatilityNode(dim=4, tonic_volatility=-4.0)

    torch.manual_seed(42)
    # Stable signal for 20 steps
    for i in range(20):
        obs = torch.randn(4) * 0.1 + 1.0  # Low noise around 1.0
        result = node(obs)
        if i % 5 == 0:
            print(f"  Step {i:2d}: mean_lr={result['effective_lr'].mean():.4f} "
                  f"precision={result['precision'].mean():.4f}")

    # Sudden change: mean shifts to 5.0
    print("  --- Signal shift to 5.0 ---")
    for i in range(20, 40):
        obs = torch.randn(4) * 0.1 + 5.0
        result = node(obs)
        if i % 5 == 0:
            print(f"  Step {i:2d}: mean_lr={result['effective_lr'].mean():.4f} "
                  f"precision={result['precision'].mean():.4f}")

    # --- Demo 2: HierarchicalVolatilityFilter ---
    print("\n--- 3-Level Hierarchical Filter ---")
    hgf = HierarchicalVolatilityFilter(dim=8, num_levels=3)

    torch.manual_seed(42)
    lrs = []
    for i in range(100):
        # Alternating stable and volatile phases
        if i < 30:
            obs = torch.randn(8) * 0.1 + 1.0
        elif i < 60:
            obs = torch.randn(8) * 2.0 + torch.sin(torch.tensor(i * 0.5)) * 3.0
        else:
            obs = torch.randn(8) * 0.1 + 1.0

        result = hgf(obs)
        lrs.append(result['effective_lr'].mean().item())

        if i % 20 == 0:
            print(f"  Step {i:3d}: lr={lrs[-1]:.4f} "
                  f"vol_pe={result['volatility_pe'].mean():.4f} "
                  f"mean_vol={result['mean_volatility']:.4f}")

    # LR should be higher during volatile phase (30-60)
    stable_early = sum(lrs[:30]) / 30
    volatile = sum(lrs[30:60]) / 30
    stable_late = sum(lrs[60:]) / 40
    print(f"\n  Mean LR stable (0-30):   {stable_early:.4f}")
    print(f"  Mean LR volatile (30-60): {volatile:.4f}")
    print(f"  Mean LR stable (60-100): {stable_late:.4f}")

    # --- Demo 3: VolatilityGatedLearning ---
    print("\n--- VolatilityGatedLearning ---")
    gate = VolatilityGatedLearning(dim=16, base_lr=0.01, min_lr=0.001, max_lr=0.1)

    torch.manual_seed(42)
    for i in range(30):
        obs = torch.randn(16) * (0.1 if i < 15 else 2.0)
        result = gate(obs)
        if i % 5 == 0:
            print(f"  Step {i:2d}: effective_lr={result['effective_lr'].mean():.5f} "
                  f"lr_mult={result['lr_multiplier'].mean():.3f}")

    print("\n[OK] Volatility filter demo passed!")
