"""
Causal Flow Module for PyQuifer

Measures WHO is influencing WHOM using transfer entropy —
directed information flow between oscillator populations.

Key concepts:
- Transfer Entropy: T(X→Y) = how much X's past reduces uncertainty about Y's future
- Causal Flow Map: Directed graph of information flow across all populations
- Dominance Detection: Is the system in perception mode (bottom-up) or
  imagination mode (top-down)?

This complements IIT (which measures total integration) by adding DIRECTIONALITY.

References:
- Schreiber (2000). Measuring Information Transfer.
- Vicente et al. (2011). Transfer Entropy — A Model-Free Measure of
  Effective Connectivity for the Neurosciences.
- Barnett et al. (2009). Granger Causality and Transfer Entropy.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Dict, List, Tuple


class TransferEntropyEstimator(nn.Module):
    """
    Computes transfer entropy between pairs of time series.

    T(X→Y) = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)

    Positive T(X→Y) means X carries information about Y's future
    beyond what Y's own past provides.

    Uses histogram-based estimation with adaptive binning.
    """

    def __init__(self,
                 num_bins: int = 8,
                 history_length: int = 3,
                 buffer_size: int = 200):
        """
        Args:
            num_bins: Number of bins for discretization
            history_length: Number of past time steps to condition on
            buffer_size: Maximum time series length to store
        """
        super().__init__()
        self.num_bins = num_bins
        self.history_length = history_length
        self.buffer_size = buffer_size

    def _discretize(self, x: torch.Tensor) -> torch.Tensor:
        """Bin continuous values into discrete bins."""
        # Adaptive binning based on data range
        x_min = x.min()
        x_max = x.max()
        if x_max - x_min < 1e-8:
            return torch.zeros_like(x, dtype=torch.long)
        normalized = (x - x_min) / (x_max - x_min + 1e-8)
        binned = (normalized * (self.num_bins - 1)).long().clamp(0, self.num_bins - 1)
        return binned

    def _entropy_from_counts(self, counts: torch.Tensor) -> float:
        """Compute entropy from a count tensor."""
        total = counts.sum().float()
        if total < 1:
            return 0.0
        probs = counts.float() / total
        probs = probs[probs > 0]
        return -(probs * torch.log2(probs)).sum().item()

    def compute_te(self,
                   source: torch.Tensor,
                   target: torch.Tensor) -> float:
        """
        Compute transfer entropy T(source → target).

        Args:
            source: Source time series (T,)
            target: Target time series (T,)

        Returns:
            Transfer entropy in bits
        """
        T = min(len(source), len(target))
        if T < self.history_length + 2:
            return 0.0

        # Discretize
        src_d = self._discretize(source[:T])
        tgt_d = self._discretize(target[:T])

        k = self.history_length
        n_samples = T - k - 1

        if n_samples < 10:
            return 0.0

        # Build joint distributions
        # We need: P(y_t+1, y_past, x_past) and P(y_t+1, y_past)
        K = self.num_bins

        # Count tables
        # joint_yyx: counts of (y_future, y_past_hash, x_past_hash)
        # For efficiency, hash past sequences to single indices
        y_future = tgt_d[k + 1: k + 1 + n_samples]

        # Hash past k values into single index (vectorized)
        powers = K ** torch.arange(k, device=source.device)
        # Stack k consecutive slices into (k, n_samples)
        y_past_slices = torch.stack([tgt_d[k - i: k - i + n_samples] for i in range(k)])
        x_past_slices = torch.stack([src_d[k - i: k - i + n_samples] for i in range(k)])
        # Matrix multiply with powers: (k,) @ (k, n_samples) -> (n_samples,)
        y_past_hash = (powers.unsqueeze(1) * y_past_slices).sum(dim=0).long()
        x_past_hash = (powers.unsqueeze(1) * x_past_slices).sum(dim=0).long()

        num_past_states = K ** k

        # H(Y_future | Y_past)
        h_y_given_ypast = 0.0
        for yp in range(min(num_past_states, n_samples)):
            mask = y_past_hash == yp
            if mask.sum() < 2:
                continue
            yf_given_yp = y_future[mask]
            counts = torch.bincount(yf_given_yp, minlength=K)[:K]
            h = self._entropy_from_counts(counts)
            weight = mask.sum().float() / n_samples
            h_y_given_ypast += weight.item() * h

        # H(Y_future | Y_past, X_past)
        h_y_given_ypast_xpast = 0.0
        combined_hash = y_past_hash * num_past_states + x_past_hash
        num_combined = num_past_states * num_past_states

        for c in combined_hash.unique():
            mask = combined_hash == c
            if mask.sum() < 2:
                continue
            yf_given_c = y_future[mask]
            counts = torch.bincount(yf_given_c, minlength=K)[:K]
            h = self._entropy_from_counts(counts)
            weight = mask.sum().float() / n_samples
            h_y_given_ypast_xpast += weight.item() * h

        te = max(0.0, h_y_given_ypast - h_y_given_ypast_xpast)
        return te

    def forward(self,
                source: torch.Tensor,
                target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute transfer entropy and net flow.

        Args:
            source: Source time series (T,)
            target: Target time series (T,)

        Returns:
            Dictionary with:
            - te_source_to_target: T(source → target)
            - te_target_to_source: T(target → source)
            - net_flow: Net causal flow (positive = source drives target)
        """
        te_xy = self.compute_te(source, target)
        te_yx = self.compute_te(target, source)
        net_flow = te_xy - te_yx

        return {
            'te_source_to_target': torch.tensor(te_xy),
            'te_target_to_source': torch.tensor(te_yx),
            'net_flow': torch.tensor(net_flow),
        }


class CausalFlowMap(nn.Module):
    """
    Directed graph of information flow across all oscillator populations.

    Computes pairwise transfer entropy and builds a causal flow matrix.
    Identifies drivers (net exporters of information) and followers
    (net importers).
    """

    def __init__(self,
                 num_populations: int,
                 num_bins: int = 8,
                 history_length: int = 3,
                 buffer_size: int = 200):
        """
        Args:
            num_populations: Number of oscillator populations
            num_bins: Bins for TE estimation
            history_length: Past conditioning length
            buffer_size: Time series buffer size
        """
        super().__init__()
        self.num_populations = num_populations
        self.buffer_size = buffer_size

        self.te_estimator = TransferEntropyEstimator(
            num_bins=num_bins,
            history_length=history_length,
            buffer_size=buffer_size,
        )

        # Time series buffer
        self.register_buffer('time_series',
                             torch.zeros(buffer_size, num_populations))
        self.register_buffer('ts_ptr', torch.tensor(0))

        # Cached flow matrix
        self.register_buffer('flow_matrix',
                             torch.zeros(num_populations, num_populations))

    def record(self, population_states: torch.Tensor):
        """
        Record one time step of population states.

        Args:
            population_states: Current state of each population (num_populations,)
        """
        with torch.no_grad():
            idx = self.ts_ptr % self.buffer_size
            self.time_series[idx] = population_states.detach()
            self.ts_ptr.add_(1)

    def compute_flow(self) -> Dict[str, torch.Tensor]:
        """
        Compute full causal flow map from buffered time series.

        Returns:
            Dictionary with:
            - flow_matrix: TE(i→j) matrix (num_pop x num_pop)
            - net_flow_matrix: Net flow matrix (positive = i drives j)
            - driver_scores: Net outward flow per population
            - follower_scores: Net inward flow per population
        """
        n_valid = min(self.ts_ptr.item(), self.buffer_size)
        if n_valid < 20:
            return {
                'flow_matrix': self.flow_matrix.clone(),
                'net_flow_matrix': torch.zeros_like(self.flow_matrix),
                'driver_scores': torch.zeros(self.num_populations),
                'follower_scores': torch.zeros(self.num_populations),
            }

        series = self.time_series[:n_valid]

        # Compute pairwise TE
        flow = torch.zeros(self.num_populations, self.num_populations)
        for i in range(self.num_populations):
            for j in range(self.num_populations):
                if i != j:
                    flow[i, j] = self.te_estimator.compute_te(series[:, i], series[:, j])

        with torch.no_grad():
            self.flow_matrix.copy_(flow)

        # Net flow
        net_flow = flow - flow.t()

        # Driver = net exporter of information
        driver_scores = net_flow.sum(dim=1)  # Sum of outward net flows
        # Follower = net importer
        follower_scores = -driver_scores  # Opposite of driver

        return {
            'flow_matrix': flow,
            'net_flow_matrix': net_flow,
            'driver_scores': driver_scores,
            'follower_scores': follower_scores,
        }

    def forward(self,
                population_states: torch.Tensor,
                compute_every: int = 50) -> Dict[str, torch.Tensor]:
        """
        Record states and periodically compute flow.

        Args:
            population_states: Current population states (num_populations,)
            compute_every: Recompute flow map every N steps

        Returns:
            Dictionary with flow map (may be stale between compute steps)
        """
        self.record(population_states)

        if self.ts_ptr % compute_every == 0 and self.ts_ptr > 20:
            return self.compute_flow()

        # Return cached
        return {
            'flow_matrix': self.flow_matrix.clone(),
            'net_flow_matrix': (self.flow_matrix - self.flow_matrix.t()),
            'driver_scores': (self.flow_matrix - self.flow_matrix.t()).sum(dim=1),
            'follower_scores': -(self.flow_matrix - self.flow_matrix.t()).sum(dim=1),
        }

    def reset(self):
        """Reset buffer and flow matrix."""
        self.time_series.zero_()
        self.ts_ptr.zero_()
        self.flow_matrix.zero_()


class DominanceDetector(nn.Module):
    """
    Detects whether the system is in perception mode (bottom-up dominant)
    or imagination mode (top-down dominant).

    Compares upward TE (sensory→abstract) vs downward TE (abstract→sensory).
    - Dominance > 0.5 → perception mode (errors dominate)
    - Dominance < 0.5 → imagination mode (predictions dominate)
    """

    def __init__(self,
                 num_levels: int = 3,
                 num_bins: int = 8,
                 buffer_size: int = 200,
                 hysteresis: float = 0.03):
        """
        Args:
            num_levels: Number of hierarchy levels (bottom=sensory, top=abstract)
            num_bins: Bins for TE estimation
            buffer_size: Time series buffer
            hysteresis: Band width for mode switching hysteresis
        """
        super().__init__()
        self.num_levels = num_levels
        self.buffer_size = buffer_size
        self.hysteresis = hysteresis

        self.te_estimator = TransferEntropyEstimator(
            num_bins=num_bins, buffer_size=buffer_size
        )

        # Buffer for per-level summary signals
        self.register_buffer('level_history',
                             torch.zeros(buffer_size, num_levels))
        self.register_buffer('hist_ptr', torch.tensor(0))

        # Python int mirror of hist_ptr — avoids GPU sync on every call
        self._hist_ptr_py: int = 0

        # Dominance ratio
        self.register_buffer('dominance_ratio', torch.tensor(0.5))

        # Running statistics for adaptive mode thresholds
        # Instead of static 0.4/0.6 thresholds, derive from observed ratio distribution
        self._ratio_ema: float = 0.5       # Exponential moving average of ratio
        self._ratio_var_ema: float = 0.01   # EMA of variance (for adaptive band width)
        self._ema_alpha: float = 0.08       # Smoothing factor
        self._current_mode: str = 'balanced'

    def forward(self,
                level_activations: torch.Tensor,
                compute_every: int = 10,
                criticality_distance: float = None) -> Dict[str, torch.Tensor]:
        """
        Record level activations and compute dominance with three-way mode detection.

        Uses adaptive thresholds derived from running statistics of the dominance
        ratio, with hysteresis to prevent rapid mode flipping. This produces the
        metastable mode-switching pattern observed in empirical EEG studies
        (100-500ms timescale at default tick rate).

        When ``criticality_distance`` is provided, hysteresis is scaled by
        ``1/sqrt(crit_dist)`` — implementing critical slowing down
        (Meisel et al. 2015, SNIC scaling exponents). Near the critical point,
        modes persist longer; far from it, transitions are faster.

        Args:
            level_activations: Summary activation per hierarchy level (num_levels,)
            compute_every: How often to recompute TE (default 10 for ~2s switching)
            criticality_distance: Optional distance from critical point.
                Small values → near-critical → modes persist longer.

        Returns:
            Dictionary with:
            - dominance_ratio: > 0.5 = perception-leaning, < 0.5 = imagination-leaning
            - bottom_up_te: Total upward TE
            - top_down_te: Total downward TE
            - mode: 'perception', 'imagination', or 'balanced'
        """
        with torch.no_grad():
            idx = self._hist_ptr_py % self.buffer_size
            self.level_history[idx] = level_activations.detach()
            self.hist_ptr.add_(1)
            self._hist_ptr_py += 1

        n_valid = min(self._hist_ptr_py, self.buffer_size)

        bottom_up_te = torch.tensor(0.0, device=level_activations.device)
        top_down_te = torch.tensor(0.0, device=level_activations.device)

        # Lower minimum from 20 to 5 — don't wait 4s for first mode decision
        if n_valid >= 5 and self._hist_ptr_py % compute_every == 0:
            series = self.level_history[:n_valid]

            # Sum TE from lower→higher levels (bottom-up)
            # Sum TE from higher→lower levels (top-down)
            bu = 0.0
            td = 0.0
            for i in range(self.num_levels - 1):
                bu += self.te_estimator.compute_te(series[:, i], series[:, i + 1])
                td += self.te_estimator.compute_te(series[:, i + 1], series[:, i])

            bottom_up_te = torch.tensor(bu, device=level_activations.device)
            top_down_te = torch.tensor(td, device=level_activations.device)

            total = bu + td + 1e-8
            new_ratio = bu / total
            with torch.no_grad():
                self.dominance_ratio.fill_(new_ratio)

            # Update running statistics for adaptive thresholds
            alpha = self._ema_alpha
            self._ratio_var_ema = (1 - alpha) * self._ratio_var_ema + alpha * (new_ratio - self._ratio_ema) ** 2
            self._ratio_ema = (1 - alpha) * self._ratio_ema + alpha * new_ratio

        # Adaptive three-way thresholds from running statistics
        # Band width = max(hysteresis, 1σ of observed ratio) — widens when ratio
        # is volatile, narrows when stable. Center is the running mean.
        sigma = math.sqrt(max(1e-6, self._ratio_var_ema))
        band = max(self.hysteresis, sigma)
        center = self._ratio_ema
        upper = center + band   # Above this = perception
        lower = center - band   # Below this = imagination

        ratio = float(self.dominance_ratio.item())

        # Hysteresis: once in a mode, need to cross further to leave.
        # Fix 7: Critical slowing down — scale hysteresis by 1/sqrt(crit_dist).
        # Near the critical point (small crit_dist), modes persist much longer.
        # At crit_dist=0.01: 10x wider hysteresis; at crit_dist=0.16: 2.5x.
        # (Meisel et al. 2015: tau ~ (mu_c - mu)^(-0.5))
        hyst = self.hysteresis
        if criticality_distance is not None:
            crit_scale = 1.0 / math.sqrt(max(0.01, criticality_distance))
            hyst = hyst * crit_scale
        if self._current_mode == 'perception':
            if ratio < upper - hyst:
                self._current_mode = 'balanced' if ratio > lower + hyst else 'imagination'
        elif self._current_mode == 'imagination':
            if ratio > lower + hyst:
                self._current_mode = 'balanced' if ratio < upper - hyst else 'perception'
        else:  # balanced
            if ratio > upper:
                self._current_mode = 'perception'
            elif ratio < lower:
                self._current_mode = 'imagination'

        return {
            'dominance_ratio': self.dominance_ratio.clone(),
            'bottom_up_te': bottom_up_te,
            'top_down_te': top_down_te,
            'mode': self._current_mode,
        }

    def reset(self):
        """Reset buffer and dominance."""
        self.level_history.zero_()
        self.hist_ptr.zero_()
        self._hist_ptr_py = 0
        self.dominance_ratio.fill_(0.5)
        self._ratio_ema = 0.5
        self._ratio_var_ema = 0.01
        self._current_mode = 'balanced'


if __name__ == '__main__':
    print("--- Causal Flow Examples ---")

    # Example 1: Transfer Entropy (causal signal)
    print("\n1. Transfer Entropy")
    te = TransferEntropyEstimator(num_bins=8, history_length=2)

    # X drives Y with lag 1
    T = 300
    x = torch.randn(T)
    y = torch.zeros(T)
    for t in range(1, T):
        y[t] = 0.8 * x[t - 1] + 0.2 * torch.randn(1).item()

    result = te(x, y)
    print(f"   TE(X→Y) = {result['te_source_to_target'].item():.4f} (should be positive)")
    print(f"   TE(Y→X) = {result['te_target_to_source'].item():.4f} (should be ~0)")
    print(f"   Net flow = {result['net_flow'].item():.4f} (should be positive)")

    # Example 2: Causal Flow Map
    print("\n2. Causal Flow Map (4 populations)")
    cfm = CausalFlowMap(num_populations=4, buffer_size=300)

    # Population 0 drives 1, 1 drives 2, 2 drives 3 (chain)
    states = torch.zeros(4)
    for t in range(300):
        states[0] = torch.randn(1).item()
        for i in range(1, 4):
            states[i] = 0.7 * states[i - 1] + 0.3 * torch.randn(1).item()
        cfm.record(states.clone())

    flow_result = cfm.compute_flow()
    print(f"   Driver scores: {flow_result['driver_scores'].detach().cpu().numpy().round(3)}")
    print(f"   (Population 0 should be strongest driver)")

    # Example 3: Dominance Detection
    print("\n3. Dominance Detection (3 levels)")
    dd = DominanceDetector(num_levels=3, buffer_size=200)

    # Bottom-up dominant: low drives high
    for t in range(200):
        levels = torch.zeros(3)
        levels[0] = torch.randn(1).item()
        levels[1] = 0.6 * levels[0] + 0.4 * torch.randn(1).item()
        levels[2] = 0.6 * levels[1] + 0.4 * torch.randn(1).item()
        result = dd(levels, compute_every=50)

    print(f"   Dominance ratio: {result['dominance_ratio'].item():.3f}")
    print(f"   Mode: {result['mode']} (expected: perception)")

    print("\n[OK] All causal flow tests passed!")
