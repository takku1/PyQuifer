"""
Criticality monitoring — detection and measurement of critical dynamics.

Contains passive monitors that measure criticality metrics without
actively adjusting system parameters:

- phase_activity_to_spikes: Convert oscillator phases to spike events
- AvalancheDetector: Detect and analyze power-law avalanches
- BranchingRatio: Compute branching ratio (sigma) from activity
- KuramotoCriticalityMonitor: Criticality via synchronization susceptibility
- KoopmanBifurcationDetector: Bifurcation detection via DMD eigenvalues
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict


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
        # Python-side counter for fast modular checks without GPU sync
        self._hist_ptr_py: int = 0

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
            idx = self._hist_ptr_py % self.window_size
            self.R_history[idx] = R.detach()
            self.hist_ptr.fill_(self._hist_ptr_py + 1)
            self._hist_ptr_py += 1

        n_valid = min(self._hist_ptr_py, self.window_size)

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
        self._hist_ptr_py = 0


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
