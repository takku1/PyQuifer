"""Mean-field reduction and phase topology caching.

- KuramotoDaidoMeanField: Ott-Antonsen mean-field reduction of the Kuramoto
  model, tracking the complex order parameter Z directly in O(1) per step.
- PhaseTopologyCache: Rotation-invariant cache for phase topology patterns
  with Bayesian outcome priors.
"""
import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn


class KuramotoDaidoMeanField(nn.Module):
    """
    Mean-field reduction of the Kuramoto model via the Ott-Antonsen ansatz.

    Instead of tracking N individual oscillator phases (O(N^2) coupling),
    tracks the complex order parameter Z directly (O(1) per step):

        dZ/dt = (-i*w_mean - Delta_eff + K/2) * Z - K/2 * |Z|^2 * Z

    Where:
    - Z = R * exp(i*Psi) is the complex order parameter
    - w_mean is the mean natural frequency
    - Delta_eff is the effective frequency spread
    - K is coupling strength

    For Cauchy/Lorentzian distributed frequencies, the Ott-Antonsen
    ansatz is exact (Delta_eff = delta, the Cauchy half-width).

    For Gaussian distributed frequencies, a scaled approximation
    (Montbrio, Pazo & Roxin 2015) is used:
    Delta_eff = delta * sqrt(2*pi) / 2, where delta is the standard
    deviation. This gives much better predictions when frequencies
    are sampled from Uniform or Gaussian distributions.

    Args:
        omega_mean: Mean natural frequency
        delta: Frequency spread (Cauchy half-width or Gaussian std)
        coupling: Coupling strength K
        dt: Integration timestep
        distribution: 'cauchy' (exact Ott-Antonsen) or 'gaussian'
                      (Montbrio-Pazo-Roxin scaled approximation)
    """

    def __init__(self,
                 omega_mean: float = 1.0,
                 delta: float = 0.1,
                 coupling: float = 1.0,
                 dt: float = 0.01,
                 distribution: str = 'cauchy'):
        super().__init__()
        self.dt = dt
        self.distribution = distribution

        # Learnable dynamics parameters
        self.omega_mean = nn.Parameter(torch.tensor(omega_mean))
        self.coupling = nn.Parameter(torch.tensor(coupling))

        # Delta (spread) as buffer — not typically learned
        self.register_buffer('delta', torch.tensor(delta))

        # Complex order parameter Z = R * exp(i*Psi)
        # Store as (real, imag) pair for buffer compatibility
        self.register_buffer('Z_real', torch.tensor(0.1))
        self.register_buffer('Z_imag', torch.tensor(0.0))

        # Step counter
        self.register_buffer('step_count', torch.tensor(0))

    @property
    def Z(self) -> torch.Tensor:
        """Complex order parameter."""
        return torch.complex(self.Z_real, self.Z_imag)

    @property
    def R(self) -> torch.Tensor:
        """Order parameter magnitude (synchronization level)."""
        return torch.abs(self.Z)

    @property
    def Psi(self) -> torch.Tensor:
        """Order parameter phase (mean phase)."""
        return torch.angle(self.Z)

    def forward(self, steps: int = 1,
                external_field: Optional[torch.Tensor] = None) -> Dict:
        """
        Evolve the mean-field ODE for given number of steps.

        Args:
            steps: Number of integration steps
            external_field: Optional complex driving field

        Returns:
            Dict with R (magnitude), Psi (phase), Z (complex)
        """
        Z = self.Z

        # Effective spread: for Gaussian, scale delta by sqrt(2*pi)/2
        # (Montbrio, Pazo & Roxin 2015 QIF mean-field reduction)
        if self.distribution == 'gaussian':
            delta_eff = self.delta * (math.sqrt(2 * math.pi) / 2)
        else:
            delta_eff = self.delta

        for _ in range(steps):
            K = self.coupling
            w = self.omega_mean

            # Ott-Antonsen mean-field ODE
            # dZ/dt = (-i*w - Delta_eff + K/2) * Z - K/2 * |Z|^2 * Z
            Z_sq_mag = (Z.real ** 2 + Z.imag ** 2)
            linear_coeff = torch.complex(-delta_eff + K / 2, -w)
            dZ = linear_coeff * Z - (K / 2) * Z_sq_mag * Z

            if external_field is not None:
                dZ = dZ + external_field

            Z = Z + dZ * self.dt

        # Update stored state
        with torch.no_grad():
            self.Z_real.copy_(Z.real.detach())
            self.Z_imag.copy_(Z.imag.detach())
            self.step_count.add_(steps)

        R = torch.abs(Z)
        Psi = torch.angle(Z)

        return {
            'R': R,
            'Psi': Psi,
            'Z': Z,
        }

    @classmethod
    def from_frequencies(cls, omega_samples: torch.Tensor,
                         coupling: float = 1.0, dt: float = 0.01,
                         distribution: str = 'auto') -> 'KuramotoDaidoMeanField':
        """Construct mean-field model by fitting distribution to frequency samples.

        For Cauchy: uses IQR-based robust fitting (Pietras & Daffertshofer 2016):
        - omega_mean = median(samples)
        - delta = IQR / 2  (half-width at half-maximum of Cauchy)

        For Gaussian: uses mean and standard deviation directly:
        - omega_mean = mean(samples)
        - delta = std(samples)

        With distribution='auto', uses excess kurtosis to choose:
        - kurtosis < 6 → Gaussian (normal kurtosis = 3, uniform = 1.8)
        - kurtosis >= 6 → Cauchy (theoretical kurtosis = infinity)

        Args:
            omega_samples: Tensor of sampled natural frequencies (N,)
            coupling: Coupling strength K
            dt: Integration timestep
            distribution: 'auto', 'cauchy', or 'gaussian'

        Returns:
            KuramotoDaidoMeanField instance with fitted parameters
        """
        if distribution == 'auto':
            # Excess kurtosis test: Cauchy has very heavy tails
            mean_val = omega_samples.mean()
            std_val = omega_samples.std()
            if std_val > 1e-8:
                centered = omega_samples - mean_val
                kurtosis = (centered ** 4).mean() / (std_val ** 4)
            else:
                kurtosis = 3.0  # Default to Gaussian
            distribution = 'gaussian' if kurtosis < 6.0 else 'cauchy'

        if distribution == 'gaussian':
            omega_mean = omega_samples.mean().item()
            delta = max(omega_samples.std().item(), 1e-6)
        else:
            omega_sorted = omega_samples.sort().values
            n = len(omega_sorted)
            q25 = omega_sorted[max(0, int(n * 0.25))].item()
            q75 = omega_sorted[min(n - 1, int(n * 0.75))].item()
            omega_mean = omega_sorted[n // 2].item()  # median
            delta = max((q75 - q25) / 2.0, 1e-6)  # IQR-based Cauchy half-width

        return cls(omega_mean=omega_mean, delta=delta, coupling=coupling,
                   dt=dt, distribution=distribution)

    def get_order_parameter(self) -> torch.Tensor:
        """Get current synchronization level R."""
        return self.R

    def get_critical_coupling(self) -> torch.Tensor:
        """Critical coupling for onset of synchronization.

        Cauchy: Kc = 2 * delta
        Gaussian: Kc = 2 * delta * sqrt(2*pi) / pi  (Montbrio et al. 2015)
        """
        if self.distribution == 'gaussian':
            return 2.0 * self.delta * math.sqrt(2 * math.pi) / math.pi
        return 2.0 * self.delta

    def is_synchronized(self, threshold: float = 0.5) -> bool:
        """Check if population is synchronized."""
        return self.R.item() > threshold

    def reset(self):
        """Reset order parameter to low-sync state."""
        self.Z_real.fill_(0.1)
        self.Z_imag.fill_(0.0)
        self.step_count.zero_()


class PhaseTopologyCache:
    """
    Cache for phase topology patterns with outcome labels.

    Stores hashes of relative phase patterns (rotation-invariant)
    with observed outcome classes. Acts as a Bayesian prior for
    confidence estimation — NOT a lookup table.

    Plain class (not nn.Module) — no learnable parameters.

    Args:
        capacity: Maximum number of cached entries
        hash_bins: Number of quantization bins for phase differences
    """

    def __init__(self, capacity: int = 1000, hash_bins: int = 64):
        self.capacity = capacity
        self.hash_bins = hash_bins
        self._store: Dict[int, Dict] = {}  # hash -> {outcome, confidence, count}
        self._access_order: List[int] = []  # LRU eviction

    def compute_hash(self, phases: torch.Tensor) -> int:
        """
        Rotation-invariant hash of relative phase pattern.

        Sorts phase differences (removes rotation ambiguity),
        quantizes into bins, then hashes.
        """
        with torch.no_grad():
            # Compute pairwise phase differences (rotation-invariant)
            diffs = (phases.unsqueeze(0) - phases.unsqueeze(1)) % (2 * math.pi)
            # Take upper triangle to avoid redundancy
            n = phases.shape[0]
            upper_diffs = []
            for i in range(n):
                for j in range(i + 1, n):
                    upper_diffs.append(diffs[i, j].item())
            if not upper_diffs:
                return 0
            # Sort for permutation invariance
            upper_diffs.sort()
            # Quantize into bins
            binned = tuple(
                int(d / (2 * math.pi) * self.hash_bins) % self.hash_bins
                for d in upper_diffs
            )
            return hash(binned)

    def store(self, phases: torch.Tensor, outcome: str, confidence: float):
        """Store an observed outcome for a phase topology."""
        h = self.compute_hash(phases)

        if h in self._store:
            entry = self._store[h]
            # Bayesian update: weighted running average
            old_count = entry['count']
            entry['confidence'] = (entry['confidence'] * old_count + confidence) / (old_count + 1)
            entry['count'] += 1
            # If outcome differs, keep the majority
            if outcome != entry['outcome'] and confidence > entry['confidence']:
                entry['outcome'] = outcome
        else:
            # Evict if at capacity
            if len(self._store) >= self.capacity:
                oldest = self._access_order.pop(0)
                self._store.pop(oldest, None)
            self._store[h] = {'outcome': outcome, 'confidence': confidence, 'count': 1}

        # Update access order
        if h in self._access_order:
            self._access_order.remove(h)
        self._access_order.append(h)

    def query(self, phases: torch.Tensor) -> Optional[Dict]:
        """Returns {outcome, confidence, count} if similar pattern seen, else None."""
        h = self.compute_hash(phases)
        entry = self._store.get(h)
        if entry is not None:
            # Update LRU
            if h in self._access_order:
                self._access_order.remove(h)
            self._access_order.append(h)
            return dict(entry)
        return None

    def get_prior(self, phases: torch.Tensor, hypothesis: str) -> float:
        """
        Bayesian prior: P(hypothesis | similar_topology_seen_before).

        Returns stored confidence weighted by count if hash matches
        and outcome matches hypothesis; else 0.5 (uninformative).
        """
        entry = self.query(phases)
        if entry is None:
            return 0.5
        if entry['outcome'] == hypothesis:
            # Confidence grows with evidence: cap at 0.95
            count_weight = min(1.0, entry['count'] / 10.0)
            return 0.5 + (entry['confidence'] - 0.5) * count_weight
        else:
            # Evidence against this hypothesis
            count_weight = min(1.0, entry['count'] / 10.0)
            return 0.5 - (entry['confidence'] - 0.5) * count_weight * 0.5
