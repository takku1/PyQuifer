"""
Integrated Information Theory (IIT) Metrics Module for PyQuifer

Implements rigorous IIT-inspired metrics for measuring integrated information
and consciousness-like properties in neural systems.

Key concepts from IIT 4.0:
- Phi (Φ): Integrated information - how much the whole exceeds its parts
- Cause-Effect Structure: The set of all distinctions a system makes
- Intrinsic Information: Information from the system's own perspective
- Earth Mover's Distance: Transport cost between probability distributions

These metrics can be used to:
1. Quantify how "integrated" a neural system's representations are
2. Detect when information is truly unified vs. merely aggregated
3. Measure the richness of a system's cause-effect repertoire

Based on work by Tononi, Koch, Oizumi, Albantakis (pyphi implementation).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass, field


@dataclass
class Concept:
    """
    A concept in IIT: a mechanism with its maximally irreducible cause and effect.

    Each concept represents a specific distinction the system makes — a way
    it constrains past and future states.
    """
    mechanism: Tuple[int, ...]           # Indices of nodes in this mechanism
    cause_purview: Tuple[int, ...]       # Nodes constrained in past
    effect_purview: Tuple[int, ...]      # Nodes constrained in future
    phi: float                           # Integrated information (scalar)
    cause_phi: float = 0.0              # MIC phi (cause irreducibility)
    effect_phi: float = 0.0             # MIE phi (effect irreducibility)


@dataclass
class SystemIrreducibilityAnalysis:
    """
    Result of a system-level Big Phi computation.

    Big Phi measures how irreducible the system's entire cause-effect structure is.
    """
    big_phi: float                       # System-level integrated information
    concepts: List[Concept] = field(default_factory=list)
    min_cut: Optional[Tuple[int, ...]] = None  # The MIP cut location
    num_concepts: int = 0


def generate_bipartitions(n: int, device: torch.device = None) -> torch.Tensor:
    """
    Generate all valid bipartitions of n elements.

    For n elements, there are 2^(n-1) - 1 valid bipartitions
    (excluding empty/full and duplicates via symmetry).

    Args:
        n: Number of elements to partition.
        device: Torch device.

    Returns:
        Boolean tensor (num_partitions, n) where True = part A.
    """
    partitions = []
    # Iterate from 1 to 2^(n-1) - 1 (half the total, avoiding symmetry duplicates)
    for mask in range(1, 2 ** (n - 1)):
        assignment = torch.zeros(n, dtype=torch.bool, device=device)
        for bit in range(n):
            assignment[bit] = bool((mask >> bit) & 1)
        partitions.append(assignment)
    return torch.stack(partitions)


def hamming_distance_matrix(n_bits: int, device: torch.device = None) -> torch.Tensor:
    """
    Create a matrix of Hamming distances between all binary states.

    For n_bits nodes, creates a 2^n x 2^n matrix where entry (i,j)
    is the Hamming distance between states i and j.

    Args:
        n_bits: Number of binary nodes
        device: Torch device

    Returns:
        Hamming distance matrix (2^n, 2^n)
    """
    n_states = 2 ** n_bits

    # Generate all possible states as binary vectors
    states = torch.zeros(n_states, n_bits, device=device)
    for i in range(n_states):
        for bit in range(n_bits):
            states[i, bit] = (i >> bit) & 1

    # Compute pairwise Hamming distances
    # |a - b|^2 for binary = Hamming distance
    distances = torch.cdist(states, states, p=0)  # L0 = Hamming

    return distances


class EarthMoverDistance(nn.Module):
    """
    Computes Earth Mover's Distance (EMD) between probability distributions.

    EMD measures the minimum "work" needed to transform one distribution
    into another, where work = mass * distance. In IIT, this is used
    with Hamming distance as the ground metric.

    For small distributions, uses the Sinkhorn algorithm for differentiability.
    """

    def __init__(self,
                 n_states: int,
                 use_hamming: bool = True,
                 sinkhorn_iterations: int = 50,
                 sinkhorn_epsilon: float = 0.1):
        """
        Args:
            n_states: Number of possible states (2^n for n binary nodes)
            use_hamming: Use Hamming distance as ground metric
            sinkhorn_iterations: Iterations for Sinkhorn algorithm
            sinkhorn_epsilon: Regularization for Sinkhorn
        """
        super().__init__()
        self.n_states = n_states
        self.sinkhorn_iterations = sinkhorn_iterations
        self.sinkhorn_epsilon = sinkhorn_epsilon

        # Precompute cost matrix
        if use_hamming:
            n_bits = int(math.log2(n_states))
            cost_matrix = hamming_distance_matrix(n_bits)
        else:
            # Simple index distance
            idx = torch.arange(n_states).float()
            cost_matrix = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))

        self.register_buffer('cost_matrix', cost_matrix)

    def sinkhorn(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        Sinkhorn algorithm for differentiable optimal transport.

        Args:
            p: Source distribution (batch, n_states)
            q: Target distribution (batch, n_states)

        Returns:
            EMD approximation (batch,)
        """
        # Add small epsilon for numerical stability
        p = p + 1e-8
        q = q + 1e-8
        p = p / p.sum(dim=-1, keepdim=True)
        q = q / q.sum(dim=-1, keepdim=True)

        # Kernel matrix
        K = torch.exp(-self.cost_matrix / self.sinkhorn_epsilon)

        # Sinkhorn iterations
        u = torch.ones_like(p)
        for _ in range(self.sinkhorn_iterations):
            v = q / (K.T @ u.unsqueeze(-1)).squeeze(-1)
            u = p / (K @ v.unsqueeze(-1)).squeeze(-1)

        # Transport plan
        P = u.unsqueeze(-1) * K * v.unsqueeze(-2)

        # EMD = sum of (transport * cost)
        emd = (P * self.cost_matrix).sum(dim=(-1, -2))

        return emd

    def forward(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        Compute EMD between two distributions.

        Args:
            p: First distribution (batch, n_states) or (n_states,)
            q: Second distribution (batch, n_states) or (n_states,)

        Returns:
            EMD values (batch,) or scalar
        """
        squeeze_output = False
        if p.dim() == 1:
            p = p.unsqueeze(0)
            q = q.unsqueeze(0)
            squeeze_output = True

        result = self.sinkhorn(p, q)

        if squeeze_output:
            result = result.squeeze(0)

        return result


class InformationDensity(nn.Module):
    """
    Computes information density (element-wise KL divergence) between distributions.

    Information density at each state shows how much that state contributes
    to distinguishing the two distributions.
    """

    def __init__(self, epsilon: float = 1e-10):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        Compute information density p * log(p/q) at each state.

        Args:
            p: First distribution
            q: Second distribution

        Returns:
            Information density (same shape as input)
        """
        p_safe = torch.clamp(p, min=self.epsilon)
        q_safe = torch.clamp(q, min=self.epsilon)

        return p_safe * torch.log2(p_safe / q_safe)


class KLDivergence(nn.Module):
    """
    Kullback-Leibler Divergence between distributions.

    KL(P||Q) measures how much information is lost when Q is used
    to approximate P. Asymmetric - not a true distance metric.
    """

    def __init__(self, epsilon: float = 1e-10):
        super().__init__()
        self.info_density = InformationDensity(epsilon)

    def forward(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        Compute KL(p || q).

        Args:
            p: First distribution
            q: Second distribution

        Returns:
            KL divergence (scalar or batch)
        """
        density = self.info_density(p, q)

        # Sum over state dimension
        if density.dim() > 1:
            return density.sum(dim=-1)
        return density.sum()


class L1Distance(nn.Module):
    """Simple L1 distance between distributions."""

    def forward(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        Compute L1 distance (sum of absolute differences).
        """
        diff = torch.abs(p - q)
        if diff.dim() > 1:
            return diff.sum(dim=-1)
        return diff.sum()


class PartitionedInformation(nn.Module):
    """
    Computes information loss when a system is partitioned.

    This is the core of Phi computation: comparing the whole system's
    information to the sum of its parts. True integration means the
    whole carries more information than the partitioned version.

    For n <= 12: enumerates ALL 2^(n-1)-1 bipartitions (exact MIP).
    For n > 12: uses random sampling with heuristic refinement.
    """

    # Threshold for exhaustive enumeration
    EXHAUSTIVE_THRESHOLD = 12

    def __init__(self,
                 state_dim: int,
                 num_partitions: int = 8,
                 distance_metric: str = 'emd'):
        """
        Args:
            state_dim: Dimension of state vectors
            num_partitions: Number of random bipartitions to try (only used if state_dim > 12)
            distance_metric: 'emd', 'kl', or 'l1'
        """
        super().__init__()
        self.state_dim = state_dim

        if state_dim <= self.EXHAUSTIVE_THRESHOLD:
            # Exhaustive: enumerate all 2^(n-1)-1 bipartitions
            partitions = generate_bipartitions(state_dim)
            self.num_partitions = partitions.shape[0]
            self.exhaustive = True
        else:
            # Sampling: use max(num_partitions, 64) random partitions
            self.num_partitions = max(num_partitions, 64)
            self.exhaustive = False
            partitions = []
            for _ in range(self.num_partitions):
                assignment = torch.rand(state_dim) > 0.5
                if assignment.sum() == 0:
                    assignment[0] = True
                if assignment.sum() == state_dim:
                    assignment[0] = False
                partitions.append(assignment)
            partitions = torch.stack(partitions)

        self.register_buffer('partitions', partitions)

        # Distance metric
        if distance_metric == 'emd':
            # For EMD, we need to discretize - use 16 states
            self.distance = EarthMoverDistance(16, use_hamming=False)
            self.discretize = True
        elif distance_metric == 'kl':
            self.distance = KLDivergence()
            self.discretize = False
        else:
            self.distance = L1Distance()
            self.discretize = False

    def _to_distribution(self, x: torch.Tensor, n_bins: int = 16) -> torch.Tensor:
        """Convert continuous values to discrete distribution via histogram."""
        batch_size = x.shape[0] if x.dim() > 1 else 1
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Normalize to [0, 1]
        x_norm = (x - x.min(dim=-1, keepdim=True)[0]) / (
            x.max(dim=-1, keepdim=True)[0] - x.min(dim=-1, keepdim=True)[0] + 1e-8
        )

        # Create histogram
        bins = torch.linspace(0, 1, n_bins + 1, device=x.device)
        hist = torch.zeros(batch_size, n_bins, device=x.device)

        for i in range(n_bins):
            mask = (x_norm >= bins[i]) & (x_norm < bins[i+1])
            hist[:, i] = mask.float().sum(dim=-1)

        # Handle edge case for x == 1
        mask = x_norm >= bins[-1]
        hist[:, -1] += mask.float().sum(dim=-1)

        # Normalize to probability
        hist = hist / (hist.sum(dim=-1, keepdim=True) + 1e-8)

        return hist

    def forward(self,
                state: torch.Tensor,
                return_min_partition: bool = False
                ) -> Dict[str, torch.Tensor]:
        """
        Compute partitioned information for a state.

        Args:
            state: State vector (batch, state_dim) or (state_dim,)
            return_min_partition: Return the minimum information partition

        Returns:
            Dictionary with:
            - phi: Integrated information (minimum over partitions)
            - partition_losses: Information loss for each partition
            - min_partition_idx: Index of minimum partition (if requested)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        batch_size = state.shape[0]

        # Compute whole system distribution
        if self.discretize:
            whole_dist = self._to_distribution(state)
        else:
            whole_dist = F.softmax(state, dim=-1)

        # Entropy helper
        def _entropy(p):
            return -(p * torch.log(p + 1e-10)).sum(dim=-1)

        H_whole = _entropy(whole_dist)

        # Compute information loss for each partition via mutual information
        # I(A;B) = H(A) + H(B) - H(AB)
        # High I = parts are correlated = partition destroys information
        partition_losses = []

        for partition in self.partitions:
            # Split state into two parts
            part_a = state[:, partition]
            part_b = state[:, ~partition]

            # Get marginal distributions for each part
            if self.discretize:
                dist_a = self._to_distribution(part_a)
                dist_b = self._to_distribution(part_b)
            else:
                dist_a = F.softmax(part_a, dim=-1)
                dist_b = F.softmax(part_b, dim=-1)

            # Mutual information: if parts are independent, H(A)+H(B) = H(AB)
            # If correlated, H(A)+H(B) > H(AB), so partition loses info
            H_a = _entropy(dist_a)
            H_b = _entropy(dist_b)
            loss = F.relu(H_a + H_b - H_whole)  # relu for numerical stability
            partition_losses.append(loss)

        partition_losses = torch.stack(partition_losses, dim=-1)

        # Phi = minimum information loss over all partitions
        # (the partition that loses least information is the "weakest link")
        phi, min_idx = partition_losses.min(dim=-1)

        result = {
            'phi': phi,
            'partition_losses': partition_losses,
            'min_partition_idx': min_idx,
            'exhaustive': torch.tensor(self.exhaustive, device=state.device),
        }

        if return_min_partition:
            # Also return the actual partition mask for the MIP
            result['min_partition_mask'] = self.partitions[min_idx[0].item()]

        return result


class IntegratedInformation(nn.Module):
    """
    Main Phi computation module.

    Phi measures how much a system is "more than the sum of its parts"
    in terms of information. High Phi = high integration = unified processing.

    This is a differentiable approximation suitable for neural networks.
    """

    def __init__(self,
                 state_dim: int,
                 num_partitions: int = 8,
                 temporal_window: int = 10,
                 distance_metric: str = 'l1'):
        """
        Args:
            state_dim: Dimension of state vectors
            num_partitions: Partitions to test for minimum information partition
            temporal_window: Time steps to consider for temporal Phi
            distance_metric: Distance metric for comparing distributions
        """
        super().__init__()
        self.state_dim = state_dim
        self.temporal_window = temporal_window

        self.partition_info = PartitionedInformation(
            state_dim, num_partitions, distance_metric
        )

        # State history for temporal integration
        self.register_buffer('state_history', torch.zeros(temporal_window, state_dim))
        self.register_buffer('history_ptr', torch.tensor(0))
        self.register_buffer('history_filled', torch.tensor(False))

    def forward(self,
                state: torch.Tensor,
                compute_temporal: bool = True) -> Dict[str, torch.Tensor]:
        """
        Compute integrated information for current state.

        Args:
            state: Current state (batch, state_dim) or (state_dim,)
            compute_temporal: Also compute temporal integration

        Returns:
            Dictionary with:
            - phi: Spatial integrated information
            - temporal_phi: Temporal integration (if compute_temporal)
            - partition_losses: Per-partition information losses
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Spatial Phi
        spatial_result = self.partition_info(state)

        result = {
            'phi': spatial_result['phi'],
            'partition_losses': spatial_result['partition_losses'],
        }

        # Update history
        with torch.no_grad():
            self.state_history[self.history_ptr % self.temporal_window] = state[0]
            self.history_ptr = self.history_ptr + 1
            if self.history_ptr >= self.temporal_window:
                self.history_filled = torch.tensor(True, device=state.device)

        # Temporal Phi (integration across time)
        if compute_temporal and self.history_filled:
            # Compute how well past predicts present
            valid_history = self.state_history

            # Temporal coherence: does the sequence have structure?
            # High temporal Phi = past and present are integrated
            past_mean = valid_history[:-1].mean(dim=0, keepdim=True)
            temporal_result = self.partition_info(
                torch.cat([past_mean, state], dim=0).mean(dim=0, keepdim=True)
            )
            result['temporal_phi'] = temporal_result['phi']
        else:
            result['temporal_phi'] = torch.tensor(0.0, device=state.device)

        return result

    def reset(self):
        """Reset state history."""
        self.state_history.zero_()
        self.history_ptr.zero_()
        self.history_filled.fill_(False)


class CauseEffectRepertoire(nn.Module):
    """
    Computes cause and effect repertoires for IIT analysis.

    In IIT, the cause repertoire P(past | present) specifies which past
    states could have caused the current state, and the effect repertoire
    P(future | present) specifies which future states the current state
    can cause.

    Direction-aware: cause and effect are tracked separately, with
    independent MIC (maximally irreducible cause) and MIE (maximally
    irreducible effect) computation.

    NOTE: Uses learned transition models as approximations.
    """

    def __init__(self,
                 state_dim: int,
                 hidden_dim: int = 64):
        """
        Args:
            state_dim: Dimension of state space
            hidden_dim: Hidden layer dimension for transition models
        """
        super().__init__()
        self.state_dim = state_dim

        # Cause model: P(past | present) — backward
        self.cause_model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
            nn.Softmax(dim=-1)
        )

        # Effect model: P(future | present) — forward
        self.effect_model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
            nn.Softmax(dim=-1)
        )

        self.kl = KLDivergence()

    def cause_repertoire(self, state: torch.Tensor) -> torch.Tensor:
        """Compute cause repertoire: P(past | present)."""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.cause_model(state)

    def effect_repertoire(self, state: torch.Tensor) -> torch.Tensor:
        """Compute effect repertoire: P(future | present)."""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.effect_model(state)

    def _unconstrained_repertoire(self, dim: int, device: torch.device) -> torch.Tensor:
        """Uniform distribution as the unconstrained baseline."""
        return torch.ones(1, dim, device=device) / dim

    def compute_mic(self, state: torch.Tensor,
                    mechanism: Optional[Tuple[int, ...]] = None) -> Dict[str, torch.Tensor]:
        """
        Compute the Maximally Irreducible Cause (MIC) for a mechanism.

        Tests partitions of the mechanism and finds the one where
        partitioning causes the least information loss. The MIC phi
        is this minimum loss — if > 0, the cause is irreducible.

        Args:
            state: Current state (state_dim,) or (1, state_dim)
            mechanism: Indices of nodes in this mechanism. Default: all.

        Returns:
            Dict with cause_phi, cause_repertoire, cause_purview.
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        device = state.device
        if mechanism is None:
            mechanism = tuple(range(self.state_dim))

        # Full cause repertoire (unpartitioned)
        full_cause = self.cause_repertoire(state)

        if len(mechanism) <= 1:
            # Single-node mechanism: phi = distance from unconstrained
            unconstrained = self._unconstrained_repertoire(self.state_dim, device)
            phi = self.kl(full_cause, unconstrained)
            return {
                'cause_phi': phi,
                'cause_repertoire': full_cause,
                'cause_purview': mechanism,
            }

        # Test bipartitions of the mechanism
        n = len(mechanism)
        min_phi = torch.tensor(float('inf'), device=device)

        for mask_val in range(1, 2 ** (n - 1)):
            # Create partitioned state: zero out one part at a time
            state_a = state.clone()
            state_b = state.clone()
            for bit_idx in range(n):
                node = mechanism[bit_idx]
                if (mask_val >> bit_idx) & 1:
                    state_b[:, node] = 0.0
                else:
                    state_a[:, node] = 0.0

            # Partitioned cause = product of marginals (approximated)
            cause_a = self.cause_repertoire(state_a)
            cause_b = self.cause_repertoire(state_b)
            partitioned_cause = (cause_a + cause_b) / 2.0

            # Information loss from partitioning
            loss = self.kl(full_cause, partitioned_cause)
            min_phi = torch.min(min_phi, loss)

        return {
            'cause_phi': min_phi,
            'cause_repertoire': full_cause,
            'cause_purview': mechanism,
        }

    def compute_mie(self, state: torch.Tensor,
                    mechanism: Optional[Tuple[int, ...]] = None) -> Dict[str, torch.Tensor]:
        """
        Compute the Maximally Irreducible Effect (MIE) for a mechanism.

        Analogous to MIC but in the forward (effect) direction.

        Args:
            state: Current state (state_dim,) or (1, state_dim)
            mechanism: Indices of nodes in this mechanism. Default: all.

        Returns:
            Dict with effect_phi, effect_repertoire, effect_purview.
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        device = state.device
        if mechanism is None:
            mechanism = tuple(range(self.state_dim))

        full_effect = self.effect_repertoire(state)

        if len(mechanism) <= 1:
            unconstrained = self._unconstrained_repertoire(self.state_dim, device)
            phi = self.kl(full_effect, unconstrained)
            return {
                'effect_phi': phi,
                'effect_repertoire': full_effect,
                'effect_purview': mechanism,
            }

        n = len(mechanism)
        min_phi = torch.tensor(float('inf'), device=device)

        for mask_val in range(1, 2 ** (n - 1)):
            state_a = state.clone()
            state_b = state.clone()
            for bit_idx in range(n):
                node = mechanism[bit_idx]
                if (mask_val >> bit_idx) & 1:
                    state_b[:, node] = 0.0
                else:
                    state_a[:, node] = 0.0

            effect_a = self.effect_repertoire(state_a)
            effect_b = self.effect_repertoire(state_b)
            partitioned_effect = (effect_a + effect_b) / 2.0

            loss = self.kl(full_effect, partitioned_effect)
            min_phi = torch.min(min_phi, loss)

        return {
            'effect_phi': min_phi,
            'effect_repertoire': full_effect,
            'effect_purview': mechanism,
        }

    def compute_concept(self, state: torch.Tensor,
                        mechanism: Tuple[int, ...]) -> Concept:
        """
        Compute a full Concept for a given mechanism.

        A concept exists only if both MIC and MIE are non-zero.
        The concept's phi is min(cause_phi, effect_phi).

        Args:
            state: Current state
            mechanism: Indices of nodes

        Returns:
            Concept dataclass
        """
        mic = self.compute_mic(state, mechanism)
        mie = self.compute_mie(state, mechanism)

        cause_phi = mic['cause_phi'].item()
        effect_phi = mie['effect_phi'].item()
        phi = min(cause_phi, effect_phi)

        return Concept(
            mechanism=mechanism,
            cause_purview=mic['cause_purview'],
            effect_purview=mie['effect_purview'],
            phi=phi,
            cause_phi=cause_phi,
            effect_phi=effect_phi,
        )

    def forward(self,
                state: torch.Tensor
                ) -> Dict[str, torch.Tensor]:
        """
        Compute cause and effect repertoires.

        Args:
            state: Current state (batch, state_dim) or (state_dim,)

        Returns:
            Dictionary with cause/effect repertoires and asymmetry.
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        cause = self.cause_model(state)
        effect = self.effect_model(state)

        # Asymmetry: how different are cause and effect directions?
        asymmetry = self.kl(cause, effect)

        return {
            'cause_repertoire': cause,
            'effect_repertoire': effect,
            'repertoire_asymmetry': asymmetry,
        }


class IITConsciousnessMonitor(nn.Module):
    """
    Complete IIT-based consciousness monitoring system.

    Combines multiple IIT metrics:
    - Phi (integrated information)
    - Cause-effect structure complexity
    - Temporal integration

    Can be used as an intrinsic reward signal or diagnostic tool.
    """

    def __init__(self,
                 state_dim: int,
                 num_partitions: int = 8,
                 temporal_window: int = 20):
        """
        Args:
            state_dim: Dimension of monitored state
            num_partitions: Partitions for Phi computation
            temporal_window: Window for temporal metrics
        """
        super().__init__()
        self.state_dim = state_dim

        # Core Phi computation
        self.phi_computer = IntegratedInformation(
            state_dim, num_partitions, temporal_window
        )

        # Cause-effect repertoires
        self.ce_repertoire = CauseEffectRepertoire(state_dim)

        # KL for repertoire comparison
        self.kl = KLDivergence()

        # Running statistics
        self.register_buffer('phi_history', torch.zeros(100))
        self.register_buffer('phi_ptr', torch.tensor(0))

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute all consciousness metrics (fast path for real-time use).

        Args:
            state: Current state (batch, state_dim) or (state_dim,)

        Returns:
            Complete metrics dictionary
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Phi
        phi_result = self.phi_computer(state)

        # Cause-effect repertoires (includes asymmetry now)
        ce_result = self.ce_repertoire(state)
        repertoire_asymmetry = ce_result['repertoire_asymmetry']

        # Combined consciousness index
        consciousness = (
            0.5 * torch.tanh(phi_result['phi']) +
            0.3 * torch.tanh(phi_result.get('temporal_phi', torch.tensor(0.0, device=state.device))) +
            0.2 * torch.tanh(repertoire_asymmetry)
        )

        # Update history
        with torch.no_grad():
            self.phi_history[self.phi_ptr % 100] = phi_result['phi'].mean()
            self.phi_ptr = self.phi_ptr + 1

        return {
            'phi': phi_result['phi'],
            'temporal_phi': phi_result.get('temporal_phi', torch.tensor(0.0, device=state.device)),
            'cause_repertoire': ce_result['cause_repertoire'],
            'effect_repertoire': ce_result['effect_repertoire'],
            'repertoire_asymmetry': repertoire_asymmetry,
            'consciousness_index': consciousness,
        }

    def compute_big_phi(self, state: torch.Tensor,
                        max_mechanism_size: int = 3) -> SystemIrreducibilityAnalysis:
        """
        Compute system-level Big Phi.

        Big Phi measures how irreducible the system's entire cause-effect
        structure (CES) is. This is expensive: it enumerates mechanisms up
        to max_mechanism_size and computes concepts for each.

        For real-time use, call forward() instead. Use this for
        detailed analysis or diagnostics.

        Args:
            state: Current state (state_dim,)
            max_mechanism_size: Largest mechanism to enumerate (default 3
                               to keep computation tractable)

        Returns:
            SystemIrreducibilityAnalysis with Big Phi and concept list
        """
        if state.dim() == 2:
            state = state[0]  # Use first batch element

        device = state.device
        n = min(self.state_dim, 12)  # Cap for tractability

        # Enumerate all mechanisms of size 1..max_mechanism_size
        concepts = []
        from itertools import combinations
        for size in range(1, min(max_mechanism_size, n) + 1):
            for mechanism in combinations(range(n), size):
                concept = self.ce_repertoire.compute_concept(state, mechanism)
                if concept.phi > 1e-6:  # Only keep non-trivial concepts
                    concepts.append(concept)

        if not concepts:
            return SystemIrreducibilityAnalysis(
                big_phi=0.0, concepts=[], num_concepts=0
            )

        # System Phi from the whole system partition
        phi_result = self.phi_computer(state.unsqueeze(0))
        system_phi = phi_result['phi'].item()

        # CES richness: sum of all concept phi values
        ces_sum = sum(c.phi for c in concepts)

        # Big Phi = system-level integration * CES richness
        # Normalized by number of possible mechanisms to keep scale manageable
        n_possible = sum(
            math.comb(n, k)
            for k in range(1, min(max_mechanism_size, n) + 1)
        )
        big_phi = system_phi * (ces_sum / max(n_possible, 1))

        # Find MIP: the partition that would destroy the most concepts
        min_cut = phi_result.get('min_partition_idx', None)
        min_cut_tuple = None
        if min_cut is not None:
            idx = min_cut.item() if hasattr(min_cut, 'item') else int(min_cut)
            min_cut_tuple = (idx,)

        return SystemIrreducibilityAnalysis(
            big_phi=big_phi,
            concepts=sorted(concepts, key=lambda c: c.phi, reverse=True),
            min_cut=min_cut_tuple,
            num_concepts=len(concepts),
        )

    def get_phi_trend(self) -> torch.Tensor:
        """Get recent Phi history for trend analysis."""
        n_valid = min(self.phi_ptr.item(), 100)
        if n_valid < 2:
            return torch.tensor(0.0, device=self.phi_history.device)
        return self.phi_history[:n_valid]

    def reset(self):
        """Reset all state."""
        self.phi_computer.reset()
        self.phi_history.zero_()
        self.phi_ptr.zero_()


if __name__ == '__main__':
    print("--- IIT Metrics Examples ---")

    # Example 1: Hamming Distance Matrix
    print("\n1. Hamming Distance Matrix (3 bits)")
    H = hamming_distance_matrix(3)
    print(f"   Shape: {H.shape}")
    print(f"   Distance(000, 111) = {H[0, 7].item()}")  # Should be 3
    print(f"   Distance(000, 001) = {H[0, 1].item()}")  # Should be 1

    # Example 2: Earth Mover's Distance
    print("\n2. Earth Mover's Distance")
    emd = EarthMoverDistance(8, use_hamming=True)

    # Two identical distributions
    p1 = torch.tensor([0.5, 0.5, 0, 0, 0, 0, 0, 0])
    q1 = torch.tensor([0.5, 0.5, 0, 0, 0, 0, 0, 0])
    print(f"   EMD(same) = {emd(p1, q1).item():.4f}")

    # Very different distributions
    p2 = torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0])
    q2 = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1.0])
    print(f"   EMD(opposite) = {emd(p2, q2).item():.4f}")

    # Example 3: Integrated Information
    print("\n3. Integrated Information (Phi)")
    phi_computer = IntegratedInformation(state_dim=16, num_partitions=8)

    # Random state
    state = torch.randn(16)
    result = phi_computer(state)
    print(f"   Phi = {result['phi'].item():.4f}")

    # Highly structured state (repeated pattern)
    structured = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0.0])
    result2 = phi_computer(structured)
    print(f"   Phi (structured) = {result2['phi'].item():.4f}")

    # Example 4: IIT Consciousness Monitor
    print("\n4. IIT Consciousness Monitor")
    monitor = IITConsciousnessMonitor(state_dim=32)

    # Simulate some states
    for i in range(25):
        state = torch.randn(32)
        metrics = monitor(state)

        if i % 5 == 0:
            print(f"   Step {i}: phi={metrics['phi'].item():.3f}, "
                  f"consciousness={metrics['consciousness_index'].item():.3f}")

    print("\n   Phi trend (recent):",
          monitor.get_phi_trend()[-5:].tolist() if monitor.phi_ptr > 5 else "N/A")
