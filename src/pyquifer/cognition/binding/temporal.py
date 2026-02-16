"""
Oscillatory Temporal Binding — AKOrN adapted for 1D sequences.

Extends the visual binding paradigm to text sequences: tokens that
"belong together" synchronize their oscillator phases, forming concept
clusters via Kuramoto dynamics with attention-based coupling.

Key classes:
- SequenceAKOrN: Multi-head Kuramoto layer for token sequences
- PhaseGrouping: Group tokens by synchronization into concept clusters
- OscillatoryChunking: Hierarchical chunking via nested oscillation

References:
- Ramsauer et al. (2025). AKOrN: Attentive Kuramoto Oscillator Networks. ICLR 2025.
- von der Malsburg (1981). The correlation theory of brain function.
- Lisman & Jensen (2013). The theta-gamma neural code. Neuron.
"""

import torch
import torch.nn as nn
import math
from typing import Any, Dict, List, Optional, Tuple


class SequenceAKOrN(nn.Module):
    """AKOrN adapted for 1D token sequences.

    Each token position gets oscillator phases. Coupling J is computed
    from attention: J_ij = softmax(q_i * k_j / sqrt(d)). Phase dynamics
    follow Kuramoto: dphi_i/dt = omega_i + sum_j J_ij sin(phi_j - phi_i).

    After several synchronization steps, in-phase tokens are "bound"
    together as belonging to the same concept.

    Args:
        dim: Token embedding dimension.
        num_heads: Number of attention heads (each with separate coupling).
        num_oscillators_per_head: Oscillators per head per token position.
        dt: Integration timestep.
        num_steps: Number of Kuramoto synchronization steps.
        causal: If True, tokens can only couple with past tokens.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        num_oscillators_per_head: int = 1,
        dt: float = 0.1,
        num_steps: int = 5,
        causal: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_osc = num_oscillators_per_head
        self.dt = dt
        self.num_steps = num_steps
        self.causal = causal
        self.head_dim = dim // num_heads

        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        # Q, K for coupling computation
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        # Value projection
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # Learnable natural frequencies per head
        self.omega = nn.Parameter(
            torch.randn(num_heads, num_oscillators_per_head) * 0.1
        )

    def _compute_coupling(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """Compute attention-based coupling matrix.

        Args:
            x: (B, T, D) input features

        Returns:
            (B, H, T, T) coupling matrix J
        """
        B, T, D = x.shape
        H = self.num_heads
        d_k = self.head_dim

        q = self.q_proj(x).view(B, T, H, d_k).transpose(1, 2)  # (B,H,T,d_k)
        k = self.k_proj(x).view(B, T, H, d_k).transpose(1, 2)

        # Scaled dot-product attention as coupling
        J = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # (B,H,T,T)

        if self.causal:
            mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
            )
            J = J.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        J = torch.softmax(J, dim=-1)
        return J

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run Kuramoto synchronization on token sequence.

        Args:
            x: (B, T, D) token embeddings

        Returns:
            (B, T, D) phase-modulated token embeddings
        """
        B, T, D = x.shape
        H = self.num_heads
        device = x.device

        # Compute coupling from attention
        J = self._compute_coupling(x)  # (B, H, T, T)

        # Initialize phases from input structure
        # Use angle of projected features as initial phase
        v = self.v_proj(x).view(B, T, H, self.head_dim).transpose(1, 2)  # (B,H,T,d)
        phases = torch.atan2(
            v[..., :self.num_osc].sum(dim=-1),
            v[..., self.num_osc:2*self.num_osc].sum(dim=-1) + 1e-8,
        )  # (B, H, T)

        # Run Kuramoto dynamics
        omega = self.omega.unsqueeze(0).unsqueeze(2)  # (1, H, 1, O)
        omega = omega.expand(B, H, T, -1).squeeze(-1)  # (B, H, T)

        for _ in range(self.num_steps):
            # Phase differences: (B, H, T, T)
            diff = phases.unsqueeze(-1) - phases.unsqueeze(-2)
            # Coupling forces: sum_j J_ij sin(phi_j - phi_i)
            coupling = (J * torch.sin(diff)).sum(dim=-1)  # (B, H, T)
            # Euler step
            phases = phases + self.dt * (omega + coupling)
            phases = phases % (2 * math.pi)

        # Modulate values by phases
        phase_mod = torch.cos(phases).unsqueeze(-1)  # (B, H, T, 1)
        v_modulated = v * phase_mod
        # Merge heads
        out = v_modulated.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)

        return x + out  # Residual connection


class PhaseGrouping(nn.Module):
    """Group tokens by oscillator phase synchronization.

    Tokens with similar phases (within threshold) are assigned to the
    same group, forming "concept clusters." This is the temporal analog
    of object segmentation via synchronization.

    Args:
        dim: Feature dimension.
        max_groups: Maximum number of groups to form.
        threshold: Phase difference threshold for grouping (radians).
    """

    def __init__(
        self,
        dim: int,
        max_groups: int = 16,
        threshold: float = 0.5,
    ):
        super().__init__()
        self.dim = dim
        self.max_groups = max_groups
        self.threshold = threshold
        # Project features to phase
        self.phase_proj = nn.Linear(dim, 1, bias=False)

    def forward(
        self,
        features: torch.Tensor,
        phases: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Group tokens by phase synchronization.

        Args:
            features: (B, T, D) token features
            phases: Optional (B, T) pre-computed phases. If None, derived from features.

        Returns:
            Dict with:
            - group_ids: (B, T) group assignment per token
            - group_features: (B, max_groups, D) pooled features per group
            - num_groups: (B,) number of groups found
            - coherence: (B,) mean within-group phase coherence
        """
        B, T, D = features.shape
        device = features.device

        if phases is None:
            phases = self.phase_proj(features).squeeze(-1)  # (B, T)
            phases = phases % (2 * math.pi)

        group_ids = torch.full((B, T), -1, device=device, dtype=torch.long)
        group_features = torch.zeros(B, self.max_groups, D, device=device)
        num_groups = torch.zeros(B, device=device, dtype=torch.long)

        # Greedy clustering per batch
        for b in range(B):
            centroids = []
            counts = []
            for t in range(T):
                phi = phases[b, t]
                assigned = False
                for g, (centroid, count) in enumerate(zip(centroids, counts)):
                    # Circular distance
                    diff = torch.abs(phi - centroid)
                    diff = torch.min(diff, 2 * math.pi - diff)
                    if diff < self.threshold:
                        group_ids[b, t] = g
                        # Update centroid (running circular mean)
                        centroids[g] = torch.atan2(
                            (count * torch.sin(centroid) + torch.sin(phi)) / (count + 1),
                            (count * torch.cos(centroid) + torch.cos(phi)) / (count + 1),
                        ) % (2 * math.pi)
                        counts[g] = count + 1
                        assigned = True
                        break
                if not assigned and len(centroids) < self.max_groups:
                    g = len(centroids)
                    centroids.append(phi)
                    counts.append(1)
                    group_ids[b, t] = g
                elif not assigned:
                    # Assign to closest group
                    dists = torch.stack([
                        torch.min(torch.abs(phi - c), 2 * math.pi - torch.abs(phi - c))
                        for c in centroids
                    ])
                    group_ids[b, t] = dists.argmin()

            num_groups[b] = len(centroids)

            # Pool features per group
            for g in range(len(centroids)):
                mask = (group_ids[b] == g)
                if mask.any():
                    group_features[b, g] = features[b][mask].mean(dim=0)

        # Compute within-group coherence
        coherence = torch.zeros(B, device=device)
        for b in range(B):
            cos_sum = 0.0
            for g in range(num_groups[b].item()):
                mask = (group_ids[b] == g)
                if mask.sum() > 1:
                    g_phases = phases[b][mask]
                    z = torch.exp(1j * g_phases.to(torch.complex64))
                    cos_sum += torch.abs(z.mean()).item()
            if num_groups[b] > 0:
                coherence[b] = cos_sum / num_groups[b].item()

        return {
            'group_ids': group_ids,
            'group_features': group_features,
            'num_groups': num_groups,
            'coherence': coherence,
        }


class OscillatoryChunking(nn.Module):
    """Hierarchical chunking via nested oscillation frequencies.

    Different oscillator levels capture different temporal granularities:
    - Level 0 (fast, ~gamma): Word-level grouping
    - Level 1 (medium, ~beta): Phrase-level grouping
    - Level 2 (slow, ~theta): Sentence/clause-level grouping

    Phase coherence peaks mark chunk boundaries at each level.

    Args:
        dim: Feature dimension.
        num_levels: Number of hierarchy levels (default 3).
        oscillators_per_level: Oscillators per level.
        dt: Integration timestep.
        num_steps: Synchronization steps per forward pass.
    """

    def __init__(
        self,
        dim: int,
        num_levels: int = 3,
        oscillators_per_level: int = 4,
        dt: float = 0.1,
        num_steps: int = 10,
    ):
        super().__init__()
        self.dim = dim
        self.num_levels = num_levels
        self.osc_per_level = oscillators_per_level
        self.dt = dt
        self.num_steps = num_steps

        # Per-level projections and frequencies
        self.level_projs = nn.ModuleList([
            nn.Linear(dim, oscillators_per_level, bias=False)
            for _ in range(num_levels)
        ])

        # Natural frequencies: faster at lower levels
        # Level 0: gamma (~40Hz), Level 1: beta (~20Hz), Level 2: theta (~6Hz)
        freq_scales = [4.0, 2.0, 0.6]
        self.omegas = nn.ParameterList([
            nn.Parameter(
                torch.randn(oscillators_per_level) * 0.1 + freq_scales[min(i, len(freq_scales)-1)]
            )
            for i in range(num_levels)
        ])

        # Intra-level coupling
        self.coupling_strengths = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0 + i * 0.5))
            for i in range(num_levels)
        ])

    def _find_boundaries(
        self, phases: torch.Tensor, threshold: float = 0.3
    ) -> List[int]:
        """Find chunk boundaries from phase coherence dips.

        Args:
            phases: (T, O) phases at each position
            threshold: Coherence dip threshold for boundary detection

        Returns:
            List of boundary indices
        """
        T = phases.shape[0]
        if T < 3:
            return []

        # Local coherence: order parameter in sliding window of 3
        coherences = []
        for t in range(1, T - 1):
            window = phases[t-1:t+2]  # (3, O)
            z = torch.exp(1j * window.to(torch.complex64))
            R = torch.abs(z.mean(dim=0)).mean()
            coherences.append(R.item())

        # Boundaries at coherence dips
        boundaries = []
        for i in range(1, len(coherences) - 1):
            if (coherences[i] < coherences[i-1] - threshold and
                coherences[i] < coherences[i+1] - threshold):
                boundaries.append(i + 1)  # +1 for offset from window

        return boundaries

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """Run multi-level oscillatory chunking.

        Args:
            x: (B, T, D) token sequence

        Returns:
            Dict with:
            - chunks: List[List[int]] boundaries at finest level (per batch)
            - hierarchy: Dict[int, List[List[int]]] boundaries per level
            - level_phases: List[Tensor] (B, T, O) phases per level
            - level_coherences: (B, num_levels) mean coherence per level
        """
        B, T, D = x.shape
        device = x.device

        all_boundaries = {}
        all_phases = []
        level_coherences = torch.zeros(B, self.num_levels, device=device)

        for level in range(self.num_levels):
            # Project features to oscillator space
            proj = self.level_projs[level](x)  # (B, T, O)
            O = self.osc_per_level

            # Initialize phases from projection
            phases = torch.atan2(proj, proj.roll(1, dims=-1) + 1e-8)
            phases = phases % (2 * math.pi)

            omega = self.omegas[level].unsqueeze(0).unsqueeze(0)  # (1, 1, O)
            K = self.coupling_strengths[level]

            # Run Kuramoto dynamics
            for _ in range(self.num_steps):
                for b in range(B):
                    # All-to-all coupling within each level
                    ph = phases[b]  # (T, O)
                    # Mean field coupling across positions
                    mean_phase = torch.atan2(
                        torch.sin(ph).mean(dim=0, keepdim=True),
                        torch.cos(ph).mean(dim=0, keepdim=True),
                    )
                    coupling = K * torch.sin(mean_phase - ph) / T
                    phases[b] = (ph + self.dt * (omega.squeeze(0) + coupling)) % (2 * math.pi)

            all_phases.append(phases.detach())

            # Find boundaries per batch
            level_boundaries = []
            for b in range(B):
                boundaries = self._find_boundaries(phases[b])
                level_boundaries.append(boundaries)

                # Compute level coherence
                z = torch.exp(1j * phases[b].to(torch.complex64))
                R = torch.abs(z.mean(dim=0)).mean()
                level_coherences[b, level] = R

            all_boundaries[level] = level_boundaries

        return {
            'chunks': all_boundaries.get(0, [[] for _ in range(B)]),
            'hierarchy': all_boundaries,
            'level_phases': all_phases,
            'level_coherences': level_coherences,
        }
