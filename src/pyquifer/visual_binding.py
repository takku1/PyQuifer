"""
Visual Binding via Attentive Kuramoto Oscillator Networks (AKOrN) for PyQuifer

Implements oscillator-based perceptual binding: making Kuramoto dynamics
perform actual visual work — object segmentation, feature binding, and
attention-driven grouping. Tokens that synchronize in phase are bound
into coherent object representations.

Core idea: replace static attention weights with dynamic Kuramoto coupling.
The coupling matrix J is derived from attention (J_ij = softmax(q_i . k_j / sqrt(d))),
so phase synchronization patterns are input-dependent. Tokens attending to each
other entrain, forming synchronized clusters that correspond to objects.

Key concepts:
- Phase binding: Tokens with similar oscillator phases belong to the same object
- Attention-driven coupling: J_ij from scaled dot-product attention
- Multi-head structure: Block-diagonal coupling for independent binding channels
- Order parameter readout: Cluster coherence R indicates binding strength

Supports both 2D spatial inputs (B, H*W, D) from vision and 1D sequential
inputs (B, T, D) from language/audio — any token-based representation.

References:
- Rusch et al. (2025). "AKOrN: Attentive Kuramoto Oscillator Networks for
  Object-Centric Learning." ICLR 2025.
- Kuramoto (1975). "Self-entrainment of a population of coupled non-linear
  oscillators." International Symposium on Mathematical Problems in
  Theoretical Physics.
- von der Malsburg (1981). "The Correlation Theory of Brain Function."
  Internal Report 81-2, MPI for Biophysical Chemistry.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple


class AKOrNLayer(nn.Module):
    """
    Multi-head Kuramoto layer with attention-based coupling.

    Each head maintains a set of oscillators per token. Coupling between
    tokens is computed via scaled dot-product attention over learned
    query/key projections, yielding an input-dependent coupling matrix:

        J_ij = softmax(q_i . k_j / sqrt(d_k))

    Phase dynamics follow the Kuramoto model with this attention coupling:

        dphi_i/dt = omega_i + sum_j J_ij sin(phi_j - phi_i)

    After integration, token features are modulated by the resulting phases,
    so synchronized tokens produce coherent representations.

    Args:
        dim: Feature dimension of input tokens.
        num_heads: Number of independent oscillator heads (block-diagonal coupling).
        num_oscillators_per_head: Oscillators per head per token.
        dt: Integration timestep for Kuramoto dynamics.
        num_steps: Number of Kuramoto integration steps per forward pass.
        coupling_mode: 'attention' for input-dependent J, 'fixed' for learnable static J.

    Shape:
        Input: (B, N, D) where B=batch, N=tokens, D=dim
        Output: (B, N, D)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        num_oscillators_per_head: int = 8,
        dt: float = 0.1,
        num_steps: int = 5,
        coupling_mode: str = "attention",
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.num_osc = num_oscillators_per_head
        self.dt = dt
        self.num_steps = num_steps
        self.coupling_mode = coupling_mode

        # Dimension per head for query/key projections
        self.d_k = dim // num_heads
        assert dim % num_heads == 0, (
            f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        )

        if coupling_mode == "attention":
            # Query and key projections for attention-based coupling
            self.W_q = nn.Linear(dim, dim, bias=False)
            self.W_k = nn.Linear(dim, dim, bias=False)
        elif coupling_mode == "fixed":
            # Learnable static coupling matrix (block-diagonal, one per head)
            self.coupling_matrix = nn.Parameter(
                torch.randn(num_heads, 1, 1) * 0.1 + 1.0
            )
        else:
            raise ValueError(
                f"coupling_mode must be 'attention' or 'fixed', got '{coupling_mode}'"
            )

        # Learnable natural frequencies: (num_heads, num_osc)
        self.omega = nn.Parameter(
            torch.randn(num_heads, num_oscillators_per_head) * 0.1
        )

        # Project features into phase space and back
        self.to_phase = nn.Linear(dim, num_heads * num_oscillators_per_head)
        self.from_phase = nn.Linear(num_heads * num_oscillators_per_head, dim)

        # Learnable coupling scale
        self.coupling_scale = nn.Parameter(torch.tensor(1.0))

    def _compute_coupling(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """Compute the coupling matrix J from input features.

        For attention mode:
            J_ij = softmax(q_i . k_j / sqrt(d_k))   shape: (B, H, N, N)

        For fixed mode:
            J = coupling_matrix expanded to (1, H, 1, 1)

        Args:
            x: Input features (B, N, D).

        Returns:
            J: Coupling matrix (B, H, N, N) for attention mode,
               or (1, H, 1, 1) for fixed mode.
        """
        if self.coupling_mode == "attention":
            B, N, D = x.shape
            # (B, N, D) -> (B, N, H, d_k) -> (B, H, N, d_k)
            q = self.W_q(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
            k = self.W_k(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)

            # Scaled dot-product attention: (B, H, N, N)
            attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
            J = F.softmax(attn, dim=-1)
            return J
        else:
            # Fixed coupling: broadcast scalar per head
            return self.coupling_matrix.unsqueeze(0)  # (1, H, 1, 1)

    def _kuramoto_step(
        self,
        phases: torch.Tensor,
        J: torch.Tensor,
    ) -> torch.Tensor:
        """Single Kuramoto integration step.

        dphi_i/dt = omega_i + coupling_scale * sum_j J_ij sin(phi_j - phi_i)

        Args:
            phases: Current phases (B, H, N, num_osc).
            J: Coupling matrix (B, H, N, N) or broadcastable.

        Returns:
            Updated phases (B, H, N, num_osc).
        """
        # Phase differences: phi_j - phi_i
        # phases: (B, H, N, num_osc) -> expand for pairwise
        # phi_j: (B, H, 1, N, num_osc), phi_i: (B, H, N, 1, num_osc)
        phase_diff = phases.unsqueeze(2) - phases.unsqueeze(3)
        # phase_diff: (B, H, N, N, num_osc) = phi_j[dim=2] - phi_i[dim=3]
        # Wait — we need J_ij * sin(phi_j - phi_i) summed over j.
        # Let dim 3 = j, dim 2 = i:
        #   phase_diff[..., i, j, :] = phases[..., j, :] - phases[..., i, :]
        # Actually let's be explicit:
        phi_i = phases.unsqueeze(3)  # (B, H, N, 1, osc)
        phi_j = phases.unsqueeze(2)  # (B, H, 1, N, osc)
        sin_diff = torch.sin(phi_j - phi_i)  # (B, H, N, N, osc)

        # J: (B, H, N, N) -> (B, H, N, N, 1) for broadcasting
        if J.dim() == 4:
            J_expanded = J.unsqueeze(-1)
        else:
            J_expanded = J.unsqueeze(-1)

        # Weighted coupling: sum_j J_ij sin(phi_j - phi_i)
        coupling = (J_expanded * sin_diff).sum(dim=3)  # (B, H, N, osc)

        # Natural frequency drift: (H, osc) -> (1, H, 1, osc)
        omega = self.omega.unsqueeze(0).unsqueeze(2)

        # Kuramoto ODE
        dphase = omega + self.coupling_scale * coupling

        # Euler integration
        new_phases = phases + self.dt * dphase

        # Wrap to [-pi, pi]
        new_phases = torch.remainder(new_phases + math.pi, 2 * math.pi) - math.pi

        return new_phases

    def order_parameter(self, phases: torch.Tensor) -> torch.Tensor:
        """Compute Kuramoto order parameter R per head.

        R = |<exp(i * phi)>| in [0, 1].
        R ~ 0: incoherent (random phases).
        R ~ 1: fully synchronized.

        Args:
            phases: (B, H, N, num_osc).

        Returns:
            R: (B, H) order parameter per head.
        """
        z = torch.exp(1j * phases.to(torch.cfloat))
        z_mean = z.mean(dim=[2, 3])
        return torch.abs(z_mean).float()

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass: run Kuramoto dynamics with attention coupling.

        Args:
            x: Input features (B, N, D).
            mask: Optional token mask (B, N), True = valid token. Currently
                  reserved for future use.

        Returns:
            Output features (B, N, D) modulated by oscillator phases.
        """
        B, N, D = x.shape

        # Compute attention-based coupling
        J = self._compute_coupling(x)  # (B, H, N, N)

        # Project features into phase space
        phase_proj = self.to_phase(x)  # (B, N, H * osc)
        phase_proj = phase_proj.view(B, N, self.num_heads, self.num_osc)
        phases = phase_proj.permute(0, 2, 1, 3)  # (B, H, N, osc)

        # Run Kuramoto dynamics
        for _ in range(self.num_steps):
            phases = self._kuramoto_step(phases, J)

        # Modulate features with synchronized phases
        # Use cos(phases) as a real-valued modulation signal
        phase_signal = torch.cos(phases)  # (B, H, N, osc)
        phase_signal = phase_signal.permute(0, 2, 1, 3)  # (B, N, H, osc)
        phase_signal = phase_signal.reshape(B, N, self.num_heads * self.num_osc)

        # Project back to feature space
        output = self.from_phase(phase_signal)  # (B, N, D)

        return output


class AKOrNBlock(nn.Module):
    """
    AKOrN layer + feedforward network, stackable like a Transformer block.

    Architecture:
        x -> LayerNorm -> AKOrN -> + (residual) -> LayerNorm -> FFN -> + (residual)

    This follows the Pre-LN Transformer convention (Xiong et al., 2020) for
    stable deep stacking.

    Args:
        dim: Feature dimension.
        num_heads: Number of oscillator heads.
        ff_dim: Hidden dimension of the feedforward network.
        num_oscillators_per_head: Oscillators per head per token.
        dt: Kuramoto integration timestep.
        num_steps: Number of Kuramoto integration steps.
        dropout: Dropout rate for FFN.
        coupling_mode: 'attention' or 'fixed' coupling.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        ff_dim: int = None,
        num_oscillators_per_head: int = 8,
        dt: float = 0.1,
        num_steps: int = 5,
        dropout: float = 0.1,
        coupling_mode: str = "attention",
    ):
        super().__init__()

        if ff_dim is None:
            ff_dim = dim * 4  # Standard Transformer ratio

        self.norm1 = nn.LayerNorm(dim)
        self.akorn = AKOrNLayer(
            dim=dim,
            num_heads=num_heads,
            num_oscillators_per_head=num_oscillators_per_head,
            dt=dt,
            num_steps=num_steps,
            coupling_mode=coupling_mode,
        )

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with pre-norm residual connections.

        Args:
            x: Input features (B, N, D).
            mask: Optional token mask (B, N).

        Returns:
            Output features (B, N, D).
        """
        # AKOrN block with residual
        x = x + self.akorn(self.norm1(x), mask=mask)

        # FFN block with residual
        x = x + self.ffn(self.norm2(x))

        return x


class AKOrNEncoder(nn.Module):
    """
    Full encoder stack of AKOrN blocks for image or sequence feature maps.

    Stacks multiple AKOrNBlocks with optional learnable positional embeddings.
    Designed for use after a patch embedding or CNN backbone — takes flattened
    spatial tokens (B, H*W, D) or sequence tokens (B, T, D).

    Args:
        dim: Feature dimension.
        depth: Number of AKOrN blocks to stack.
        num_heads: Number of oscillator heads per block.
        ff_dim: FFN hidden dimension (default: 4 * dim).
        num_oscillators_per_head: Oscillators per head per token.
        dt: Kuramoto integration timestep.
        num_steps: Kuramoto integration steps per block.
        max_tokens: Maximum sequence length for positional embedding.
            If None, no positional embedding is added.
        dropout: Dropout rate.
        coupling_mode: 'attention' or 'fixed' coupling.
    """

    def __init__(
        self,
        dim: int,
        depth: int = 4,
        num_heads: int = 4,
        ff_dim: int = None,
        num_oscillators_per_head: int = 8,
        dt: float = 0.1,
        num_steps: int = 5,
        max_tokens: Optional[int] = None,
        dropout: float = 0.1,
        coupling_mode: str = "attention",
    ):
        super().__init__()

        if ff_dim is None:
            ff_dim = dim * 4

        # Optional positional embedding
        self.pos_embed = None
        if max_tokens is not None:
            self.pos_embed = nn.Parameter(
                torch.randn(1, max_tokens, dim) * 0.02
            )

        self.blocks = nn.ModuleList([
            AKOrNBlock(
                dim=dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                num_oscillators_per_head=num_oscillators_per_head,
                dt=dt,
                num_steps=num_steps,
                dropout=dropout,
                coupling_mode=coupling_mode,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the full encoder stack.

        Args:
            x: Input features (B, N, D).
            mask: Optional token mask (B, N).

        Returns:
            Encoded features (B, N, D).
        """
        B, N, D = x.shape

        # Add positional embeddings if available
        if self.pos_embed is not None:
            x = x + self.pos_embed[:, :N, :]

        # Pass through all blocks
        for block in self.blocks:
            x = block(x, mask=mask)

        # Final layer norm
        x = self.norm(x)

        return x


class OscillatorySegmenter(nn.Module):
    """
    Object discovery via oscillator synchronization clusters.

    The binding-by-synchrony hypothesis (von der Malsburg, 1981): neurons
    representing the same object fire in synchrony, while neurons representing
    different objects are desynchronized. This module operationalizes that
    idea for modern deep learning.

    Process:
    1. Run AKOrN dynamics to synchronize related tokens
    2. Extract mean phase per token (across oscillators within each head)
    3. Cluster tokens by phase similarity using a coherence threshold
    4. Assign segment IDs: tokens in the same phase cluster = same object

    The number of segments is not fixed — it emerges from the dynamics.

    Args:
        dim: Feature dimension.
        num_heads: Number of oscillator heads.
        num_oscillators_per_head: Oscillators per head.
        num_steps: Kuramoto integration steps (more steps = sharper clustering).
        threshold: Phase distance threshold for grouping (radians).
            Tokens with mean phase distance < threshold are assigned to the
            same segment. Lower = stricter clustering (more segments).
        dt: Kuramoto integration timestep.
        coupling_mode: 'attention' or 'fixed' coupling.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        num_oscillators_per_head: int = 8,
        num_steps: int = 10,
        threshold: float = 0.5,
        dt: float = 0.1,
        coupling_mode: str = "attention",
    ):
        super().__init__()

        self.threshold = threshold

        self.akorn = AKOrNLayer(
            dim=dim,
            num_heads=num_heads,
            num_oscillators_per_head=num_oscillators_per_head,
            dt=dt,
            num_steps=num_steps,
            coupling_mode=coupling_mode,
        )

        # Project features to phase space for clustering
        self.to_phase = nn.Linear(dim, num_heads * num_oscillators_per_head)
        self.num_heads = num_heads
        self.num_osc = num_oscillators_per_head

    def _extract_mean_phase(self, x: torch.Tensor) -> torch.Tensor:
        """Compute mean phase per token by running AKOrN and averaging.

        Uses the circular mean: angle(mean(exp(i * phi))).

        Args:
            x: Input features (B, N, D).

        Returns:
            mean_phase: (B, N) mean phase per token in [-pi, pi].
        """
        B, N, D = x.shape

        # Compute coupling from features
        J = self.akorn._compute_coupling(x)

        # Project to phase space
        phase_proj = self.to_phase(x)  # (B, N, H * osc)
        phase_proj = phase_proj.view(B, N, self.num_heads, self.num_osc)
        phases = phase_proj.permute(0, 2, 1, 3)  # (B, H, N, osc)

        # Run Kuramoto dynamics
        for _ in range(self.akorn.num_steps):
            phases = self.akorn._kuramoto_step(phases, J)

        # Circular mean across heads and oscillators: angle(mean(exp(i*phi)))
        z = torch.exp(1j * phases.to(torch.cfloat))  # (B, H, N, osc)
        z_mean = z.mean(dim=[1, 3])  # (B, N)
        mean_phase = torch.angle(z_mean).float()  # (B, N) in [-pi, pi]

        return mean_phase, phases

    def _cluster_by_phase(
        self, mean_phase: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assign segment IDs by greedy phase clustering.

        Uses circular distance: d(a, b) = min(|a - b|, 2*pi - |a - b|).
        Greedy sequential assignment: each token is assigned to the nearest
        existing cluster if within threshold, otherwise starts a new cluster.

        Args:
            mean_phase: (B, N) mean phase per token.

        Returns:
            segment_ids: (B, N) integer segment assignments.
            num_segments: (B,) number of segments per sample.
            cluster_coherence: (B,) mean within-cluster phase coherence.
        """
        B, N = mean_phase.shape
        device = mean_phase.device

        # Pairwise circular distance matrix: (B, N, N)
        phase_i = mean_phase.unsqueeze(2)  # (B, N, 1)
        phase_j = mean_phase.unsqueeze(1)  # (B, 1, N)
        abs_diff = torch.abs(phase_i - phase_j)
        circ_dist = torch.min(abs_diff, 2 * math.pi - abs_diff)  # (B, N, N)

        # Greedy clustering via adjacency
        # Two tokens are in the same cluster if circ_dist < threshold
        adjacency = (circ_dist < self.threshold).float()  # (B, N, N)

        # Use a simple connected-components approach via iterative propagation
        # Initialize each token as its own segment
        segment_ids = torch.arange(N, device=device).unsqueeze(0).expand(B, -1).clone()

        # Propagate: each token takes the minimum ID among its neighbors
        # Iterate until convergence (at most N steps, but usually ~log(N))
        for _ in range(int(math.log2(N)) + 2):
            # For each token, find minimum segment ID among connected tokens
            # adjacency: (B, N, N), segment_ids: (B, N)
            # Expand segment_ids for gathering
            expanded = segment_ids.unsqueeze(1).expand(B, N, N)  # (B, N, N)
            # Where adjacency is 0, replace with a large ID so it won't be min
            masked = expanded.clone()
            masked[adjacency < 0.5] = N  # sentinel: won't be chosen as min
            new_ids = masked.min(dim=2).values  # (B, N)
            if torch.equal(new_ids, segment_ids):
                break
            segment_ids = new_ids

        # Renumber segments to be contiguous 0, 1, 2, ...
        # Process per batch element
        num_segments = torch.zeros(B, dtype=torch.long, device=device)
        for b in range(B):
            unique_ids = segment_ids[b].unique(sorted=True)
            num_segments[b] = unique_ids.shape[0]
            for new_id, old_id in enumerate(unique_ids):
                segment_ids[b][segment_ids[b] == old_id] = new_id

        # Compute within-cluster coherence
        # For each cluster: R = |mean(exp(i * phase))| among members
        coherence_sum = torch.zeros(B, device=device)
        for b in range(B):
            n_seg = num_segments[b].item()
            if n_seg == 0:
                continue
            for s in range(n_seg):
                member_mask = (segment_ids[b] == s)
                if member_mask.sum() < 2:
                    coherence_sum[b] += 1.0  # Single-token cluster is trivially coherent
                    continue
                member_phases = mean_phase[b][member_mask]
                z = torch.exp(1j * member_phases.to(torch.cfloat))
                R = torch.abs(z.mean()).float()
                coherence_sum[b] += R
            coherence_sum[b] /= n_seg

        return segment_ids, num_segments, coherence_sum

    def forward(
        self, features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Segment tokens into objects via phase synchronization.

        Args:
            features: Input features (B, N, D).

        Returns:
            Dict with:
                'segments': (B, N) integer segment IDs.
                'num_segments': (B,) number of segments per sample.
                'cluster_coherence': (B,) mean within-cluster coherence.
                'mean_phase': (B, N) mean phase per token.
                'phases': (B, H, N, osc) raw oscillator phases after dynamics.
        """
        mean_phase, phases = self._extract_mean_phase(features)
        segment_ids, num_segments, coherence = self._cluster_by_phase(mean_phase)

        return {
            "segments": segment_ids,
            "num_segments": num_segments,
            "cluster_coherence": coherence,
            "mean_phase": mean_phase,
            "phases": phases,
        }


class BindingReadout(nn.Module):
    """
    Extract per-object representations from phase-synchronized clusters.

    Given features and segment assignments from OscillatorySegmenter,
    pools features within each segment to produce fixed-size object
    representations. Segments are ordered by size (largest first).

    For segments fewer than max_objects, remaining slots are zero-padded.
    For segments more than max_objects, only the largest are kept.

    Args:
        dim: Feature dimension.
        max_objects: Maximum number of object slots in the output.
        pool_mode: How to aggregate features within a segment.
            'mean' — average pooling (default).
            'attention' — learnable attention pooling within each segment.
    """

    def __init__(
        self,
        dim: int,
        max_objects: int = 8,
        pool_mode: str = "mean",
    ):
        super().__init__()

        self.dim = dim
        self.max_objects = max_objects
        self.pool_mode = pool_mode

        if pool_mode == "attention":
            # Learnable query for attention pooling within each segment
            self.pool_query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
            self.pool_attn = nn.MultiheadAttention(
                embed_dim=dim, num_heads=1, batch_first=True
            )
        elif pool_mode != "mean":
            raise ValueError(f"pool_mode must be 'mean' or 'attention', got '{pool_mode}'")

        # Optional projection for object representations
        self.obj_proj = nn.Linear(dim, dim)
        self.obj_norm = nn.LayerNorm(dim)

    def forward(
        self,
        features: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Pool features by segment to produce per-object representations.

        Args:
            features: Token features (B, N, D).
            segment_ids: Segment assignments (B, N) from OscillatorySegmenter.

        Returns:
            Object representations (B, max_objects, D). Unused slots are
            zero-filled. Objects ordered by segment size (largest first).
        """
        B, N, D = features.shape
        device = features.device

        output = torch.zeros(B, self.max_objects, D, device=device)

        for b in range(B):
            unique_segs = segment_ids[b].unique(sorted=True)
            n_segs = unique_segs.shape[0]

            if n_segs == 0:
                continue

            # Collect pooled representations and their sizes
            seg_reps = []
            seg_sizes = []
            for seg_id in unique_segs:
                member_mask = (segment_ids[b] == seg_id)
                member_features = features[b][member_mask]  # (M, D)

                if self.pool_mode == "mean":
                    pooled = member_features.mean(dim=0)  # (D,)
                else:
                    # Attention pooling: query attends to segment members
                    query = self.pool_query.expand(1, -1, -1)  # (1, 1, D)
                    kv = member_features.unsqueeze(0)  # (1, M, D)
                    pooled, _ = self.pool_attn(query, kv, kv)  # (1, 1, D)
                    pooled = pooled.squeeze(0).squeeze(0)  # (D,)

                seg_reps.append(pooled)
                seg_sizes.append(member_mask.sum().item())

            # Sort by segment size descending (largest objects first)
            size_order = sorted(
                range(len(seg_sizes)), key=lambda i: seg_sizes[i], reverse=True
            )

            # Fill output slots
            n_fill = min(len(seg_reps), self.max_objects)
            for slot_idx in range(n_fill):
                output[b, slot_idx] = seg_reps[size_order[slot_idx]]

        # Project and normalize
        output = self.obj_norm(self.obj_proj(output))

        return output
