"""
Visual Binding via Attentive Kuramoto Oscillator Networks (AKOrN) for PyQuifer

Implements oscillator-based perceptual binding using generalized vector-valued
Kuramoto dynamics on unit hyperspheres. Tokens that synchronize in phase are
bound into coherent object representations.

Core idea: replace static attention weights with dynamic Kuramoto coupling.
The coupling matrix J is derived from attention (J_ij = softmax(q_i . k_j / sqrt(d))),
so phase synchronization patterns are input-dependent. Tokens attending to each
other entrain, forming synchronized clusters that correspond to objects.

Key concepts:
- Vector-valued oscillators: Each oscillator is an n-dimensional unit vector on S^(n-1),
  not a scalar phase angle. This is the generalized Kuramoto model.
- Tangent projection: Updates are projected onto the tangent space of the hypersphere
  before integration, ensuring oscillator states remain on the manifold.
- Phase-invariant readout: Output features are extracted via L2 norm of n-dim groups,
  yielding representations invariant to the absolute phase.
- Conditioning bias: Input features serve as a bias term in the coupling, driving
  oscillators toward input-dependent attractors.
- Random initialization: Oscillator states are initialized randomly on the unit sphere,
  breaking symmetry so dynamics can discover grouping structure.

Supports both 2D spatial inputs (B, H*W, D) from vision and 1D sequential
inputs (B, T, D) from language/audio -- any token-based representation.

References:
- Rusch et al. (2025). "AKOrN: Attentive Kuramoto Oscillator Networks for
  Object-Centric Learning." ICLR 2025 (Oral).
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


# ============================================================
# Hypersphere utilities (following Rusch et al. 2025)
# ============================================================

def _reshape_to_groups(x: torch.Tensor, n: int) -> torch.Tensor:
    """Unflatten channel dim into (C//n, n) groups of n-dim vectors.

    (B, C, ...) -> (B, C//n, n, ...)
    """
    return x.unflatten(1, (-1, n))


def _reshape_from_groups(x: torch.Tensor) -> torch.Tensor:
    """Flatten (C//n, n) groups back to channel dim.

    (B, C//n, n, ...) -> (B, C, ...)
    """
    return x.flatten(1, 2)


def _normalize_to_sphere(x: torch.Tensor, n: int) -> torch.Tensor:
    """Project oscillator states onto the unit hypersphere S^(n-1).

    Groups the channel dimension into n-dim vectors and L2-normalizes each.
    """
    x = _reshape_to_groups(x, n)
    x = F.normalize(x, dim=2)
    x = _reshape_from_groups(x)
    return x


def _tangent_project(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Project y onto the tangent space of x on the unit sphere.

    T_x(S^{n-1}) = {v : <v, x> = 0}

    Formula: y_tan = y - <y, x> * x

    Both y and x should have shape (B, G, n, ...) where G = num groups.
    Returns projected y with the same shape.
    """
    # Inner product along the n-dim axis (dim=2), keep dims for broadcasting
    dot = (y * x).sum(dim=2, keepdim=True)
    return y - dot * x


class AKOrNLayer(nn.Module):
    """
    Generalized vector-valued Kuramoto layer with attention-based coupling.

    Each oscillator is an n-dimensional unit vector on S^(n-1). The coupling
    between tokens is computed via scaled dot-product attention, yielding an
    input-dependent coupling matrix J. Input features serve as a conditioning
    bias that drives oscillators toward input-dependent attractors.

    Dynamics (per Kuramoto step):
        y = J(x) + c                       (connectivity + conditioning bias)
        y_tan = y - <y, x>x                (tangent projection)
        x <- normalize(x + gamma * y_tan)  (update + re-project to sphere)

    where x are oscillator states, c is the conditioning bias from input
    features, and J is the attention-derived coupling.

    Args:
        dim: Feature dimension of input tokens (must be divisible by n).
        n: Dimension of each oscillator vector (lives on S^(n-1)).
            n=2 is equivalent to classical scalar Kuramoto (circle S^1).
            n=4 is the default used in Rusch et al. (2025).
        num_heads: Number of attention heads for coupling computation.
        gamma: Step size for Kuramoto update (default 1.0).
        num_steps: Number of Kuramoto integration steps per forward pass.
        coupling_mode: 'attention' for input-dependent J, 'fixed' for learnable static J.
        c_norm: Normalization for conditioning bias. 'gn'=GroupNorm, 'none'=identity.
        apply_proj: Whether to apply tangent projection (True for proper geometry).

    Shape:
        Input: (B, N, D) where B=batch, N=tokens, D=dim
        Output: (B, N, D)
    """

    def __init__(
        self,
        dim: int,
        n: int = 4,
        num_heads: int = 4,
        gamma: float = 1.0,
        num_steps: int = 5,
        coupling_mode: str = "attention",
        c_norm: str = "gn",
        apply_proj: bool = True,
    ):
        super().__init__()

        self.dim = dim
        self.n = n
        self.num_heads = num_heads
        self.gamma = gamma
        self.num_steps = num_steps
        self.coupling_mode = coupling_mode
        self.apply_proj = apply_proj

        assert dim % n == 0, f"dim ({dim}) must be divisible by n ({n})"
        self.num_groups = dim // n  # Number of n-dim oscillator groups

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
            # Learnable static coupling scale
            self.coupling_matrix = nn.Parameter(
                torch.randn(num_heads, 1, 1) * 0.1 + 1.0
            )
        else:
            raise ValueError(
                f"coupling_mode must be 'attention' or 'fixed', got '{coupling_mode}'"
            )

        # Conditioning bias normalization
        if c_norm == "gn":
            # GroupNorm with groups matching oscillator groups
            self.c_norm = nn.GroupNorm(self.num_groups, dim, affine=True)
        elif c_norm == "none" or c_norm is None:
            self.c_norm = nn.Identity()
        else:
            raise ValueError(f"c_norm must be 'gn' or 'none', got '{c_norm}'")

        # Conditioning projection: features -> bias c
        self.cond_proj = nn.Linear(dim, dim)

        # Phase-invariant readout (following Rusch et al. 2025):
        # 1. Learned linear: (D) -> (D * n) -- creates n-dim groups with non-trivial norms
        # 2. Unflatten to (D, n) groups
        # 3. L2 norm along n-dim: (D,) -- phase-invariant but information-rich
        # 4. Add learnable bias
        # The key insight: the linear projection before norm creates groups whose
        # norms encode phase relationships, unlike raw normalized states (all norm=1).
        self.readout_linear = nn.Linear(dim, dim * n, bias=False)
        self.readout_bias = nn.Parameter(torch.zeros(dim))

    def _compute_coupling(
        self, x: torch.Tensor, x_state: torch.Tensor
    ) -> torch.Tensor:
        """Compute connectivity: sum_j J_ij * x_j.

        For attention mode: uses learned Q/K from input features, applies
        attention weights to oscillator states.

        Args:
            x: Input features (B, N, D) for computing attention weights.
            x_state: Oscillator states (B, N, D) to be mixed by coupling.

        Returns:
            Coupled states (B, N, D).
        """
        if self.coupling_mode == "attention":
            B, N, D = x.shape
            # Compute attention weights from input features
            q = self.W_q(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
            k = self.W_k(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
            attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
            attn = F.softmax(attn, dim=-1)  # (B, H, N, N)

            # Apply attention to oscillator states
            # x_state: (B, N, D) -> (B, H, N, d_k) for per-head mixing
            v = x_state.view(B, N, self.num_heads, self.d_k).transpose(1, 2)
            coupled = torch.matmul(attn, v)  # (B, H, N, d_k)
            coupled = coupled.transpose(1, 2).reshape(B, N, D)
            return coupled
        else:
            # Fixed coupling: scale oscillator states
            return self.coupling_matrix.mean() * x_state

    def _kuramoto_step(
        self,
        x_state: torch.Tensor,
        c: torch.Tensor,
        x_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single generalized Kuramoto step on the unit hypersphere.

        1. Compute connectivity: y = J(x_state) + c
        2. Reshape into n-dim groups
        3. Tangent projection: y_tan = y - <y, x>x
        4. Update: x <- normalize(x + gamma * y_tan)

        Args:
            x_state: Current oscillator states (B, N, D), on unit sphere.
            c: Conditioning bias (B, N, D), normalized.
            x_features: Original input features (B, N, D) for coupling computation.

        Returns:
            Updated oscillator states (B, N, D), on unit sphere.
            Similarity (energy proxy) (B, N, D).
        """
        # Connectivity: sum_j J_ij x_j
        y = self._compute_coupling(x_features, x_state)

        # Add conditioning bias
        y = y + c

        # Reshape into n-dim groups for tangent projection
        y_grouped = _reshape_to_groups(y, self.n)        # (B, N, G, n) -- wait
        x_grouped = _reshape_to_groups(x_state, self.n)

        # Wait -- our tensors are (B, N, D). _reshape_to_groups expects (B, C, ...).
        # We need to transpose to (B, D, N) first, then group, then transpose back.
        # Actually let's do the grouping on the last dim instead.
        # Simpler: reshape (B, N, D) -> (B, N, G, n) directly
        B, N, D = x_state.shape
        y_g = y.view(B, N, self.num_groups, self.n)
        x_g = x_state.view(B, N, self.num_groups, self.n)

        # Tangent projection on dim=-1 (the n-dim)
        if self.apply_proj:
            dot = (y_g * x_g).sum(dim=-1, keepdim=True)  # (B, N, G, 1)
            y_tan = y_g - dot * x_g
            sim = dot.squeeze(-1)  # (B, N, G) for energy tracking
        else:
            y_tan = y_g
            sim = (y_g * x_g).sum(dim=-1)

        # Reshape back to (B, N, D)
        dxdt = y_tan.view(B, N, D)

        # Update + re-project to sphere
        x_new = x_state + self.gamma * dxdt
        # Normalize each n-dim group to unit sphere
        x_new_g = x_new.view(B, N, self.num_groups, self.n)
        x_new_g = F.normalize(x_new_g, dim=-1)
        x_new = x_new_g.view(B, N, D)

        return x_new, sim.view(B, N, self.num_groups)

    def order_parameter(self, x_state: torch.Tensor) -> torch.Tensor:
        """Compute generalized order parameter R per oscillator group.

        For vector-valued Kuramoto: R = ||mean(x_i)||_2 where x_i are unit vectors.
        R ~ 0: incoherent (random orientations).
        R ~ 1: fully synchronized (all pointing same direction).

        Args:
            x_state: Oscillator states (B, N, D) on unit sphere.

        Returns:
            R: (B, G) order parameter per oscillator group.
        """
        B, N, D = x_state.shape
        x_g = x_state.view(B, N, self.num_groups, self.n)  # (B, N, G, n)
        x_mean = x_g.mean(dim=1)  # (B, G, n) mean over tokens
        R = torch.linalg.norm(x_mean, dim=-1)  # (B, G)
        return R

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass: run generalized Kuramoto dynamics with attention coupling.

        1. Compute conditioning bias from input features
        2. Initialize oscillator states randomly on the unit sphere
        3. Run T Kuramoto steps with tangent projection
        4. Phase-invariant readout via L2 norm of oscillator groups

        Args:
            x: Input features (B, N, D).
            mask: Optional token mask (B, N), True = valid token. Currently
                  reserved for future use.

        Returns:
            Output features (B, N, D) from phase-invariant readout.
        """
        B, N, D = x.shape

        # Conditioning bias from input features (normalized)
        c = self.cond_proj(x)  # (B, N, D)
        # Apply GroupNorm: need (B, D, N) format for nn.GroupNorm
        c = c.transpose(1, 2)          # (B, D, N)
        c = self.c_norm(c)             # (B, D, N)
        c = c.transpose(1, 2)          # (B, N, D)

        # Random initialization on unit sphere (key design choice)
        x_state = torch.randn(B, N, D, device=x.device, dtype=x.dtype)
        # Normalize each n-dim group to unit sphere
        x_state_g = x_state.view(B, N, self.num_groups, self.n)
        x_state_g = F.normalize(x_state_g, dim=-1)
        x_state = x_state_g.view(B, N, D)

        # Run Kuramoto dynamics
        for _ in range(self.num_steps):
            x_state, _ = self._kuramoto_step(x_state, c, x)

        # Phase-invariant readout (Rusch et al. 2025):
        # Linear projection creates n-dim groups with informative norms,
        # then L2 norm extracts phase-invariant features.
        readout = self.readout_linear(x_state)  # (B, N, D*n)
        readout = readout.view(B, N, self.dim, self.n)  # (B, N, D, n)
        output = torch.linalg.norm(readout, dim=-1)  # (B, N, D)
        output = output + self.readout_bias  # (B, N, D)

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
        n: Oscillator vector dimension (default 4, lives on S^(n-1)).
        num_heads: Number of attention heads for coupling.
        ff_dim: Hidden dimension of the feedforward network.
        gamma: Kuramoto step size.
        num_steps: Number of Kuramoto integration steps.
        dropout: Dropout rate for FFN.
        coupling_mode: 'attention' or 'fixed' coupling.
    """

    def __init__(
        self,
        dim: int,
        n: int = 4,
        num_heads: int = 4,
        ff_dim: int = None,
        gamma: float = 1.0,
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
            n=n,
            num_heads=num_heads,
            gamma=gamma,
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
        n: Oscillator vector dimension (default 4).
        num_heads: Number of attention heads per block.
        ff_dim: FFN hidden dimension (default: 4 * dim).
        gamma: Kuramoto step size.
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
        n: int = 4,
        num_heads: int = 4,
        ff_dim: int = None,
        gamma: float = 1.0,
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
                n=n,
                num_heads=num_heads,
                ff_dim=ff_dim,
                gamma=gamma,
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
        n: Oscillator vector dimension (default 4).
        num_heads: Number of attention heads.
        num_steps: Kuramoto integration steps (more steps = sharper clustering).
        threshold: Phase distance threshold for grouping (radians).
            Tokens with mean phase distance < threshold are assigned to the
            same segment. Lower = stricter clustering (more segments).
        gamma: Kuramoto step size.
        coupling_mode: 'attention' or 'fixed' coupling.
    """

    def __init__(
        self,
        dim: int,
        n: int = 4,
        num_heads: int = 4,
        num_steps: int = 10,
        threshold: float = 0.5,
        gamma: float = 1.0,
        coupling_mode: str = "attention",
    ):
        super().__init__()

        self.threshold = threshold
        self.n = n

        self.akorn = AKOrNLayer(
            dim=dim,
            n=n,
            num_heads=num_heads,
            gamma=gamma,
            num_steps=num_steps,
            coupling_mode=coupling_mode,
        )

    def _extract_mean_phase(self, x: torch.Tensor) -> torch.Tensor:
        """Compute mean phase per token by running AKOrN dynamics.

        For vector-valued Kuramoto, we extract a scalar phase per token by
        computing the angle of the mean oscillator vector (using atan2 on
        the first two components of each n-dim group, then averaging groups).

        Args:
            x: Input features (B, N, D).

        Returns:
            mean_phase: (B, N) mean phase per token in [-pi, pi].
            x_state: (B, N, D) final oscillator states on the sphere.
        """
        B, N, D = x.shape

        # Run full AKOrN forward to get oscillator states
        # We need access to internals, so replicate the forward pass
        c = self.akorn.cond_proj(x)
        c = c.transpose(1, 2)
        c = self.akorn.c_norm(c)
        c = c.transpose(1, 2)

        x_state = torch.randn(B, N, D, device=x.device, dtype=x.dtype)
        x_g = x_state.view(B, N, self.akorn.num_groups, self.n)
        x_g = F.normalize(x_g, dim=-1)
        x_state = x_g.view(B, N, D)

        for _ in range(self.akorn.num_steps):
            x_state, _ = self.akorn._kuramoto_step(x_state, c, x)

        # Extract scalar phase: atan2 of first two components of each group
        x_g = x_state.view(B, N, self.akorn.num_groups, self.n)  # (B, N, G, n)
        # Use atan2(y, x) on first two dims of each group
        phases_per_group = torch.atan2(x_g[..., 1], x_g[..., 0])  # (B, N, G)
        # Average phase across groups (circular mean)
        z = torch.exp(1j * phases_per_group.to(torch.cfloat))  # (B, N, G)
        z_mean = z.mean(dim=-1)  # (B, N)
        mean_phase = torch.angle(z_mean).float()  # (B, N)

        return mean_phase, x_state

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
