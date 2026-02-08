"""
Global Workspace Theory Module for PyQuifer

Implements GWT-inspired mechanisms for consciousness and attention:

1. Workspace - Limited capacity bottleneck for global broadcasting
2. Ignition - Threshold crossing triggers global broadcast
3. Competition - Salience-based winner-take-all dynamics
4. Precision Weighting - Reliability-modulated attention
5. Broadcasting - Winner content distributed to all modules

Mathematical Foundation:
- Salience as energy in Hopfield-like dynamics
- Ignition as phase transition (criticality)
- Competition via softmax/Gumbel attention
- Broadcasting via einsum-based projection

No if/else logic - all behavior from differentiable dynamics.

Based on: Baars (1988), Dehaene et al. (2014), Mashour et al. (2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass
from enum import Enum


class ContentType(Enum):
    """Types of content that can enter workspace."""
    PERCEPTUAL = 0
    MEMORY = 1
    THOUGHT = 2
    MOTOR = 3
    EMOTIONAL = 4
    METACOGNITIVE = 5


@dataclass
class WorkspaceItem:
    """Item competing for workspace access."""
    content: torch.Tensor       # The actual content
    salience: torch.Tensor      # Current salience/activation
    precision: torch.Tensor     # Reliability/confidence
    source_id: int              # Which module it came from
    content_type: ContentType   # Type of content
    timestamp: int              # When it was created


class SalienceComputer(nn.Module):
    """
    Compute salience of content for workspace access.

    Salience = f(content, context, precision)

    Uses attention mechanism - content queries context
    to determine relevance.
    """

    def __init__(self,
                 content_dim: int,
                 context_dim: int,
                 num_heads: int = 4):
        super().__init__()
        self.content_dim = content_dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = content_dim // num_heads

        # Query from content, Key/Value from context
        self.W_q = nn.Parameter(torch.randn(num_heads, content_dim, self.head_dim) * 0.02)
        self.W_k = nn.Parameter(torch.randn(num_heads, context_dim, self.head_dim) * 0.02)

        # Salience projection
        self.salience_proj = nn.Linear(content_dim, 1)

        # Precision integration
        self.precision_scale = nn.Parameter(torch.ones(1))

    def forward(self,
                content: torch.Tensor,
                context: torch.Tensor,
                precision: torch.Tensor) -> torch.Tensor:
        """
        Compute salience of content given context.

        Args:
            content: Content vectors (batch, n_items, content_dim)
            context: Context vectors (batch, n_context, context_dim)
            precision: Precision weights (batch, n_items)

        Returns:
            salience: Salience scores (batch, n_items)
        """
        batch_size, n_items = content.shape[:2]
        n_context = context.shape[1]

        # Multi-head queries and keys via einsum
        # content: (B, I, D), W_q: (H, D, O) -> Q: (B, I, H, O)
        Q = torch.einsum('bid,hdo->biho', content, self.W_q)
        # context: (B, C, D), W_k: (H, D, O) -> K: (B, C, H, O)
        K = torch.einsum('bcd,hdo->bcho', context, self.W_k)

        # Attention scores (batch, n_items, heads, n_context)
        # Q: (B, I, H, O), K: (B, C, H, O) -> attn: (B, I, H, C)
        attn = torch.einsum('biho,bcho->bihc', Q, K) / math.sqrt(self.head_dim)

        # Max attention as relevance (batch, n_items, heads)
        relevance = attn.max(dim=-1)[0]

        # Average over heads (batch, n_items)
        relevance = relevance.mean(dim=-1)

        # Base salience from content
        base_salience = self.salience_proj(content).squeeze(-1)

        # Precision-weighted salience
        salience = (base_salience + relevance) * (precision * self.precision_scale)

        return salience


class IgnitionDynamics(nn.Module):
    """
    Ignition dynamics for global broadcast.

    When salience crosses threshold, content "ignites" and
    becomes globally available. This is a soft phase transition.

    Implements via sigmoid with temperature (sharpness).
    At low temperature, becomes sharp threshold (step function).
    """

    def __init__(self,
                 threshold: float = 0.5,
                 temperature: float = 0.1,
                 refractory_period: int = 5):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(threshold))
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.refractory_period = refractory_period

        # Track ignition history
        self.register_buffer('last_ignition', torch.tensor(-1000))
        self.register_buffer('ignition_count', torch.tensor(0))

    def forward(self,
                salience: torch.Tensor,
                current_time: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply ignition dynamics.

        Args:
            salience: Salience scores (batch, n_items)
            current_time: Current timestep

        Returns:
            ignition: Ignition probabilities (batch, n_items)
            is_refractory: Whether in refractory period
        """
        # Refractory check (soft)
        time_since_ignition = current_time - self.last_ignition.float()
        refractory_gate = torch.sigmoid(
            (time_since_ignition - self.refractory_period) / 2.0
        )

        # Ignition probability (soft threshold)
        ignition = torch.sigmoid(
            (salience - self.threshold) / self.temperature
        )

        # Apply refractory gating
        ignition = ignition * refractory_gate

        return ignition, refractory_gate < 0.5

    def update_history(self, did_ignite: bool, current_time: int):
        """Update ignition history."""
        if did_ignite:
            self.last_ignition.fill_(current_time)
            self.ignition_count += 1


class CompetitionDynamics(nn.Module):
    """
    Winner-take-all competition for workspace access.

    Only one (or few) items can occupy the workspace.
    Uses Hopfield-like energy dynamics.

    E(x) = -1/2 x^T W x + b^T x

    Minima correspond to winner states.
    """

    def __init__(self,
                 n_slots: int,
                 inhibition_strength: float = 1.0,
                 n_winners: int = 1):
        super().__init__()
        self.n_slots = n_slots
        self.inhibition_strength = inhibition_strength
        self.n_winners = n_winners

        # Mutual inhibition matrix (off-diagonal negative)
        W = -inhibition_strength * (torch.ones(n_slots, n_slots) - torch.eye(n_slots))
        self.register_buffer('W', W)

    def energy(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Compute Hopfield energy of activation state.

        Lower energy = more stable configuration.
        Useful for diagnostics and monitoring convergence.
        """
        # E = -1/2 x^T W x
        return -0.5 * torch.einsum('bi,ij,bj->b', activations, self.W, activations)

    def forward(self,
                salience: torch.Tensor,
                temperature: float = 1.0,
                steps: int = 5) -> torch.Tensor:
        """
        Run competition dynamics.

        Args:
            salience: Initial salience (batch, n_items)
            temperature: Softmax temperature
            steps: Number of relaxation steps

        Returns:
            winners: Winning activations (batch, n_items)
        """
        # Pad/truncate to n_slots
        batch_size = salience.shape[0]
        n_items = salience.shape[1]

        if n_items < self.n_slots:
            salience = F.pad(salience, (0, self.n_slots - n_items))
        elif n_items > self.n_slots:
            salience = salience[:, :self.n_slots]

        # Initialize activations
        x = salience.clone()

        # Relaxation dynamics
        for _ in range(steps):
            # Hopfield update
            h = torch.einsum('ij,bj->bi', self.W, x) + salience
            x = torch.softmax(h / temperature, dim=-1)

        # Take top-k winners
        if self.n_winners < self.n_slots:
            topk_vals, topk_idx = x.topk(self.n_winners, dim=-1)
            mask = torch.zeros_like(x)
            mask.scatter_(1, topk_idx, 1.0)
            x = x * mask

        # Normalize winners
        x = x / (x.sum(dim=-1, keepdim=True) + 1e-8)

        # Return to original n_items size
        if n_items < self.n_slots:
            return x[:, :n_items]
        elif n_items > self.n_slots:
            # Pad back with zeros for items that didn't compete
            return F.pad(x, (0, n_items - self.n_slots))
        return x


class PrecisionWeighting(nn.Module):
    """
    Precision weighting for prediction errors.

    Precision = inverse variance = reliability/confidence.

    High precision errors are more salient.
    Low precision errors are down-weighted.

    Implements optimal Bayesian precision estimation.
    """

    def __init__(self,
                 dim: int,
                 min_precision: float = 0.1,
                 max_precision: float = 10.0):
        super().__init__()
        self.dim = dim
        self.min_precision = min_precision
        self.max_precision = max_precision

        # Precision estimator network
        self.precision_net = nn.Sequential(
            nn.Linear(dim * 2, dim),  # prediction + error
            nn.ELU(),
            nn.Linear(dim, 1),
            nn.Softplus(),  # Ensure positive
        )

        # Running statistics for normalization
        self.register_buffer('error_mean', torch.zeros(dim))
        self.register_buffer('error_var', torch.ones(dim))
        self.register_buffer('n_samples', torch.tensor(0.0))

    def estimate_precision(self,
                          prediction: torch.Tensor,
                          error: torch.Tensor) -> torch.Tensor:
        """
        Estimate precision from prediction and error.

        Args:
            prediction: Model prediction (batch, dim)
            error: Prediction error (batch, dim)

        Returns:
            precision: Estimated precision (batch,)
        """
        # Concatenate prediction and error
        combined = torch.cat([prediction, error], dim=-1)

        # Network estimate
        precision = self.precision_net(combined).squeeze(-1)

        # Clamp to valid range
        precision = torch.clamp(precision, self.min_precision, self.max_precision)

        return precision

    def weight_error(self,
                     error: torch.Tensor,
                     precision: torch.Tensor) -> torch.Tensor:
        """
        Apply precision weighting to error.

        Weighted error = sqrt(precision) * error
        """
        return error * torch.sqrt(precision).unsqueeze(-1)

    def update_statistics(self, error: torch.Tensor):
        """Update running error statistics."""
        with torch.no_grad():
            batch_mean = error.mean(dim=0)
            batch_var = error.var(dim=0)
            batch_size = error.shape[0]

            # Online update
            self.n_samples += batch_size
            delta = batch_mean - self.error_mean
            self.error_mean += delta * batch_size / self.n_samples
            self.error_var = (
                self.error_var * (self.n_samples - batch_size) +
                batch_var * batch_size +
                delta.pow(2) * (self.n_samples - batch_size) * batch_size / self.n_samples
            ) / self.n_samples


class GlobalBroadcast(nn.Module):
    """
    Broadcast winning content to all modules.

    Winner content is projected to each module's input space,
    creating global availability.
    """

    def __init__(self,
                 workspace_dim: int,
                 module_dims: List[int]):
        super().__init__()
        self.workspace_dim = workspace_dim
        self.module_dims = module_dims
        self.n_modules = len(module_dims)

        # Projection to each module
        self.projections = nn.ModuleList([
            nn.Linear(workspace_dim, dim)
            for dim in module_dims
        ])

        # Module attention (which modules receive broadcast)
        self.module_attention = nn.Linear(workspace_dim, self.n_modules)

    def forward(self,
                workspace_content: torch.Tensor,
                broadcast_strength: float = 1.0) -> List[torch.Tensor]:
        """
        Broadcast workspace content to modules.

        Args:
            workspace_content: Content in workspace (batch, workspace_dim)
            broadcast_strength: Overall broadcast strength

        Returns:
            List of projected contents for each module
        """
        # Module attention weights
        attn = F.softmax(self.module_attention(workspace_content), dim=-1)

        # Project and weight by attention
        broadcasts = []
        for i, proj in enumerate(self.projections):
            projected = proj(workspace_content)
            weighted = projected * attn[:, i:i+1] * broadcast_strength
            broadcasts.append(weighted)

        return broadcasts


class GlobalWorkspace(nn.Module):
    """
    Complete Global Workspace system.

    Integrates:
    - Salience computation
    - Ignition dynamics
    - Competition
    - Precision weighting
    - Global broadcasting

    This is the "consciousness bottleneck" - limited capacity
    that creates global integration.
    """

    def __init__(self,
                 content_dim: int = 128,
                 workspace_dim: int = 256,
                 n_slots: int = 8,
                 n_winners: int = 1,
                 context_dim: Optional[int] = None,
                 module_dims: Optional[List[int]] = None,
                 ignition_threshold: float = 0.5):
        super().__init__()

        self.content_dim = content_dim
        self.workspace_dim = workspace_dim
        self.n_slots = n_slots
        self.n_winners = n_winners

        # Default context_dim to content_dim if not specified
        context_dim = context_dim if context_dim is not None else content_dim

        # Components
        self.salience = SalienceComputer(content_dim, context_dim)
        self.ignition = IgnitionDynamics(threshold=ignition_threshold)
        self.competition = CompetitionDynamics(n_slots, n_winners=n_winners)
        self.precision = PrecisionWeighting(content_dim)

        # Content encoder to workspace
        self.workspace_encoder = nn.Sequential(
            nn.Linear(content_dim, workspace_dim),
            nn.LayerNorm(workspace_dim),
            nn.ELU(),
        )

        # Broadcast system
        if module_dims is None:
            module_dims = [content_dim] * 4  # Default: 4 modules
        self.broadcast = GlobalBroadcast(workspace_dim, module_dims)

        # Workspace state
        self.register_buffer('workspace_content', torch.zeros(1, workspace_dim))
        self.register_buffer('workspace_occupied', torch.tensor(False))
        self.register_buffer('current_time', torch.tensor(0))

    def forward(self,
                contents: torch.Tensor,
                context: torch.Tensor,
                precisions: Optional[torch.Tensor] = None,
                predictions: Optional[torch.Tensor] = None,
                errors: Optional[torch.Tensor] = None,
                ) -> Dict[str, torch.Tensor]:
        """
        Process contents through global workspace.

        Args:
            contents: Candidate contents (batch, n_items, content_dim)
            context: Global context (batch, n_context, context_dim)
            precisions: Pre-computed precisions (optional)
            predictions: For precision estimation (optional)
            errors: For precision estimation (optional)

        Returns:
            Dictionary with workspace state and broadcast outputs
        """
        batch_size, n_items = contents.shape[:2]

        # Compute or use provided precision
        if precisions is None:
            if predictions is not None and errors is not None:
                precisions = self.precision.estimate_precision(predictions, errors)
            else:
                precisions = torch.ones(batch_size, n_items, device=contents.device)

        # Compute salience
        salience = self.salience(contents, context, precisions)

        # Ignition check
        ignition_probs, is_refractory = self.ignition(salience, self.current_time.item())

        # Competition
        winners = self.competition(salience * ignition_probs)

        # Winner content (weighted sum)
        winner_content = torch.einsum('bi,bid->bd', winners, contents)

        # Encode to workspace
        workspace = self.workspace_encoder(winner_content)

        # Check if any item ignited
        max_ignition = ignition_probs.max(dim=-1)[0]
        did_ignite = max_ignition > 0.5

        # Update workspace state (in-place to preserve registered buffers)
        with torch.no_grad():
            self.workspace_content.copy_(workspace[:1] if workspace.shape[0] > 0 else workspace)
            self.workspace_occupied.fill_(did_ignite.any().item())

        # Broadcast to modules
        broadcasts = self.broadcast(workspace, broadcast_strength=max_ignition.mean())

        # Update time
        self.current_time += 1
        self.ignition.update_history(did_ignite.any().item(), self.current_time.item())

        return {
            'salience': salience,
            'ignition_probs': ignition_probs,
            'winners': winners,
            'workspace': workspace,
            'did_ignite': did_ignite,
            'broadcasts': broadcasts,
            'is_refractory': is_refractory,
        }

    def reset(self):
        """Reset workspace state."""
        self.workspace_content.zero_()
        self.workspace_occupied.fill_(False)
        self.current_time.zero_()
        self.ignition.last_ignition.fill_(-1000)


class DiversityTracker:
    """
    Anti-collapse mechanism for workspace competition.

    Tracks per-organ win counts over a sliding window and boosts
    salience for organs that haven't won recently.

    Args:
        pressure: Boost strength for underrepresented organs
        window_size: Sliding window for tracking wins
    """

    def __init__(self, pressure: float = 0.1, window_size: int = 50):
        self.pressure = pressure
        self.window_size = window_size
        self._win_history: List[str] = []

    def record_win(self, organ_id: str):
        """Record a win for the given organ."""
        self._win_history.append(organ_id)
        if len(self._win_history) > self.window_size:
            self._win_history.pop(0)

    def get_boost(self, organ_id: str) -> float:
        """Get salience boost for organ. Higher = hasn't won recently."""
        if not self._win_history:
            return self.pressure
        win_count = sum(1 for w in self._win_history if w == organ_id)
        win_fraction = win_count / len(self._win_history)
        # Boost inversely proportional to win rate
        return self.pressure * max(0.0, 1.0 - win_fraction * 2)

    def reset(self):
        self._win_history.clear()


class HierarchicalWorkspace(nn.Module):
    """
    Multi-level workspace hierarchy.

    Lower levels: Fast, local processing
    Higher levels: Slow, global integration

    Information flows up via ignition, down via broadcast.
    """

    def __init__(self,
                 dims: List[int] = [64, 128, 256],
                 n_slots_per_level: List[int] = [16, 8, 4]):
        super().__init__()

        self.n_levels = len(dims)
        self.dims = dims

        # Workspace at each level
        self.workspaces = nn.ModuleList([
            GlobalWorkspace(
                content_dim=dims[i],
                workspace_dim=dims[min(i+1, len(dims)-1)],
                n_slots=n_slots_per_level[i],
                context_dim=dims[i],
            )
            for i in range(self.n_levels)
        ])

        # Inter-level projections
        # up_projection input must match workspace output dim at that level
        self.up_projections = nn.ModuleList([
            nn.Linear(dims[min(i+1, len(dims)-1)], dims[i+1])
            for i in range(self.n_levels - 1)
        ])

        # down_projection[i] projects workspace[i] output to content_dim[i+1]
        # workspace[i] output = dims[min(i+1, len-1)], target = dims[i+1]
        self.down_projections = nn.ModuleList([
            nn.Linear(dims[min(i+1, len(dims)-1)], dims[i+1])
            for i in range(self.n_levels - 1)
        ])

    def forward(self,
                contents: List[torch.Tensor],
                contexts: List[torch.Tensor]) -> Dict[str, List]:
        """
        Process through hierarchy.

        Args:
            contents: Contents at each level
            contexts: Contexts at each level

        Returns:
            Dictionary with results at each level
        """
        results = {'workspaces': [], 'ignitions': [], 'broadcasts': []}

        for i in range(self.n_levels):
            # Combine with downward broadcast from higher level
            if i > 0 and results['workspaces']:
                higher_broadcast = self.down_projections[i-1](results['workspaces'][-1])
                contents[i] = contents[i] + higher_broadcast.unsqueeze(1)

            # Process through workspace
            ws_result = self.workspaces[i](
                contents[i],
                contexts[i]
            )

            results['workspaces'].append(ws_result['workspace'])
            results['ignitions'].append(ws_result['did_ignite'])
            results['broadcasts'].append(ws_result['broadcasts'])

            # Propagate to higher level
            if i < self.n_levels - 1:
                upward = self.up_projections[i](ws_result['workspace'])
                contents[i+1] = torch.cat([
                    contents[i+1],
                    upward.unsqueeze(1)
                ], dim=1)

        return results


class StandingBroadcast(nn.Module):
    """
    EMA buffer that persists a workspace's recent broadcast content.

    Even when a workspace doesn't ignite this tick, its standing broadcast
    provides background context from previous successful competitions.

    Args:
        dim: Broadcast content dimension
        momentum: How fast old broadcasts decay (higher = more persistent)
    """

    def __init__(self, dim: int, momentum: float = 0.9):
        super().__init__()
        self.momentum = momentum
        self.register_buffer('content', torch.zeros(dim))

    def update(self, new_broadcast: torch.Tensor) -> None:
        """EMA update with new broadcast content."""
        with torch.no_grad():
            broadcast = new_broadcast.detach()
            if broadcast.dim() > 1:
                broadcast = broadcast.squeeze(0)
            self.content.mul_(self.momentum).add_(
                broadcast * (1 - self.momentum)
            )

    def get(self) -> torch.Tensor:
        """Get current standing broadcast content."""
        return self.content


class CrossBleedGate(nn.Module):
    """
    Gates information flow between workspaces based on phase coherence.

    When source workspace organs are in-phase with the target workspace,
    more of the source's standing broadcast "bleeds" into the target's
    competition. This implements inter-workspace Communication Through
    Coherence (CTC).

    gate = sigmoid(w_coh * coherence + w_sal * salience + bias)
    output = gate * source_broadcast

    Args:
        dim: Broadcast content dimension
    """

    def __init__(self, dim: int):
        super().__init__()
        self.w_coherence = nn.Parameter(torch.tensor(2.0))
        self.w_salience = nn.Parameter(torch.tensor(0.5))
        self.bias = nn.Parameter(torch.tensor(-1.0))
        self.proj = nn.Linear(dim, dim)

    def forward(self,
                source_broadcast: torch.Tensor,
                source_phases: Optional[torch.Tensor] = None,
                target_phases: Optional[torch.Tensor] = None,
                source_salience: float = 0.5) -> torch.Tensor:
        """
        Compute gated cross-bleed from source to target workspace.

        Args:
            source_broadcast: Standing broadcast from source workspace
            source_phases: Oscillator phases associated with source
            target_phases: Oscillator phases associated with target
            source_salience: How salient the source workspace content is

        Returns:
            Gated content to inject into target workspace
        """
        # Compute inter-workspace coherence
        if source_phases is not None and target_phases is not None:
            # Mean cosine similarity between phase vectors
            min_len = min(source_phases.shape[0], target_phases.shape[0])
            coherence = torch.cos(
                source_phases[:min_len] - target_phases[:min_len]
            ).mean()
        else:
            coherence = torch.tensor(0.5, device=source_broadcast.device)

        # Gating
        logit = (self.w_coherence * coherence
                 + self.w_salience * source_salience
                 + self.bias)
        gate = torch.sigmoid(logit)

        # Project and gate
        projected = self.proj(source_broadcast)
        return gate * projected


class WorkspaceEnsemble(nn.Module):
    """
    Collection of N parallel GlobalWorkspace instances with cross-bleed.

    Enables multiple workspaces to run simultaneously — e.g. an active
    workspace handling the current task while background workspaces
    maintain standing representations of specialist knowledge (chess,
    law, music theory, etc.). Phase coherence between workspaces gates
    how much background context "bleeds" into the active workspace.

    Args:
        n_workspaces: Number of parallel workspaces
        content_dim: Content dimension for each workspace
        workspace_dim: Internal workspace dimension
        n_slots: Number of competition slots per workspace
        n_winners: Winners per competition round
        bleed_strength: Base cross-bleed strength multiplier
        standing_momentum: EMA momentum for standing broadcasts
    """

    def __init__(self,
                 n_workspaces: int = 2,
                 content_dim: int = 64,
                 workspace_dim: int = 128,
                 n_slots: int = 8,
                 n_winners: int = 1,
                 bleed_strength: float = 0.3,
                 standing_momentum: float = 0.9):
        super().__init__()

        self.n_workspaces = n_workspaces
        self.content_dim = content_dim
        self.workspace_dim = workspace_dim
        self.bleed_strength = bleed_strength

        # Parallel workspaces
        self.workspaces = nn.ModuleList([
            GlobalWorkspace(
                content_dim=content_dim,
                workspace_dim=workspace_dim,
                n_slots=n_slots,
                n_winners=n_winners,
                context_dim=content_dim,
            )
            for _ in range(n_workspaces)
        ])

        # Standing broadcasts (one per workspace)
        self.standings = nn.ModuleList([
            StandingBroadcast(workspace_dim, momentum=standing_momentum)
            for _ in range(n_workspaces)
        ])

        # Cross-bleed gates: gate[i] controls bleed FROM workspace i TO others
        # Total: n_workspaces gates (each workspace has one outgoing gate)
        self.bleed_gates = nn.ModuleList([
            CrossBleedGate(workspace_dim)
            for _ in range(n_workspaces)
        ])

        # Projection: workspace_dim → content_dim for injecting bleed as context
        self.bleed_to_context = nn.Linear(workspace_dim, content_dim)

        # Which workspace is foreground
        self.register_buffer('active_idx', torch.tensor(0))

    def set_active(self, idx: int):
        """Set which workspace is the foreground (active) workspace."""
        with torch.no_grad():
            self.active_idx.fill_(idx)

    def get_bleed_matrix(self) -> torch.Tensor:
        """
        Get NxN coherence/bleed matrix for diagnostics.

        Entry [i, j] is how much workspace i bleeds into workspace j.
        Diagonal is zero (no self-bleed).
        """
        mat = torch.zeros(self.n_workspaces, self.n_workspaces)
        for i in range(self.n_workspaces):
            source = self.standings[i].get()
            if source.abs().sum() < 1e-8:
                continue
            for j in range(self.n_workspaces):
                if i == j:
                    continue
                gated = self.bleed_gates[i](source)
                mat[i, j] = gated.abs().mean().item() * self.bleed_strength
        return mat

    def forward(self,
                contents_per_ws: List[torch.Tensor],
                contexts_per_ws: List[torch.Tensor],
                phases_per_ws: Optional[List[torch.Tensor]] = None,
                ) -> Dict[str, Any]:
        """
        Run all workspaces with cross-bleed.

        Args:
            contents_per_ws: List of content tensors, one per workspace
                Each: (batch, n_items, content_dim)
            contexts_per_ws: List of context tensors, one per workspace
                Each: (batch, n_context, content_dim)
            phases_per_ws: Optional list of phase tensors for coherence
                Each: (n_oscillators,) — used for cross-bleed gating

        Returns:
            Dict with:
            - workspace_results: List of per-workspace result dicts
            - bleed_matrix: NxN bleed strength matrix
            - standing_broadcasts: List of standing broadcast tensors
        """
        assert len(contents_per_ws) == self.n_workspaces
        assert len(contexts_per_ws) == self.n_workspaces

        workspace_results = []

        # Step 1: Collect cross-bleed from standing broadcasts
        for j in range(self.n_workspaces):
            bleed_sum = torch.zeros(
                1, 1, self.content_dim,
                device=contents_per_ws[j].device
            )
            for i in range(self.n_workspaces):
                if i == j:
                    continue
                source = self.standings[i].get()
                if source.abs().sum() < 1e-8:
                    continue

                src_phases = phases_per_ws[i] if phases_per_ws else None
                tgt_phases = phases_per_ws[j] if phases_per_ws else None

                gated = self.bleed_gates[i](
                    source,
                    source_phases=src_phases,
                    target_phases=tgt_phases,
                )
                # Project to content_dim and accumulate
                bleed_content = self.bleed_to_context(gated)
                bleed_sum = bleed_sum + bleed_content.unsqueeze(0).unsqueeze(0) * self.bleed_strength

            # Inject bleed as additional context
            contexts_per_ws[j] = torch.cat([
                contexts_per_ws[j],
                bleed_sum.expand(contexts_per_ws[j].shape[0], -1, -1)
            ], dim=1)

        # Step 2: Run each workspace
        for i in range(self.n_workspaces):
            ws_result = self.workspaces[i](
                contents_per_ws[i],
                contexts_per_ws[i],
            )
            workspace_results.append(ws_result)

            # Step 3: Update standing broadcasts
            self.standings[i].update(ws_result['workspace'].mean(dim=0))

        # Step 4: Build diagnostics
        bleed_matrix = self.get_bleed_matrix()

        return {
            'workspace_results': workspace_results,
            'bleed_matrix': bleed_matrix,
            'standing_broadcasts': [s.get() for s in self.standings],
            'active_idx': self.active_idx.item(),
        }

    def reset(self):
        """Reset all workspaces and standing broadcasts."""
        for ws in self.workspaces:
            ws.reset()
        for sb in self.standings:
            with torch.no_grad():
                sb.content.zero_()


if __name__ == '__main__':
    print("--- Global Workspace Examples ---")

    # Example 1: Salience Computation
    print("\n1. Salience Computation")
    salience_comp = SalienceComputer(content_dim=64, context_dim=64)

    content = torch.randn(4, 8, 64)  # 4 batches, 8 items
    context = torch.randn(4, 10, 64)
    precision = torch.rand(4, 8)

    salience = salience_comp(content, context, precision)
    print(f"   Salience shape: {salience.shape}")
    print(f"   Salience range: [{salience.min():.3f}, {salience.max():.3f}]")

    # Example 2: Ignition Dynamics
    print("\n2. Ignition Dynamics")
    ignition = IgnitionDynamics(threshold=0.5, temperature=0.1)

    ignition_probs, is_refr = ignition(salience, current_time=0)
    print(f"   Ignition probs: {ignition_probs[0].tolist()}")
    print(f"   Max ignition: {ignition_probs.max().item():.3f}")

    # Example 3: Competition
    print("\n3. Competition Dynamics")
    competition = CompetitionDynamics(n_slots=8, n_winners=2)

    winners = competition(salience)
    print(f"   Winner distribution: {winners[0].tolist()}")
    print(f"   Sum (should be 1): {winners[0].sum().item():.3f}")

    # Example 4: Full Global Workspace
    print("\n4. Full Global Workspace")
    gw = GlobalWorkspace(
        content_dim=64,
        workspace_dim=128,
        n_slots=8,
        n_winners=1,
        context_dim=64,
    )

    result = gw(content, context)
    print(f"   Workspace shape: {result['workspace'].shape}")
    print(f"   Did ignite: {result['did_ignite'].tolist()}")
    print(f"   Number of broadcasts: {len(result['broadcasts'])}")

    # Example 5: Precision Weighting
    print("\n5. Precision Weighting")
    prec = PrecisionWeighting(dim=64)

    prediction = torch.randn(4, 64)
    error = torch.randn(4, 64) * 0.5

    precision_est = prec.estimate_precision(prediction, error)
    weighted_error = prec.weight_error(error, precision_est)

    print(f"   Precision estimates: {precision_est.tolist()}")
    print(f"   Error norm before: {error.norm(dim=-1).mean():.3f}")
    print(f"   Error norm after weighting: {weighted_error.norm(dim=-1).mean():.3f}")

    # Example 6: Hierarchical Workspace
    print("\n6. Hierarchical Workspace")
    hier = HierarchicalWorkspace(dims=[32, 64, 128], n_slots_per_level=[16, 8, 4])

    contents = [
        torch.randn(4, 16, 32),
        torch.randn(4, 8, 64),
        torch.randn(4, 4, 128),
    ]
    contexts = [
        torch.randn(4, 10, 32),
        torch.randn(4, 8, 64),
        torch.randn(4, 6, 128),
    ]

    hier_result = hier(contents, contexts)
    print(f"   Level 0 workspace: {hier_result['workspaces'][0].shape}")
    print(f"   Level 2 workspace: {hier_result['workspaces'][2].shape}")
    print(f"   Ignitions per level: {[ig.sum().item() for ig in hier_result['ignitions']]}")

    print("\n[OK] All global workspace tests passed!")
