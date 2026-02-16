"""
Workspace ensemble and hierarchy for Global Workspace Theory.

Contains multi-workspace and hierarchical workspace architectures:

- DiversityTracker: Anti-collapse mechanism boosting underrepresented organs
- HierarchicalWorkspace: Multi-level workspace with up/down information flow
- CrossBleedGate: Phase-coherence-gated inter-workspace information flow
- WorkspaceEnsemble: N parallel workspaces with cross-bleed via standing broadcasts
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Any

from pyquifer.workspace.workspace import GlobalWorkspace
from pyquifer.workspace.broadcast import StandingBroadcast


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
