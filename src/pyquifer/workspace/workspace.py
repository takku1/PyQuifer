"""
Main Global Workspace implementation.

Contains the complete GlobalWorkspace system that integrates salience
computation, ignition dynamics, competition, precision weighting,
and global broadcasting into a unified consciousness bottleneck.

Based on: Baars (1988), Dehaene et al. (2014), Mashour et al. (2020)
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from pyquifer.workspace.broadcast import GlobalBroadcast
from pyquifer.workspace.competition import (
    CompetitionDynamics,
    IgnitionDynamics,
    PrecisionWeighting,
    SalienceComputer,
)


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
