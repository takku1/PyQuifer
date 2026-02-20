"""
Workspace broadcasting mechanisms for Global Workspace Theory.

Contains the broadcast systems that distribute winning workspace
content to all subscribed modules:

- GlobalBroadcast: Projects workspace content to module-specific spaces
- StandingBroadcast: EMA buffer persisting recent broadcast content
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


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
