"""
MCP-as-Organ Protocol for PyQuifer.

Wraps external MCP (Model Context Protocol) resources as workspace-competing
organs in the Global Workspace. Each MCP server's resource endpoint becomes
a first-class specialist that can win broadcast attention alongside internal
organs (HPC, Motivation, Selection).

Design principles:
- Non-blocking: observe() reads from a pre-fetched tensor cache, never I/O.
- Salience from change: big state change = high salience (something happened).
- Freshness decay: stale cached state → decaying salience → graceful dropout.
- Simple encoder: Linear + Tanh, trainable via PreGWAdapter cycle consistency.

The async poller that feeds the cache lives in Mizuki's runtime layer,
not here. This module is pure PyTorch, no asyncio or MCP imports.

References:
- Baars (1988). A Cognitive Theory of Consciousness.
- Fries (2005). Communication Through Coherence (CTC).
"""

import time
from dataclasses import dataclass, field
from typing import Optional, Set

import torch
import torch.nn as nn

from pyquifer.workspace.organ_base import Organ, Proposal


@dataclass
class MCPOrganConfig:
    """Configuration for one MCP organ instance.

    Attributes:
        organ_id: Unique identifier, e.g. "mcp:github:notifications".
        resource_uri: MCP resource URI to poll, e.g. "github://notifications".
        latent_dim: Native dimension of this organ's internal state.
        base_salience: Resting salience when nothing has changed (0.0-1.0).
        salience_decay: Multiplicative decay per tick when state is stale.
        cost: Computational cost estimate for workspace competition.
        tags: Domain tags for workspace diversity tracking.
        poll_stale_after: Seconds before cached state is considered stale.
    """
    organ_id: str
    resource_uri: str
    latent_dim: int = 64
    base_salience: float = 0.2
    salience_decay: float = 0.95
    cost: float = 0.1
    tags: Set[str] = field(default_factory=lambda: {"external", "mcp"})
    poll_stale_after: float = 5.0


class MCPOrgan(Organ):
    """Wraps an MCP resource endpoint as a PyQuifer workspace-competing organ.

    Lifecycle per tick:
    1. observe() — reads cached resource tensor, encodes to latent
    2. propose() — computes salience from state change + freshness
    3. accept() — updates standing latent from broadcast winner

    Resource polling happens OUTSIDE the tick loop (async, in Mizuki runtime).
    The organ reads from a pre-fetched cache, never blocks the tick.

    Thread safety: update_cache() clones tensors. Worst case the tick reads
    one poll cycle behind. This is acceptable for 1 Hz external polling.
    """

    def __init__(self, config: MCPOrganConfig):
        super().__init__(
            organ_id=config.organ_id,
            latent_dim=config.latent_dim,
        )
        self.config = config

        # State cache (written by async poller, read by observe)
        self.register_buffer('_cached_state', torch.zeros(config.latent_dim))
        self._has_state = False
        self._prev_state: Optional[torch.Tensor] = None
        self._state_timestamp: float = 0.0
        self._change_magnitude: float = 0.0

        # Encoder: raw MCP state → latent_dim
        # Simple projection, trainable via PreGWAdapter cycle consistency
        self._encoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim),
            nn.Tanh(),
        )

        # Internal latent (observe output, propose input)
        self.register_buffer('_latent', torch.zeros(config.latent_dim))

    def update_cache(self, state_tensor: torch.Tensor, timestamp: float):
        """Called by async poller (NOT during tick). Thread-safe via clone.

        Args:
            state_tensor: Tensor of shape (latent_dim,) encoded from MCP resource.
            timestamp: time.time() when the resource was polled.
        """
        with torch.no_grad():
            # Save previous for change detection
            if self._has_state:
                self._prev_state = self._cached_state.clone()

            # Pad/trim to latent_dim
            if state_tensor.shape[-1] >= self.config.latent_dim:
                self._cached_state.copy_(state_tensor[:self.config.latent_dim])
            else:
                self._cached_state.zero_()
                self._cached_state[:state_tensor.shape[-1]] = state_tensor

            self._state_timestamp = timestamp
            self._has_state = True

            # Compute change magnitude for salience
            if self._prev_state is not None:
                self._change_magnitude = float(
                    (self._cached_state - self._prev_state).norm().item()
                )
            else:
                self._change_magnitude = self.config.base_salience

    def observe(self, sensory_input: torch.Tensor,
                global_broadcast: Optional[torch.Tensor] = None) -> None:
        """Read cached state into local latent. Never blocks on I/O."""
        if self._has_state:
            self._latent = self._encoder(self._cached_state)
        # Also incorporate broadcast if available
        if global_broadcast is not None:
            self.update_standing(global_broadcast)

    def propose(self, gw_state: Optional[torch.Tensor] = None) -> Proposal:
        """Compute salience from state change magnitude + staleness.

        Salience formula:
            raw = base_salience * 0.3 + change_magnitude * 0.7
            freshness = clamp(1.0 - staleness / poll_stale_after, 0, 1)
            salience = raw * freshness
        """
        if not self._has_state:
            return Proposal(
                content=self.standing_latent,
                salience=0.0,
                tags=self.config.tags,
                cost=self.config.cost,
                organ_id=self.config.organ_id,
            )

        # Freshness: 1.0 when just polled, decays to 0.0 when stale
        staleness = time.time() - self._state_timestamp
        freshness = max(0.0, min(1.0, 1.0 - staleness / self.config.poll_stale_after))

        salience = (
            self.config.base_salience * 0.3
            + self._change_magnitude * 0.7
        ) * freshness

        return Proposal(
            content=self._latent.detach(),
            salience=float(salience),
            tags=self.config.tags,
            cost=self.config.cost,
            organ_id=self.config.organ_id,
        )

    def accept(self, global_broadcast: torch.Tensor) -> None:
        """Update standing latent from broadcast winner."""
        self.update_standing(global_broadcast)
