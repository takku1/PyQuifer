"""
Organ Protocol + GWT Wiring for PyQuifer

Implements the Oscillating Global Workspace Network (OGWN) protocol:
- Organ ABC: Specialist modules that compete for workspace broadcast
- OscillatoryWriteGate: Phase-coherence-based gating (CTC)
- PreGWAdapter: Projects organ-specific latent to common workspace dim
- Proposal: Dataclass for workspace competition entries

Example concrete organs wrapping existing PyQuifer modules:
- HPCOrgan: Wraps HierarchicalPredictiveCoding
- MotivationOrgan: Wraps IntrinsicMotivationSystem
- SelectionOrgan: Wraps SelectionArena

References:
- Baars (1988). A Cognitive Theory of Consciousness.
- Dehaene & Naccache (2001). Towards a cognitive neuroscience of consciousness.
- Fries (2005). Communication Through Coherence (CTC).
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Set

import torch
import torch.nn as nn


@dataclass
class Proposal:
    """Entry for global workspace competition."""
    content: torch.Tensor         # Latent payload (workspace_dim after projection)
    salience: float               # Urgency/importance
    tags: Set[str] = field(default_factory=set)  # Domain labels
    cost: float = 0.0            # Predicted compute
    organ_id: str = ""           # Source organ


class Organ(nn.Module, ABC):
    """
    Abstract base class for specialist modules that compete for
    global workspace broadcast.

    Each Organ:
    - Has its own local oscillator phase/frequency
    - Observes sensory input + global broadcast
    - Proposes content for workspace competition
    - Accepts winner's broadcast to learn from it

    The local oscillator enables Communication Through Coherence (CTC):
    organs in phase with the global rhythm have higher salience.
    """

    def __init__(self, organ_id: str, latent_dim: int,
                 frequency: float = 1.0,
                 standing_momentum: float = 0.9):
        super().__init__()
        self.organ_id = organ_id
        self.latent_dim = latent_dim
        self.standing_momentum = standing_momentum

        # Local oscillator state (not learned — evolves by dynamics)
        self.register_buffer('phase', torch.tensor(0.0))
        self.register_buffer('frequency', torch.tensor(frequency))

        # Standing latent: EMA of recent broadcasts received
        # Always present even when organ isn't actively competing
        self.register_buffer('standing_latent', torch.zeros(latent_dim))

    def step_oscillator(self, dt: float = 0.01,
                        global_phase: Optional[torch.Tensor] = None,
                        coupling: float = 0.1):
        """Advance local oscillator phase, optionally coupled to global rhythm."""
        with torch.no_grad():
            d_phase = self.frequency * 2 * math.pi * dt
            if global_phase is not None:
                # Kuramoto coupling to global phase
                d_phase = d_phase + coupling * torch.sin(global_phase - self.phase)
            self.phase.add_(d_phase).remainder_(2 * math.pi)

    def update_standing(self, new_content: torch.Tensor):
        """Update standing latent representation via EMA.

        Called by accept() so the organ's representation is always present
        even when it's not actively competing this tick.
        """
        with torch.no_grad():
            content = new_content.detach()
            if content.dim() > 1:
                content = content.squeeze(0)
            # Trim/pad to latent_dim
            if content.shape[-1] > self.latent_dim:
                content = content[:self.latent_dim]
            elif content.shape[-1] < self.latent_dim:
                padded = torch.zeros(self.latent_dim, device=content.device)
                padded[:content.shape[-1]] = content
                content = padded
            self.standing_latent.mul_(self.standing_momentum).add_(
                content * (1 - self.standing_momentum)
            )

    @abstractmethod
    def observe(self, sensory_input: torch.Tensor,
                global_broadcast: Optional[torch.Tensor] = None) -> None:
        """Update local state from sensory input and global broadcast."""
        ...

    @abstractmethod
    def propose(self, gw_state: Optional[torch.Tensor] = None) -> Proposal:
        """Generate a proposal for workspace competition."""
        ...

    @abstractmethod
    def accept(self, global_broadcast: torch.Tensor) -> None:
        """Receive and learn from the winning broadcast."""
        ...


class OscillatoryWriteGate(nn.Module):
    """
    Gate that modulates proposal salience based on phase coherence
    between organ and global rhythm (Communication Through Coherence).

    gate = sigmoid(w_coh * cos(organ_phase - global_phase)
                   + w_nov * novelty + w_cost * (-cost) + bias)

    High coherence → high gate → more likely to win broadcast.
    """

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.w_coherence = nn.Parameter(torch.tensor(2.0))
        self.w_novelty = nn.Parameter(torch.tensor(1.0))
        self.w_cost = nn.Parameter(torch.tensor(-0.5))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, organ_phase: torch.Tensor,
                global_phase: torch.Tensor,
                novelty: float = 0.5,
                cost: float = 0.0) -> torch.Tensor:
        """
        Compute write gate value.

        Args:
            organ_phase: Local oscillator phase
            global_phase: Global workspace rhythm phase
            novelty: Novelty/surprise of the proposal content
            cost: Estimated computation cost

        Returns:
            gate: Scalar gate value in [0, 1]
        """
        coherence = torch.cos(organ_phase - global_phase)
        logit = (self.w_coherence * coherence
                 + self.w_novelty * novelty
                 + self.w_cost * cost
                 + self.bias)
        return torch.sigmoid(logit)


class PreGWAdapter(nn.Module):
    """
    Projects organ-specific latent → common workspace dimension.

    Encoder: organ_dim → workspace_dim (Linear + LayerNorm + ELU)
    Decoder: workspace_dim → organ_dim (for cycle-consistency loss)
    """

    def __init__(self, organ_dim: int, workspace_dim: int):
        super().__init__()
        self.organ_dim = organ_dim
        self.workspace_dim = workspace_dim

        self.encoder = nn.Sequential(
            nn.Linear(organ_dim, workspace_dim),
            nn.LayerNorm(workspace_dim),
            nn.ELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(workspace_dim, organ_dim),
            nn.LayerNorm(organ_dim),
            nn.ELU(),
        )

    def encode(self, organ_latent: torch.Tensor) -> torch.Tensor:
        """Project organ latent to workspace dimension."""
        return self.encoder(organ_latent)

    def decode(self, workspace_latent: torch.Tensor) -> torch.Tensor:
        """Project workspace latent back to organ dimension."""
        return self.decoder(workspace_latent)

    def cycle_consistency_loss(self, organ_latent: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss: encode → decode → compare."""
        workspace = self.encode(organ_latent)
        reconstructed = self.decode(workspace)
        return (organ_latent - reconstructed).pow(2).mean()


# ---------------------------------------------------------------------------
# Concrete Organ Examples
# ---------------------------------------------------------------------------

class HPCOrgan(Organ):
    """
    Wraps HierarchicalPredictiveCoding as a workspace-competing organ.

    Proposes prediction errors (bottom-up) — high free energy = high salience.
    Accepts broadcast to update top-level beliefs (top-down).
    """

    def __init__(self, latent_dim: int = 64, hierarchy_dims: Optional[list] = None,
                 frequency: float = 10.0):
        super().__init__(organ_id="hpc", latent_dim=latent_dim, frequency=frequency)
        if hierarchy_dims is None:
            hierarchy_dims = [latent_dim, latent_dim // 2, latent_dim // 4]

        from pyquifer.cognition.predictive.hierarchical import HierarchicalPredictiveCoding
        self.hpc = HierarchicalPredictiveCoding(
            level_dims=hierarchy_dims, lr=0.05, gen_lr=0.01
        )
        self._last_result = None
        self.register_buffer('_latent', torch.zeros(latent_dim))

    def observe(self, sensory_input: torch.Tensor,
                global_broadcast: Optional[torch.Tensor] = None) -> None:
        if sensory_input.dim() == 1:
            sensory_input = sensory_input.unsqueeze(0)
        # Trim/pad to match hierarchy bottom dim
        bottom_dim = self.hpc.levels[0].input_dim
        if sensory_input.shape[-1] != bottom_dim:
            if sensory_input.shape[-1] > bottom_dim:
                sensory_input = sensory_input[..., :bottom_dim]
            else:
                pad = torch.zeros(*sensory_input.shape[:-1],
                                  bottom_dim - sensory_input.shape[-1],
                                  device=sensory_input.device)
                sensory_input = torch.cat([sensory_input, pad], dim=-1)

        self._last_result = self.hpc(sensory_input)
        with torch.no_grad():
            self._latent.copy_(self._last_result['errors'][0].squeeze(0)[:self.latent_dim].detach())

    def propose(self, gw_state: Optional[torch.Tensor] = None) -> Proposal:
        if self._last_result is None:
            return Proposal(
                content=self._latent,
                salience=0.0,
                tags={"prediction_error"},
                organ_id=self.organ_id,
            )
        # Salience = free energy (high error = high salience)
        fe = self._last_result['free_energy']
        salience = fe.item() if isinstance(fe, torch.Tensor) else fe
        return Proposal(
            content=self._latent,
            salience=min(salience, 10.0),
            tags={"prediction_error", "hierarchical"},
            organ_id=self.organ_id,
        )

    def accept(self, global_broadcast: torch.Tensor) -> None:
        self.update_standing(global_broadcast)
        # Use broadcast as top-down prior (update top-level beliefs)
        if self._last_result is not None:
            top_dim = self.hpc.levels[-1].belief_dim
            with torch.no_grad():
                top_signal = global_broadcast[:top_dim] if global_broadcast.shape[-1] >= top_dim else global_broadcast
                self.hpc.levels[-1].beliefs.copy_(
                    self.hpc.levels[-1].beliefs * 0.9 + top_signal * 0.1
                )


class MotivationOrgan(Organ):
    """
    Wraps IntrinsicMotivationSystem as a workspace-competing organ.

    Proposes high-novelty/high-motivation signals for broadcast.
    """

    def __init__(self, latent_dim: int = 64, frequency: float = 0.5):
        super().__init__(organ_id="motivation", latent_dim=latent_dim,
                         frequency=frequency)
        from pyquifer.cognition.control.motivation import IntrinsicMotivationSystem
        self.motivation = IntrinsicMotivationSystem(state_dim=latent_dim)
        self._last_result = None
        self.register_buffer('_latent', torch.zeros(latent_dim))

    def observe(self, sensory_input: torch.Tensor,
                global_broadcast: Optional[torch.Tensor] = None) -> None:
        if sensory_input.dim() > 1:
            sensory_input = sensory_input.squeeze(0)
        # Pad or trim to match latent_dim (motivation expects state_dim=latent_dim)
        if sensory_input.shape[-1] > self.latent_dim:
            inp = sensory_input[:self.latent_dim]
        elif sensory_input.shape[-1] < self.latent_dim:
            inp = torch.zeros(self.latent_dim, device=sensory_input.device)
            inp[:sensory_input.shape[-1]] = sensory_input
        else:
            inp = sensory_input
        self._last_result = self.motivation(inp)
        with torch.no_grad():
            self._latent.copy_(inp.detach())

    def propose(self, gw_state: Optional[torch.Tensor] = None) -> Proposal:
        if self._last_result is None:
            return Proposal(content=self._latent, salience=0.0,
                            tags={"motivation"}, organ_id=self.organ_id)
        motiv = self._last_result['motivation']
        salience = motiv.item() if isinstance(motiv, torch.Tensor) else motiv
        return Proposal(
            content=self._latent,
            salience=max(0.0, salience),
            tags={"motivation", "novelty"},
            organ_id=self.organ_id,
        )

    def accept(self, global_broadcast: torch.Tensor) -> None:
        self.update_standing(global_broadcast)


class SelectionOrgan(Organ):
    """
    Wraps SelectionArena as a workspace-competing organ.

    Proposes the best neuronal group's output for broadcast.
    """

    def __init__(self, latent_dim: int = 64, num_groups: int = 6,
                 group_dim: int = 16, frequency: float = 2.0):
        super().__init__(organ_id="selection", latent_dim=latent_dim,
                         frequency=frequency)
        from pyquifer.identity.neural_darwinism import SelectionArena
        self.arena = SelectionArena(
            num_groups=num_groups, group_dim=group_dim, total_budget=10.0
        )
        self.group_dim = group_dim
        self._last_result = None
        self.register_buffer('_latent', torch.zeros(latent_dim))

    def observe(self, sensory_input: torch.Tensor,
                global_broadcast: Optional[torch.Tensor] = None) -> None:
        if sensory_input.dim() > 1:
            sensory_input = sensory_input.squeeze(0)
        inp = sensory_input[:self.group_dim] if sensory_input.shape[-1] > self.group_dim else sensory_input
        coherence = global_broadcast[:self.group_dim] if global_broadcast is not None else None
        self._last_result = self.arena(inp, global_coherence=coherence)
        with torch.no_grad():
            # Pad/trim arena output to latent_dim
            best_group = self._last_result['group_outputs'][0]
            if best_group.shape[-1] >= self.latent_dim:
                self._latent.copy_(best_group[:self.latent_dim].detach())
            else:
                self._latent[:best_group.shape[-1]].copy_(best_group.detach())

    def propose(self, gw_state: Optional[torch.Tensor] = None) -> Proposal:
        if self._last_result is None:
            return Proposal(content=self._latent, salience=0.0,
                            tags={"selection"}, organ_id=self.organ_id)
        fitness = self._last_result['mean_fitness']
        salience = fitness.item() if isinstance(fitness, torch.Tensor) else fitness
        return Proposal(
            content=self._latent,
            salience=max(0.0, salience),
            tags={"selection", "competition"},
            organ_id=self.organ_id,
        )

    def accept(self, global_broadcast: torch.Tensor) -> None:
        self.update_standing(global_broadcast)
