"""
Differentiable Memory with NMDA Gating.

Biological gating inspired by NMDA receptor dynamics — memory write
operations are gated by oscillator phase (write during theta peak).

Key classes:
- NMDAGate: Biological voltage+ligand gated memory gate
- DifferentiableMemoryBank: Read/write via attention + gating
- MemoryConsolidationLoop: Oscillation-gated replay

References:
- Whittington et al. (2023). Relating transformers to models and neural
  representations of the hippocampal formation. Nature 2023.
- Baddeley (2003). Working memory and language: an overview.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class NMDAGate(nn.Module):
    """NMDA receptor-inspired memory gate.

    The NMDA receptor requires BOTH pre-synaptic glutamate (ligand)
    AND post-synaptic depolarization (voltage) to open — a coincidence
    detector. We model this as:

    gate = sigma(ligand) * sigma(voltage - threshold)

    In our context:
    - ligand = input salience (how important is this input?)
    - voltage = oscillator phase proximity to theta peak

    This ensures writes only happen when input is salient AND
    the oscillator is at the right phase.

    Args:
        dim: Gate dimension.
        threshold: Voltage threshold for gate opening.
    """

    def __init__(self, dim: int, threshold: float = 0.0):
        super().__init__()
        self.dim = dim
        self.threshold = threshold

        # Ligand pathway: input salience
        self.ligand_proj = nn.Linear(dim, dim)
        # Voltage pathway: oscillator phase
        self.voltage_proj = nn.Linear(1, dim)
        # Combined gate output
        self.gate_proj = nn.Linear(dim, dim)

    def forward(
        self,
        content: torch.Tensor,
        phase: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute NMDA gate value.

        Args:
            content: (..., dim) input content (provides ligand signal)
            phase: (...,) oscillator phase (provides voltage signal)

        Returns:
            (gate_value, gated_content): gate in [0,1] and gated content
        """
        # Ligand: input salience
        ligand = torch.sigmoid(self.ligand_proj(content))

        # Voltage: phase proximity to theta peak (peak at π)
        # cos(phase - π) is maximal when phase ≈ π
        phase_signal = torch.cos(phase - math.pi).unsqueeze(-1)  # (..., 1)
        voltage = torch.sigmoid(self.voltage_proj(phase_signal) - self.threshold)

        # Coincidence detection: both must be high
        gate = ligand * voltage
        gate = torch.sigmoid(self.gate_proj(gate))

        gated = content * gate
        return gate, gated


class DifferentiableMemoryBank(nn.Module):
    """Read/write memory bank with attention-based access and NMDA gating.

    Supports differentiable read (attention) and gated write operations.
    Memory slots are updated via soft attention + NMDA gate.

    Args:
        num_slots: Number of memory slots.
        slot_dim: Dimension of each slot.
        num_heads: Number of attention heads for reading.
    """

    def __init__(self, num_slots: int = 128, slot_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_heads = num_heads

        # Memory matrix
        self.register_buffer('memory', torch.randn(num_slots, slot_dim) * 0.01)
        self.register_buffer('usage', torch.zeros(num_slots))

        # NMDA gate for writes
        self.write_gate = NMDAGate(slot_dim)

        # Read attention
        self.read_query = nn.Linear(slot_dim, slot_dim)
        self.read_key = nn.Linear(slot_dim, slot_dim)
        self.read_value = nn.Linear(slot_dim, slot_dim)

        # Write address computation
        self.write_address = nn.Linear(slot_dim, num_slots)

        # Erase/add vectors
        self.erase_head = nn.Linear(slot_dim, slot_dim)
        self.add_head = nn.Linear(slot_dim, slot_dim)

    def read(self, query: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Read from memory via attention.

        Args:
            query: (B, dim) or (dim,) read query

        Returns:
            Dict with 'content' (read content), 'weights' (attention weights)
        """
        was_1d = query.dim() == 1
        if was_1d:
            query = query.unsqueeze(0)

        B = query.shape[0]

        # Attention over memory slots
        q = self.read_query(query)  # (B, dim)
        k = self.read_key(self.memory)  # (N, dim)
        v = self.read_value(self.memory)  # (N, dim)

        # Scaled dot product attention
        attn = torch.matmul(q, k.T) / math.sqrt(self.slot_dim)  # (B, N)
        weights = F.softmax(attn, dim=-1)

        content = torch.matmul(weights, v)  # (B, dim)

        if was_1d:
            content = content.squeeze(0)
            weights = weights.squeeze(0)

        return {
            'content': content,
            'weights': weights,
        }

    def write(
        self,
        content: torch.Tensor,
        phase: torch.Tensor,
        strength: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Write to memory with NMDA gating.

        Args:
            content: (B, dim) or (dim,) content to write
            phase: scalar or (B,) oscillator phase
            strength: Write strength multiplier

        Returns:
            Dict with 'gate_value', 'write_address', 'updated_slots'
        """
        was_1d = content.dim() == 1
        if was_1d:
            content = content.unsqueeze(0)
        if phase.dim() == 0:
            phase = phase.unsqueeze(0)

        B = content.shape[0]

        # NMDA gate
        gate, gated_content = self.write_gate(content, phase)

        # Write address: where to write
        address_logits = self.write_address(gated_content)  # (B, N)
        # Combine with usage (prefer unused slots)
        address_logits = address_logits - self.usage.unsqueeze(0) * 0.1
        address_weights = F.softmax(address_logits, dim=-1)  # (B, N)

        # Erase and add
        erase = torch.sigmoid(self.erase_head(gated_content))  # (B, dim)
        add = self.add_head(gated_content)  # (B, dim)

        with torch.no_grad():
            for b in range(B):
                w = address_weights[b]  # (N,)
                # Erase: M = M * (1 - w * erase)
                erase_matrix = torch.outer(w, erase[b])  # (N, dim)
                self.memory.mul_(1.0 - strength * erase_matrix)
                # Add: M = M + w * add
                add_matrix = torch.outer(w, add[b])
                self.memory.add_(strength * add_matrix)
                # Update usage
                self.usage.add_(w)

        gate_value = gate.mean()
        if was_1d:
            address_weights = address_weights.squeeze(0)

        return {
            'gate_value': gate_value,
            'write_address': address_weights,
            'updated_slots': (address_weights > 0.01).sum().item(),
        }

    def reset(self):
        self.memory.normal_(0, 0.01)
        self.usage.zero_()


class MemoryConsolidationLoop(nn.Module):
    """Oscillation-gated memory replay and consolidation.

    Theta rhythm triggers consolidation: during theta peaks, episodic
    traces are replayed into the memory bank for long-term storage.

    Args:
        dim: Memory dimension.
        num_replay_steps: Number of replay iterations per consolidation.
        replay_noise: Noise added during replay (generalization).
        theta_frequency: Theta oscillation frequency (Hz).
    """

    def __init__(
        self,
        dim: int,
        num_replay_steps: int = 10,
        replay_noise: float = 0.01,
        theta_frequency: float = 6.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_replay_steps = num_replay_steps
        self.replay_noise = replay_noise
        self.theta_frequency = theta_frequency

        # Replay transformation: slightly modify memory during replay
        self.replay_transform = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

        self.memory_bank = DifferentiableMemoryBank(
            num_slots=256, slot_dim=dim,
        )

        # Phase tracker
        self.register_buffer('_theta_phase', torch.tensor(0.0))

    def step_theta(self, dt: float = 0.01) -> float:
        """Advance theta oscillation. Returns current phase."""
        with torch.no_grad():
            self._theta_phase.add_(2 * math.pi * self.theta_frequency * dt)
            self._theta_phase.remainder_(2 * math.pi)
        return self._theta_phase.item()

    def consolidate(
        self,
        episodic_traces: torch.Tensor,
        priorities: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run consolidation replay.

        Args:
            episodic_traces: (N, dim) memories to consolidate
            priorities: Optional (N,) replay priorities

        Returns:
            Dict with 'num_replayed', 'mean_gate', 'theta_phase'
        """
        N = episodic_traces.shape[0]
        if N == 0:
            return {
                'num_replayed': 0,
                'mean_gate': torch.tensor(0.0),
                'theta_phase': self._theta_phase,
            }

        # Priority-weighted sampling
        if priorities is not None:
            probs = F.softmax(priorities, dim=0)
        else:
            probs = torch.ones(N, device=episodic_traces.device) / N

        total_gate = 0.0
        replayed = 0

        for _ in range(self.num_replay_steps):
            # Step theta
            phase = self.step_theta()

            # Sample a trace to replay
            idx = torch.multinomial(probs, 1).item()
            trace = episodic_traces[idx]

            # Add replay noise for generalization
            noisy = trace + torch.randn_like(trace) * self.replay_noise

            # Transform through replay network
            transformed = self.replay_transform(noisy.unsqueeze(0)).squeeze(0)

            # Write to memory bank (gated by theta phase)
            result = self.memory_bank.write(
                transformed, self._theta_phase, strength=0.5,
            )
            total_gate += result['gate_value'].item()
            replayed += 1

        return {
            'num_replayed': replayed,
            'mean_gate': torch.tensor(total_gate / max(1, replayed)),
            'theta_phase': self._theta_phase.clone(),
        }

    def recall(self, query: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Recall from consolidated memory."""
        return self.memory_bank.read(query)

    def reset(self):
        self.memory_bank.reset()
        self._theta_phase.zero_()
