"""
Complementary Learning Systems (CLS) Memory Architecture.

Implements the two-system memory theory: a fast hippocampal system for
one-shot episodic learning and a slow neocortical system for gradual
schema extraction, connected by consolidation during "sleep."

Key classes:
- HippocampalModule: Fast episodic learning with pattern separation
- NeocorticalModule: Slow semantic learning with schema extraction
- ConsolidationScheduler: Trigger replay during sleep
- ForgettingCurve: Exponential decay with rehearsal-based rescue
- ImportanceScorer: Multi-factor importance weighting
- MemoryInterference: Detect and prevent catastrophic interference

References:
- McClelland et al. (1995). Why there are complementary learning systems.
- Kumaran et al. (2016). What learning systems do intelligent agents need?
- O'Reilly & Norman (2002). Hippocampal and neocortical contributions to memory.
"""

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


@dataclass
class MemoryTrace:
    """Single memory trace with metadata."""
    content: torch.Tensor
    importance: float = 1.0
    creation_time: float = 0.0
    last_access: float = 0.0
    access_count: int = 0
    emotional_salience: float = 0.0
    novelty: float = 1.0
    context: Optional[torch.Tensor] = None


class ForgettingCurve:
    """Exponential decay with rehearsal-based rescue.

    Implements Ebbinghaus forgetting curve: retention = exp(-t/S)
    where S (stability) increases with each rehearsal.

    Args:
        initial_stability: Base stability for new memories (seconds).
        rehearsal_boost: Multiplicative boost per rehearsal.
        min_retention: Minimum retention (never fully forget).
    """

    def __init__(
        self,
        initial_stability: float = 3600.0,
        rehearsal_boost: float = 2.0,
        min_retention: float = 0.01,
    ):
        self.initial_stability = initial_stability
        self.rehearsal_boost = rehearsal_boost
        self.min_retention = min_retention

    def retention(self, time_elapsed: float, access_count: int) -> float:
        """Compute retention probability.

        Args:
            time_elapsed: Time since encoding (seconds).
            access_count: Number of rehearsals.

        Returns:
            Retention probability in [min_retention, 1.0].
        """
        stability = self.initial_stability * (self.rehearsal_boost ** access_count)
        ret = math.exp(-time_elapsed / stability)
        return max(self.min_retention, ret)

    def should_forget(self, time_elapsed: float, access_count: int, threshold: float = 0.1) -> bool:
        """Check if a memory should be forgotten."""
        return self.retention(time_elapsed, access_count) < threshold


class ImportanceScorer:
    """Multi-factor importance weighting for memories.

    Combines emotional salience, novelty, utility, and recency
    into a single importance score.

    Args:
        emotion_weight: Weight for emotional salience.
        novelty_weight: Weight for novelty.
        utility_weight: Weight for utility/reward.
        recency_weight: Weight for recency.
    """

    def __init__(
        self,
        emotion_weight: float = 0.3,
        novelty_weight: float = 0.25,
        utility_weight: float = 0.3,
        recency_weight: float = 0.15,
    ):
        self.emotion_weight = emotion_weight
        self.novelty_weight = novelty_weight
        self.utility_weight = utility_weight
        self.recency_weight = recency_weight

    def score(
        self,
        emotional_salience: float = 0.0,
        novelty: float = 0.0,
        utility: float = 0.0,
        recency: float = 0.0,
    ) -> float:
        """Compute importance score in [0, 1]."""
        raw = (
            self.emotion_weight * emotional_salience +
            self.novelty_weight * novelty +
            self.utility_weight * utility +
            self.recency_weight * recency
        )
        return max(0.0, min(1.0, raw))


class HippocampalModule(nn.Module):
    """Fast episodic learning with pattern separation.

    Stores memories in a single exposure (one-shot) with pattern
    separation to minimize interference between similar memories.

    Uses a sparse, high-dimensional representation where each memory
    gets a distinct encoding (orthogonal-ish storage).

    Args:
        dim: Memory dimension.
        capacity: Maximum number of stored memories.
        separation_factor: Controls pattern separation strength.
    """

    def __init__(self, dim: int, capacity: int = 1000, separation_factor: float = 1.0):
        super().__init__()
        self.dim = dim
        self.capacity = capacity
        self.separation_factor = separation_factor

        # Pattern separation: expand to higher dim, then sparse encode
        self.encoder = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim),
        )
        # Pattern completion: reconstruct from partial cue
        self.decoder = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim),
        )

        # Memory storage
        self.register_buffer('memories', torch.zeros(capacity, dim))
        self.register_buffer('memory_mask', torch.zeros(capacity, dtype=torch.bool))
        self.register_buffer('num_stored', torch.tensor(0))
        self._write_ptr = 0

        self.importance_scorer = ImportanceScorer()
        self.forgetting_curve = ForgettingCurve()
        self._metadata: List[Dict] = []

    def store(
        self,
        content: torch.Tensor,
        importance: float = 1.0,
        emotional_salience: float = 0.0,
        novelty: float = 1.0,
    ) -> int:
        """Store a memory in one shot.

        Args:
            content: (dim,) memory content
            importance: Base importance
            emotional_salience: Emotional weight
            novelty: How novel this memory is

        Returns:
            Index where memory was stored
        """
        # Pattern separation: encode to minimize interference
        with torch.no_grad():
            separated = self.encoder(content.unsqueeze(0)).squeeze(0)
            # Add sparsity: keep only top-k activations
            k = max(1, int(self.dim * 0.3))
            topk = separated.abs().topk(k)
            sparse = torch.zeros_like(separated)
            sparse[topk.indices] = separated[topk.indices]

        idx = self._write_ptr
        self.memories[idx] = sparse
        self.memory_mask[idx] = True

        # Update metadata
        if idx >= len(self._metadata):
            self._metadata.append({})
        self._metadata[idx] = {
            'importance': importance,
            'emotional_salience': emotional_salience,
            'novelty': novelty,
            'creation_time': time.time(),
            'last_access': time.time(),
            'access_count': 0,
        }

        self._write_ptr = (self._write_ptr + 1) % self.capacity
        with torch.no_grad():
            self.num_stored.copy_(torch.tensor(
                min(self.num_stored.item() + 1, self.capacity)
            ))
        return idx

    def recall(self, cue: torch.Tensor, top_k: int = 5) -> Dict[str, torch.Tensor]:
        """Pattern completion: retrieve memory from partial cue.

        Args:
            cue: (dim,) query cue
            top_k: Number of memories to retrieve

        Returns:
            Dict with 'memories', 'similarities', 'indices'
        """
        if self.num_stored.item() == 0:
            return {
                'memories': torch.zeros(0, self.dim, device=cue.device),
                'similarities': torch.zeros(0, device=cue.device),
                'indices': torch.zeros(0, dtype=torch.long, device=cue.device),
            }

        with torch.no_grad():
            encoded_cue = self.encoder(cue.unsqueeze(0)).squeeze(0)

        # Cosine similarity with stored memories
        valid = self.memory_mask.nonzero(as_tuple=True)[0]
        if valid.numel() == 0:
            return {
                'memories': torch.zeros(0, self.dim, device=cue.device),
                'similarities': torch.zeros(0, device=cue.device),
                'indices': torch.zeros(0, dtype=torch.long, device=cue.device),
            }

        stored = self.memories[valid]
        sims = nn.functional.cosine_similarity(
            encoded_cue.unsqueeze(0), stored, dim=-1
        )

        k = min(top_k, valid.shape[0])
        topk_sims, topk_local = sims.topk(k)
        topk_indices = valid[topk_local]

        # Decode memories
        decoded = self.decoder(stored[topk_local])

        # Update access counts
        for idx in topk_indices:
            i = idx.item()
            if i < len(self._metadata):
                self._metadata[i]['access_count'] += 1
                self._metadata[i]['last_access'] = time.time()

        return {
            'memories': decoded,
            'similarities': topk_sims,
            'indices': topk_indices,
        }

    def reset(self):
        self.memories.zero_()
        self.memory_mask.zero_()
        self.num_stored.zero_()
        self._write_ptr = 0
        self._metadata.clear()


class NeocorticalModule(nn.Module):
    """Slow semantic learning with gradual schema extraction.

    Learns abstract schemas from replayed episodic memories over many
    exposures. Uses a low learning rate to avoid catastrophic forgetting.

    Args:
        dim: Input dimension.
        schema_dim: Schema representation dimension.
        num_schemas: Number of schema slots.
        lr: Learning rate for schema updates.
    """

    def __init__(self, dim: int, schema_dim: int = 32, num_schemas: int = 50, lr: float = 0.01):
        super().__init__()
        self.dim = dim
        self.schema_dim = schema_dim
        self.num_schemas = num_schemas

        # Schema extraction: compress episodic traces into schemas
        self.schema_encoder = nn.Sequential(
            nn.Linear(dim, schema_dim * 2),
            nn.ReLU(),
            nn.Linear(schema_dim * 2, schema_dim),
        )

        # Schema storage
        self.register_buffer('schemas', torch.randn(num_schemas, schema_dim) * 0.01)
        self.register_buffer('schema_usage', torch.zeros(num_schemas))
        self.register_buffer('num_active', torch.tensor(0))

        self.lr = lr

    def consolidate(self, episodic_traces: torch.Tensor) -> Dict[str, Any]:
        """Gradually extract schemas from episodic replays.

        Args:
            episodic_traces: (N, dim) batch of episodic memories to consolidate

        Returns:
            Dict with 'num_updated', 'mean_update_norm'
        """
        if episodic_traces.shape[0] == 0:
            return {'num_updated': 0, 'mean_update_norm': 0.0}

        with torch.no_grad():
            encoded = self.schema_encoder(episodic_traces)  # (N, schema_dim)

            # Assign each trace to nearest schema
            sims = nn.functional.cosine_similarity(
                encoded.unsqueeze(1),
                self.schemas.unsqueeze(0),
                dim=-1,
            )  # (N, num_schemas)

            assignments = sims.argmax(dim=-1)  # (N,)
            update_norms = []

            for s in range(self.num_schemas):
                mask = (assignments == s)
                if not mask.any():
                    continue

                # Gradual update toward mean of assigned traces
                target = encoded[mask].mean(dim=0)
                delta = self.lr * (target - self.schemas[s])
                self.schemas[s] += delta
                self.schema_usage[s] += mask.sum()
                update_norms.append(delta.norm().item())

            active = (self.schema_usage > 0).sum()
            self.num_active.copy_(active)

        return {
            'num_updated': len(update_norms),
            'mean_update_norm': sum(update_norms) / max(1, len(update_norms)),
        }

    def query(self, cue: torch.Tensor, top_k: int = 3) -> Dict[str, torch.Tensor]:
        """Query schemas by similarity.

        Args:
            cue: (dim,) query
            top_k: Number of schemas to return

        Returns:
            Dict with 'schemas', 'similarities'
        """
        with torch.no_grad():
            encoded = self.schema_encoder(cue.unsqueeze(0)).squeeze(0)
            sims = nn.functional.cosine_similarity(
                encoded.unsqueeze(0), self.schemas, dim=-1
            )
            k = min(top_k, self.num_schemas)
            topk_sims, topk_idx = sims.topk(k)

        return {
            'schemas': self.schemas[topk_idx],
            'similarities': topk_sims,
            'indices': topk_idx,
        }

    def reset(self):
        self.schemas.normal_(0, 0.01)
        self.schema_usage.zero_()
        self.num_active.zero_()


class MemoryInterference(nn.Module):
    """Detect and prevent catastrophic interference.

    Measures similarity between new and existing memories to flag
    potential interference. High similarity + different content = danger.

    Args:
        dim: Memory dimension.
        interference_threshold: Similarity above which interference is flagged.
    """

    def __init__(self, dim: int, interference_threshold: float = 0.8):
        super().__init__()
        self.dim = dim
        self.threshold = interference_threshold

    def check(
        self,
        new_memory: torch.Tensor,
        existing_memories: torch.Tensor,
    ) -> Dict[str, Any]:
        """Check for potential interference.

        Args:
            new_memory: (dim,) new memory to store
            existing_memories: (N, dim) existing memories

        Returns:
            Dict with 'interference_risk', 'most_similar_idx', 'max_similarity'
        """
        if existing_memories.shape[0] == 0:
            return {
                'interference_risk': False,
                'most_similar_idx': -1,
                'max_similarity': 0.0,
            }

        sims = nn.functional.cosine_similarity(
            new_memory.unsqueeze(0), existing_memories, dim=-1
        )
        max_sim, max_idx = sims.max(dim=0)

        return {
            'interference_risk': max_sim.item() > self.threshold,
            'most_similar_idx': max_idx.item(),
            'max_similarity': max_sim.item(),
        }


class ConsolidationScheduler:
    """Schedule memory consolidation events.

    Triggers replay during "sleep" periods based on time elapsed,
    number of new memories, and importance of pending memories.

    Args:
        min_interval: Minimum time between consolidations (seconds).
        memory_threshold: Trigger consolidation after this many new memories.
        importance_threshold: Trigger if any memory exceeds this importance.
    """

    def __init__(
        self,
        min_interval: float = 300.0,
        memory_threshold: int = 50,
        importance_threshold: float = 0.9,
    ):
        self.min_interval = min_interval
        self.memory_threshold = memory_threshold
        self.importance_threshold = importance_threshold
        self._last_consolidation = time.time()
        self._new_memories_since = 0

    def should_consolidate(
        self,
        sleep_signal: float = 0.0,
        max_importance: float = 0.0,
    ) -> bool:
        """Check if consolidation should be triggered.

        Args:
            sleep_signal: External sleep signal in [0, 1].
            max_importance: Maximum importance of pending memories.

        Returns:
            True if consolidation should occur.
        """
        elapsed = time.time() - self._last_consolidation

        # Always consolidate during sleep
        if sleep_signal > 0.5 and elapsed > self.min_interval:
            return True

        # Consolidate if many new memories
        if self._new_memories_since >= self.memory_threshold:
            return True

        # Consolidate for very important memories
        if max_importance >= self.importance_threshold and elapsed > self.min_interval / 2:
            return True

        return False

    def record_memory(self):
        """Record that a new memory was stored."""
        self._new_memories_since += 1

    def record_consolidation(self):
        """Record that consolidation occurred."""
        self._last_consolidation = time.time()
        self._new_memories_since = 0
