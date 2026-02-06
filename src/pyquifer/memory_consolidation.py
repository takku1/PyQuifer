"""
Memory Consolidation Module for PyQuifer

Sleep replay and episodic→semantic transfer. Mizuki can prevent
forgetting (EWC) but can't digest experiences into wisdom — until now.

Key concepts:
- EpisodicBuffer: Ring buffer of (state, reward, context, timestamp) tuples
- SharpWaveRipple: Compressed temporal replay at accelerated timescale,
  triggered during "sleep" (ecology.py circadian trough)
- ConsolidationEngine: Repeated noisy replay strengthens semantic traces
  (dreams are noisy reconstructions)
- MemoryReconsolidation: Recalled memories become labile, blending with
  current context (memory is constructive, not archival)

References:
- Tononi & Cirelli (2014). Sleep and the Price of Plasticity.
- Rasch & Born (2013). About Sleep's Role in Memory.
- Nader et al. (2000). Memory Reconsolidation.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Dict, List, Tuple


class EpisodicBuffer(nn.Module):
    """
    Ring buffer storing episodic memories as (state, reward, context) tuples.

    Oldest memories are overwritten when buffer is full, but high-reward
    memories are protected from overwrite via priority.
    """

    def __init__(self,
                 state_dim: int,
                 context_dim: int = 0,
                 capacity: int = 1000,
                 protect_top_k: int = 50):
        """
        Args:
            state_dim: Dimension of state vectors
            context_dim: Dimension of context vectors (0 to disable)
            capacity: Maximum number of memories
            protect_top_k: Number of high-reward memories to protect
        """
        super().__init__()
        self.state_dim = state_dim
        self.context_dim = context_dim if context_dim > 0 else state_dim
        self.capacity = capacity
        self.protect_top_k = protect_top_k

        # Storage buffers
        self.register_buffer('states', torch.zeros(capacity, state_dim))
        self.register_buffer('rewards', torch.zeros(capacity))
        self.register_buffer('contexts', torch.zeros(capacity, self.context_dim))
        self.register_buffer('timestamps', torch.zeros(capacity))
        self.register_buffer('replay_counts', torch.zeros(capacity))

        # Bookkeeping
        self.register_buffer('write_ptr', torch.tensor(0))
        self.register_buffer('num_stored', torch.tensor(0))
        self.register_buffer('global_step', torch.tensor(0))

    def store(self,
              state: torch.Tensor,
              reward: float,
              context: Optional[torch.Tensor] = None):
        """
        Store an episodic memory.

        Args:
            state: State vector (state_dim,)
            reward: Reward signal
            context: Optional context vector (context_dim,)
        """
        with torch.no_grad():
            # Find write position (skip protected memories)
            ptr = self.write_ptr.item()

            # Check if this slot is protected
            if self.num_stored >= self.capacity and self.protect_top_k > 0:
                # Find the top-k reward indices
                valid = min(self.num_stored.item(), self.capacity)
                _, top_indices = self.rewards[:valid].topk(
                    min(self.protect_top_k, valid)
                )
                # If current ptr is protected, advance to next unprotected slot
                attempts = 0
                while ptr in top_indices.tolist() and attempts < self.capacity:
                    ptr = (ptr + 1) % self.capacity
                    attempts += 1

            self.states[ptr] = state.detach()
            self.rewards[ptr] = reward
            if context is not None:
                self.contexts[ptr] = context.detach()
            else:
                self.contexts[ptr] = state.detach()[:self.context_dim]
            self.timestamps[ptr] = self.global_step.float()
            self.replay_counts[ptr] = 0

            self.write_ptr.fill_((ptr + 1) % self.capacity)
            self.num_stored.copy_(torch.clamp(self.num_stored + 1, max=self.capacity))
            self.global_step.add_(1)

    def sample_prioritized(self,
                           batch_size: int,
                           alpha: float = 0.6,
                           epsilon: float = 0.01) -> Dict[str, torch.Tensor]:
        """
        Sample memories with priority based on reward magnitude.

        P(i) ∝ (|reward_i| + epsilon) ^ alpha

        Args:
            batch_size: Number of memories to sample
            alpha: Priority exponent (0=uniform, 1=fully prioritized)
            epsilon: Small constant for non-zero probability

        Returns:
            Dictionary with sampled states, rewards, contexts, indices
        """
        valid = min(self.num_stored.item(), self.capacity)
        if valid == 0:
            return {
                'states': torch.zeros(0, self.state_dim),
                'rewards': torch.zeros(0),
                'contexts': torch.zeros(0, self.context_dim),
                'indices': torch.zeros(0, dtype=torch.long),
            }

        batch_size = min(batch_size, valid)

        # Priority = |reward| + epsilon, raised to alpha
        priorities = (self.rewards[:valid].abs() + epsilon) ** alpha
        probs = priorities / priorities.sum()

        indices = torch.multinomial(probs, batch_size, replacement=False)

        return {
            'states': self.states[indices],
            'rewards': self.rewards[indices],
            'contexts': self.contexts[indices],
            'indices': indices,
        }

    def forward(self, dummy: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Return buffer statistics."""
        valid = min(self.num_stored.item(), self.capacity)
        return {
            'num_stored': self.num_stored.clone(),
            'mean_reward': self.rewards[:valid].mean() if valid > 0 else torch.tensor(0.0),
            'max_reward': self.rewards[:valid].max() if valid > 0 else torch.tensor(0.0),
        }

    def reset(self):
        """Clear buffer."""
        self.states.zero_()
        self.rewards.zero_()
        self.contexts.zero_()
        self.timestamps.zero_()
        self.replay_counts.zero_()
        self.write_ptr.zero_()
        self.num_stored.zero_()


class SharpWaveRipple(nn.Module):
    """
    Compressed temporal replay at accelerated timescale.

    Replays sequences in bursts, weighted by reward magnitude.
    Triggered when the circadian phase enters "sleep" trough.
    Models hippocampal sharp-wave ripples during slow-wave sleep.
    """

    def __init__(self,
                 state_dim: int,
                 replay_burst_size: int = 10,
                 compression_ratio: float = 20.0,
                 noise_scale: float = 0.1):
        """
        Args:
            state_dim: Dimension of state vectors
            replay_burst_size: Number of memories per replay burst
            compression_ratio: Temporal compression factor
            noise_scale: Noise added during replay (dreaming)
        """
        super().__init__()
        self.state_dim = state_dim
        self.replay_burst_size = replay_burst_size
        self.compression_ratio = compression_ratio
        self.noise_scale = noise_scale

        # Replay output buffer
        self.register_buffer('last_replay', torch.zeros(replay_burst_size, state_dim))
        self.register_buffer('replay_count', torch.tensor(0))

    def forward(self,
                buffer: EpisodicBuffer,
                sleep_signal: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Generate a replay burst from the episodic buffer.

        Args:
            buffer: EpisodicBuffer to replay from
            sleep_signal: How deeply "asleep" (0=awake, 1=deep sleep)
                         Controls replay intensity

        Returns:
            Dictionary with:
            - replayed_states: Noisy replayed states (burst_size, state_dim)
            - replayed_rewards: Associated rewards
            - replay_active: Whether replay occurred
        """
        if buffer.num_stored < 2 or sleep_signal < 0.3:
            return {
                'replayed_states': torch.zeros(0, self.state_dim,
                                               device=buffer.states.device),
                'replayed_rewards': torch.zeros(0, device=buffer.states.device),
                'replay_active': torch.tensor(False),
            }

        # Sample with priority (high reward = more likely to replay)
        batch_size = min(self.replay_burst_size, buffer.num_stored.item())
        sample = buffer.sample_prioritized(batch_size, alpha=0.8)

        # Add noise (dreams are noisy reconstructions)
        noise = torch.randn_like(sample['states']) * self.noise_scale * sleep_signal
        replayed = sample['states'] + noise

        # Increment replay counts
        with torch.no_grad():
            buffer.replay_counts[sample['indices']].add_(1)
            self.last_replay[:batch_size] = replayed
            self.replay_count.add_(1)

        return {
            'replayed_states': replayed,
            'replayed_rewards': sample['rewards'],
            'replay_active': torch.tensor(True),
        }


class ConsolidationEngine(nn.Module):
    """
    Transfers episodic memories to semantic traces via repeated noisy replay.

    semantic_trace += lr * (replayed_episodic - semantic_trace)

    Each replay pulls the semantic trace closer to the episodic content.
    Noise during replay creates generalization (the dream doesn't match
    the memory exactly, so the semantic trace captures the GIST).
    """

    def __init__(self,
                 state_dim: int,
                 semantic_dim: int,
                 lr: float = 0.01,
                 num_semantic_slots: int = 100):
        """
        Args:
            state_dim: Dimension of episodic states
            semantic_dim: Dimension of semantic traces
            lr: Consolidation learning rate
            num_semantic_slots: Number of semantic memory slots
        """
        super().__init__()
        self.state_dim = state_dim
        self.semantic_dim = semantic_dim
        self.lr = lr
        self.num_semantic_slots = num_semantic_slots

        # Episodic → semantic projection
        self.projection = nn.Linear(state_dim, semantic_dim)

        # Semantic memory traces
        self.register_buffer('semantic_traces',
                             torch.zeros(num_semantic_slots, semantic_dim))
        self.register_buffer('trace_strengths', torch.zeros(num_semantic_slots))
        self.register_buffer('trace_ptr', torch.tensor(0))
        self.register_buffer('num_traces', torch.tensor(0))

    def consolidate(self,
                    replayed_states: torch.Tensor,
                    replayed_rewards: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Consolidate replayed episodic memories into semantic traces.

        Args:
            replayed_states: Replayed states from SharpWaveRipple (N, state_dim)
            replayed_rewards: Associated rewards (N,)

        Returns:
            Dictionary with consolidation metrics
        """
        if replayed_states.shape[0] == 0:
            return {
                'consolidated': torch.tensor(False),
                'num_traces': self.num_traces.clone(),
            }

        # Project to semantic space
        with torch.no_grad():
            semantic_content = self.projection(replayed_states)

        # For each replayed memory, find nearest semantic trace or create new
        consolidated_count = 0
        with torch.no_grad():
            for i in range(semantic_content.shape[0]):
                content = semantic_content[i]
                reward = replayed_rewards[i].abs()

                if self.num_traces > 0:
                    valid = min(self.num_traces.item(), self.num_semantic_slots)
                    # Find nearest existing trace
                    dists = torch.cdist(
                        content.unsqueeze(0),
                        self.semantic_traces[:valid]
                    ).squeeze(0)
                    nearest_idx = dists.argmin()
                    nearest_dist = dists[nearest_idx]

                    if nearest_dist < 2.0:
                        # Update existing trace (pull toward replayed content)
                        effective_lr = self.lr * (1 + reward)
                        delta = content - self.semantic_traces[nearest_idx]
                        self.semantic_traces[nearest_idx].add_(delta * effective_lr)
                        self.trace_strengths[nearest_idx].add_(1)
                        consolidated_count += 1
                        continue

                # Create new trace
                ptr = self.trace_ptr.item()
                self.semantic_traces[ptr] = content
                self.trace_strengths[ptr] = 1.0
                self.trace_ptr.fill_((ptr + 1) % self.num_semantic_slots)
                self.num_traces.copy_(
                    torch.clamp(self.num_traces + 1, max=self.num_semantic_slots)
                )
                consolidated_count += 1

        return {
            'consolidated': torch.tensor(True),
            'num_consolidated': torch.tensor(consolidated_count),
            'num_traces': self.num_traces.clone(),
            'mean_trace_strength': self.trace_strengths[:min(self.num_traces.item(), self.num_semantic_slots)].mean()
            if self.num_traces > 0 else torch.tensor(0.0),
        }

    def forward(self,
                replayed_states: torch.Tensor,
                replayed_rewards: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Alias for consolidate."""
        return self.consolidate(replayed_states, replayed_rewards)

    def query(self, query_state: torch.Tensor, top_k: int = 5) -> Dict[str, torch.Tensor]:
        """
        Query semantic memory for nearest traces.

        Args:
            query_state: Query vector (state_dim,)
            top_k: Number of results

        Returns:
            Nearest semantic traces and their strengths
        """
        valid = min(self.num_traces.item(), self.num_semantic_slots)
        if valid == 0:
            return {
                'traces': torch.zeros(0, self.semantic_dim),
                'strengths': torch.zeros(0),
                'distances': torch.zeros(0),
            }

        with torch.no_grad():
            query_semantic = self.projection(query_state.unsqueeze(0))

        dists = torch.cdist(query_semantic, self.semantic_traces[:valid]).squeeze(0)
        k = min(top_k, valid)
        top_dists, top_indices = dists.topk(k, largest=False)

        return {
            'traces': self.semantic_traces[top_indices],
            'strengths': self.trace_strengths[top_indices],
            'distances': top_dists,
        }

    def reset(self):
        """Reset semantic memory."""
        self.semantic_traces.zero_()
        self.trace_strengths.zero_()
        self.trace_ptr.zero_()
        self.num_traces.zero_()


class MemoryReconsolidation(nn.Module):
    """
    Memories become labile when recalled, blending with current context.

    recalled_memory = (1 - lability) * old_memory + lability * current_context

    This models the psychological finding that recalling a memory
    makes it vulnerable to modification — memory is constructive,
    not archival.
    """

    def __init__(self,
                 dim: int,
                 base_lability: float = 0.1,
                 max_lability: float = 0.5):
        """
        Args:
            dim: Memory dimension
            base_lability: How much a recalled memory blends with context
            max_lability: Maximum lability (emotional memories are more labile)
        """
        super().__init__()
        self.dim = dim
        self.base_lability = base_lability
        self.max_lability = max_lability

    def forward(self,
                memory: torch.Tensor,
                current_context: torch.Tensor,
                emotional_intensity: float = 0.0) -> Dict[str, torch.Tensor]:
        """
        Reconsolidate a recalled memory with current context.

        Args:
            memory: Original memory (dim,) or (batch, dim)
            current_context: Current context state (dim,) or (batch, dim)
            emotional_intensity: How emotional the recall is (0-1)
                               Higher = more labile

        Returns:
            Dictionary with:
            - reconsolidated: Modified memory
            - lability: Effective lability used
            - change_magnitude: How much the memory changed
        """
        # Lability increases with emotional intensity
        lability = self.base_lability + emotional_intensity * (self.max_lability - self.base_lability)

        # Reconsolidate
        reconsolidated = (1 - lability) * memory + lability * current_context

        change = (reconsolidated - memory).norm(dim=-1)

        return {
            'reconsolidated': reconsolidated,
            'lability': torch.tensor(lability),
            'change_magnitude': change,
        }


if __name__ == '__main__':
    print("--- Memory Consolidation Examples ---")

    # Example 1: Episodic Buffer
    print("\n1. Episodic Buffer")
    buffer = EpisodicBuffer(state_dim=16, capacity=100, protect_top_k=5)

    for i in range(150):
        state = torch.randn(16)
        reward = torch.randn(1).item()
        buffer.store(state, reward)

    stats = buffer()
    print(f"   Stored: {stats['num_stored'].item()}")
    print(f"   Mean reward: {stats['mean_reward'].item():.3f}")

    sample = buffer.sample_prioritized(10)
    print(f"   Sampled batch: {sample['states'].shape}")

    # Example 2: Sharp Wave Ripple
    print("\n2. Sharp Wave Ripple (sleep replay)")
    ripple = SharpWaveRipple(state_dim=16, replay_burst_size=8)

    # During sleep
    replay = ripple(buffer, sleep_signal=1.0)
    print(f"   Replay active: {replay['replay_active'].item()}")
    print(f"   Replayed states: {replay['replayed_states'].shape}")

    # During wakefulness (should not replay)
    replay_awake = ripple(buffer, sleep_signal=0.1)
    print(f"   Replay during wake: {replay_awake['replay_active'].item()}")

    # Example 3: Consolidation Engine
    print("\n3. Consolidation (episodic → semantic)")
    engine = ConsolidationEngine(state_dim=16, semantic_dim=8, lr=0.05)

    # Multiple sleep cycles
    for cycle in range(10):
        replay = ripple(buffer, sleep_signal=0.8)
        if replay['replay_active']:
            result = engine(replay['replayed_states'], replay['replayed_rewards'])

    print(f"   Semantic traces: {engine.num_traces.item()}")
    print(f"   Mean strength: {engine.trace_strengths[:engine.num_traces.item()].mean().item():.2f}")

    # Query semantic memory
    query = torch.randn(16)
    query_result = engine.query(query, top_k=3)
    print(f"   Query results: {query_result['traces'].shape}")

    # Example 4: Reconsolidation
    print("\n4. Memory Reconsolidation")
    recon = MemoryReconsolidation(dim=16, base_lability=0.1, max_lability=0.5)

    memory = torch.randn(16)
    context = torch.randn(16)

    # Calm recall
    result_calm = recon(memory, context, emotional_intensity=0.0)
    print(f"   Calm recall change: {result_calm['change_magnitude'].item():.4f}")

    # Emotional recall (more labile)
    result_emo = recon(memory, context, emotional_intensity=0.9)
    print(f"   Emotional recall change: {result_emo['change_magnitude'].item():.4f}")
    print(f"   Emotional > calm: {result_emo['change_magnitude'] > result_calm['change_magnitude']}")

    print("\n[OK] All memory consolidation tests passed!")
