"""
Graph Reasoning Layer — structured reasoning over relational knowledge.

Attention and message passing over time-varying graphs, modulated by
oscillator phase. Enables structured reasoning over entity-relationship
knowledge that evolves over time.

Key classes:
- DynamicGraphAttention: Attention over time-varying graph
- MessagePassingWithPhase: GNN message passing modulated by oscillator phase
- TemporalGraphTransformer: Transformer on temporal knowledge graph

References:
- Velickovic et al. (2018). Graph Attention Networks. ICLR.
- Xu et al. (2020). Inductive Representation Learning on Temporal Graphs.
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicGraphAttention(nn.Module):
    """Attention over time-varying graph structure.

    Computes attention between nodes where the graph connectivity
    changes over time. Edges are weighted by both content similarity
    and temporal recency.

    Args:
        dim: Node feature dimension.
        num_heads: Number of attention heads.
        dropout: Attention dropout rate.
        temporal_decay: Exponential decay rate for temporal weighting.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        temporal_decay: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.temporal_decay = temporal_decay

        assert dim % num_heads == 0

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # Temporal encoding
        self.time_proj = nn.Linear(1, num_heads)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        edge_times: Optional[torch.Tensor] = None,
        current_time: float = 0.0,
    ) -> torch.Tensor:
        """Compute dynamic graph attention.

        Args:
            node_features: (B, N, dim) node features
            adjacency: (B, N, N) or (N, N) adjacency matrix (0/1 or weighted)
            edge_times: Optional (B, N, N) or (N, N) edge timestamps
            current_time: Current time for temporal weighting

        Returns:
            (B, N, dim) updated node features
        """
        B, N, D = node_features.shape
        H = self.num_heads
        d_k = self.head_dim

        q = self.q_proj(node_features).view(B, N, H, d_k).transpose(1, 2)
        k = self.k_proj(node_features).view(B, N, H, d_k).transpose(1, 2)
        v = self.v_proj(node_features).view(B, N, H, d_k).transpose(1, 2)

        # Content attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        # Graph mask: only attend to connected nodes
        if adjacency.dim() == 2:
            adjacency = adjacency.unsqueeze(0).expand(B, -1, -1)
        graph_mask = (adjacency == 0)
        attn = attn.masked_fill(
            graph_mask.unsqueeze(1), float('-inf')
        )

        # Temporal weighting
        if edge_times is not None:
            if edge_times.dim() == 2:
                edge_times = edge_times.unsqueeze(0).expand(B, -1, -1)
            time_diff = (current_time - edge_times).clamp(min=0)
            temporal_weight = torch.exp(-self.temporal_decay * time_diff)
            temporal_bias = self.time_proj(
                temporal_weight.unsqueeze(-1)
            ).permute(0, 3, 1, 2)  # (B, H, N, N)
            attn = attn + temporal_bias

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)

        return node_features + out


class MessagePassingWithPhase(nn.Module):
    """GNN message passing modulated by oscillator phase.

    Messages between nodes are gated by the phase difference
    between sender and receiver oscillators. In-phase nodes
    exchange messages freely; out-of-phase nodes are blocked.

    Args:
        dim: Node feature dimension.
        num_oscillators: Oscillators per node.
        aggregation: Aggregation type ('mean', 'sum', 'max').
    """

    def __init__(
        self,
        dim: int,
        num_oscillators: int = 4,
        aggregation: str = 'mean',
    ):
        super().__init__()
        self.dim = dim
        self.num_oscillators = num_oscillators
        self.aggregation = aggregation

        # Message computation
        self.message_net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

        # Phase-based gate
        self.phase_gate = nn.Sequential(
            nn.Linear(num_oscillators, dim),
            nn.Sigmoid(),
        )

        # Node update
        self.update_net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        node_phases: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run phase-gated message passing.

        Args:
            node_features: (B, N, dim) or (N, dim) node features
            adjacency: (N, N) adjacency matrix
            node_phases: Optional (B, N, num_osc) or (N, num_osc) oscillator phases

        Returns:
            Updated node features, same shape as input
        """
        was_2d = node_features.dim() == 2
        if was_2d:
            node_features = node_features.unsqueeze(0)
            if node_phases is not None and node_phases.dim() == 2:
                node_phases = node_phases.unsqueeze(0)

        B, N, D = node_features.shape

        # Compute messages for each edge
        aggregated = torch.zeros_like(node_features)
        counts = torch.zeros(B, N, 1, device=node_features.device)

        for i in range(N):
            neighbors = adjacency[i].nonzero(as_tuple=True)[0]
            if neighbors.numel() == 0:
                continue

            # Compute messages from neighbors
            receiver = node_features[:, i:i+1].expand(-1, neighbors.shape[0], -1)
            sender = node_features[:, neighbors]
            messages = self.message_net(
                torch.cat([receiver, sender], dim=-1)
            )

            # Phase gating
            if node_phases is not None:
                phase_diff = node_phases[:, i:i+1] - node_phases[:, neighbors]
                # Coherence: cos(delta_phase) high when in-phase
                coherence = torch.cos(phase_diff)  # (B, num_neighbors, num_osc)
                gate = self.phase_gate(coherence)
                messages = messages * gate

            # Aggregate
            if self.aggregation == 'sum':
                aggregated[:, i] = messages.sum(dim=1)
            elif self.aggregation == 'max':
                aggregated[:, i] = messages.max(dim=1).values
            else:
                aggregated[:, i] = messages.mean(dim=1)

            counts[:, i] = neighbors.shape[0]

        # Update nodes
        updated = self.update_net(
            torch.cat([node_features, aggregated], dim=-1)
        )
        result = node_features + updated

        if was_2d:
            result = result.squeeze(0)

        return result


class TemporalGraphTransformer(nn.Module):
    """Transformer over temporal knowledge graph.

    Combines graph structure with temporal information using:
    1. Dynamic graph attention (content + structure + time)
    2. Phase-gated message passing for oscillator integration
    3. Feedforward with temporal encoding

    Args:
        dim: Feature dimension.
        num_heads: Number of attention heads.
        num_layers: Number of transformer layers.
        num_oscillators: Oscillators per node.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        num_oscillators: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers

        self.attention_layers = nn.ModuleList([
            DynamicGraphAttention(dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.message_layers = nn.ModuleList([
            MessagePassingWithPhase(dim, num_oscillators)
            for _ in range(num_layers)
        ])

        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * 4, dim),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(dim) for _ in range(num_layers * 2)
        ])

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        edge_times: Optional[torch.Tensor] = None,
        node_phases: Optional[torch.Tensor] = None,
        current_time: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """Process temporal graph.

        Args:
            node_features: (B, N, dim) node features
            adjacency: (B, N, N) or (N, N) adjacency matrix
            edge_times: Optional edge timestamps
            node_phases: Optional (B, N, num_osc) oscillator phases
            current_time: Current time for temporal weighting

        Returns:
            Dict with 'node_features', 'graph_embedding'
        """
        x = node_features

        for i in range(self.num_layers):
            # Dynamic graph attention
            x = self.norms[i * 2](x)
            x = self.attention_layers[i](x, adjacency, edge_times, current_time)

            # Phase-gated message passing
            adj = adjacency if adjacency.dim() == 2 else adjacency[0]
            x = self.norms[i * 2 + 1](x)
            x = self.message_layers[i](x, adj, node_phases)

            # Feedforward
            x = x + self.ffn_layers[i](x)

        # Global graph embedding (mean pooling)
        graph_emb = x.mean(dim=1)

        return {
            'node_features': x,
            'graph_embedding': graph_emb,
        }
