"""
Temporal Knowledge Graph — time-aware entity-relationship reasoning.

Provides graph structures where nodes and edges carry temporal validity
intervals, enabling queries like "what happened before/after X?" and
"what was true at time T?"

Key classes:
- TemporalNode: Entity with time-stamped attributes
- TemporalEdge: Relationship with validity interval
- TemporalKnowledgeGraph: Graph with time-aware queries
- EventTimeline: Ordered event sequence with causal links
- TemporalReasoner: Answer temporal questions

References:
- Lacroix et al. (2020). Tensor Decompositions for Temporal Knowledge Graphs.
- Jin et al. (2020). Recurrent Event Network for Reasoning over Temporal KGs.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn


@dataclass
class TemporalNode:
    """Entity with time-stamped attributes.

    Attributes:
        node_id: Unique identifier.
        entity_type: Type classification (e.g., 'person', 'event', 'concept').
        embedding: Learnable embedding vector.
        attributes: Time-stamped attribute dict: {attr_name: [(value, t_start, t_end)]}.
        created_at: Creation timestamp.
    """
    node_id: str
    entity_type: str = "entity"
    embedding: Optional[torch.Tensor] = None
    attributes: Dict[str, List[Tuple[Any, float, float]]] = field(default_factory=dict)
    created_at: float = 0.0

    def get_attribute(self, name: str, at_time: float) -> Optional[Any]:
        """Get attribute value valid at a given time."""
        if name not in self.attributes:
            return None
        for value, t_start, t_end in self.attributes[name]:
            if t_start <= at_time <= t_end:
                return value
        return None

    def set_attribute(self, name: str, value: Any, t_start: float, t_end: float = float('inf')):
        """Set a time-stamped attribute."""
        if name not in self.attributes:
            self.attributes[name] = []
        self.attributes[name].append((value, t_start, t_end))


@dataclass
class TemporalEdge:
    """Relationship with temporal validity interval.

    Attributes:
        source_id: Source node ID.
        target_id: Target node ID.
        relation: Relationship type.
        t_start: Start of validity interval.
        t_end: End of validity interval.
        confidence: Confidence in [0, 1].
        metadata: Additional metadata.
    """
    source_id: str
    target_id: str
    relation: str
    t_start: float = 0.0
    t_end: float = float('inf')
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_valid_at(self, t: float) -> bool:
        """Check if edge is valid at time t."""
        return self.t_start <= t <= self.t_end


class TemporalKnowledgeGraph:
    """Knowledge graph with temporal awareness.

    Supports time-scoped queries, temporal pattern detection,
    and causal ordering of events.

    Args:
        embedding_dim: Dimension of node/edge embeddings.
    """

    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self._nodes: Dict[str, TemporalNode] = {}
        self._edges: List[TemporalEdge] = []
        self._adjacency: Dict[str, List[int]] = {}  # node_id → edge indices
        self._reverse_adjacency: Dict[str, List[int]] = {}

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    @property
    def num_edges(self) -> int:
        return len(self._edges)

    def add_node(self, node: TemporalNode) -> None:
        """Add a node to the graph."""
        self._nodes[node.node_id] = node
        if node.node_id not in self._adjacency:
            self._adjacency[node.node_id] = []
        if node.node_id not in self._reverse_adjacency:
            self._reverse_adjacency[node.node_id] = []

    def add_edge(self, edge: TemporalEdge) -> int:
        """Add a temporal edge. Returns edge index."""
        idx = len(self._edges)
        self._edges.append(edge)
        if edge.source_id not in self._adjacency:
            self._adjacency[edge.source_id] = []
        self._adjacency[edge.source_id].append(idx)
        if edge.target_id not in self._reverse_adjacency:
            self._reverse_adjacency[edge.target_id] = []
        self._reverse_adjacency[edge.target_id].append(idx)
        return idx

    def get_node(self, node_id: str) -> Optional[TemporalNode]:
        return self._nodes.get(node_id)

    def get_neighbors(
        self,
        node_id: str,
        at_time: Optional[float] = None,
        relation: Optional[str] = None,
    ) -> List[Tuple[TemporalNode, TemporalEdge]]:
        """Get neighbors of a node, optionally filtered by time and relation.

        Args:
            node_id: Source node.
            at_time: If set, only return edges valid at this time.
            relation: If set, only return edges of this type.

        Returns:
            List of (neighbor_node, edge) tuples.
        """
        results = []
        for idx in self._adjacency.get(node_id, []):
            edge = self._edges[idx]
            if at_time is not None and not edge.is_valid_at(at_time):
                continue
            if relation is not None and edge.relation != relation:
                continue
            target = self._nodes.get(edge.target_id)
            if target is not None:
                results.append((target, edge))
        return results

    def query_time_range(
        self, t_start: float, t_end: float
    ) -> List[TemporalEdge]:
        """Get all edges active during a time range."""
        return [
            e for e in self._edges
            if e.t_start <= t_end and e.t_end >= t_start
        ]

    def get_history(
        self, node_id: str, before_time: float
    ) -> List[Tuple[TemporalEdge, str]]:
        """Get all historical edges involving a node before a time.

        Returns:
            List of (edge, direction) tuples where direction is 'out' or 'in'.
        """
        history = []
        for idx in self._adjacency.get(node_id, []):
            e = self._edges[idx]
            if e.t_start < before_time:
                history.append((e, 'out'))
        for idx in self._reverse_adjacency.get(node_id, []):
            e = self._edges[idx]
            if e.t_start < before_time:
                history.append((e, 'in'))
        history.sort(key=lambda x: x[0].t_start)
        return history

    def snapshot(self, at_time: float) -> Dict[str, Any]:
        """Get a snapshot of the graph at a specific time.

        Returns only nodes and edges valid at the given time.
        """
        valid_edges = [e for e in self._edges if e.is_valid_at(at_time)]
        involved_ids: Set[str] = set()
        for e in valid_edges:
            involved_ids.add(e.source_id)
            involved_ids.add(e.target_id)
        valid_nodes = {nid: self._nodes[nid] for nid in involved_ids if nid in self._nodes}
        return {
            'nodes': valid_nodes,
            'edges': valid_edges,
            'time': at_time,
            'num_nodes': len(valid_nodes),
            'num_edges': len(valid_edges),
        }


@dataclass
class Event:
    """A timestamped event with optional causal links."""
    event_id: str
    description: str
    timestamp: float
    embedding: Optional[torch.Tensor] = None
    causes: List[str] = field(default_factory=list)  # event_ids
    effects: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class EventTimeline:
    """Ordered sequence of events with causal links.

    Args:
        embedding_dim: Dimension for event embeddings.
    """

    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self._events: Dict[str, Event] = {}
        self._timeline: List[str] = []  # Sorted by timestamp

    def add_event(self, event: Event) -> None:
        """Add an event to the timeline."""
        self._events[event.event_id] = event
        self._timeline.append(event.event_id)
        self._timeline.sort(key=lambda eid: self._events[eid].timestamp)

    def get_before(self, timestamp: float, limit: int = 10) -> List[Event]:
        """Get events before a timestamp."""
        result = []
        for eid in reversed(self._timeline):
            e = self._events[eid]
            if e.timestamp < timestamp:
                result.append(e)
                if len(result) >= limit:
                    break
        return list(reversed(result))

    def get_after(self, timestamp: float, limit: int = 10) -> List[Event]:
        """Get events after a timestamp."""
        result = []
        for eid in self._timeline:
            e = self._events[eid]
            if e.timestamp > timestamp:
                result.append(e)
                if len(result) >= limit:
                    break
        return result

    def get_causal_chain(self, event_id: str, direction: str = 'backward', max_depth: int = 5) -> List[Event]:
        """Trace causal chain from an event.

        Args:
            event_id: Starting event.
            direction: 'backward' (find causes) or 'forward' (find effects).
            max_depth: Maximum chain length.

        Returns:
            Ordered list of events in causal chain.
        """
        chain = []
        visited: Set[str] = set()
        queue = [event_id]

        for _ in range(max_depth):
            if not queue:
                break
            current_id = queue.pop(0)
            if current_id in visited or current_id not in self._events:
                continue
            visited.add(current_id)
            event = self._events[current_id]
            chain.append(event)

            if direction == 'backward':
                queue.extend(event.causes)
            else:
                queue.extend(event.effects)

        return chain


class TemporalReasoner(nn.Module):
    """Answer temporal questions over a knowledge graph.

    Uses learned embeddings to score temporal relations and answer
    queries like "what happened before X?", "was Y true at time T?"

    Args:
        dim: Embedding dimension.
        num_relations: Number of relation types.
        num_temporal_features: Number of temporal features.
    """

    def __init__(self, dim: int, num_relations: int = 32, num_temporal_features: int = 8):
        super().__init__()
        self.dim = dim

        # Temporal encoding: encode timestamps as features
        self.time_encoder = nn.Sequential(
            nn.Linear(num_temporal_features, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

        # Relation scoring
        self.relation_embeddings = nn.Embedding(num_relations, dim)
        self.scoring = nn.Sequential(
            nn.Linear(dim * 3, dim),  # subject + relation + time → score
            nn.ReLU(),
            nn.Linear(dim, 1),
        )

    def encode_time(self, t: torch.Tensor) -> torch.Tensor:
        """Encode a timestamp using sinusoidal features.

        Args:
            t: (...,) timestamps

        Returns:
            (..., dim) temporal embeddings
        """
        # Sinusoidal time encoding (like positional encoding)
        freqs = torch.arange(4, device=t.device, dtype=t.dtype) * 0.5
        features = t.unsqueeze(-1) * freqs.unsqueeze(0) if t.dim() > 0 else t * freqs
        encoded = torch.cat([torch.sin(features), torch.cos(features)], dim=-1)
        return self.time_encoder(encoded)

    def score_triple(
        self,
        subject: torch.Tensor,
        relation_id: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        """Score a temporal triple (subject, relation, time).

        Args:
            subject: (B, dim) subject embedding
            relation_id: (B,) relation indices
            time: (B,) timestamps

        Returns:
            (B,) scores
        """
        rel = self.relation_embeddings(relation_id)
        t_emb = self.encode_time(time)
        combined = torch.cat([subject, rel, t_emb], dim=-1)
        return self.scoring(combined).squeeze(-1)
