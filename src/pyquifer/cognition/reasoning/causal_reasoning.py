"""
Causal Reasoning Engine — interventional and counterfactual queries.

Implements Pearl's causal hierarchy: observation (seeing), intervention
(doing), and counterfactual (imagining). Moves beyond correlation
(transfer entropy) to true causal reasoning with do-calculus.

Key classes:
- CausalGraph: DAG with interventional semantics
- DoOperator: Pearl's do(X=x) for causal intervention
- CounterfactualEngine: "What if X had been Y?"
- CausalDiscovery: Learn causal structure from data
- InterventionalQuery: Answer causal questions

References:
- Pearl (2009). Causality: Models, Reasoning, and Inference.
- Peters, Janzing & Scholkopf (2017). Elements of Causal Inference.
- Scholkopf et al. (2021). Towards Causal Representation Learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from collections import deque


@dataclass
class CausalVariable:
    """A variable in the causal graph."""
    name: str
    dim: int = 1
    observed: bool = True
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)


class CausalGraph:
    """Directed Acyclic Graph with causal semantics.

    Stores causal relationships between variables and supports
    graph operations needed for causal inference (d-separation,
    topological sort, ancestral sets).

    Args:
        variables: Optional list of variable names to initialize.
    """

    def __init__(self, variables: Optional[List[str]] = None):
        self._variables: Dict[str, CausalVariable] = {}
        self._adjacency: Dict[str, Set[str]] = {}  # parent → children
        self._reverse: Dict[str, Set[str]] = {}     # child → parents

        if variables:
            for v in variables:
                self.add_variable(v)

    @property
    def num_variables(self) -> int:
        return len(self._variables)

    @property
    def num_edges(self) -> int:
        return sum(len(ch) for ch in self._adjacency.values())

    def add_variable(self, name: str, dim: int = 1, observed: bool = True) -> None:
        """Add a variable to the causal graph."""
        self._variables[name] = CausalVariable(name=name, dim=dim, observed=observed)
        if name not in self._adjacency:
            self._adjacency[name] = set()
        if name not in self._reverse:
            self._reverse[name] = set()

    def add_edge(self, parent: str, child: str) -> None:
        """Add a directed causal edge parent → child."""
        if parent not in self._variables:
            self.add_variable(parent)
        if child not in self._variables:
            self.add_variable(child)

        # Check for cycles
        if self._would_create_cycle(parent, child):
            raise ValueError(f"Edge {parent} → {child} would create a cycle")

        self._adjacency[parent].add(child)
        self._reverse[child].add(parent)
        self._variables[parent].children.append(child)
        self._variables[child].parents.append(parent)

    def _would_create_cycle(self, parent: str, child: str) -> bool:
        """Check if adding parent → child creates a cycle."""
        if parent == child:
            return True
        visited = set()
        queue = deque([parent])
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            for p in self._reverse.get(current, set()):
                if p == child:
                    return True
                queue.append(p)
        return False

    def get_parents(self, node: str) -> Set[str]:
        """Get direct parents of a node."""
        return self._reverse.get(node, set())

    def get_children(self, node: str) -> Set[str]:
        """Get direct children of a node."""
        return self._adjacency.get(node, set())

    def get_ancestors(self, node: str) -> Set[str]:
        """Get all ancestors of a node."""
        ancestors = set()
        queue = deque(list(self._reverse.get(node, set())))
        while queue:
            current = queue.popleft()
            if current not in ancestors:
                ancestors.add(current)
                queue.extend(self._reverse.get(current, set()))
        return ancestors

    def get_descendants(self, node: str) -> Set[str]:
        """Get all descendants of a node."""
        descendants = set()
        queue = deque(list(self._adjacency.get(node, set())))
        while queue:
            current = queue.popleft()
            if current not in descendants:
                descendants.add(current)
                queue.extend(self._adjacency.get(current, set()))
        return descendants

    def topological_sort(self) -> List[str]:
        """Return variables in topological order."""
        in_degree = {v: len(self._reverse.get(v, set())) for v in self._variables}
        queue = deque([v for v, d in in_degree.items() if d == 0])
        result = []
        while queue:
            node = queue.popleft()
            result.append(node)
            for child in self._adjacency.get(node, set()):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        return result

    def d_separated(
        self, x: str, y: str, conditioning: Optional[Set[str]] = None
    ) -> bool:
        """Test d-separation between x and y given conditioning set.

        Uses Bayes-Ball algorithm for efficient d-separation testing.

        Args:
            x: Source variable.
            y: Target variable.
            conditioning: Set of conditioned variables.

        Returns:
            True if x and y are d-separated given conditioning.
        """
        if conditioning is None:
            conditioning = set()

        # Bayes-Ball: check if ball can travel from x to y
        reachable = set()
        queue = deque([(x, 'up')])  # (node, direction)
        visited = set()

        while queue:
            node, direction = queue.popleft()
            if (node, direction) in visited:
                continue
            visited.add((node, direction))

            if node not in conditioning:
                reachable.add(node)

            # Ball traveling up (from child to parent)
            if direction == 'up' and node not in conditioning:
                for parent in self._reverse.get(node, set()):
                    queue.append((parent, 'up'))
                for child in self._adjacency.get(node, set()):
                    queue.append((child, 'down'))

            # Ball traveling down (from parent to child)
            elif direction == 'down':
                if node not in conditioning:
                    for child in self._adjacency.get(node, set()):
                        queue.append((child, 'down'))
                if node in conditioning or self._has_conditioned_descendant(node, conditioning):
                    for parent in self._reverse.get(node, set()):
                        queue.append((parent, 'up'))

        return y not in reachable

    def _has_conditioned_descendant(self, node: str, conditioning: Set[str]) -> bool:
        """Check if node has any descendant in conditioning set."""
        return bool(self.get_descendants(node) & conditioning)

    def interventional_graph(self, intervention_targets: Set[str]) -> 'CausalGraph':
        """Return mutilated graph after do-intervention.

        Removes all incoming edges to intervention targets
        (graph surgery for do-operator).
        """
        new_graph = CausalGraph()
        for v in self._variables.values():
            new_graph.add_variable(v.name, v.dim, v.observed)

        for parent, children in self._adjacency.items():
            for child in children:
                if child not in intervention_targets:
                    new_graph.add_edge(parent, child)

        return new_graph


class DoOperator(nn.Module):
    """Pearl's do(X=x) intervention operator.

    Computes the effect of setting variable X to value x,
    using the truncated factorization formula:
    P(Y|do(X=x)) = sum_z P(Y|X=x, Z=z) P(Z=z)

    Args:
        dim: Variable embedding dimension.
        num_variables: Maximum number of variables.
    """

    def __init__(self, dim: int, num_variables: int = 32):
        super().__init__()
        self.dim = dim
        self.num_variables = num_variables

        # Structural equation models: parent embeddings → child value
        self.structural_eq = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

        # Noise model (exogenous variables)
        self.noise_encoder = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim),
        )

    def intervene(
        self,
        variable_embeddings: torch.Tensor,
        target_idx: int,
        intervention_value: torch.Tensor,
        parent_indices: List[int],
    ) -> torch.Tensor:
        """Apply do-intervention.

        Args:
            variable_embeddings: (N, dim) current variable states
            target_idx: Index of variable to intervene on
            intervention_value: (dim,) value to set
            parent_indices: Indices of parent variables

        Returns:
            (N, dim) updated variable embeddings after intervention
        """
        result = variable_embeddings.clone()

        # Set intervention target directly (ignoring parents)
        result[target_idx] = intervention_value

        return result

    def propagate(
        self,
        variable_embeddings: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        """Propagate effects through causal graph.

        Args:
            variable_embeddings: (N, dim) variable states
            adjacency: (N, N) binary adjacency matrix (i→j)

        Returns:
            (N, dim) propagated variable states
        """
        N = variable_embeddings.shape[0]
        result = variable_embeddings.clone()

        for j in range(N):
            parent_mask = adjacency[:, j] > 0
            if parent_mask.any():
                parent_embs = variable_embeddings[parent_mask]
                parent_agg = parent_embs.mean(dim=0)
                combined = torch.cat([result[j], parent_agg], dim=-1)
                noise = self.noise_encoder(result[j])
                result[j] = self.structural_eq(combined) + noise

        return result


class CounterfactualEngine(nn.Module):
    """Answer counterfactual questions: "What if X had been Y?"

    Uses abduction-action-prediction:
    1. Abduction: Infer exogenous noise from observed evidence
    2. Action: Apply counterfactual intervention
    3. Prediction: Propagate through modified model

    Args:
        dim: Variable embedding dimension.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        # Abduction: infer exogenous variables from observations
        self.abductor = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

        # Prediction: propagate counterfactual
        self.predictor = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def counterfactual(
        self,
        factual_state: torch.Tensor,
        counterfactual_intervention: torch.Tensor,
        parent_state: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute counterfactual outcome.

        Args:
            factual_state: (dim,) actual observed state
            counterfactual_intervention: (dim,) "what if" value
            parent_state: (dim,) aggregated parent state

        Returns:
            Dict with 'counterfactual_outcome', 'exogenous_noise',
            'divergence' (how different from factual)
        """
        # Step 1: Abduction — infer noise
        combined = torch.cat([factual_state, parent_state], dim=-1)
        noise = self.abductor(combined)

        # Step 2 & 3: Action + Prediction
        cf_input = torch.cat([counterfactual_intervention, noise], dim=-1)
        outcome = self.predictor(cf_input)

        divergence = F.mse_loss(outcome, factual_state, reduction='none').mean()

        return {
            'counterfactual_outcome': outcome,
            'exogenous_noise': noise,
            'divergence': divergence,
        }


class CausalDiscovery(nn.Module):
    """Learn causal structure from observational data.

    Uses a differentiable approach inspired by NOTEARS:
    learn a weighted adjacency matrix W such that W is a DAG.

    DAG constraint: tr(exp(W * W)) - d = 0

    Args:
        num_variables: Number of variables.
        dim: Variable dimension.
        lambda_sparse: Sparsity regularization weight.
        lambda_dag: DAG constraint weight.
    """

    def __init__(
        self,
        num_variables: int,
        dim: int = 1,
        lambda_sparse: float = 0.01,
        lambda_dag: float = 1.0,
    ):
        super().__init__()
        self.num_variables = num_variables
        self.dim = dim
        self.lambda_sparse = lambda_sparse
        self.lambda_dag = lambda_dag

        # Learnable adjacency matrix
        self.W = nn.Parameter(torch.zeros(num_variables, num_variables))

        # Structural equation parameters
        self.linear_sem = nn.Linear(num_variables * dim, num_variables * dim)

    def dag_constraint(self) -> torch.Tensor:
        """Compute DAG-ness constraint: tr(exp(W*W)) - d.

        Returns 0 iff W is a DAG.
        """
        d = self.num_variables
        W_sq = self.W * self.W
        # Matrix exponential via power series (truncated)
        M = torch.eye(d, device=self.W.device)
        power = torch.eye(d, device=self.W.device)
        for k in range(1, d + 1):
            power = power @ W_sq / k
            M = M + power
        return torch.trace(M) - d

    def forward(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Fit causal model to data and return loss.

        Args:
            data: (B, num_variables * dim) observation batch

        Returns:
            Dict with 'loss', 'reconstruction', 'sparsity',
            'dag_penalty', 'adjacency'
        """
        B = data.shape[0]

        # Apply SEM: X = W^T X + noise
        predicted = self.linear_sem(data)
        recon_loss = F.mse_loss(predicted, data)

        # Sparsity
        sparsity = self.W.abs().sum()

        # DAG constraint
        dag_penalty = self.dag_constraint()

        loss = (
            recon_loss +
            self.lambda_sparse * sparsity +
            self.lambda_dag * dag_penalty
        )

        # Threshold adjacency for structure
        adjacency = (self.W.abs() > 0.1).float()

        return {
            'loss': loss,
            'reconstruction': recon_loss,
            'sparsity': sparsity,
            'dag_penalty': dag_penalty,
            'adjacency': adjacency,
            'weighted_adjacency': self.W.detach(),
        }


class InterventionalQuery(nn.Module):
    """Answer causal queries at Pearl's three levels.

    Level 1 (Association): P(Y|X) — seeing
    Level 2 (Intervention): P(Y|do(X)) — doing
    Level 3 (Counterfactual): P(Y_x|X'=x') — imagining

    Args:
        dim: Variable embedding dimension.
        num_variables: Number of causal variables.
    """

    def __init__(self, dim: int, num_variables: int = 32):
        super().__init__()
        self.dim = dim
        self.num_variables = num_variables

        self.do_operator = DoOperator(dim, num_variables)
        self.cf_engine = CounterfactualEngine(dim)

        # Association query (level 1): conditional probability
        self.association_net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def query_association(
        self,
        evidence: torch.Tensor,
        query_var: torch.Tensor,
    ) -> torch.Tensor:
        """Level 1: P(Y|X=x) — observational query.

        Args:
            evidence: (dim,) observed evidence
            query_var: (dim,) query variable embedding

        Returns:
            (dim,) predicted query variable value
        """
        combined = torch.cat([evidence, query_var], dim=-1)
        return self.association_net(combined)

    def query_intervention(
        self,
        variable_states: torch.Tensor,
        target_idx: int,
        intervention_value: torch.Tensor,
        query_idx: int,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        """Level 2: P(Y|do(X=x)) — interventional query.

        Args:
            variable_states: (N, dim) current states
            target_idx: Variable to intervene on
            intervention_value: (dim,) intervention value
            query_idx: Variable to query
            adjacency: (N, N) causal adjacency matrix

        Returns:
            (dim,) predicted query variable value after intervention
        """
        # Apply intervention
        parent_indices = (adjacency[:, target_idx] > 0).nonzero(as_tuple=True)[0].tolist()
        intervened = self.do_operator.intervene(
            variable_states, target_idx, intervention_value, parent_indices
        )

        # Propagate
        propagated = self.do_operator.propagate(intervened, adjacency)

        return propagated[query_idx]

    def query_counterfactual(
        self,
        factual_state: torch.Tensor,
        counterfactual_value: torch.Tensor,
        parent_state: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Level 3: P(Y_x|X'=x') — counterfactual query.

        Args:
            factual_state: (dim,) what actually happened
            counterfactual_value: (dim,) what we imagine instead
            parent_state: (dim,) parent variable state

        Returns:
            Dict with counterfactual results
        """
        return self.cf_engine.counterfactual(
            factual_state, counterfactual_value, parent_state
        )
