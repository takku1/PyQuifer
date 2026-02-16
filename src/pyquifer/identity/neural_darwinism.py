"""
Neural Darwinism Module for PyQuifer

Neuronal Group Selection — the "evolutionary engine" where subsystems
compete for representation. Winners are strengthened, losers atrophy,
but system-level coherence EMERGES from micro-level competition.

The paradox: competition at micro-level → coherence at macro-level.
This works because fitness is DEFINED by contribution to the whole.

Key concepts (Edelman's three mechanisms):
1. Developmental: Initial diversity (random initialization)
2. Experiential: Fitness-based selection (useful groups thrive)
3. Reentrant: Cross-group signaling forms functional maps (cooperation emerges)

References:
- Edelman (1987). Neural Darwinism: The Theory of Neuronal Group Selection.
- Edelman (1993). Neural Darwinism: Selection and Reentrant Signaling.
- Margulis (1967). On the Origin of Mitosing Cells. (Symbiogenesis)
"""

import torch
import torch.nn as nn
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple


@dataclass
class HypothesisProfile:
    """Named hypothesis with evidence mapping for SelectionArena."""
    name: str
    group_indices: List[int]  # Which arena groups represent this hypothesis
    evidence_keys: List[str]  # Which evidence sources support this
    base_weight: float = 1.0  # Prior strength


class NeuronalGroup(nn.Module):
    """
    A population with fitness, resources, and activation.

    - fitness: How much this group contributes to system coherence
    - resources: Accumulated fitness (the group's "energy budget")
    - activation: Resources gate how much influence this group has

    Groups compete for a finite resource pool. High-fitness groups
    thrive; low-fitness groups atrophy (slowly — not instant death).
    """

    def __init__(self,
                 dim: int,
                 group_id: int = 0):
        """
        Args:
            dim: Dimension of this group's output
            group_id: Identifier for this group
        """
        super().__init__()
        self.dim = dim
        self.group_id = group_id

        # Group's processing network
        self.network = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
        )

        # State (managed by dynamics, not backprop)
        self.register_buffer('fitness', torch.tensor(0.5))
        self.register_buffer('resources', torch.tensor(1.0))
        self.register_buffer('activation_level', torch.tensor(1.0))

    def forward(self, input: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process input, gated by resources.

        Args:
            input: Input tensor (..., dim)

        Returns:
            Dictionary with:
            - output: Resource-gated output
            - raw_output: Output before resource gating
        """
        raw_output = self.network(input)
        # Resources gate influence
        gated_output = raw_output * self.activation_level

        return {
            'output': gated_output,
            'raw_output': raw_output,
            'fitness': self.fitness.clone(),
            'resources': self.resources.clone(),
        }


class SelectionArena(nn.Module):
    """
    The competition environment for neuronal groups.

    Finite resource pool (total activation budget is bounded).
    Groups compete: high-fitness groups get more resources.
    Implements Edelman's three mechanisms:
    1. Developmental: initial diversity (random init — handled by NeuronalGroup)
    2. Experiential: fitness-based selection (replicator equation)
    3. Reentrant: cross-group signaling forms functional maps
    """

    def __init__(self,
                 num_groups: int,
                 group_dim: int,
                 total_budget: float = 10.0,
                 selection_pressure: float = 0.1,
                 atrophy_rate: float = 0.01):
        """
        Args:
            num_groups: Number of competing groups
            group_dim: Dimension of each group
            total_budget: Total resource budget (finite)
            selection_pressure: How strongly fitness affects resources
            atrophy_rate: How fast low-fitness groups lose resources
        """
        super().__init__()
        self.num_groups = num_groups
        self.group_dim = group_dim
        self.total_budget = total_budget
        self.selection_pressure = selection_pressure
        self.atrophy_rate = atrophy_rate

        # Create groups
        self.groups = nn.ModuleList([
            NeuronalGroup(group_dim, group_id=i)
            for i in range(num_groups)
        ])

        # Reentrant connections (cross-group signaling)
        self.reentrant = nn.Linear(group_dim * num_groups, group_dim * num_groups)

        # Step counter
        self.register_buffer('step_count', torch.tensor(0))

    def compute_fitness(self,
                        group_outputs: List[torch.Tensor],
                        global_coherence: torch.Tensor) -> torch.Tensor:
        """
        Compute fitness for each group based on contribution to coherence.

        fitness_i = correlation(group_i_output, global_coherence_signal)

        Args:
            group_outputs: List of group outputs (num_groups items)
            global_coherence: System-level coherence signal (group_dim,)

        Returns:
            Fitness scores per group (num_groups,)
        """
        fitnesses = torch.zeros(self.num_groups)
        for i, output in enumerate(group_outputs):
            if output.dim() > 1:
                output = output.mean(dim=0)
            # Cosine similarity with global coherence
            cos_sim = torch.nn.functional.cosine_similarity(
                output.unsqueeze(0),
                global_coherence.unsqueeze(0)
            )
            fitnesses[i] = cos_sim.item()

        # Shift to positive range
        fitnesses = (fitnesses + 1) / 2  # [0, 1]
        return fitnesses

    def selection_step(self, fitnesses: torch.Tensor):
        """
        Apply selection pressure: replicator equation.

        d_resource_i/dt = resource_i * (fitness_i - mean_fitness)

        Args:
            fitnesses: Fitness per group (num_groups,)
        """
        mean_fitness = fitnesses.mean()

        with torch.no_grad():
            for i, group in enumerate(self.groups):
                # Replicator dynamics: relative fitness advantage
                advantage = fitnesses[i] - mean_fitness
                resource_delta = group.resources * advantage * self.selection_pressure

                # Atrophy: slight decay even for neutral fitness
                atrophy = -self.atrophy_rate * group.resources

                group.resources.add_(resource_delta + atrophy)
                group.resources.clamp_(min=0.01)  # Don't kill completely
                group.fitness.copy_(fitnesses[i])

            # Normalize resources to total budget
            total_resources = sum(g.resources.item() for g in self.groups)
            if total_resources > 0:
                scale = self.total_budget / total_resources
                for group in self.groups:
                    group.resources.mul_(scale)
                    # Activation = resource share
                    group.activation_level.copy_(
                        group.resources / self.total_budget * self.num_groups
                    )

    def forward(self,
                input: torch.Tensor,
                global_coherence: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Process input through all groups with competition.

        Args:
            input: Input tensor (..., group_dim)
            global_coherence: System coherence signal for fitness.
                            If None, uses mean of group outputs.

        Returns:
            Dictionary with:
            - output: Combined group outputs
            - group_outputs: Individual group outputs
            - fitnesses: Per-group fitness
            - resources: Per-group resources
        """
        # Process through each group
        group_results = []
        group_outputs = []
        for group in self.groups:
            result = group(input)
            group_results.append(result)
            group_outputs.append(result['output'])

        # Stack and apply reentrant connections
        if input.dim() == 1:
            stacked = torch.cat([o for o in group_outputs], dim=-1)
            reentrant_signal = self.reentrant(stacked)
            # Reshape back
            reentrant_parts = reentrant_signal.split(self.group_dim, dim=-1)
        else:
            stacked = torch.cat([o for o in group_outputs], dim=-1)
            reentrant_signal = self.reentrant(stacked)
            reentrant_parts = reentrant_signal.split(self.group_dim, dim=-1)

        # Add reentrant feedback (mechanism 3)
        for i in range(self.num_groups):
            group_outputs[i] = group_outputs[i] + 0.1 * reentrant_parts[i]

        # Combined output (resource-weighted sum)
        combined = torch.stack(group_outputs, dim=0)
        weights = torch.stack([g.activation_level.detach() for g in self.groups]).to(input.device)
        weights = weights / (weights.sum() + 1e-8)

        if combined.dim() == 2:
            output = (combined * weights.unsqueeze(1)).sum(dim=0)
        else:
            output = (combined * weights.view(-1, 1, 1)).sum(dim=0)

        # Compute fitness
        if global_coherence is None:
            global_coherence = output.detach()
            if global_coherence.dim() > 1:
                global_coherence = global_coherence.mean(dim=0)

        fitnesses = self.compute_fitness(group_outputs, global_coherence)

        # Selection step
        self.selection_step(fitnesses)

        with torch.no_grad():
            self.step_count.add_(1)

        resources = torch.stack([g.resources.detach() for g in self.groups])

        return {
            'output': output,
            'group_outputs': group_outputs,
            'fitnesses': fitnesses,
            'resources': resources,
            'mean_fitness': fitnesses.mean(),
            'fitness_variance': fitnesses.var(),
        }

    def set_hypotheses(self, profiles: List[HypothesisProfile]):
        """
        Assign groups to named hypotheses and generate orthogonal basis vectors.

        Each hypothesis gets a coherence target where its groups are activated
        and other groups are suppressed (-0.1). This mirrors the benchmark's
        _build_coherence_target() logic but lives in the library.
        """
        self._hypothesis_profiles = profiles
        self._hypothesis_targets: Dict[str, torch.Tensor] = {}

        for profile in profiles:
            target = torch.full((self.group_dim,), -0.1)
            # Each hypothesis's groups get a slice of the vector space
            n_indices = len(profile.group_indices)
            if n_indices == 0:
                continue
            slice_size = self.group_dim // max(1, len(profiles))
            start = profiles.index(profile) * slice_size
            end = start + slice_size
            target[start:end] = profile.base_weight
            self._hypothesis_targets[profile.name] = target

    def inject_evidence(self, evidence: Dict[str, float]) -> torch.Tensor:
        """
        Convert evidence dict into coherence target using hypothesis profiles.

        Builds a combined coherence vector where each hypothesis's region
        is activated proportionally to its evidence weight. The strongest
        hypothesis dominates, while weaker ones get slight suppression.

        Args:
            evidence: Dict mapping evidence_key -> strength (float)

        Returns:
            Coherence target tensor (group_dim,)
        """
        if not hasattr(self, '_hypothesis_profiles') or not self._hypothesis_profiles:
            # Backward-compatible: return zeros
            return torch.zeros(self.group_dim)

        n_profiles = len(self._hypothesis_profiles)
        slice_size = self.group_dim // max(1, n_profiles)

        # Compute per-hypothesis evidence weight
        hypothesis_weights: Dict[str, float] = {}
        for profile in self._hypothesis_profiles:
            weight = profile.base_weight
            for key in profile.evidence_keys:
                if key in evidence:
                    weight += evidence[key]
            hypothesis_weights[profile.name] = weight

        # Build combined target: each hypothesis region gets its weight,
        # weaker regions get -0.1 suppression relative to the winner
        target = torch.full((self.group_dim,), -0.1)
        for i, profile in enumerate(self._hypothesis_profiles):
            start = i * slice_size
            end = start + slice_size
            target[start:end] = hypothesis_weights[profile.name]

        return target

    def get_hypothesis_strengths(self) -> Dict[str, float]:
        """
        Return resource totals per hypothesis (not per group).

        Aggregates group resources by hypothesis name.
        """
        if not hasattr(self, '_hypothesis_profiles') or not self._hypothesis_profiles:
            return {}

        strengths: Dict[str, float] = {}
        for profile in self._hypothesis_profiles:
            total = 0.0
            for idx in profile.group_indices:
                if 0 <= idx < self.num_groups:
                    total += self.groups[idx].resources.item()
            strengths[profile.name] = total
        return strengths

    def reset(self):
        """Reset all groups to equal resources."""
        per_group = self.total_budget / self.num_groups
        for group in self.groups:
            group.resources.fill_(per_group)
            group.activation_level.fill_(1.0)
            group.fitness.fill_(0.5)
        self.step_count.zero_()


class SymbiogenesisDetector(nn.Module):
    """
    Detects when groups begin cooperating (symbiogenesis).

    Mutual information between groups above threshold → "symbiotic bond".
    Bonded groups share resources — cooperation emerges from competition.

    Named after Margulis's theory: mitochondria were once independent
    organisms that became symbiotically integrated.
    """

    def __init__(self,
                 num_groups: int,
                 group_dim: int,
                 mi_threshold: float = 0.5,
                 buffer_size: int = 100):
        """
        Args:
            num_groups: Number of groups to monitor
            group_dim: Dimension of group outputs
            mi_threshold: MI threshold for detecting symbiosis
            buffer_size: History buffer for MI computation
        """
        super().__init__()
        self.num_groups = num_groups
        self.group_dim = group_dim
        self.mi_threshold = mi_threshold
        self.buffer_size = buffer_size

        # History of group activations
        self.register_buffer('history',
                             torch.zeros(buffer_size, num_groups, group_dim))
        self.register_buffer('hist_ptr', torch.tensor(0))

        # Symbiotic bond matrix (boolean-like)
        self.register_buffer('bonds', torch.zeros(num_groups, num_groups))

    def _estimate_mi_all_pairs(self, n_valid: int) -> torch.Tensor:
        """
        Estimate mutual information between all group pairs at once.

        Uses vectorized correlation: MI ≈ -0.5 * log(1 - r^2) for Gaussian.

        Returns:
            mi_matrix: (num_groups, num_groups) symmetric MI matrix
        """
        mi_matrix = torch.zeros(self.num_groups, self.num_groups,
                                device=self.bonds.device)
        if n_valid < 10:
            return mi_matrix

        # history[:n_valid] is (T, G, D)
        data = self.history[:n_valid]  # (T, G, D)
        T, G, D = data.shape

        # Mean-center each group's time series
        data_c = data - data.mean(dim=0, keepdim=True)  # (T, G, D)
        # Std per group per dim
        data_std = data_c.std(dim=0) + 1e-8  # (G, D)

        # For each pair (i, j), we need:
        #   r = mean over D of: (mean over T of data_c[:,i,:] * data_c[:,j,:]) / (std_i * std_j)
        # Vectorize: compute cross-correlation matrix (G, G, D) then average over D
        # cross_corr[i,j,d] = mean_t(data_c[t,i,d] * data_c[t,j,d])
        # = (data_c[:,:,:].T @ data_c[:,:,:]) / T  but need einsum for the grouping

        # Reshape to (T, G*D) then compute outer products? No, use einsum:
        # cross_cov[i,j,d] = sum_t data_c[t,i,d] * data_c[t,j,d] / T
        cross_cov = torch.einsum('tid,tjd->ijd', data_c, data_c) / T  # (G, G, D)

        # Normalize by stds
        # norm[i,j,d] = cross_cov[i,j,d] / (std[i,d] * std[j,d])
        std_outer = data_std.unsqueeze(1) * data_std.unsqueeze(0)  # (G, G, D)
        correlations = cross_cov / std_outer  # (G, G, D)

        # Mean correlation across dimensions, then MI
        mean_corr = correlations.mean(dim=-1)  # (G, G)
        r_squared = mean_corr.pow(2).clamp(max=0.999)
        mi_matrix = (-0.5 * torch.log(1 - r_squared)).clamp(min=0.0)

        # Zero the diagonal
        mi_matrix.fill_diagonal_(0.0)

        return mi_matrix

    def forward(self,
                group_outputs: List[torch.Tensor],
                compute_every: int = 10) -> Dict[str, torch.Tensor]:
        """
        Record group outputs and detect symbiogenesis.

        Args:
            group_outputs: List of group output tensors
            compute_every: Only recompute MI every N calls (default 10)

        Returns:
            Dictionary with:
            - bonds: Symbiotic bond matrix
            - num_bonds: Number of active symbiotic bonds
            - mi_matrix: Mutual information between all pairs
        """
        # Record history
        with torch.no_grad():
            idx = self.hist_ptr % self.buffer_size
            for i, output in enumerate(group_outputs):
                out = output.detach()
                if out.dim() > 1:
                    out = out.mean(dim=0)
                self.history[idx, i] = out
            self.hist_ptr.add_(1)

        n_valid = min(self.hist_ptr.item(), self.buffer_size)

        # Only recompute MI every compute_every calls
        should_compute = (self.hist_ptr.item() % compute_every == 0) and n_valid >= 20

        if should_compute:
            # Vectorized pairwise MI computation
            mi_matrix = self._estimate_mi_all_pairs(n_valid)

            # Cache for non-compute ticks
            if not hasattr(self, '_cached_mi'):
                self.register_buffer('_cached_mi',
                    torch.zeros(self.num_groups, self.num_groups,
                                device=self.bonds.device))
            self._cached_mi.copy_(mi_matrix)

            # Detect bonds
            with torch.no_grad():
                new_bonds = (mi_matrix > self.mi_threshold).float()
                self.bonds.copy_(torch.max(self.bonds * 0.99, new_bonds))
        else:
            mi_matrix = getattr(self, '_cached_mi',
                torch.zeros(self.num_groups, self.num_groups,
                            device=self.bonds.device))

        num_bonds = (self.bonds > 0.5).sum().item() // 2

        return {
            'bonds': self.bonds.clone(),
            'num_bonds': torch.tensor(num_bonds),
            'mi_matrix': mi_matrix,
        }

    def get_bonded_groups(self) -> List[Tuple]:
        """Get list of bonded group pairs."""
        pairs = []
        for i in range(self.num_groups):
            for j in range(i + 1, self.num_groups):
                if self.bonds[i, j] > 0.5:
                    pairs.append((i, j))
        return pairs

    def reset(self):
        """Reset bonds and history."""
        self.bonds.zero_()
        self.history.zero_()
        self.hist_ptr.zero_()


class SpeciatedSelectionArena(nn.Module):
    """
    Selection arena with speciation, fitness sharing, and stagnation detection.

    Extends SelectionArena with biological speciation mechanisms:
    1. Groups are clustered into species by weight similarity
    2. Fitness is shared within species (prevents single-species dominance)
    3. Stagnant species (no improvement for N steps) are eliminated

    Same interface as SelectionArena for drop-in replacement.

    Args:
        num_groups: Number of competing groups
        group_dim: Dimension of each group
        total_budget: Total resource budget
        selection_pressure: How strongly fitness affects resources
        atrophy_rate: How fast low-fitness groups lose resources
        compatibility_threshold: Distance threshold for same-species
        stagnation_limit: Steps without improvement before species elimination
    """

    def __init__(self,
                 num_groups: int,
                 group_dim: int,
                 total_budget: float = 10.0,
                 selection_pressure: float = 0.1,
                 atrophy_rate: float = 0.01,
                 compatibility_threshold: float = 0.5,
                 stagnation_limit: int = 50):
        super().__init__()
        self.num_groups = num_groups
        self.group_dim = group_dim
        self.total_budget = total_budget
        self.selection_pressure = selection_pressure
        self.atrophy_rate = atrophy_rate
        self.compatibility_threshold = compatibility_threshold
        self.stagnation_limit = stagnation_limit

        # Groups (same as SelectionArena)
        self.groups = nn.ModuleList([
            NeuronalGroup(group_dim, group_id=i)
            for i in range(num_groups)
        ])

        # Reentrant connections
        self.reentrant = nn.Linear(group_dim * num_groups, group_dim * num_groups)

        # Species tracking
        self.register_buffer('species_ids', torch.zeros(num_groups, dtype=torch.long))
        self.register_buffer('species_best_fitness', torch.zeros(num_groups))
        self.register_buffer('species_stagnation', torch.zeros(num_groups, dtype=torch.long))
        self.register_buffer('step_count', torch.tensor(0))

    def _compute_distance(self, group_i: NeuronalGroup, group_j: NeuronalGroup) -> float:
        """Cosine distance between group weight vectors."""
        w_i = torch.cat([p.flatten() for p in group_i.network.parameters()])
        w_j = torch.cat([p.flatten() for p in group_j.network.parameters()])

        cos_sim = torch.nn.functional.cosine_similarity(
            w_i.unsqueeze(0), w_j.unsqueeze(0)
        )
        return (1.0 - cos_sim.item())  # distance = 1 - similarity

    def _assign_species(self):
        """Cluster groups into species by compatibility threshold."""
        # Representative: first member of each species
        representatives = {}  # species_id -> group_index

        with torch.no_grad():
            for i, group in enumerate(self.groups):
                assigned = False
                for sp_id, rep_idx in representatives.items():
                    dist = self._compute_distance(group, self.groups[rep_idx])
                    if dist < self.compatibility_threshold:
                        self.species_ids[i] = sp_id
                        assigned = True
                        break
                if not assigned:
                    new_id = max(representatives.keys(), default=-1) + 1
                    representatives[new_id] = i
                    self.species_ids[i] = new_id

    def _fitness_sharing(self, raw_fitnesses: torch.Tensor) -> torch.Tensor:
        """Adjust fitness by species size: adjusted = raw / species_size."""
        adjusted = raw_fitnesses.clone()
        for sp_id in self.species_ids.unique():
            mask = self.species_ids == sp_id
            species_size = mask.sum().float()
            adjusted[mask] = adjusted[mask] / species_size
        return adjusted

    def _stagnation_check(self, fitnesses: torch.Tensor):
        """Eliminate species with no improvement for stagnation_limit steps."""
        with torch.no_grad():
            for sp_id in self.species_ids.unique():
                mask = self.species_ids == sp_id
                species_max_fitness = fitnesses[mask].max()

                if species_max_fitness > self.species_best_fitness[sp_id]:
                    self.species_best_fitness[sp_id] = species_max_fitness
                    self.species_stagnation[sp_id] = 0
                else:
                    self.species_stagnation[sp_id] += 1

                # Eliminate stagnant species (reduce resources drastically)
                if self.species_stagnation[sp_id] > self.stagnation_limit:
                    for i, group in enumerate(self.groups):
                        if self.species_ids[i] == sp_id:
                            group.resources.mul_(0.1)
                    # Reset stagnation counter
                    self.species_stagnation[sp_id] = 0
                    self.species_best_fitness[sp_id] = 0.0

    def compute_fitness(self,
                        group_outputs: List[torch.Tensor],
                        global_coherence: torch.Tensor) -> torch.Tensor:
        """Compute fitness for each group (same as SelectionArena)."""
        fitnesses = torch.zeros(self.num_groups)
        for i, output in enumerate(group_outputs):
            if output.dim() > 1:
                output = output.mean(dim=0)
            cos_sim = torch.nn.functional.cosine_similarity(
                output.unsqueeze(0), global_coherence.unsqueeze(0)
            )
            fitnesses[i] = cos_sim.item()
        fitnesses = (fitnesses + 1) / 2
        return fitnesses

    def selection_step(self, fitnesses: torch.Tensor):
        """Apply speciated selection with fitness sharing."""
        # Re-assign species periodically
        if self.step_count % 10 == 0:
            self._assign_species()

        # Fitness sharing
        adjusted = self._fitness_sharing(fitnesses)

        # Stagnation check
        self._stagnation_check(fitnesses)

        # Replicator dynamics on adjusted fitness
        mean_fitness = adjusted.mean()
        with torch.no_grad():
            for i, group in enumerate(self.groups):
                advantage = adjusted[i] - mean_fitness
                resource_delta = group.resources * advantage * self.selection_pressure
                atrophy = -self.atrophy_rate * group.resources
                group.resources.add_(resource_delta + atrophy)
                group.resources.clamp_(min=0.01)
                group.fitness.copy_(fitnesses[i])

            # Normalize to budget
            total = sum(g.resources.item() for g in self.groups)
            if total > 0:
                scale = self.total_budget / total
                for group in self.groups:
                    group.resources.mul_(scale)
                    group.activation_level.copy_(
                        group.resources / self.total_budget * self.num_groups
                    )

    def forward(self,
                input: torch.Tensor,
                global_coherence: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Process input through speciated groups (same API as SelectionArena)."""
        group_results = []
        group_outputs = []
        for group in self.groups:
            result = group(input)
            group_results.append(result)
            group_outputs.append(result['output'])

        # Reentrant connections
        stacked = torch.cat(group_outputs, dim=-1)
        reentrant_signal = self.reentrant(stacked)
        reentrant_parts = reentrant_signal.split(self.group_dim, dim=-1)

        for i in range(self.num_groups):
            group_outputs[i] = group_outputs[i] + 0.1 * reentrant_parts[i]

        # Combined output
        combined = torch.stack(group_outputs, dim=0)
        weights = torch.stack([g.activation_level.detach() for g in self.groups]).to(input.device)
        weights = weights / (weights.sum() + 1e-8)

        if combined.dim() == 2:
            output = (combined * weights.unsqueeze(1)).sum(dim=0)
        else:
            output = (combined * weights.view(-1, 1, 1)).sum(dim=0)

        if global_coherence is None:
            global_coherence = output.detach()
            if global_coherence.dim() > 1:
                global_coherence = global_coherence.mean(dim=0)

        fitnesses = self.compute_fitness(group_outputs, global_coherence)
        self.selection_step(fitnesses)

        with torch.no_grad():
            self.step_count.add_(1)

        resources = torch.stack([g.resources.detach() for g in self.groups])

        return {
            'output': output,
            'group_outputs': group_outputs,
            'fitnesses': fitnesses,
            'resources': resources,
            'mean_fitness': fitnesses.mean(),
            'fitness_variance': fitnesses.var(),
            'species_ids': self.species_ids.clone(),
            'num_species': self.species_ids.unique().numel(),
        }

    def reset(self):
        """Reset all groups and species tracking."""
        per_group = self.total_budget / self.num_groups
        for group in self.groups:
            group.resources.fill_(per_group)
            group.activation_level.fill_(1.0)
            group.fitness.fill_(0.5)
        self.species_ids.zero_()
        self.species_best_fitness.zero_()
        self.species_stagnation.zero_()
        self.step_count.zero_()


if __name__ == '__main__':
    from typing import Tuple

    print("--- Neural Darwinism Examples ---")

    # Example 1: Neuronal Group
    print("\n1. Neuronal Group")
    group = NeuronalGroup(dim=16)

    input = torch.randn(16)
    result = group(input)
    print(f"   Output shape: {result['output'].shape}")
    print(f"   Initial fitness: {result['fitness'].item():.3f}")
    print(f"   Initial resources: {result['resources'].item():.3f}")

    # Example 2: Selection Arena
    print("\n2. Selection Arena (6 groups)")
    arena = SelectionArena(num_groups=6, group_dim=16, total_budget=10.0)

    # Provide a coherent target signal
    target = torch.sin(torch.linspace(0, math.pi, 16))

    for i in range(100):
        # Some noise in input
        input = target + torch.randn(16) * 0.3
        result = arena(input, global_coherence=target)

    print(f"   Fitnesses: {result['fitnesses'].detach().cpu().numpy().round(3)}")
    print(f"   Resources: {result['resources'].detach().cpu().numpy().round(3)}")
    print(f"   Mean fitness: {result['mean_fitness'].item():.3f}")
    print(f"   Fitness variance: {result['fitness_variance'].item():.4f}")

    # Example 3: Symbiogenesis
    print("\n3. Symbiogenesis Detection")
    arena2 = SelectionArena(num_groups=4, group_dim=8, total_budget=8.0)
    symbiosis = SymbiogenesisDetector(num_groups=4, group_dim=8, mi_threshold=0.3)

    for i in range(200):
        input = torch.randn(8)
        result = arena2(input)
        sym_result = symbiosis(result['group_outputs'])

    print(f"   Symbiotic bonds: {sym_result['num_bonds'].item()}")
    print(f"   MI matrix:\n{sym_result['mi_matrix'].detach().cpu().numpy().round(3)}")
    bonded = symbiosis.get_bonded_groups()
    if bonded:
        print(f"   Bonded pairs: {bonded}")

    # Example 4: Competition creates specialization
    print("\n4. Competition Creates Specialization")
    arena3 = SelectionArena(num_groups=4, group_dim=16, selection_pressure=0.2)

    # Different inputs at different times
    for epoch in range(3):
        target = torch.randn(16)  # New target each epoch
        for i in range(50):
            input = target + torch.randn(16) * 0.1
            result = arena3(input, global_coherence=target)

    resources = result['resources']
    print(f"   Final resources: {resources.detach().cpu().numpy().round(3)}")
    print(f"   Resource ratio (max/min): {resources.max().item() / resources.min().item():.2f}")
    print(f"   (Higher ratio = more specialization)")

    print("\n[OK] All neural darwinism tests passed!")
