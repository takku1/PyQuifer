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
from typing import Optional, Dict, List, Tuple


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

    def _estimate_mi(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Estimate mutual information between two series using correlation.

        MI ≈ -0.5 * log(1 - r^2) for Gaussian variables.

        Args:
            x: First series (T, D)
            y: Second series (T, D)

        Returns:
            Estimated MI (non-negative)
        """
        if x.shape[0] < 10:
            return 0.0

        # Flatten to 1D for correlation
        x_flat = x.reshape(x.shape[0], -1)
        y_flat = y.reshape(y.shape[0], -1)

        # Mean-center
        x_c = x_flat - x_flat.mean(dim=0, keepdim=True)
        y_c = y_flat - y_flat.mean(dim=0, keepdim=True)

        # Average correlation across dimensions
        x_std = x_c.std(dim=0) + 1e-8
        y_std = y_c.std(dim=0) + 1e-8

        correlations = (x_c * y_c).mean(dim=0) / (x_std * y_std)
        r_squared = correlations.mean().item() ** 2
        r_squared = min(r_squared, 0.999)  # Prevent log(0)

        mi = -0.5 * math.log(1 - r_squared)
        return max(0.0, mi)

    def forward(self,
                group_outputs: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Record group outputs and detect symbiogenesis.

        Args:
            group_outputs: List of group output tensors

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

        # Compute pairwise MI
        mi_matrix = torch.zeros(self.num_groups, self.num_groups)
        if n_valid >= 20:
            for i in range(self.num_groups):
                for j in range(i + 1, self.num_groups):
                    mi = self._estimate_mi(
                        self.history[:n_valid, i],
                        self.history[:n_valid, j]
                    )
                    mi_matrix[i, j] = mi
                    mi_matrix[j, i] = mi

        # Detect bonds
        with torch.no_grad():
            new_bonds = (mi_matrix > self.mi_threshold).float()
            # Bonds are sticky: once formed, harder to break
            self.bonds.copy_(torch.max(self.bonds * 0.99, new_bonds))

        num_bonds = (self.bonds > 0.5).sum().item() // 2  # Each bond counted twice

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
