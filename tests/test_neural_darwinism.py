"""Tests for neural_darwinism module."""
import torch
import math
import pytest
from pyquifer.neural_darwinism import NeuronalGroup, SelectionArena, SymbiogenesisDetector


class TestNeuronalGroup:
    def test_construction(self):
        group = NeuronalGroup(dim=16)
        assert group.fitness.item() == 0.5
        assert group.resources.item() == 1.0

    def test_output_keys(self):
        group = NeuronalGroup(dim=8)
        r = group(torch.randn(8))
        assert all(k in r for k in ['output', 'raw_output', 'fitness', 'resources'])

    def test_output_shape(self):
        group = NeuronalGroup(dim=16)
        r = group(torch.randn(16))
        assert r['output'].shape == (16,)

    def test_batch_support(self):
        group = NeuronalGroup(dim=8)
        r = group(torch.randn(4, 8))
        assert r['output'].shape == (4, 8)

    def test_resource_gating(self):
        group = NeuronalGroup(dim=8)
        x = torch.randn(8)
        r_normal = group(x)

        with torch.no_grad():
            group.activation_level.fill_(0.0)
        r_gated = group(x)
        assert r_gated['output'].abs().sum().item() == 0.0
        assert r_gated['raw_output'].abs().sum().item() > 0.0


class TestSelectionArena:
    def test_construction(self):
        arena = SelectionArena(num_groups=4, group_dim=16)
        assert len(arena.groups) == 4

    def test_output_keys(self):
        arena = SelectionArena(num_groups=3, group_dim=8)
        r = arena(torch.randn(8))
        assert all(k in r for k in ['output', 'group_outputs', 'fitnesses',
                                      'resources', 'mean_fitness', 'fitness_variance'])

    def test_output_shape(self):
        arena = SelectionArena(num_groups=4, group_dim=16)
        r = arena(torch.randn(16))
        assert r['output'].shape == (16,)

    def test_batch_output_shape(self):
        arena = SelectionArena(num_groups=3, group_dim=8)
        r = arena(torch.randn(4, 8))
        assert r['output'].shape == (4, 8)

    def test_resources_sum_to_budget(self):
        arena = SelectionArena(num_groups=4, group_dim=8, total_budget=10.0)
        arena(torch.randn(8))
        total = sum(g.resources.item() for g in arena.groups)
        assert abs(total - 10.0) < 0.1

    def test_selection_favors_fit_groups(self):
        torch.manual_seed(42)
        arena = SelectionArena(num_groups=4, group_dim=16, selection_pressure=0.3)
        target = torch.sin(torch.linspace(0, math.pi, 16))
        for _ in range(200):
            arena(target + torch.randn(16) * 0.1, global_coherence=target)
        resources = [g.resources.item() for g in arena.groups]
        assert max(resources) > min(resources) * 1.1

    def test_step_counter_increments(self):
        arena = SelectionArena(num_groups=2, group_dim=8)
        for _ in range(10):
            arena(torch.randn(8))
        assert arena.step_count.item() == 10

    def test_reset(self):
        arena = SelectionArena(num_groups=4, group_dim=8, total_budget=8.0)
        for _ in range(50):
            arena(torch.randn(8))
        arena.reset()
        per_group = 8.0 / 4
        for g in arena.groups:
            assert abs(g.resources.item() - per_group) < 0.01
        assert arena.step_count.item() == 0


class TestSymbiogenesisDetector:
    def test_construction(self):
        sd = SymbiogenesisDetector(num_groups=4, group_dim=8)
        assert sd.bonds.shape == (4, 4)

    def test_output_keys(self):
        sd = SymbiogenesisDetector(num_groups=3, group_dim=8)
        outputs = [torch.randn(8) for _ in range(3)]
        r = sd(outputs)
        assert all(k in r for k in ['bonds', 'num_bonds', 'mi_matrix'])

    def test_no_bonds_with_few_samples(self):
        sd = SymbiogenesisDetector(num_groups=3, group_dim=4)
        for _ in range(5):
            outputs = [torch.randn(4) for _ in range(3)]
            r = sd(outputs)
        assert r['num_bonds'].item() == 0

    def test_correlated_groups_form_bonds(self):
        torch.manual_seed(42)
        sd = SymbiogenesisDetector(num_groups=3, group_dim=4,
                                    mi_threshold=0.1, buffer_size=200)
        for _ in range(200):
            base = torch.randn(4)
            outputs = [
                base + torch.randn(4) * 0.1,  # Group 0: tracks base
                base + torch.randn(4) * 0.1,  # Group 1: tracks base
                torch.randn(4),                 # Group 2: independent
            ]
            r = sd(outputs)
        # Groups 0 and 1 should have higher MI than with group 2
        assert r['mi_matrix'][0, 1] > r['mi_matrix'][0, 2]

    def test_history_wraps(self):
        sd = SymbiogenesisDetector(num_groups=2, group_dim=4, buffer_size=10)
        for _ in range(25):
            sd([torch.randn(4), torch.randn(4)])
        assert sd.hist_ptr.item() == 25

    def test_get_bonded_groups_empty(self):
        sd = SymbiogenesisDetector(num_groups=3, group_dim=4)
        assert sd.get_bonded_groups() == []

    def test_reset(self):
        sd = SymbiogenesisDetector(num_groups=3, group_dim=4)
        for _ in range(30):
            sd([torch.randn(4) for _ in range(3)])
        sd.reset()
        assert sd.hist_ptr.item() == 0
        assert sd.bonds.sum().item() == 0
