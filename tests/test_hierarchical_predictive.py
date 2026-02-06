"""Tests for hierarchical_predictive module."""
import torch
import math
import pytest
from pyquifer.hierarchical_predictive import PredictiveLevel, HierarchicalPredictiveCoding


class TestPredictiveLevel:
    def test_construction(self):
        level = PredictiveLevel(input_dim=16, belief_dim=8)
        assert level.beliefs.shape == (8,)

    def test_output_keys(self):
        level = PredictiveLevel(input_dim=16, belief_dim=8)
        r = level(torch.randn(2, 16))
        assert all(k in r for k in ['prediction', 'error', 'weighted_error', 'beliefs'])

    def test_shapes(self):
        level = PredictiveLevel(input_dim=16, belief_dim=8)
        r = level(torch.randn(2, 16))
        assert r['prediction'].shape == (2, 16)
        assert r['beliefs'].shape == (8,)

    def test_reset(self):
        level = PredictiveLevel(input_dim=16, belief_dim=8)
        level(torch.randn(2, 16))
        level.reset()
        assert level.beliefs.abs().sum().item() == 0


class TestHierarchicalPredictiveCoding:
    def test_construction(self):
        hpc = HierarchicalPredictiveCoding(level_dims=[64, 32, 16])
        assert hpc.num_levels == 3

    def test_rejects_single_level(self):
        with pytest.raises(ValueError):
            HierarchicalPredictiveCoding(level_dims=[32])

    def test_output_keys(self):
        hpc = HierarchicalPredictiveCoding(level_dims=[32, 16])
        r = hpc(torch.randn(2, 32))
        assert all(k in r for k in ['predictions', 'errors', 'beliefs',
                                      'total_error', 'free_energy',
                                      'top_level_beliefs', 'bottom_prediction'])

    def test_level_count(self):
        hpc = HierarchicalPredictiveCoding(level_dims=[64, 32, 16])
        r = hpc(torch.randn(2, 64))
        assert len(r['predictions']) == 3
        assert len(r['errors']) == 3
        assert len(r['beliefs']) == 3

    def test_shapes(self):
        hpc = HierarchicalPredictiveCoding(level_dims=[64, 32, 16])
        r = hpc(torch.randn(2, 64))
        assert r['predictions'][0].shape == (2, 64)
        assert r['top_level_beliefs'].shape == (16,)

    def test_beliefs_converge(self):
        hpc = HierarchicalPredictiveCoding(level_dims=[32, 16], lr=0.1, num_iterations=3)
        pattern = torch.sin(torch.linspace(0, 4 * math.pi, 32)).unsqueeze(0)
        norms = []
        for _ in range(100):
            r = hpc(pattern + torch.randn(1, 32) * 0.05)
            norms.append(r['beliefs'][0].norm().item())
        late_var = torch.tensor(norms[-20:]).var().item()
        early_var = torch.tensor(norms[:20]).var().item()
        assert late_var < early_var + 0.1

    def test_backprop_training(self):
        # gen_lr=0: disable internal learning, use external optimizer only
        hpc = HierarchicalPredictiveCoding(level_dims=[32, 16], lr=0.1, gen_lr=0, num_iterations=1)
        optimizer = torch.optim.Adam(hpc.parameters(), lr=0.01)
        pattern = torch.sin(torch.linspace(0, 4 * math.pi, 32)).unsqueeze(0)
        errors = []
        for _ in range(100):
            optimizer.zero_grad()
            r = hpc(pattern + torch.randn(1, 32) * 0.05)
            loss = r['errors'][0].pow(2).mean()
            loss.backward()
            optimizer.step()
            errors.append(loss.item())
        assert errors[-1] < errors[0] * 0.9

    def test_external_precision(self):
        hpc = HierarchicalPredictiveCoding(level_dims=[16, 8])
        prec = torch.ones(16)
        prec[8:] = 0.01
        r = hpc(torch.randn(1, 16), precisions=[prec, torch.ones(16)])
        assert r['total_error'].item() > 0

    def test_reset(self):
        hpc = HierarchicalPredictiveCoding(level_dims=[32, 16])
        hpc(torch.randn(2, 32))
        hpc.reset()
        assert all(l.beliefs.abs().sum().item() == 0 for l in hpc.levels)
