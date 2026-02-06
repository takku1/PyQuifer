"""Tests for the Volatility Filter module (HGF-inspired adaptive learning)."""
import torch
import pytest
import statistics
from pyquifer.volatility_filter import (
    VolatilityNode,
    HierarchicalVolatilityFilter,
    VolatilityGatedLearning,
)


class TestVolatilityNode:
    @pytest.fixture
    def node(self):
        return VolatilityNode(dim=8, tonic_volatility=-4.0)

    def test_construction(self, node):
        assert node.dim == 8
        assert node.mean.shape == (8,)
        assert node.precision.shape == (8,)

    def test_output_keys(self, node):
        result = node(torch.randn(8))
        expected = {'value_pe', 'volatility_pe', 'mean', 'precision',
                    'effective_lr', 'effective_precision'}
        assert set(result.keys()) == expected

    def test_output_shapes(self, node):
        result = node(torch.randn(8))
        assert result['value_pe'].shape == (8,)
        assert result['volatility_pe'].shape == (8,)
        assert result['mean'].shape == (8,)
        assert result['precision'].shape == (8,)
        assert result['effective_lr'].shape == (8,)

    def test_mean_tracks_signal(self, node):
        """Mean should converge toward a constant signal."""
        target = torch.ones(8) * 3.0
        for _ in range(50):
            result = node(target + torch.randn(8) * 0.01)
        # Mean should be close to 3.0
        assert (result['mean'] - 3.0).abs().mean().item() < 0.5

    def test_precision_increases_for_stable_signal(self, node):
        """Precision should increase when signal is consistent."""
        initial_prec = node.precision.mean().item()
        for _ in range(30):
            result = node(torch.ones(8) * 2.0)
        final_prec = result['precision'].mean().item()
        assert final_prec > initial_prec

    def test_effective_lr_in_range(self, node):
        """Effective LR (sigmoid of vol PE) should be in (0, 1)."""
        for _ in range(10):
            result = node(torch.randn(8))
            lr = result['effective_lr']
            assert (lr > 0).all() and (lr < 1).all()

    def test_step_count_increments(self, node):
        assert node.step_count.item() == 0
        node(torch.randn(8))
        assert node.step_count.item() == 1
        node(torch.randn(8))
        assert node.step_count.item() == 2

    def test_predict_with_parent(self, node):
        """Prediction with volatility parent should not crash."""
        parent_mean = torch.zeros(8)
        node.predict(volatility_parent_mean=parent_mean, kappa=1.0)
        result = node.update(torch.randn(8))
        assert result['mean'].shape == (8,)

    def test_reset(self, node):
        for _ in range(10):
            node(torch.randn(8))
        node.reset()
        assert node.step_count.item() == 0
        assert node.mean.abs().sum().item() == 0.0


class TestHierarchicalVolatilityFilter:
    @pytest.fixture
    def hgf2(self):
        return HierarchicalVolatilityFilter(dim=8, num_levels=2)

    @pytest.fixture
    def hgf3(self):
        return HierarchicalVolatilityFilter(dim=8, num_levels=3)

    def test_construction_2level(self, hgf2):
        assert hgf2.num_levels == 2
        assert len(hgf2.levels) == 2

    def test_construction_3level(self, hgf3):
        assert hgf3.num_levels == 3
        assert len(hgf3.levels) == 3

    def test_rejects_invalid_levels(self):
        with pytest.raises(AssertionError):
            HierarchicalVolatilityFilter(dim=8, num_levels=1)
        with pytest.raises(AssertionError):
            HierarchicalVolatilityFilter(dim=8, num_levels=4)

    def test_output_keys(self, hgf2):
        result = hgf2(torch.randn(8))
        expected = {'effective_lr', 'value_pe', 'volatility_pe',
                    'mean_volatility', 'precision', 'level_results'}
        assert set(result.keys()) == expected

    def test_effective_lr_shape(self, hgf2):
        result = hgf2(torch.randn(8))
        assert result['effective_lr'].shape == (8,)

    def test_level_results_count(self, hgf3):
        result = hgf3(torch.randn(8))
        assert len(result['level_results']) == 3

    def test_lr_increases_for_volatile_signal(self, hgf2):
        """
        Learning rate should increase when signal becomes volatile.
        Stable -> volatile transition should push LR up.
        """
        torch.manual_seed(42)
        # Stable phase
        stable_lrs = []
        for _ in range(30):
            obs = torch.randn(8) * 0.1 + 1.0
            result = hgf2(obs)
            stable_lrs.append(result['effective_lr'].mean().item())

        # Volatile phase (big jumps)
        volatile_lrs = []
        for i in range(30):
            obs = torch.randn(8) * 3.0 + (5.0 if i % 2 == 0 else -5.0)
            result = hgf2(obs)
            volatile_lrs.append(result['effective_lr'].mean().item())

        stable_mean = statistics.mean(stable_lrs[-10:])
        volatile_mean = statistics.mean(volatile_lrs[-10:])

        # Volatile phase should have higher learning rate
        assert volatile_mean > stable_mean, \
            f"Volatile LR ({volatile_mean:.4f}) should exceed stable LR ({stable_mean:.4f})"

    def test_lr_decreases_after_stabilization(self, hgf2):
        """
        After a volatile period, returning to stable signal should
        reduce learning rate back down.
        """
        torch.manual_seed(42)
        # Volatile phase — track last 10 LRs
        volatile_lrs = []
        for i in range(30):
            obs = torch.randn(8) * 3.0 + (5.0 if i % 2 == 0 else -5.0)
            result = hgf2(obs)
            volatile_lrs.append(result['effective_lr'].mean().item())
        volatile_mean = statistics.mean(volatile_lrs[-10:])

        # Stable phase (longer to let filter settle) — track last 10 LRs
        stable_lrs = []
        for _ in range(50):
            obs = torch.randn(8) * 0.1 + 1.0
            result = hgf2(obs)
            stable_lrs.append(result['effective_lr'].mean().item())
        stable_mean = statistics.mean(stable_lrs[-10:])

        # After stabilization, LR should be lower
        assert stable_mean < volatile_mean, \
            f"Post-stable LR ({stable_mean:.4f}) should be below volatile ({volatile_mean:.4f})"

    def test_get_volatility_summary(self, hgf3):
        hgf3(torch.randn(8))
        summary = hgf3.get_volatility_summary()
        assert 'level_0_mean' in summary
        assert 'level_2_precision' in summary

    def test_reset(self, hgf2):
        for _ in range(10):
            hgf2(torch.randn(8))
        hgf2.reset()
        for level in hgf2.levels:
            assert level.step_count.item() == 0

    def test_no_nan_extended_run(self, hgf3):
        """100 steps with varied input should produce no NaN."""
        torch.manual_seed(42)
        for i in range(100):
            obs = torch.randn(8) * (0.1 + (i % 20) * 0.2)
            result = hgf3(obs)
            assert not result['effective_lr'].isnan().any(), f"NaN at step {i}"
            assert not result['precision'].isnan().any(), f"NaN precision at step {i}"


class TestVolatilityGatedLearning:
    @pytest.fixture
    def gate(self):
        return VolatilityGatedLearning(
            dim=16, base_lr=0.01, min_lr=0.001, max_lr=0.1
        )

    def test_construction(self, gate):
        assert gate.dim == 16
        assert gate.base_lr == 0.01

    def test_output_keys(self, gate):
        result = gate(torch.randn(16))
        expected = {'effective_lr', 'lr_multiplier', 'volatility_pe',
                    'mean_volatility', 'precision'}
        assert set(result.keys()) == expected

    def test_lr_in_bounds(self, gate):
        """Effective LR should stay within [min_lr, max_lr]."""
        for _ in range(20):
            result = gate(torch.randn(16) * 5.0)
            lr = result['effective_lr']
            assert (lr >= gate.min_lr - 1e-6).all(), f"LR below min: {lr.min()}"
            assert (lr <= gate.max_lr + 1e-6).all(), f"LR above max: {lr.max()}"

    def test_get_mean_lr(self, gate):
        gate(torch.randn(16))
        mean_lr = gate.get_mean_lr()
        assert 0.001 <= mean_lr <= 0.1

    def test_volatile_signal_increases_lr(self, gate):
        """Switching from stable to volatile should increase mean LR."""
        torch.manual_seed(42)
        # Stable
        for _ in range(30):
            gate(torch.randn(16) * 0.05 + 1.0)
        stable_lr = gate.get_mean_lr()

        # Volatile
        for i in range(30):
            gate(torch.randn(16) * 5.0 + (10.0 if i % 2 == 0 else -10.0))
        volatile_lr = gate.get_mean_lr()

        assert volatile_lr > stable_lr

    def test_reset(self, gate):
        for _ in range(10):
            gate(torch.randn(16))
        gate.reset()
        # After reset, filter state is zeroed
        assert gate.filter.levels[0].step_count.item() == 0

    def test_lazy_import(self):
        """Should be importable via top-level pyquifer package."""
        from pyquifer import VolatilityNode, HierarchicalVolatilityFilter, VolatilityGatedLearning
        assert VolatilityNode is not None
        assert HierarchicalVolatilityFilter is not None
        assert VolatilityGatedLearning is not None
