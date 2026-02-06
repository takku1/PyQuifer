"""Tests for precision-weighted oscillator coupling (Phase 4C)."""
import torch
import pytest
import math
from pyquifer.oscillators import LearnableKuramotoBank
from pyquifer.precision_weighting import PrecisionEstimator


class TestPrecisionBuffers:
    """Test that precision tracking infrastructure exists."""

    @pytest.fixture
    def bank(self):
        return LearnableKuramotoBank(num_oscillators=10, dt=0.01)

    def test_precision_buffer_exists(self, bank):
        assert hasattr(bank, 'precision')
        assert bank.precision.shape == (10,)

    def test_phase_velocity_var_buffer(self, bank):
        assert hasattr(bank, 'phase_velocity_var')
        assert bank.phase_velocity_var.shape == (10,)

    def test_prev_phases_buffer(self, bank):
        assert hasattr(bank, 'prev_phases')
        assert bank.prev_phases.shape == (10,)

    def test_initial_precision_is_ones(self, bank):
        """Initial precision should be 1.0 (before any steps)."""
        assert torch.allclose(bank.precision, torch.ones(10), atol=0.1)


class TestPrecisionUpdates:
    """Test that precision updates correctly from phase dynamics."""

    def test_precision_changes_after_steps(self):
        bank = LearnableKuramotoBank(num_oscillators=10, dt=0.01)
        initial_prec = bank.precision.clone()
        bank(steps=50)
        assert not torch.allclose(bank.precision, initial_prec)

    def test_stable_oscillators_get_high_precision(self):
        """Oscillators with low phase velocity variance -> high precision."""
        torch.manual_seed(42)
        # With strong coupling, oscillators sync -> low velocity variance -> high precision
        bank = LearnableKuramotoBank(num_oscillators=10, dt=0.01, topology='global')
        # Start synchronized (all same phase)
        bank.phases.fill_(0.0)
        bank.prev_phases.fill_(0.0)
        # Run many steps
        for _ in range(200):
            bank(steps=1)
        # Should have high precision since they're synchronized
        assert bank.precision.mean().item() > 1.0

    def test_precision_positive(self):
        """Precision should always be positive."""
        bank = LearnableKuramotoBank(num_oscillators=10, dt=0.01)
        for _ in range(20):
            bank(steps=5)
            assert (bank.precision > 0).all()

    def test_precision_bounded(self):
        """Precision should not exceed the cap (100.0)."""
        bank = LearnableKuramotoBank(num_oscillators=10, dt=0.01)
        for _ in range(50):
            bank(steps=5)
            assert (bank.precision <= 100.0 + 1e-6).all()


class TestPrecisionWeightedCoupling:
    """Test that precision weighting affects coupling dynamics."""

    def test_use_precision_flag(self):
        """Should work with precision on and off."""
        bank = LearnableKuramotoBank(num_oscillators=10, dt=0.01)
        phases_with = bank(steps=10, use_precision=True)
        assert phases_with.shape == (10,)

        bank2 = LearnableKuramotoBank(num_oscillators=10, dt=0.01)
        phases_without = bank2(steps=10, use_precision=False)
        assert phases_without.shape == (10,)

    def test_precision_affects_dynamics(self):
        """With vs without precision should produce different phase trajectories."""
        torch.manual_seed(42)
        bank1 = LearnableKuramotoBank(num_oscillators=10, dt=0.01)
        bank2 = LearnableKuramotoBank(num_oscillators=10, dt=0.01)
        # Same initial conditions
        bank2.phases.copy_(bank1.phases)
        bank2.prev_phases.copy_(bank1.prev_phases)

        # Run with different precision weightings
        # Manually set precision to be non-uniform
        with torch.no_grad():
            bank1.precision.copy_(torch.tensor([10., 10., 10., 10., 10.,
                                                 0.1, 0.1, 0.1, 0.1, 0.1]))
        phases1 = bank1(steps=50, use_precision=True)
        phases2 = bank2(steps=50, use_precision=False)

        # Should produce different results
        assert not torch.allclose(phases1, phases2, atol=0.01)

    def test_works_with_ring_topology(self):
        """Precision-weighted coupling should work with non-global topologies."""
        bank = LearnableKuramotoBank(
            num_oscillators=10, dt=0.01,
            topology='ring', topology_params={'k': 2}
        )
        phases = bank(steps=20, use_precision=True)
        assert phases.shape == (10,)
        assert not phases.isnan().any()

    def test_works_with_small_world(self):
        bank = LearnableKuramotoBank(
            num_oscillators=10, dt=0.01,
            topology='small_world', topology_params={'k': 3, 'p': 0.2}
        )
        phases = bank(steps=20, use_precision=True)
        assert not phases.isnan().any()

    def test_works_with_external_input(self):
        bank = LearnableKuramotoBank(num_oscillators=10, dt=0.01)
        ext = torch.randn(10) * 0.1
        phases = bank(external_input=ext, steps=10, use_precision=True)
        assert phases.shape == (10,)
        assert not phases.isnan().any()

    def test_no_nan_extended_run(self):
        """100 steps with precision coupling should produce no NaN."""
        bank = LearnableKuramotoBank(num_oscillators=20, dt=0.01)
        for _ in range(100):
            phases = bank(steps=1, use_precision=True)
            assert not phases.isnan().any()
            assert not bank.precision.isnan().any()


class TestPrecisionEstimatorOscillator:
    """Test the oscillator precision mapping in PrecisionEstimator."""

    @pytest.fixture
    def estimator(self):
        return PrecisionEstimator(num_channels=8)

    def test_get_oscillator_precision_same_dim(self, estimator):
        # Feed some errors
        for _ in range(10):
            estimator(torch.randn(8))
        prec = estimator.get_oscillator_precision(8)
        assert prec.shape == (8,)
        assert (prec > 0).all()

    def test_get_oscillator_precision_different_dim(self, estimator):
        for _ in range(10):
            estimator(torch.randn(8))
        prec = estimator.get_oscillator_precision(16)
        assert prec.shape == (16,)
        assert (prec > 0).all()

    def test_get_oscillator_precision_smaller_dim(self, estimator):
        for _ in range(10):
            estimator(torch.randn(8))
        prec = estimator.get_oscillator_precision(4)
        assert prec.shape == (4,)
        assert (prec > 0).all()

    def test_high_variance_low_precision(self, estimator):
        """Channels with high error variance should have low precision."""
        for _ in range(50):
            errors = torch.randn(8)
            errors[:4] *= 0.1  # Low variance
            errors[4:] *= 5.0  # High variance
            estimator(errors)
        prec = estimator.get_oscillator_precision(8)
        assert prec[:4].mean() > prec[4:].mean()


class TestOrderParameterWithPrecision:
    """Test that order parameter still works correctly."""

    def test_order_parameter_unchanged(self):
        """get_order_parameter should not be affected by precision addition."""
        bank = LearnableKuramotoBank(num_oscillators=10, dt=0.01)
        bank.phases.fill_(0.0)  # All synchronized
        r = bank.get_order_parameter()
        assert abs(r.item() - 1.0) < 0.01

    def test_local_order_parameter_unchanged(self):
        bank = LearnableKuramotoBank(
            num_oscillators=10, dt=0.01,
            topology='ring', topology_params={'k': 2}
        )
        bank(steps=10)
        local_r = bank.get_local_order_parameters()
        assert local_r.shape == (10,)
        assert not local_r.isnan().any()
