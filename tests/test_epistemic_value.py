"""Tests for EpistemicValue (information gain as epistemic drive) in motivation.py."""
import torch
import pytest
from pyquifer.motivation import (
    EpistemicValue,
    IntrinsicMotivationSystem,
)


class TestEpistemicValue:
    @pytest.fixture
    def ev(self):
        return EpistemicValue(dim=8, num_bins=16)

    def test_construction(self, ev):
        assert ev.dim == 8
        assert ev.num_bins == 16
        assert ev.belief_counts.shape == (8, 16)

    def test_output_keys(self, ev):
        result = ev(torch.randn(8))
        expected = {'info_gain', 'mean_info_gain', 'prior_entropy',
                    'posterior_entropy', 'epistemic_value'}
        assert set(result.keys()) == expected

    def test_info_gain_shape(self, ev):
        result = ev(torch.randn(8))
        assert result['info_gain'].shape == (8,)

    def test_mean_info_gain_scalar(self, ev):
        result = ev(torch.randn(8))
        assert result['mean_info_gain'].dim() == 0

    def test_epistemic_value_in_range(self, ev):
        """Epistemic value (sigmoid output) should be in (0, 1)."""
        for _ in range(20):
            result = ev(torch.randn(8))
            val = result['epistemic_value'].item()
            assert 0.0 <= val <= 1.0

    def test_info_gain_nonnegative(self, ev):
        """Information gain should always be >= 0."""
        for _ in range(20):
            result = ev(torch.randn(8))
            assert (result['info_gain'] >= -1e-6).all()

    def test_novel_input_high_info_gain(self, ev):
        """First observation after reset should have high info gain."""
        ev.reset()
        # Feed many similar inputs to build a strong prior
        for _ in range(50):
            ev(torch.ones(8) * 0.5)

        # Novel input should have higher info gain than familiar one
        familiar_result = ev(torch.ones(8) * 0.5)
        ev_familiar = familiar_result['mean_info_gain'].item()

        novel_result = ev(torch.ones(8) * 5.0)
        ev_novel = novel_result['mean_info_gain'].item()

        assert ev_novel >= ev_familiar

    def test_entropy_decreases_with_observations(self, ev):
        """Entropy should decrease as we observe more consistent data."""
        initial_entropy = ev.get_current_entropy().item()
        # Feed consistent signal
        for _ in range(100):
            ev(torch.ones(8) * 1.0)
        final_entropy = ev.get_current_entropy().item()
        # Entropy should decrease (beliefs become more concentrated)
        assert final_entropy < initial_entropy

    def test_batch_input(self, ev):
        """Should handle batch input by averaging."""
        batch = torch.randn(4, 8)
        result = ev(batch)
        assert result['mean_info_gain'].dim() == 0

    def test_step_count_increments(self, ev):
        assert ev.step_count.item() == 0
        ev(torch.randn(8))
        assert ev.step_count.item() == 1

    def test_reset(self, ev):
        for _ in range(10):
            ev(torch.randn(8))
        ev.reset()
        assert ev.step_count.item() == 0
        # Belief counts reset to uniform
        assert torch.allclose(ev.belief_counts, torch.ones(8, 16))


class TestIntrinsicMotivationWithEpistemic:
    @pytest.fixture
    def ims(self):
        return IntrinsicMotivationSystem(state_dim=8, performance_dim=1)

    def test_has_epistemic_value(self, ims):
        assert hasattr(ims, 'epistemic_value')
        assert isinstance(ims.epistemic_value, EpistemicValue)

    def test_forward_includes_epistemic(self, ims):
        result = ims(torch.randn(8))
        assert 'epistemic_value' in result
        assert 'info_gain' in result

    def test_epistemic_affects_motivation(self, ims):
        """Epistemic value should contribute to combined motivation."""
        result = ims(torch.randn(8))
        motivation = result['motivation'].item()
        # Should be at least baseline_arousal
        assert motivation >= 0.1

    def test_exploration_drive_includes_epistemic(self, ims):
        result = ims(torch.randn(8))
        explore = ims.get_exploration_drive(result)
        assert 0.0 <= explore.item() <= 1.0

    def test_reset_clears_epistemic(self, ims):
        for _ in range(10):
            ims(torch.randn(8))
        ims.reset()
        assert ims.epistemic_value.step_count.item() == 0

    def test_lazy_import(self):
        from pyquifer import EpistemicValue
        assert EpistemicValue is not None
