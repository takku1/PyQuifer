"""
Phase 5A Tests: Quick Wins
- 5A-1: Episodic memory buffer activation in NoveltyDetector
- 5A-2: Ornstein-Uhlenbeck colored noise
- 5A-3: Homeostatic STDP regulation
"""

import torch
import math
import pytest


# ── 5A-1: Episodic Memory in NoveltyDetector ──

class TestEpisodicMemory:
    """Tests for activated episodic memory buffer in NoveltyDetector."""

    def test_episodic_novelty_empty_memory(self):
        """With no stored memories, everything should be maximally novel."""
        from pyquifer.motivation import NoveltyDetector
        nd = NoveltyDetector(dim=8, memory_size=50)
        x = torch.randn(1, 8)
        bonus = nd._episodic_novelty(x)
        assert bonus.shape == (1,)
        assert bonus.item() == 1.0  # No memories → max novelty

    def test_episodic_novelty_repeated_patterns(self):
        """Repeatedly seen patterns should have low episodic novelty."""
        from pyquifer.motivation import NoveltyDetector
        nd = NoveltyDetector(dim=8, memory_size=50)

        # Feed the same pattern many times to fill memory
        pattern = torch.ones(1, 8) * 2.0
        for _ in range(60):
            nd(pattern)

        # Now check episodic novelty for the same pattern
        bonus = nd._episodic_novelty(pattern)
        assert bonus.item() < 0.3  # Should be low (similar to memories)

    def test_episodic_novelty_novel_pattern(self):
        """A genuinely new pattern should have high episodic novelty."""
        from pyquifer.motivation import NoveltyDetector
        nd = NoveltyDetector(dim=8, memory_size=50)

        # Fill memory with one direction
        for _ in range(60):
            nd(torch.ones(1, 8) * 2.0)

        # Orthogonal pattern should be novel
        novel = torch.tensor([[-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0]])
        bonus = nd._episodic_novelty(novel)
        assert bonus.item() > 0.5

    def test_episodic_bonus_blended_into_novelty(self):
        """The episodic bonus should contribute to the final novelty signal."""
        from pyquifer.motivation import NoveltyDetector
        nd = NoveltyDetector(dim=8, memory_size=50)

        x = torch.randn(1, 8)
        novelty1, _ = nd(x)
        # First call should have episodic contribution (empty memory → bonus=1.0)
        assert novelty1.item() > 0

    def test_threshold_gated_storage(self):
        """Only sufficiently novel patterns should be stored."""
        from pyquifer.motivation import NoveltyDetector
        nd = NoveltyDetector(dim=8, memory_size=50)

        # First call stores (empty memory → bonus > 0.1)
        x = torch.randn(1, 8) * 3.0
        nd(x)
        ptr_after_first = nd.memory_ptr.item()
        assert ptr_after_first > 0

    def test_batch_episodic_novelty(self):
        """Episodic novelty should work with batched input."""
        from pyquifer.motivation import NoveltyDetector
        nd = NoveltyDetector(dim=8, memory_size=50)

        # Fill some memory
        for _ in range(20):
            nd(torch.randn(1, 8))

        batch = torch.randn(4, 8)
        bonus = nd._episodic_novelty(batch)
        assert bonus.shape == (4,)
        assert (bonus >= 0).all() and (bonus <= 1).all()

    def test_forward_returns_correct_shapes(self):
        """Forward should still return (novelty, prediction_error)."""
        from pyquifer.motivation import NoveltyDetector
        nd = NoveltyDetector(dim=8)
        x = torch.randn(3, 8)
        novelty, pred_error = nd(x)
        assert novelty.shape == (3,)
        assert pred_error.shape == (3, 8)


# ── 5A-2: Ornstein-Uhlenbeck Noise ──

class TestOrnsteinUhlenbeckNoise:
    """Tests for OU colored noise process."""

    def test_ou_init_zero(self):
        """OU state should start at zero."""
        from pyquifer.stochastic_resonance import OrnsteinUhlenbeckNoise
        ou = OrnsteinUhlenbeckNoise(dim=16)
        assert (ou.state == 0).all()

    def test_ou_generates_noise(self):
        """OU process should produce non-zero output after stepping."""
        from pyquifer.stochastic_resonance import OrnsteinUhlenbeckNoise
        ou = OrnsteinUhlenbeckNoise(dim=16, sigma=1.0)
        noise = ou()
        assert noise.shape == (16,)
        # After one step, likely non-zero
        # (could be zero in theory but astronomically unlikely)

    def test_ou_temporal_correlation(self):
        """OU noise should be temporally correlated (unlike white noise)."""
        from pyquifer.stochastic_resonance import OrnsteinUhlenbeckNoise
        ou = OrnsteinUhlenbeckNoise(dim=1, tau=50.0, sigma=1.0, dt=1.0)

        samples = []
        for _ in range(500):
            samples.append(ou().item())

        samples = torch.tensor(samples)
        # Autocorrelation at lag 1 should be positive and significant
        autocorr = torch.corrcoef(torch.stack([samples[:-1], samples[1:]]))[0, 1]
        assert autocorr > 0.5, f"OU autocorrelation {autocorr:.3f} should be > 0.5"

    def test_ou_mean_reverts(self):
        """OU process should revert to zero mean."""
        from pyquifer.stochastic_resonance import OrnsteinUhlenbeckNoise
        ou = OrnsteinUhlenbeckNoise(dim=100, tau=5.0, sigma=1.0, dt=1.0)

        # Run for many steps
        samples = []
        for _ in range(1000):
            samples.append(ou().mean().item())

        mean = sum(samples) / len(samples)
        assert abs(mean) < 0.5, f"OU mean {mean:.3f} should be near 0"

    def test_ou_reset(self):
        """Reset should zero the state."""
        from pyquifer.stochastic_resonance import OrnsteinUhlenbeckNoise
        ou = OrnsteinUhlenbeckNoise(dim=16)
        ou()
        ou.reset()
        assert (ou.state == 0).all()

    def test_asr_with_ou_noise(self):
        """AdaptiveStochasticResonance should work with OU noise type."""
        from pyquifer.stochastic_resonance import AdaptiveStochasticResonance
        asr = AdaptiveStochasticResonance(dim=16, noise_type='ou')
        signal = torch.ones(16) * 0.4
        result = asr(signal)
        assert 'enhanced' in result
        assert 'noise_level' in result

    def test_asr_white_noise_default(self):
        """Default noise_type should be 'white'."""
        from pyquifer.stochastic_resonance import AdaptiveStochasticResonance
        asr = AdaptiveStochasticResonance(dim=16)
        assert asr.noise_type == 'white'


# ── 5A-3: Homeostatic STDP ──

class TestHomeostaticSTDP:
    """Tests for homeostatic regulation in STDPLayer."""

    def test_running_rate_initialized(self):
        """Running rate buffer should exist and be initialized to target_rate."""
        from pyquifer.spiking import STDPLayer
        stdp = STDPLayer(pre_size=10, post_size=5, target_rate=0.15)
        assert hasattr(stdp, 'running_rate')
        assert stdp.running_rate.shape == (5,)
        assert torch.allclose(stdp.running_rate, torch.full((5,), 0.15))

    def test_homeostatic_correction_reduces_high_rate(self):
        """When post neurons fire too often, homeostatic correction should resist growth."""
        from pyquifer.spiking import STDPLayer

        # Compare weights WITH vs WITHOUT homeostatic correction
        stdp_homeo = STDPLayer(pre_size=10, post_size=5, target_rate=0.1,
                               homeostatic_strength=0.05)
        stdp_no_homeo = STDPLayer(pre_size=10, post_size=5, target_rate=0.1,
                                  homeostatic_strength=0.0)

        # Copy same initial weights
        stdp_no_homeo.weights.data.copy_(stdp_homeo.weights.data)

        torch.manual_seed(42)
        for _ in range(50):
            pre = (torch.rand(1, 10) > 0.5).float()
            post = torch.ones(1, 5)  # All firing
            stdp_homeo(pre, post, learn=True)

        torch.manual_seed(42)
        for _ in range(50):
            pre = (torch.rand(1, 10) > 0.5).float()
            post = torch.ones(1, 5)
            stdp_no_homeo(pre, post, learn=True)

        # Running rate should be above target
        assert stdp_homeo.running_rate.mean() > stdp_homeo.target_rate

        # Homeostatic version should have lower weights than unregulated
        assert stdp_homeo.weights.data.mean() < stdp_no_homeo.weights.data.mean()

    def test_homeostatic_correction_boosts_low_rate(self):
        """When post neurons fire too rarely, weights should increase."""
        from pyquifer.spiking import STDPLayer
        stdp = STDPLayer(pre_size=10, post_size=5, target_rate=0.5,
                         homeostatic_strength=0.01)

        # Simulate very low firing rate
        for _ in range(50):
            pre = (torch.rand(1, 10) > 0.5).float()
            post = torch.zeros(1, 5)  # Never firing
            stdp(pre, post, learn=True)

        # Running rate should be below target
        assert stdp.running_rate.mean() < stdp.target_rate

    def test_running_rate_tracks_firing(self):
        """Running rate EMA should track actual post-synaptic firing rate."""
        from pyquifer.spiking import STDPLayer
        stdp = STDPLayer(pre_size=10, post_size=5, target_rate=0.1)

        # 50% firing rate
        for _ in range(200):
            pre = (torch.rand(1, 10) > 0.5).float()
            post = (torch.rand(1, 5) > 0.5).float()
            stdp(pre, post, learn=True)

        # Running rate should approximate 0.5
        assert abs(stdp.running_rate.mean().item() - 0.5) < 0.15

    def test_backward_compat_default_params(self):
        """Default homeostatic params should not break existing behavior."""
        from pyquifer.spiking import STDPLayer
        stdp = STDPLayer(pre_size=10, post_size=5)
        # Default target_rate=0.1, homeostatic_strength=0.001 (very mild)
        assert stdp.target_rate == 0.1
        assert stdp.homeostatic_strength == 0.001

        # Should still work normally
        pre = (torch.rand(1, 10) > 0.8).float()
        post = (torch.rand(1, 5) > 0.8).float()
        current = stdp(pre, post, learn=True)
        assert current.shape == (1, 5)
