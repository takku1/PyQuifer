"""Tests for stochastic_resonance module."""
import torch
import pytest
from pyquifer.stochastic_resonance import AdaptiveStochasticResonance, ResonanceMonitor


class TestAdaptiveStochasticResonance:
    def test_construction(self):
        asr = AdaptiveStochasticResonance(dim=16)
        assert abs(asr.noise_level.item() - 0.3) < 1e-6

    def test_output_keys(self):
        asr = AdaptiveStochasticResonance(dim=8)
        r = asr(torch.randn(8))
        assert all(k in r for k in ['enhanced', 'detected', 'noise_level', 'snr'])

    def test_enhanced_shape(self):
        asr = AdaptiveStochasticResonance(dim=32)
        r = asr(torch.ones(32) * 0.4)
        assert r['enhanced'].shape == (32,)

    def test_noise_adapts(self):
        torch.manual_seed(42)
        asr = AdaptiveStochasticResonance(dim=16, threshold=0.5, initial_noise=0.1)
        initial = asr.noise_level.item()
        for _ in range(200):
            asr(torch.ones(16) * 0.4)
        assert abs(asr.noise_level.item() - initial) > 0.001

    def test_criticality_distance_affects_noise(self):
        asr1 = AdaptiveStochasticResonance(dim=8, threshold=0.3)
        for _ in range(50):
            asr1(torch.randn(8) * 0.2, criticality_distance=0.1)
        near = asr1.noise_level.item()

        asr2 = AdaptiveStochasticResonance(dim=8, threshold=0.3)
        for _ in range(50):
            asr2(torch.randn(8) * 0.2, criticality_distance=2.0)
        far = asr2.noise_level.item()
        assert abs(near - far) > 0.01

    def test_reset(self):
        asr = AdaptiveStochasticResonance(dim=8, initial_noise=0.3)
        for _ in range(20):
            asr(torch.randn(8))
        asr.reset()
        assert abs(asr.noise_level.item() - 0.3) < 0.01


class TestResonanceMonitor:
    def test_output_keys(self):
        rm = ResonanceMonitor(window_size=20)
        r = rm(torch.tensor(0.3), torch.tensor(1.5))
        assert all(k in r for k in ['noise_mean', 'noise_std', 'snr_mean', 'regime_shift'])

    def test_reset(self):
        rm = ResonanceMonitor()
        for _ in range(10):
            rm(torch.tensor(0.3), torch.tensor(1.0))
        rm.reset()
        assert rm.history_ptr.item() == 0
