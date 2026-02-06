"""Tests for precision_weighting module."""
import torch
import pytest
from pyquifer.precision_weighting import PrecisionEstimator, PrecisionGate, AttentionAsPrecision


class TestPrecisionEstimator:
    def test_construction(self):
        pe = PrecisionEstimator(num_channels=8)
        assert pe.running_var.shape == (8,)

    def test_output_keys(self):
        pe = PrecisionEstimator(num_channels=4)
        r = pe(torch.randn(2, 4))
        assert all(k in r for k in ['precision', 'log_precision', 'running_var'])

    def test_precision_shape(self):
        pe = PrecisionEstimator(num_channels=8)
        r = pe(torch.randn(4, 8))
        assert r['precision'].shape == (8,)

    def test_low_variance_gets_high_precision(self):
        pe = PrecisionEstimator(num_channels=8, tau=10.0)
        for _ in range(200):
            errs = torch.randn(4, 8)
            errs[:, :4] *= 0.1
            errs[:, 4:] *= 3.0
            r = pe(errs)
        assert r['precision'][:4].mean() > r['precision'][4:].mean() * 2

    def test_neuromodulator_boost(self):
        pe = PrecisionEstimator(num_channels=4, tau=5.0)
        for _ in range(50):
            pe(torch.randn(2, 4))
        base = pe(torch.randn(2, 4))['precision'].mean()
        boosted = pe(torch.randn(2, 4), acetylcholine=1.0, norepinephrine=1.0)['precision'].mean()
        assert boosted > base

    def test_reset(self):
        pe = PrecisionEstimator(num_channels=4)
        for _ in range(20):
            pe(torch.randn(2, 4))
        pe.reset()
        assert pe.running_var.mean().item() == 1.0
        assert pe.step_count.item() == 0


class TestPrecisionGate:
    def test_output_keys(self):
        pg = PrecisionGate(num_channels=4, base_lr=0.01)
        r = pg(torch.ones(4), torch.tensor([0.1, 1.0, 5.0, 10.0]))
        assert 'weighted_signal' in r and 'effective_lr' in r

    def test_gating(self):
        pg = PrecisionGate(num_channels=4, base_lr=0.01)
        signal = torch.ones(4)
        precision = torch.tensor([0.0, 1.0, 5.0, 10.0])
        r = pg(signal, precision)
        assert r['weighted_signal'][0].item() == 0.0
        assert r['weighted_signal'][3].item() == 10.0


class TestAttentionAsPrecision:
    def test_attention_sums_to_one(self):
        aap = AttentionAsPrecision(num_channels=4)
        for _ in range(50):
            aap(torch.randn(2, 4), torch.randn(2, 4))
        r = aap(torch.randn(2, 4), torch.randn(2, 4))
        assert abs(r['attention_map'].sum().item() - 1.0) < 0.01

    def test_full_pipeline(self):
        aap = AttentionAsPrecision(num_channels=8, base_lr=0.01)
        signal = torch.randn(2, 8)
        errors = torch.randn(2, 8)
        r = aap(signal, errors, acetylcholine=0.5, norepinephrine=0.3)
        assert r['weighted_signal'].shape == (2, 8)
        assert r['effective_lr'].shape == (8,)
