"""Tests for memory_consolidation module."""
import torch
import pytest
from pyquifer.memory_consolidation import (
    EpisodicBuffer, SharpWaveRipple, ConsolidationEngine, MemoryReconsolidation
)


class TestEpisodicBuffer:
    def test_store_and_count(self):
        buf = EpisodicBuffer(state_dim=16, capacity=100)
        for i in range(50):
            buf.store(torch.randn(16), reward=float(i))
        assert buf.num_stored.item() == 50

    def test_capacity_overflow(self):
        buf = EpisodicBuffer(state_dim=8, capacity=20)
        for i in range(50):
            buf.store(torch.randn(8), reward=float(i))
        assert buf.num_stored.item() == 20

    def test_priority_protection(self):
        buf = EpisodicBuffer(state_dim=8, capacity=20, protect_top_k=3)
        for i in range(50):
            buf.store(torch.randn(8), reward=float(i))
        assert buf.rewards.max().item() >= 47.0

    def test_prioritized_sampling(self):
        buf = EpisodicBuffer(state_dim=16, capacity=50)
        for i in range(30):
            buf.store(torch.randn(16), reward=float(i))
        sample = buf.sample_prioritized(10)
        assert sample['states'].shape == (10, 16)
        assert sample['rewards'].shape == (10,)

    def test_empty_buffer_sampling(self):
        buf = EpisodicBuffer(state_dim=8, capacity=10)
        sample = buf.sample_prioritized(5)
        assert sample['states'].shape[0] == 0

    def test_reset(self):
        buf = EpisodicBuffer(state_dim=8, capacity=20)
        buf.store(torch.randn(8), reward=1.0)
        buf.reset()
        assert buf.num_stored.item() == 0


class TestSharpWaveRipple:
    def test_replay_during_sleep(self):
        buf = EpisodicBuffer(state_dim=16, capacity=50)
        for i in range(20):
            buf.store(torch.randn(16), reward=float(i))
        swr = SharpWaveRipple(state_dim=16)
        r = swr(buf, sleep_signal=1.0)
        assert r['replay_active'].item() == True
        assert r['replayed_states'].shape[0] > 0

    def test_no_replay_when_awake(self):
        buf = EpisodicBuffer(state_dim=16, capacity=50)
        for i in range(20):
            buf.store(torch.randn(16), reward=float(i))
        swr = SharpWaveRipple(state_dim=16)
        r = swr(buf, sleep_signal=0.1)
        assert r['replay_active'].item() == False

    def test_empty_buffer_no_replay(self):
        buf = EpisodicBuffer(state_dim=8, capacity=10)
        swr = SharpWaveRipple(state_dim=8)
        r = swr(buf, sleep_signal=1.0)
        assert r['replay_active'].item() == False


class TestConsolidationEngine:
    def test_consolidation_creates_traces(self):
        engine = ConsolidationEngine(state_dim=16, semantic_dim=8)
        states = torch.randn(5, 16)
        rewards = torch.randn(5)
        r = engine(states, rewards)
        assert engine.num_traces.item() > 0

    def test_query(self):
        engine = ConsolidationEngine(state_dim=16, semantic_dim=8)
        engine(torch.randn(10, 16), torch.randn(10))
        qr = engine.query(torch.randn(16), top_k=3)
        assert qr['traces'].shape[0] > 0

    def test_empty_query(self):
        engine = ConsolidationEngine(state_dim=8, semantic_dim=4)
        qr = engine.query(torch.randn(8))
        assert qr['traces'].shape[0] == 0

    def test_reset(self):
        engine = ConsolidationEngine(state_dim=8, semantic_dim=4)
        engine(torch.randn(5, 8), torch.randn(5))
        engine.reset()
        assert engine.num_traces.item() == 0


class TestMemoryReconsolidation:
    def test_emotional_more_labile(self):
        recon = MemoryReconsolidation(dim=16, base_lability=0.1, max_lability=0.5)
        mem = torch.randn(16)
        ctx = torch.randn(16)
        calm = recon(mem, ctx, emotional_intensity=0.0)
        emotional = recon(mem, ctx, emotional_intensity=0.9)
        assert emotional['change_magnitude'] > calm['change_magnitude']

    def test_batch_shape(self):
        recon = MemoryReconsolidation(dim=8)
        r = recon(torch.randn(4, 8), torch.randn(4, 8))
        assert r['reconsolidated'].shape == (4, 8)
