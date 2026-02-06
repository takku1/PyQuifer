"""Tests for the CognitiveCycle integration module."""
import torch
import pytest
from pyquifer.integration import CycleConfig, CognitiveCycle


class TestCycleConfig:
    def test_default_factory(self):
        c = CycleConfig.default()
        assert c.state_dim == 64
        assert c.hierarchy_dims == [64, 32, 16]

    def test_small_factory(self):
        c = CycleConfig.small()
        assert c.state_dim == 32
        assert c.num_oscillators == 16

    def test_custom_config(self):
        c = CycleConfig(state_dim=128, num_oscillators=64)
        assert c.state_dim == 128
        assert c.num_oscillators == 64


class TestCognitiveCycle:
    @pytest.fixture
    def cycle(self):
        return CognitiveCycle(CycleConfig.small())

    def test_construction(self, cycle):
        assert cycle.tick_count.item() == 0
        assert hasattr(cycle, 'oscillators')
        assert hasattr(cycle, 'hpc')
        assert hasattr(cycle, 'arena')

    def test_tick_output_keys(self, cycle):
        sensory = torch.randn(cycle.config.state_dim)
        result = cycle.tick(sensory)
        assert 'modulation' in result
        assert 'consciousness' in result
        assert 'self_state' in result
        assert 'learning' in result
        assert 'diagnostics' in result

    def test_modulation_keys(self, cycle):
        sensory = torch.randn(cycle.config.state_dim)
        result = cycle.tick(sensory)
        m = result['modulation']
        assert 'temperature' in m
        assert 'personality_blend' in m
        assert 'attention_bias' in m
        assert 'processing_mode' in m
        assert 'coherence' in m

    def test_temperature_range(self, cycle):
        sensory = torch.randn(cycle.config.state_dim)
        result = cycle.tick(sensory)
        temp = result['modulation']['temperature']
        assert 0.1 <= temp <= 2.0

    def test_tick_increments(self, cycle):
        sensory = torch.randn(cycle.config.state_dim)
        assert cycle.tick_count.item() == 0
        cycle.tick(sensory)
        assert cycle.tick_count.item() == 1
        cycle.tick(sensory)
        assert cycle.tick_count.item() == 2

    def test_multiple_ticks(self, cycle):
        """Run multiple ticks without errors."""
        torch.manual_seed(42)
        for i in range(10):
            sensory = torch.randn(cycle.config.state_dim) * 0.5
            result = cycle.tick(sensory, reward=0.1 * i)
        assert cycle.tick_count.item() == 10

    def test_reward_affects_memory(self, cycle):
        sensory = torch.randn(cycle.config.state_dim)
        for _ in range(5):
            cycle.tick(sensory, reward=1.0)
        assert cycle.episodic_buffer.num_stored.item() == 5

    def test_sleep_triggers_consolidation(self, cycle):
        """Sleep signal should trigger memory consolidation."""
        torch.manual_seed(42)
        sensory = torch.randn(cycle.config.state_dim)
        # Build up memories first (awake)
        for _ in range(20):
            cycle.tick(sensory, reward=1.0, sleep_signal=0.0)
        # Now sleep
        result = cycle.tick(sensory, reward=0.0, sleep_signal=1.0)
        # Consolidation info should be present
        assert 'consolidation' in result['learning']

    def test_reset(self, cycle):
        sensory = torch.randn(cycle.config.state_dim)
        for _ in range(5):
            cycle.tick(sensory)
        cycle.reset()
        assert cycle.tick_count.item() == 0
        assert cycle.episodic_buffer.num_stored.item() == 0

    def test_get_state_summary(self, cycle):
        sensory = torch.randn(cycle.config.state_dim)
        cycle.tick(sensory)
        summary = cycle.get_state_summary()
        assert 'tick' in summary
        assert 'coherence' in summary
        assert 'num_memories' in summary
        assert summary['tick'] == 1

    def test_consciousness_metrics(self, cycle):
        sensory = torch.randn(cycle.config.state_dim)
        result = cycle.tick(sensory)
        c = result['consciousness']
        assert 'free_energy' in c
        assert 'coherence' in c
        assert 'criticality_distance' in c
        assert 'branching_ratio' in c
        assert 'sr_noise_level' in c

    def test_self_state_metrics(self, cycle):
        sensory = torch.randn(cycle.config.state_dim)
        result = cycle.tick(sensory)
        s = result['self_state']
        assert 'identity_strength' in s
        assert 'self_prediction_error' in s
        assert 'sensory_flow' in s

    def test_diagnostics_contain_tensors(self, cycle):
        sensory = torch.randn(cycle.config.state_dim)
        result = cycle.tick(sensory)
        d = result['diagnostics']
        assert isinstance(d['phases'], torch.Tensor)
        assert isinstance(d['precision'], torch.Tensor)
        assert isinstance(d['resources'], torch.Tensor)

    def test_lazy_import(self):
        """Verify CognitiveCycle can be imported from pyquifer."""
        from pyquifer import CognitiveCycle, CycleConfig
        c = CycleConfig.small()
        assert c.state_dim == 32

    def test_personality_blend_structure(self, cycle):
        sensory = torch.randn(cycle.config.state_dim)
        result = cycle.tick(sensory)
        blend = result['modulation']['personality_blend']
        assert 'facet_weights' in blend
        assert 'stability' in blend
        assert 'dominant_facet' in blend
        assert len(blend['facet_weights']) == cycle.config.num_populations
        assert abs(sum(blend['facet_weights']) - 1.0) < 1e-5
