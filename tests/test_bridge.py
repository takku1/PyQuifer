"""Tests for the PyQuiferBridge LLM modulation API."""
import torch
import pytest
from pyquifer.bridge import PyQuiferBridge, ModulationState


class TestModulationState:
    def test_defaults(self):
        s = ModulationState()
        assert s.temperature == 1.0
        assert s.top_p == 0.9
        assert s.coherence == 0.5

    def test_custom(self):
        s = ModulationState(temperature=0.7, coherence=0.9)
        assert s.temperature == 0.7
        assert s.coherence == 0.9


class TestPyQuiferBridge:
    @pytest.fixture
    def bridge(self):
        return PyQuiferBridge.small()

    def test_construction(self, bridge):
        assert hasattr(bridge, 'cycle')
        assert hasattr(bridge, 'trait_vectors')

    def test_step_returns_modulation_state(self, bridge):
        sensory = torch.randn(bridge.config.state_dim)
        state = bridge.step(sensory)
        assert isinstance(state, ModulationState)

    def test_temperature_in_range(self, bridge):
        sensory = torch.randn(bridge.config.state_dim)
        state = bridge.step(sensory)
        assert 0.1 <= state.temperature <= 2.0

    def test_top_p_in_range(self, bridge):
        sensory = torch.randn(bridge.config.state_dim)
        state = bridge.step(sensory)
        assert 0.5 <= state.top_p <= 1.0

    def test_repetition_penalty_in_range(self, bridge):
        sensory = torch.randn(bridge.config.state_dim)
        state = bridge.step(sensory)
        assert 1.0 <= state.repetition_penalty <= 1.5

    def test_phases_returned(self, bridge):
        sensory = torch.randn(bridge.config.state_dim)
        state = bridge.step(sensory)
        assert state.phases is not None
        assert state.phases.shape == (bridge.config.num_oscillators,)

    def test_neuromodulator_levels_returned(self, bridge):
        sensory = torch.randn(bridge.config.state_dim)
        state = bridge.step(sensory)
        assert state.neuromodulator_levels is not None
        assert state.neuromodulator_levels.shape == (5,)  # DA, 5HT, NE, ACh, Cortisol

    def test_tick_increments(self, bridge):
        sensory = torch.randn(bridge.config.state_dim)
        s1 = bridge.step(sensory)
        s2 = bridge.step(sensory)
        assert s2.tick == s1.tick + 1

    def test_input_projection(self, bridge):
        """Input with wrong dimension should be auto-projected."""
        wrong_dim = torch.randn(128)  # Not state_dim
        state = bridge.step(wrong_dim)
        assert isinstance(state, ModulationState)

    def test_modulate_logits(self, bridge):
        sensory = torch.randn(bridge.config.state_dim)
        state = bridge.step(sensory)
        logits = torch.randn(1, 1000)
        modified = bridge.modulate_logits(logits, state)
        assert modified.shape == logits.shape

    def test_modulate_hidden(self, bridge):
        sensory = torch.randn(bridge.config.state_dim)
        state = bridge.step(sensory)
        hidden = torch.randn(1, 10, bridge.config.state_dim)
        modified = bridge.modulate_hidden(hidden, state)
        assert modified.shape == hidden.shape
        # Perturbation should be small
        diff = (modified - hidden).norm().item()
        assert diff < hidden.norm().item() * 0.1

    def test_modulate_hidden_different_dim(self, bridge):
        """Hidden state dim != state_dim should still work."""
        sensory = torch.randn(bridge.config.state_dim)
        state = bridge.step(sensory)
        hidden = torch.randn(1, 10, 256)  # Different dim
        modified = bridge.modulate_hidden(hidden, state)
        assert modified.shape == hidden.shape

    def test_latency_tracking(self, bridge):
        sensory = torch.randn(bridge.config.state_dim)
        for _ in range(5):
            bridge.step(sensory)
        stats = bridge.get_latency_stats()
        assert stats['num_steps'] == 5
        assert stats['mean_latency_ms'] > 0

    def test_reset(self, bridge):
        sensory = torch.randn(bridge.config.state_dim)
        for _ in range(5):
            bridge.step(sensory)
        bridge.reset()
        stats = bridge.get_latency_stats()
        assert stats['num_steps'] == 0

    def test_multiple_steps(self, bridge):
        """Run 20 steps without errors."""
        torch.manual_seed(42)
        for i in range(20):
            sensory = torch.randn(bridge.config.state_dim) * 0.5
            state = bridge.step(sensory, reward=0.1 * i)
        assert state.tick == 20

    def test_sleep_consolidation(self, bridge):
        sensory = torch.randn(bridge.config.state_dim)
        # Build memories
        for _ in range(10):
            bridge.step(sensory, reward=1.0)
        # Sleep
        state = bridge.step(sensory, sleep_signal=1.0)
        assert isinstance(state, ModulationState)

    def test_lazy_import(self):
        from pyquifer import PyQuiferBridge, ModulationState
        b = PyQuiferBridge.small()
        assert isinstance(ModulationState(), ModulationState)

    def test_latency_under_target(self, bridge):
        """Latency should be under 50ms per step (conservative target)."""
        torch.manual_seed(42)
        sensory = torch.randn(bridge.config.state_dim)
        # Warmup
        for _ in range(3):
            bridge.step(sensory)
        # Measure
        latencies = []
        for _ in range(10):
            state = bridge.step(sensory)
            latencies.append(state.step_latency_ms)
        mean_latency = sum(latencies) / len(latencies)
        assert mean_latency < 50, f"Mean latency {mean_latency:.1f}ms exceeds 50ms target"
