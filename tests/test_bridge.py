"""Tests for the PyQuiferBridge LLM modulation API."""
import torch
import pytest
from pyquifer.bridge import (
    PyQuiferBridge, ModulationState, SteppedModulator,
    PyQuiferLogitsProcessor, _interpolate_state,
)


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

    # -- Shape contract tests --

    def test_modulate_logits_2d(self, bridge):
        """modulate_logits accepts (batch, vocab) shape."""
        sensory = torch.randn(bridge.config.state_dim)
        state = bridge.step(sensory)
        logits = torch.randn(2, 500)
        modified = bridge.modulate_logits(logits, state)
        assert modified.shape == (2, 500)

    def test_modulate_logits_3d(self, bridge):
        """modulate_logits accepts (batch, seq, vocab) shape."""
        sensory = torch.randn(bridge.config.state_dim)
        state = bridge.step(sensory)
        logits = torch.randn(2, 10, 500)
        modified = bridge.modulate_logits(logits, state)
        assert modified.shape == (2, 10, 500)

    def test_modulate_logits_rejects_1d(self, bridge):
        """modulate_logits rejects 1D input."""
        sensory = torch.randn(bridge.config.state_dim)
        state = bridge.step(sensory)
        with pytest.raises(AssertionError, match="2D.*or 3D"):
            bridge.modulate_logits(torch.randn(500), state)

    # -- Device safety tests --

    def test_modulate_hidden_cpu_state(self, bridge):
        """modulate_hidden works when state tensors are on CPU."""
        sensory = torch.randn(bridge.config.state_dim)
        state = bridge.step(sensory)
        # Explicitly ensure state tensors are CPU
        if state.phases is not None:
            state.phases = state.phases.cpu()
        if state.neuromodulator_levels is not None:
            state.neuromodulator_levels = state.neuromodulator_levels.cpu()
        hidden = torch.randn(1, 10, bridge.config.state_dim)
        modified = bridge.modulate_hidden(hidden, state)
        assert modified.shape == hidden.shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
    def test_modulate_hidden_cross_device(self, bridge):
        """modulate_hidden works with CPU state and CUDA hidden."""
        bridge = bridge.cpu()
        sensory = torch.randn(bridge.config.state_dim)
        state = bridge.step(sensory)
        # State tensors stay CPU, hidden goes CUDA
        hidden = torch.randn(1, 10, bridge.config.state_dim, device="cuda")
        bridge_cuda = bridge.to("cuda")
        modified = bridge_cuda.modulate_hidden(hidden, state)
        assert modified.device == hidden.device
        assert modified.shape == hidden.shape

    # -- prepare_sensory_input tests --

    def test_prepare_sensory_input_1d(self, bridge):
        """prepare_sensory_input handles 1D input."""
        x = torch.randn(128)
        s = bridge.prepare_sensory_input(x)
        assert s.shape == (bridge.config.state_dim,)

    def test_prepare_sensory_input_2d(self, bridge):
        """prepare_sensory_input collapses (B, D) → (D,) → (state_dim,)."""
        x = torch.randn(4, 256)
        s = bridge.prepare_sensory_input(x)
        assert s.shape == (bridge.config.state_dim,)

    def test_prepare_sensory_input_3d(self, bridge):
        """prepare_sensory_input collapses (B, T, D) → (D,) → (state_dim,)."""
        x = torch.randn(2, 10, 512)
        s = bridge.prepare_sensory_input(x)
        assert s.shape == (bridge.config.state_dim,)

    def test_prepare_sensory_input_exact_dim(self, bridge):
        """No projection when dim already matches."""
        x = torch.randn(bridge.config.state_dim)
        s = bridge.prepare_sensory_input(x)
        assert s.shape == (bridge.config.state_dim,)
        assert torch.allclose(s, x)

    # -- Determinism test --

    def test_deterministic_modulation(self, bridge):
        """Fixed seed + fixed input → identical modulation deltas."""
        import random
        def _run_bridge():
            torch.manual_seed(123)
            random.seed(123)
            b = PyQuiferBridge.small()
            s = torch.randn(b.config.state_dim)
            state = b.step(s)
            hidden = torch.ones(1, 5, b.config.state_dim)
            return b.modulate_hidden(hidden, state)

        r1 = _run_bridge()
        r2 = _run_bridge()
        assert torch.allclose(r1, r2, atol=1e-6), "Modulation not deterministic"


class TestSteppedModulator:
    @pytest.fixture
    def bridge(self):
        return PyQuiferBridge.small()

    def test_steps_only_every_n(self, bridge):
        """bridge.step() is called once per step_every tokens."""
        stepper = SteppedModulator(bridge, step_every=4)
        sensory = torch.randn(bridge.config.state_dim)

        ticks = []
        for _ in range(12):
            state = stepper.step(sensory)
            ticks.append(state.tick)
        # Ticks should change only at positions 0, 4, 8
        # Interpolated states between share a tick from s2
        assert ticks[0] == ticks[1]  # token 0 and 1 share the first step
        assert ticks[4] != ticks[0]  # token 4 triggers a new step

    def test_interpolated_temperature(self, bridge):
        """Temperature is interpolated between steps."""
        stepper = SteppedModulator(bridge, step_every=4)
        sensory = torch.randn(bridge.config.state_dim)

        # First step: token 0
        s0 = stepper.step(sensory)
        t0 = s0.temperature

        # Tokens 1-3: interpolated
        for i in range(1, 4):
            si = stepper.step(sensory)
            # Temperature should be between s0 and the (upcoming) s1 values
            assert isinstance(si.temperature, float)
            assert si.temperature > 0

    def test_step_every_1_no_interpolation(self, bridge):
        """step_every=1 means every token calls bridge.step()."""
        stepper = SteppedModulator(bridge, step_every=1)
        sensory = torch.randn(bridge.config.state_dim)

        states = [stepper.step(sensory) for _ in range(5)]
        ticks = [s.tick for s in states]
        # Each call should produce a new tick
        assert ticks[0] < ticks[4]

    def test_reset_clears_state(self, bridge):
        """reset() clears token counter and cached states."""
        stepper = SteppedModulator(bridge, step_every=4)
        sensory = torch.randn(bridge.config.state_dim)

        stepper.step(sensory)
        stepper.step(sensory)
        assert stepper._token_count == 2

        stepper.reset()
        assert stepper._token_count == 0
        assert stepper._prev_state is None
        assert stepper._curr_state is None

    def test_current_state_property(self, bridge):
        """current_state returns the last full (non-interpolated) state."""
        stepper = SteppedModulator(bridge, step_every=4)
        sensory = torch.randn(bridge.config.state_dim)

        assert stepper.current_state is None
        s0 = stepper.step(sensory)
        assert stepper.current_state is not None
        assert stepper.current_state.tick == s0.tick


class TestInterpolateState:
    def test_scalar_interpolation(self):
        s1 = ModulationState(temperature=0.5, coherence=0.2, motivation=0.1)
        s2 = ModulationState(temperature=1.0, coherence=0.8, motivation=0.9)
        mid = _interpolate_state(s1, s2, alpha=0.5)
        assert abs(mid.temperature - 0.75) < 1e-6
        assert abs(mid.coherence - 0.5) < 1e-6
        assert abs(mid.motivation - 0.5) < 1e-6

    def test_tensor_interpolation(self):
        s1 = ModulationState(phases=torch.zeros(8))
        s2 = ModulationState(phases=torch.ones(8))
        mid = _interpolate_state(s1, s2, alpha=0.25)
        assert torch.allclose(mid.phases, torch.full((8,), 0.25))

    def test_alpha_zero_returns_s1(self):
        s1 = ModulationState(temperature=0.3)
        s2 = ModulationState(temperature=0.9)
        result = _interpolate_state(s1, s2, alpha=0.0)
        assert result.temperature == 0.3

    def test_alpha_one_returns_s2(self):
        s1 = ModulationState(temperature=0.3)
        s2 = ModulationState(temperature=0.9)
        result = _interpolate_state(s1, s2, alpha=1.0)
        assert abs(result.temperature - 0.9) < 1e-6


class TestPyQuiferLogitsProcessor:
    @pytest.fixture
    def bridge(self):
        return PyQuiferBridge.small()

    def test_fixed_state(self, bridge):
        """LogitsProcessor with a fixed ModulationState modifies logits."""
        sensory = torch.randn(bridge.config.state_dim)
        state = bridge.step(sensory)

        proc = PyQuiferLogitsProcessor(bridge, state=state)
        input_ids = torch.randint(0, 100, (1, 10))
        scores = torch.randn(1, 32000)
        modified = proc(input_ids, scores)
        assert modified.shape == scores.shape
        # Should actually modify (not identity)
        assert not torch.equal(modified, scores)

    def test_with_stepper(self, bridge):
        """LogitsProcessor with SteppedModulator steps the bridge."""
        sensory = torch.randn(bridge.config.state_dim)
        stepper = SteppedModulator(bridge, step_every=4)

        proc = PyQuiferLogitsProcessor(bridge, stepper=stepper, sensory=sensory)
        input_ids = torch.randint(0, 100, (1, 5))
        scores = torch.randn(1, 32000)

        # Call multiple times
        for _ in range(8):
            modified = proc(input_ids, scores)
            assert modified.shape == scores.shape

    def test_no_state_passthrough(self, bridge):
        """No state and no stepper → pass-through."""
        proc = PyQuiferLogitsProcessor(bridge)
        input_ids = torch.randint(0, 100, (1, 10))
        scores = torch.randn(1, 32000)
        result = proc(input_ids, scores)
        assert torch.equal(result, scores)
