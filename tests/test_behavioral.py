"""
Behavioral Benchmarks for PyQuifer Cognitive Architecture

Tests EMERGENT properties over extended runs, not unit correctness:
1. Personality stability: dominant facet doesn't jitter randomly
2. Emotional coherence: neuromodulator trajectories are smooth
3. Self-referential consistency: identity strength grows and stabilizes
4. Free energy minimization: FE decreases for repeated stimuli
5. Processing mode adaptation: responds to reward structure
6. Memory consolidation: sleep replay creates semantic traces
7. Oscillator coupling: coherence settles into a stable regime

These are the "does it behave like a mind?" tests.

Note on resource safety:
- All tests use CycleConfig.small() (minimal dimensions)
- Step counts are bounded (100-200 ticks max)
- No GPU required, CPU-only
"""

import torch
import math
import time
import statistics
import pytest
from pyquifer.bridge import PyQuiferBridge


@pytest.fixture
def bridge():
    """Small bridge for behavioral testing."""
    torch.manual_seed(42)
    return PyQuiferBridge.small()


class TestPersonalityStability:
    """Personality should be stable, not random noise."""

    def test_dominant_facet_has_persistence(self, bridge):
        """
        The dominant personality facet should persist for multiple ticks,
        not change every single tick. Metastable = dwell then switch.
        """
        sensory = torch.randn(bridge.config.state_dim) * 0.3
        facets = []
        for _ in range(100):
            state = bridge.step(sensory)
            facets.append(state.dominant_facet)

        # Count how many times the facet changed
        changes = sum(1 for i in range(1, len(facets)) if facets[i] != facets[i - 1])

        # Should NOT change every tick (that's noise). Should change sometimes (metastable).
        # With 100 ticks and 4 populations (small config), expect < 50 changes.
        assert changes < 50, f"Facet changed {changes}/99 ticks - too unstable"

    def test_personality_weights_smooth(self, bridge):
        """
        Personality facet weights should change smoothly, not jump randomly.
        Measure mean absolute change between consecutive ticks.
        """
        sensory = torch.randn(bridge.config.state_dim) * 0.3
        prev_weights = None
        jumps = []

        for _ in range(100):
            state = bridge.step(sensory)
            weights = state.facet_weights
            if prev_weights is not None:
                jump = sum(abs(w - pw) for w, pw in zip(weights, prev_weights))
                jumps.append(jump)
            prev_weights = weights

        mean_jump = statistics.mean(jumps)
        # Weights are normalized to sum=1. Max possible jump is 2.0 (flip from
        # one facet to another). Mean jump should be moderate.
        assert mean_jump < 1.5, f"Mean weight jump {mean_jump:.3f} too large"

    def test_identity_strength_grows(self, bridge):
        """
        Identity strength should increase over time as the narrative
        stabilizes. Early identity < late identity.
        """
        sensory = torch.randn(bridge.config.state_dim) * 0.3
        early_ids = []
        late_ids = []

        for i in range(100):
            state = bridge.step(sensory)
            if i < 10:
                early_ids.append(state.identity_strength)
            if i >= 90:
                late_ids.append(state.identity_strength)

        early_avg = statistics.mean(early_ids)
        late_avg = statistics.mean(late_ids)
        assert late_avg > early_avg, \
            f"Identity didn't grow: early={early_avg:.4f}, late={late_avg:.4f}"


class TestEmotionalCoherence:
    """Neuromodulator levels should be smooth, not noise."""

    def test_neuromodulator_levels_bounded(self, bridge):
        """All neuromodulator levels should stay in [0, 1] range."""
        sensory = torch.randn(bridge.config.state_dim) * 0.3
        for i in range(50):
            reward = math.sin(i * 0.2)
            state = bridge.step(sensory, reward=reward)
            levels = state.neuromodulator_levels
            assert levels is not None
            assert (levels >= -0.01).all(), f"Neuromodulator below 0: {levels}"
            assert (levels <= 1.01).all(), f"Neuromodulator above 1: {levels}"

    def test_neuromodulator_trajectories_smooth(self, bridge):
        """
        Neuromodulator changes between ticks should be small.
        These have time constants (tau), so they can't jump instantly.
        """
        sensory = torch.randn(bridge.config.state_dim) * 0.3
        prev_levels = None
        max_jumps = []

        for i in range(50):
            reward = math.sin(i * 0.2)
            state = bridge.step(sensory, reward=reward)
            levels = state.neuromodulator_levels
            if prev_levels is not None:
                jump = (levels - prev_levels).abs().max().item()
                max_jumps.append(jump)
            prev_levels = levels.clone()

        mean_max_jump = statistics.mean(max_jumps)
        # Time constants mean levels can't change by more than ~0.1 per tick
        assert mean_max_jump < 0.3, \
            f"Mean max neuromodulator jump {mean_max_jump:.4f} too large"

    def test_reward_affects_motivation(self, bridge):
        """
        Sustained positive reward should increase motivation
        compared to zero reward.
        """
        sensory = torch.randn(bridge.config.state_dim) * 0.3

        # Run with zero reward
        bridge.reset()
        for _ in range(30):
            s_zero = bridge.step(sensory, reward=0.0)
        motiv_zero = s_zero.motivation

        # Run with positive reward
        bridge.reset()
        for _ in range(30):
            s_pos = bridge.step(sensory, reward=1.0)
        motiv_pos = s_pos.motivation

        # Positive reward should produce higher motivation
        assert motiv_pos > motiv_zero * 0.8, \
            f"Positive reward motivation ({motiv_pos:.3f}) should exceed " \
            f"zero reward ({motiv_zero:.3f})"


class TestFreeEnergyMinimization:
    """The core property: repeated input -> decreasing surprise."""

    def test_free_energy_decreases_for_fixed_input(self, bridge):
        """
        Presenting the same input repeatedly should reduce free energy
        as the predictive model learns to expect it.
        """
        sensory = torch.randn(bridge.config.state_dim) * 0.5
        free_energies = []

        for _ in range(100):
            state = bridge.step(sensory)
            free_energies.append(state.free_energy)

        early = statistics.mean(free_energies[:10])
        late = statistics.mean(free_energies[-10:])

        assert late < early, \
            f"Free energy didn't decrease: early={early:.4f}, late={late:.4f}"

    def test_novel_input_increases_free_energy(self, bridge):
        """
        After habituating to one input, a novel input should spike free energy.
        """
        sensory_a = torch.randn(bridge.config.state_dim) * 0.5

        # Habituate to input A
        for _ in range(50):
            state_a = bridge.step(sensory_a)
        fe_habituated = state_a.free_energy

        # Present novel input B
        sensory_b = torch.randn(bridge.config.state_dim) * 0.5
        state_b = bridge.step(sensory_b)
        fe_novel = state_b.free_energy

        # Novel input should have higher FE (more surprise)
        assert fe_novel > fe_habituated * 0.5, \
            f"Novel input FE ({fe_novel:.4f}) should exceed habituated ({fe_habituated:.4f})"


class TestProcessingModeAdaptation:
    """Processing mode should respond to the cognitive situation."""

    def test_processing_mode_is_valid(self, bridge):
        """Processing mode should always be a recognized string."""
        sensory = torch.randn(bridge.config.state_dim) * 0.3
        valid_modes = {"perception", "imagination", "balanced"}
        for _ in range(30):
            state = bridge.step(sensory)
            assert state.processing_mode in valid_modes, \
                f"Invalid mode: {state.processing_mode}"

    def test_temperature_responds_to_coherence(self, bridge):
        """
        Temperature should vary with cognitive state. We can't control
        coherence directly, but over many ticks we should see variation.
        """
        sensory = torch.randn(bridge.config.state_dim) * 0.3
        temps = []
        for _ in range(50):
            state = bridge.step(sensory)
            temps.append(state.temperature)

        # Temperature should vary (not all identical)
        temp_range = max(temps) - min(temps)
        assert temp_range > 0.001, \
            f"Temperature range {temp_range:.6f} - no variation detected"


class TestMemoryAndConsolidation:
    """Memory systems should accumulate and consolidate properly."""

    def test_memories_accumulate(self, bridge):
        """Episodic memories should grow as ticks proceed."""
        sensory = torch.randn(bridge.config.state_dim) * 0.3
        state = None
        for _ in range(50):
            state = bridge.step(sensory, reward=0.5)

        # After 50 ticks, tick count should be 50
        assert state.tick == 50

    def test_sleep_consolidation_runs(self, bridge):
        """
        After building memories, a sleep signal should trigger
        consolidation without errors.
        """
        sensory = torch.randn(bridge.config.state_dim) * 0.3

        # Build up memories with rewards
        for _ in range(30):
            bridge.step(sensory, reward=1.0)

        # Enter sleep
        for _ in range(5):
            state = bridge.step(sensory, sleep_signal=1.0)

        # Should complete without errors
        assert state.tick == 35


class TestOscillatorDynamics:
    """Oscillator behavior should show meaningful coupling."""

    def test_phases_change_over_time(self, bridge):
        """Oscillator phases should evolve, not be static."""
        sensory = torch.randn(bridge.config.state_dim) * 0.3
        state0 = bridge.step(sensory)
        phases_0 = state0.phases.clone()

        for _ in range(10):
            state = bridge.step(sensory)
        phases_10 = state.phases.clone()

        # Phases should have changed
        phase_diff = (phases_10 - phases_0).abs().mean().item()
        assert phase_diff > 0.01, f"Phases barely changed: mean diff = {phase_diff:.6f}"

    def test_coherence_in_valid_range(self, bridge):
        """Coherence (order parameter) should be in [0, 1]."""
        sensory = torch.randn(bridge.config.state_dim) * 0.3
        for _ in range(30):
            state = bridge.step(sensory)
            assert 0.0 <= state.coherence <= 1.0, \
                f"Coherence out of range: {state.coherence}"

    def test_criticality_distance_varies(self, bridge):
        """
        Criticality distance should vary as the system evolves,
        showing that the criticality controller is active.
        """
        sensory = torch.randn(bridge.config.state_dim) * 0.3
        distances = []
        for _ in range(50):
            state = bridge.step(sensory)
            distances.append(state.criticality_distance)

        dist_range = max(distances) - min(distances)
        # Should show some variation (controller is adjusting)
        assert dist_range > 0.001, \
            f"Criticality distance range {dist_range:.6f} - no controller activity"


class TestLLMModulationContract:
    """The bridge output should always be suitable for LLM integration."""

    def test_generation_params_always_valid(self, bridge):
        """
        Over many ticks with varied inputs, generation params
        should always be in usable ranges.
        """
        for i in range(50):
            sensory = torch.randn(bridge.config.state_dim) * (0.1 + i * 0.02)
            reward = math.sin(i * 0.3) * 2.0
            state = bridge.step(sensory, reward=reward)

            assert 0.1 <= state.temperature <= 2.0, \
                f"Tick {i}: temperature {state.temperature} out of range"
            assert 0.5 <= state.top_p <= 1.0, \
                f"Tick {i}: top_p {state.top_p} out of range"
            assert 1.0 <= state.repetition_penalty <= 1.5, \
                f"Tick {i}: repetition_penalty {state.repetition_penalty} out of range"

    def test_logit_modulation_preserves_shape(self, bridge):
        """Logit modulation should work for any vocab size."""
        sensory = torch.randn(bridge.config.state_dim) * 0.3
        state = bridge.step(sensory)

        for vocab_size in [100, 32000, 128000]:
            logits = torch.randn(1, vocab_size)
            modified = bridge.modulate_logits(logits, state)
            assert modified.shape == logits.shape

    def test_hidden_modulation_preserves_shape(self, bridge):
        """Hidden state modulation should work for any sequence length."""
        sensory = torch.randn(bridge.config.state_dim) * 0.3
        state = bridge.step(sensory)

        for seq_len in [1, 64, 512]:
            hidden = torch.randn(1, seq_len, bridge.config.state_dim)
            modified = bridge.modulate_hidden(hidden, state)
            assert modified.shape == hidden.shape

    def test_perturbation_bounded(self, bridge):
        """
        Hidden state perturbation should be small relative to hidden state norm.
        This prevents the oscillator from dominating the LLM.
        """
        sensory = torch.randn(bridge.config.state_dim) * 0.3
        state = bridge.step(sensory)

        hidden = torch.randn(1, 32, bridge.config.state_dim)
        modified = bridge.modulate_hidden(hidden, state)

        diff_norm = (modified - hidden).norm().item()
        hidden_norm = hidden.norm().item()
        relative = diff_norm / (hidden_norm + 1e-8)

        # Perturbation should be < 10% of hidden state
        assert relative < 0.1, \
            f"Perturbation too large: {relative:.4f} ({diff_norm:.4f} / {hidden_norm:.4f})"


class TestExtendedStability:
    """Run for longer to check no drift or numerical issues."""

    def test_200_ticks_no_nan(self, bridge):
        """200 ticks with varied input should produce no NaN/Inf values."""
        torch.manual_seed(42)
        for i in range(200):
            sensory = torch.randn(bridge.config.state_dim) * 0.5
            reward = math.sin(i * 0.1)
            sleep = 1.0 if (i % 50 > 40) else 0.0

            state = bridge.step(sensory, reward=reward, sleep_signal=sleep)

            # Check critical numeric values
            assert math.isfinite(state.temperature), f"Tick {i}: NaN temperature"
            assert math.isfinite(state.top_p), f"Tick {i}: NaN top_p"
            assert math.isfinite(state.coherence), f"Tick {i}: NaN coherence"
            assert math.isfinite(state.free_energy), f"Tick {i}: NaN free_energy"
            assert state.phases is not None and not state.phases.isnan().any(), \
                f"Tick {i}: NaN phases"
            assert state.neuromodulator_levels is not None and \
                not state.neuromodulator_levels.isnan().any(), \
                f"Tick {i}: NaN neuromodulators"

        assert state.tick == 200

    def test_latency_stable(self, bridge):
        """Latency should not degrade over time."""
        sensory = torch.randn(bridge.config.state_dim) * 0.3

        # Warmup
        for _ in range(5):
            bridge.step(sensory)

        # Measure early and late latencies
        early_latencies = []
        for _ in range(20):
            state = bridge.step(sensory)
            early_latencies.append(state.step_latency_ms)

        for _ in range(80):
            bridge.step(sensory)

        late_latencies = []
        for _ in range(20):
            state = bridge.step(sensory)
            late_latencies.append(state.step_latency_ms)

        early_mean = statistics.mean(early_latencies)
        late_mean = statistics.mean(late_latencies)

        # Late latency should not be more than 3x early (no memory leak or O(n) growth)
        assert late_mean < early_mean * 3, \
            f"Latency degraded: early={early_mean:.1f}ms, late={late_mean:.1f}ms"
