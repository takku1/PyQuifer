"""Cross-module integration tests for Phase 2 modules.

Tests that modules work together as designed — precision feeds into
predictive coding, metastability connects to stochastic resonance,
causal flow measures information direction, etc.
"""
import torch
import math
import pytest


class TestPrecisionIntoPredictiveCoding:
    """Precision weighting feeds into hierarchical predictive coding."""

    def test_precision_weights_prediction_errors(self):
        from pyquifer.precision_weighting import PrecisionEstimator, PrecisionGate
        from pyquifer.hierarchical_predictive import PredictiveLevel

        level = PredictiveLevel(input_dim=8, belief_dim=8)
        estimator = PrecisionEstimator(num_channels=8)
        gate = PrecisionGate(num_channels=8)

        # Run predictive level
        result = level(torch.randn(8))
        error = result['error']

        # Precision-weight the error
        precision = estimator(error)
        weighted = gate(error, precision['precision'])
        assert weighted['weighted_signal'].shape == error.shape

    def test_precision_modulates_effective_lr(self):
        from pyquifer.precision_weighting import PrecisionEstimator

        # Use batched inputs so var(dim=0) captures real variance
        estimator = PrecisionEstimator(num_channels=4, tau=5.0, max_precision=1000.0)
        # Feed consistent errors in batches (low batch variance → high precision)
        for _ in range(100):
            estimator(torch.ones(8, 4) * 0.1)  # All same → var ≈ 0
        high_precision = estimator(torch.ones(8, 4) * 0.1)

        # Feed noisy errors in batches (high batch variance → low precision)
        estimator2 = PrecisionEstimator(num_channels=4, tau=5.0, max_precision=1000.0)
        for _ in range(100):
            estimator2(torch.randn(8, 4) * 5.0)  # Diverse → var >> 0
        low_precision = estimator2(torch.randn(8, 4) * 5.0)

        assert high_precision['precision'].mean() > low_precision['precision'].mean()


class TestMetastabilityAndResonance:
    """Metastability dynamics connect to stochastic resonance."""

    def test_metastable_activations_as_resonance_input(self):
        from pyquifer.metastability import WinnerlessCompetition
        from pyquifer.stochastic_resonance import AdaptiveStochasticResonance

        wlc = WinnerlessCompetition(num_populations=8, noise_scale=0.02)
        asr = AdaptiveStochasticResonance(dim=8, threshold=0.3)

        for _ in range(100):
            comp = wlc()
            enhanced = asr(comp['activations'])
        assert enhanced['enhanced'].shape == (8,)
        assert enhanced['snr'].item() > 0

    def test_criticality_distance_from_metastability(self):
        from pyquifer.metastability import MetastabilityIndex
        from pyquifer.stochastic_resonance import AdaptiveStochasticResonance

        mi = MetastabilityIndex(num_populations=4)
        asr = AdaptiveStochasticResonance(dim=4, threshold=0.3)

        for _ in range(200):
            r = mi()
            # Metastability index as proxy for distance to criticality
            crit_dist = abs(r['metastability_index'].item() - 1.0)
            asr(r['activations'], criticality_distance=crit_dist)
        assert asr.noise_level.item() > 0


class TestCausalFlowMeasuresDirectionality:
    """Causal flow can detect which module drives which."""

    def test_driven_signal_detected(self):
        from pyquifer.causal_flow import TransferEntropyEstimator

        torch.manual_seed(42)
        te = TransferEntropyEstimator(num_bins=8, history_length=2)
        T = 400
        driver = torch.randn(T)
        follower = torch.zeros(T)
        for t in range(1, T):
            follower[t] = 0.8 * driver[t-1] + 0.2 * torch.randn(1).item()

        r = te(driver, follower)
        assert r['net_flow'].item() > 0  # driver→follower detected

    def test_causal_flow_with_metastable_populations(self):
        from pyquifer.metastability import WinnerlessCompetition
        from pyquifer.causal_flow import CausalFlowMap

        wlc = WinnerlessCompetition(num_populations=3, noise_scale=0.02)
        cfm = CausalFlowMap(num_populations=3, buffer_size=300)

        for _ in range(300):
            r = wlc()
            cfm.record(r['activations'])

        flow = cfm.compute_flow()
        assert flow['flow_matrix'].shape == (3, 3)


class TestMemoryConsolidationPipeline:
    """Episodic buffer → sharp-wave ripple → consolidation → reconsolidation."""

    def test_full_consolidation_pipeline(self):
        from pyquifer.memory_consolidation import (
            EpisodicBuffer, SharpWaveRipple, ConsolidationEngine, MemoryReconsolidation
        )

        dim = 16
        buf = EpisodicBuffer(state_dim=dim, capacity=50)
        swr = SharpWaveRipple(state_dim=dim)
        engine = ConsolidationEngine(state_dim=dim, semantic_dim=8)
        recon = MemoryReconsolidation(dim=8)

        # Store experiences
        for i in range(30):
            buf.store(torch.randn(dim), reward=float(i))

        # Sharp-wave ripple replay (sleep)
        replay = swr(buf, sleep_signal=1.0)
        assert replay['replay_active'].item() == True

        # Consolidate replayed memories
        if replay['replayed_states'].shape[0] > 0:
            rewards = torch.ones(replay['replayed_states'].shape[0])
            engine(replay['replayed_states'], rewards)

        # Query semantic memory
        query = engine.query(torch.randn(dim), top_k=3)
        if query['traces'].shape[0] > 0:
            # Reconsolidate a retrieved memory
            trace = query['traces'][0]
            context = torch.randn(8)
            result = recon(trace, context, emotional_intensity=0.5)
            assert result['reconsolidated'].shape == (8,)


class TestSelfModelIntegration:
    """Self-model integrates body, personality, capability signals."""

    def test_self_model_with_markov_blanket(self):
        from pyquifer.self_model import MarkovBlanket, SelfModel, NarrativeIdentity

        mb = MarkovBlanket(internal_dim=16, sensory_dim=8, active_dim=8)
        sm = SelfModel(self_dim=16)
        ni = NarrativeIdentity(dim=16, tau=0.1)

        # Sensory input → Markov blanket → Self-model → Narrative
        for _ in range(50):
            sensory = torch.randn(8)
            blanket_result = mb(sensory)
            self_result = sm(blanket_result['internal_state'])
            narrative_result = ni(self_result['self_summary'])

        assert narrative_result['identity_strength'].item() > 0
        assert narrative_result['deviation'].item() >= 0


class TestNeuralDarwinismWithCoherence:
    """Neural darwinism uses global coherence for fitness."""

    def test_selection_arena_differentiates_groups(self):
        from pyquifer.neural_darwinism import SelectionArena

        torch.manual_seed(42)
        arena = SelectionArena(num_groups=4, group_dim=8,
                                selection_pressure=0.2, total_budget=8.0)
        target = torch.sin(torch.linspace(0, math.pi, 8))

        for _ in range(100):
            arena(target + torch.randn(8) * 0.1, global_coherence=target)

        resources = [g.resources.item() for g in arena.groups]
        # After selection, resources should not be perfectly equal
        assert max(resources) / (min(resources) + 1e-8) > 1.0

    def test_arena_with_symbiogenesis(self):
        from pyquifer.neural_darwinism import SelectionArena, SymbiogenesisDetector

        arena = SelectionArena(num_groups=3, group_dim=8)
        symb = SymbiogenesisDetector(num_groups=3, group_dim=8, buffer_size=50)

        for _ in range(50):
            r = arena(torch.randn(8))
            symb(r['group_outputs'])

        # Should run without errors, bonds may or may not form
        assert symb.hist_ptr.item() == 50


class TestPredictiveCodingHierarchy:
    """Hierarchical predictive coding with multiple levels."""

    def test_multi_level_predictions_flow_down(self):
        from pyquifer.hierarchical_predictive import HierarchicalPredictiveCoding

        hpc = HierarchicalPredictiveCoding(
            level_dims=[16, 12, 8],
            lr=0.05
        )
        # Feed sensory input, get hierarchical result
        r = hpc(torch.randn(4, 16))
        assert len(r['predictions']) == 3
        assert r['total_error'].item() >= 0

    def test_beliefs_converge_with_repeated_input(self):
        from pyquifer.hierarchical_predictive import HierarchicalPredictiveCoding

        hpc = HierarchicalPredictiveCoding(level_dims=[8, 6], lr=0.1)
        pattern = torch.randn(8)

        errors = []
        for _ in range(50):
            r = hpc(pattern)
            errors.append(r['total_error'].item())

        # Belief variance should stabilize
        early_var = torch.tensor(errors[:10]).var().item()
        late_var = torch.tensor(errors[-10:]).var().item()
        assert late_var <= early_var + 0.1  # Allow small margin
