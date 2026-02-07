"""
Phase 7: Benchmark Gap-Closure Tests

Tests for the five improvements that push Penrose confidence from 0.63 to >0.85:
1. Attractor Stability Index (ASI) — oscillators.py
2. No-Progress Detector — criticality.py
3. Evidence Aggregator + Confidence — metacognitive.py
4. Evidence-Weighted Hypothesis Arena — neural_darwinism.py
5. Phase Topology Cache — oscillators.py
"""

import math
import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pyquifer.oscillators import LearnableKuramotoBank, PhaseTopologyCache
from pyquifer.criticality import NoProgressDetector
from pyquifer.metacognitive import (
    EvidenceSource, EvidenceAggregator, ReasoningMonitor, ReasoningStep,
)
from pyquifer.neural_darwinism import (
    HypothesisProfile, SelectionArena,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Attractor Stability Index (ASI)
# ═══════════════════════════════════════════════════════════════════════════════

class TestAttractorStabilityIndex:

    def test_asi_high_for_synchronized_oscillators(self):
        """Pre-synchronized oscillators should have high stability."""
        torch.manual_seed(0)
        bank = LearnableKuramotoBank(16, dt=0.05)
        with torch.no_grad():
            bank.coupling_strength.fill_(10.0)
        # Drive to synchrony
        for _ in range(300):
            bank(steps=1)
        result = bank.compute_attractor_stability(
            perturbation_scale=0.1, n_trials=8, recovery_steps=20,
        )
        assert result['stability_index'].item() > 0.5, (
            f"Synchronized bank should have ASI > 0.5, got {result['stability_index'].item():.3f}"
        )

    def test_asi_lower_for_weakly_coupled(self):
        """Weakly coupled oscillators should have lower mean_recovery_R than strongly coupled."""
        torch.manual_seed(0)
        # Strongly coupled — sync then measure
        strong = LearnableKuramotoBank(16, dt=0.05)
        with torch.no_grad():
            strong.coupling_strength.fill_(10.0)
        for _ in range(300):
            strong(steps=1)
        strong_result = strong.compute_attractor_stability(
            perturbation_scale=0.2, n_trials=8, recovery_steps=20,
        )

        # Weakly coupled — barely sync
        torch.manual_seed(0)
        weak = LearnableKuramotoBank(16, dt=0.05)
        with torch.no_grad():
            weak.coupling_strength.fill_(0.05)
        for _ in range(50):
            weak(steps=1)
        weak_result = weak.compute_attractor_stability(
            perturbation_scale=0.2, n_trials=8, recovery_steps=20,
        )

        assert strong_result['mean_recovery_R'].item() > weak_result['mean_recovery_R'].item(), (
            f"Strong coupling should recover higher R: "
            f"strong={strong_result['mean_recovery_R'].item():.3f}, "
            f"weak={weak_result['mean_recovery_R'].item():.3f}"
        )

    def test_asi_does_not_modify_phases(self):
        """Phases must be unchanged after ASI computation."""
        torch.manual_seed(42)
        bank = LearnableKuramotoBank(8, dt=0.05)
        for _ in range(50):
            bank(steps=1)
        phases_before = bank.phases.clone()
        bank.compute_attractor_stability(n_trials=5, recovery_steps=10)
        assert torch.allclose(bank.phases, phases_before, atol=1e-6), (
            "ASI should restore phases after computation"
        )

    def test_asi_returns_correct_keys(self):
        """Return dict must have all documented keys."""
        bank = LearnableKuramotoBank(8, dt=0.05)
        result = bank.compute_attractor_stability(n_trials=3, recovery_steps=5)
        expected_keys = {'stability_index', 'escape_probability',
                         'mean_recovery_R', 'R_variance'}
        assert set(result.keys()) == expected_keys


# ═══════════════════════════════════════════════════════════════════════════════
# 2. No-Progress Detector
# ═══════════════════════════════════════════════════════════════════════════════

class TestNoProgressDetector:

    def test_no_progress_on_constant_signal(self):
        """Constant evaluation → stalled."""
        det = NoProgressDetector(window_size=10, progress_threshold=0.01)
        for _ in range(15):
            result = det(torch.tensor(5.0))
        assert result['progress_stalled'].item() is True, (
            "Constant signal should trigger progress_stalled"
        )

    def test_progress_on_increasing_signal(self):
        """Monotonically increasing → not stalled."""
        det = NoProgressDetector(window_size=10, progress_threshold=0.01)
        for i in range(15):
            result = det(torch.tensor(float(i)))
        assert result['progress_stalled'].item() is False, (
            "Increasing signal should NOT trigger progress_stalled"
        )

    def test_stagnation_duration_grows(self):
        """Duration counter should increase while stalled."""
        det = NoProgressDetector(window_size=10, progress_threshold=0.01)
        for _ in range(20):
            result = det(torch.tensor(3.0))
        duration = result['stagnation_duration']
        if isinstance(duration, torch.Tensor):
            duration = duration.item()
        assert duration > 5, (
            f"Stagnation duration should grow, got {duration}"
        )

    def test_reset_clears_history(self):
        """Reset should zero everything."""
        det = NoProgressDetector(window_size=10)
        for _ in range(15):
            det(torch.tensor(3.0))
        det.reset()
        assert det.hist_ptr.item() == 0
        assert det.stagnation_count.item() == 0
        assert det.history.sum().item() == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Evidence Aggregator + Confidence
# ═══════════════════════════════════════════════════════════════════════════════

class TestEvidenceAggregator:

    def test_evidence_aggregator_single_source(self):
        """Single source should return its value scaled."""
        agg = EvidenceAggregator()
        agg.add_evidence(EvidenceSource(
            name="test", value=0.9, supports="DRAW", reasoning_type="deduction",
        ))
        conf = agg.get_confidence("DRAW")
        assert conf > 0.5, f"Single strong evidence should give > 0.5, got {conf:.3f}"

    def test_evidence_aggregator_agreement_high_when_unanimous(self):
        """All sources agree → agreement near 1.0."""
        agg = EvidenceAggregator()
        for name in ["a", "b", "c"]:
            agg.add_evidence(EvidenceSource(
                name=name, value=0.8, supports="DRAW", reasoning_type="deduction",
            ))
        assert agg.get_agreement() > 0.9, (
            f"Unanimous sources should have agreement > 0.9, got {agg.get_agreement():.3f}"
        )

    def test_evidence_aggregator_agreement_low_when_mixed(self):
        """Sources disagreeing → lower agreement."""
        agg = EvidenceAggregator()
        agg.add_evidence(EvidenceSource(
            name="a", value=0.8, supports="DRAW", reasoning_type="deduction",
        ))
        agg.add_evidence(EvidenceSource(
            name="b", value=0.8, supports="BLACK_WINS", reasoning_type="deduction",
        ))
        assert agg.get_agreement() < 0.5, (
            f"Disagreeing sources should have agreement < 0.5, got {agg.get_agreement():.3f}"
        )

    def test_evidence_aggregator_coverage(self):
        """Coverage = fraction of sources with value > 0."""
        agg = EvidenceAggregator()
        agg.add_evidence(EvidenceSource(
            name="a", value=0.8, supports="DRAW", reasoning_type="deduction",
        ))
        agg.add_evidence(EvidenceSource(
            name="b", value=0.0, supports="DRAW", reasoning_type="deduction",
        ))
        assert agg.get_coverage() == 0.5

    def test_analyze_chain_with_evidence_includes_extra_keys(self):
        """analyze_chain_with_evidence should include evidence metrics."""
        monitor = ReasoningMonitor()
        monitor.add_step(ReasoningStep(
            step_id=0, content="test", confidence=0.8,
            reasoning_type="deduction", evidence=["x"],
        ))
        agg = EvidenceAggregator()
        agg.add_evidence(EvidenceSource(
            name="a", value=0.8, supports="DRAW", reasoning_type="deduction",
        ))
        result = monitor.analyze_chain_with_evidence(agg)
        assert 'evidence_agreement' in result
        assert 'evidence_coverage' in result
        assert 'evidence_num_sources' in result
        assert 'evidence_hypotheses' in result


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Evidence-Weighted Hypothesis Arena
# ═══════════════════════════════════════════════════════════════════════════════

class TestHypothesisArena:

    def test_hypothesis_profile_creation(self):
        """HypothesisProfile should store all fields."""
        hp = HypothesisProfile(
            name="DRAW", group_indices=[0, 1],
            evidence_keys=["bishop", "king"],
            base_weight=1.5,
        )
        assert hp.name == "DRAW"
        assert hp.group_indices == [0, 1]
        assert hp.base_weight == 1.5

    def test_set_hypotheses_assigns_groups(self):
        """set_hypotheses should create internal targets."""
        arena = SelectionArena(num_groups=3, group_dim=16)
        arena.set_hypotheses([
            HypothesisProfile("A", [0], ["ev1"]),
            HypothesisProfile("B", [1], ["ev2"]),
            HypothesisProfile("C", [2], ["ev3"]),
        ])
        assert hasattr(arena, '_hypothesis_profiles')
        assert len(arena._hypothesis_profiles) == 3
        assert len(arena._hypothesis_targets) == 3

    def test_inject_evidence_returns_correct_shape(self):
        """inject_evidence should return (group_dim,) tensor."""
        arena = SelectionArena(num_groups=3, group_dim=16)
        arena.set_hypotheses([
            HypothesisProfile("A", [0], ["ev1"]),
            HypothesisProfile("B", [1], ["ev2"]),
        ])
        target = arena.inject_evidence({"ev1": 2.0, "ev2": 0.5})
        assert target.shape == (16,)

    def test_evidence_driven_competition(self):
        """Strong evidence for one hypothesis should make it win resources."""
        torch.manual_seed(42)
        arena = SelectionArena(num_groups=3, group_dim=16, selection_pressure=0.2)
        arena.set_hypotheses([
            HypothesisProfile("LOSE", [0], ["weak"]),
            HypothesisProfile("WIN", [1], ["strong", "very_strong"]),
            HypothesisProfile("NEUTRAL", [2], []),
        ])

        evidence = {"strong": 3.0, "very_strong": 3.0, "weak": 0.1}
        input_feat = torch.randn(16)

        for _ in range(50):
            target = arena.inject_evidence(evidence)
            arena(input_feat, global_coherence=target)

        strengths = arena.get_hypothesis_strengths()
        assert strengths["WIN"] > strengths["LOSE"], (
            f"WIN should have more resources: WIN={strengths['WIN']:.2f}, "
            f"LOSE={strengths['LOSE']:.2f}"
        )

    def test_backward_compat_no_hypotheses(self):
        """Without set_hypotheses, inject_evidence returns zeros."""
        arena = SelectionArena(num_groups=3, group_dim=16)
        target = arena.inject_evidence({"a": 1.0})
        assert target.sum().item() == 0.0
        strengths = arena.get_hypothesis_strengths()
        assert strengths == {}


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Phase Topology Cache
# ═══════════════════════════════════════════════════════════════════════════════

class TestPhaseTopologyCache:

    def test_cache_store_and_query(self):
        """Store then query should return the stored entry."""
        cache = PhaseTopologyCache(capacity=100)
        phases = torch.tensor([0.0, 1.0, 2.0, 3.0])
        cache.store(phases, "DRAW", 0.8)
        result = cache.query(phases)
        assert result is not None
        assert result['outcome'] == "DRAW"
        assert abs(result['confidence'] - 0.8) < 1e-6
        assert result['count'] == 1

    def test_cache_rotation_invariant(self):
        """Rotated phases should match the same hash."""
        cache = PhaseTopologyCache(capacity=100)
        phases = torch.tensor([0.5, 1.5, 2.5, 3.5])
        rotated = (phases + 1.0) % (2 * math.pi)  # Shift all by 1.0

        h1 = cache.compute_hash(phases)
        h2 = cache.compute_hash(rotated)
        assert h1 == h2, (
            f"Rotated phases should produce same hash: {h1} != {h2}"
        )

    def test_cache_capacity_eviction(self):
        """Cache should evict oldest when full."""
        cache = PhaseTopologyCache(capacity=3)
        for i in range(5):
            phases = torch.tensor([float(i), float(i + 1), float(i + 2)])
            cache.store(phases, f"outcome_{i}", 0.5)
        assert len(cache._store) <= 3

    def test_prior_uninformative_for_unseen(self):
        """Unseen topology should return 0.5 (uninformative prior)."""
        cache = PhaseTopologyCache(capacity=100)
        phases = torch.tensor([0.1, 0.2, 0.3])
        prior = cache.get_prior(phases, "DRAW")
        assert prior == 0.5

    def test_prior_grows_with_evidence(self):
        """Repeated storage should increase prior confidence."""
        cache = PhaseTopologyCache(capacity=100)
        phases = torch.tensor([0.0, 1.0, 2.0, 3.0])

        # Store the same topology multiple times with high confidence
        for _ in range(15):
            cache.store(phases, "DRAW", 0.9)

        prior = cache.get_prior(phases, "DRAW")
        assert prior > 0.7, (
            f"Repeated evidence should raise prior above 0.7, got {prior:.3f}"
        )

        # Prior for the opposite hypothesis should be lower
        anti_prior = cache.get_prior(phases, "BLACK_WINS")
        assert anti_prior < 0.5, (
            f"Evidence against should lower prior below 0.5, got {anti_prior:.3f}"
        )
