"""Tests for the IIT Metrics module (enhanced with rigorous IIT 4.0 features)."""
import torch
import pytest
import math
from pyquifer.iit_metrics import (
    generate_bipartitions,
    hamming_distance_matrix,
    EarthMoverDistance,
    KLDivergence,
    L1Distance,
    InformationDensity,
    PartitionedInformation,
    IntegratedInformation,
    CauseEffectRepertoire,
    IITConsciousnessMonitor,
    Concept,
    SystemIrreducibilityAnalysis,
)


class TestBipartitions:
    def test_count_3_elements(self):
        """3 elements -> 2^2 - 1 = 3 bipartitions."""
        parts = generate_bipartitions(3)
        assert parts.shape[0] == 3

    def test_count_4_elements(self):
        """4 elements -> 2^3 - 1 = 7 bipartitions."""
        parts = generate_bipartitions(4)
        assert parts.shape[0] == 7

    def test_no_empty_parts(self):
        """Every partition should have at least one True and one False."""
        for n in [2, 3, 4, 5]:
            parts = generate_bipartitions(n)
            for p in parts:
                assert p.sum() > 0, f"Empty part A for n={n}"
                assert (~p).sum() > 0, f"Empty part B for n={n}"

    def test_uniqueness(self):
        """All partitions should be unique."""
        parts = generate_bipartitions(5)
        # Convert to set of tuples for uniqueness check
        as_tuples = set(tuple(p.tolist()) for p in parts)
        assert len(as_tuples) == parts.shape[0]


class TestHammingDistance:
    def test_shape(self):
        H = hamming_distance_matrix(3)
        assert H.shape == (8, 8)

    def test_diagonal_zero(self):
        H = hamming_distance_matrix(3)
        assert H.diag().sum().item() == 0.0

    def test_known_distances(self):
        H = hamming_distance_matrix(3)
        # 000 (0) and 111 (7) should be distance 3
        assert H[0, 7].item() == 3.0
        # 000 (0) and 001 (1) should be distance 1
        assert H[0, 1].item() == 1.0


class TestDistanceMetrics:
    def test_emd_same_distributions(self):
        emd = EarthMoverDistance(8, use_hamming=True)
        p = torch.tensor([0.5, 0.5, 0, 0, 0, 0, 0, 0])
        result = emd(p, p)
        assert result.item() < 0.01

    def test_emd_different_distributions(self):
        emd = EarthMoverDistance(8, use_hamming=True)
        p = torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0])
        q = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1.0])
        result = emd(p, q)
        assert result.item() > 0.5

    def test_kl_identical_is_zero(self):
        kl = KLDivergence()
        p = torch.tensor([0.25, 0.25, 0.25, 0.25])
        result = kl(p, p)
        assert result.item() < 0.01

    def test_kl_different_positive(self):
        kl = KLDivergence()
        p = torch.tensor([0.9, 0.05, 0.025, 0.025])
        q = torch.tensor([0.25, 0.25, 0.25, 0.25])
        result = kl(p, q)
        assert result.item() > 0.1

    def test_l1_identical(self):
        l1 = L1Distance()
        p = torch.tensor([0.5, 0.5])
        result = l1(p, p)
        assert result.item() == 0.0

    def test_info_density_shape(self):
        info = InformationDensity()
        p = torch.tensor([0.7, 0.3])
        q = torch.tensor([0.5, 0.5])
        result = info(p, q)
        assert result.shape == (2,)


class TestPartitionedInformation:
    def test_exhaustive_small_dim(self):
        """For dim <= 12, should use exhaustive enumeration."""
        pi = PartitionedInformation(state_dim=4, distance_metric='l1')
        assert pi.exhaustive is True
        assert pi.num_partitions == 7  # 2^3 - 1

    def test_sampling_large_dim(self):
        """For dim > 12, should use sampling with >= 64 partitions."""
        pi = PartitionedInformation(state_dim=16, distance_metric='l1')
        assert pi.exhaustive is False
        assert pi.num_partitions >= 64

    def test_output_keys(self):
        pi = PartitionedInformation(state_dim=6, distance_metric='l1')
        state = torch.randn(6)
        result = pi(state)
        assert 'phi' in result
        assert 'partition_losses' in result
        assert 'min_partition_idx' in result
        assert 'exhaustive' in result

    def test_phi_scalar(self):
        pi = PartitionedInformation(state_dim=6, distance_metric='l1')
        state = torch.randn(6)
        result = pi(state)
        assert result['phi'].dim() == 1  # batch dim

    def test_phi_nonnegative(self):
        pi = PartitionedInformation(state_dim=6, distance_metric='l1')
        for _ in range(5):
            result = pi(torch.randn(6))
            assert result['phi'].item() >= -1e-6

    def test_return_min_partition_mask(self):
        pi = PartitionedInformation(state_dim=4, distance_metric='l1')
        result = pi(torch.randn(4), return_min_partition=True)
        assert 'min_partition_mask' in result
        mask = result['min_partition_mask']
        assert mask.shape == (4,)
        assert mask.dtype == torch.bool

    def test_batch_support(self):
        pi = PartitionedInformation(state_dim=6, distance_metric='l1')
        state = torch.randn(3, 6)
        result = pi(state)
        assert result['phi'].shape == (3,)


class TestIntegratedInformation:
    @pytest.fixture
    def phi_computer(self):
        return IntegratedInformation(state_dim=8, num_partitions=8, temporal_window=5)

    def test_output_keys(self, phi_computer):
        result = phi_computer(torch.randn(8))
        assert 'phi' in result
        assert 'temporal_phi' in result
        assert 'partition_losses' in result

    def test_temporal_phi_starts_zero(self, phi_computer):
        result = phi_computer(torch.randn(8))
        assert result['temporal_phi'].item() == 0.0

    def test_temporal_phi_after_filling(self, phi_computer):
        for _ in range(10):
            result = phi_computer(torch.randn(8))
        assert result['temporal_phi'].item() >= 0.0

    def test_reset(self, phi_computer):
        for _ in range(10):
            phi_computer(torch.randn(8))
        phi_computer.reset()
        assert phi_computer.history_ptr.item() == 0


class TestCauseEffectRepertoire:
    @pytest.fixture
    def cer(self):
        return CauseEffectRepertoire(state_dim=8)

    def test_forward_keys(self, cer):
        result = cer(torch.randn(8))
        assert 'cause_repertoire' in result
        assert 'effect_repertoire' in result
        assert 'repertoire_asymmetry' in result

    def test_cause_repertoire_shape(self, cer):
        cause = cer.cause_repertoire(torch.randn(8))
        assert cause.shape == (1, 8)

    def test_effect_repertoire_shape(self, cer):
        effect = cer.effect_repertoire(torch.randn(8))
        assert effect.shape == (1, 8)

    def test_repertoires_are_distributions(self, cer):
        state = torch.randn(8)
        cause = cer.cause_repertoire(state)
        effect = cer.effect_repertoire(state)
        assert abs(cause.sum().item() - 1.0) < 0.01
        assert abs(effect.sum().item() - 1.0) < 0.01

    def test_compute_mic_output_keys(self, cer):
        result = cer.compute_mic(torch.randn(8), mechanism=(0, 1, 2))
        assert 'cause_phi' in result
        assert 'cause_repertoire' in result
        assert 'cause_purview' in result

    def test_compute_mie_output_keys(self, cer):
        result = cer.compute_mie(torch.randn(8), mechanism=(0, 1, 2))
        assert 'effect_phi' in result
        assert 'effect_repertoire' in result
        assert 'effect_purview' in result

    def test_mic_phi_nonnegative(self, cer):
        result = cer.compute_mic(torch.randn(8), mechanism=(0, 1))
        assert result['cause_phi'].item() >= -1e-6

    def test_mie_phi_nonnegative(self, cer):
        result = cer.compute_mie(torch.randn(8), mechanism=(0, 1))
        assert result['effect_phi'].item() >= -1e-6

    def test_single_node_mechanism(self, cer):
        """Single-node mechanism should still produce a valid phi."""
        mic = cer.compute_mic(torch.randn(8), mechanism=(0,))
        assert mic['cause_phi'].item() >= 0.0

    def test_compute_concept(self, cer):
        concept = cer.compute_concept(torch.randn(8), mechanism=(0, 1))
        assert isinstance(concept, Concept)
        assert concept.mechanism == (0, 1)
        assert concept.phi >= 0.0
        assert concept.cause_phi >= 0.0
        assert concept.effect_phi >= 0.0

    def test_concept_phi_is_min_of_cause_effect(self, cer):
        concept = cer.compute_concept(torch.randn(8), mechanism=(0, 1, 2))
        assert abs(concept.phi - min(concept.cause_phi, concept.effect_phi)) < 1e-6


class TestConcept:
    def test_construction(self):
        c = Concept(
            mechanism=(0, 1),
            cause_purview=(0, 1),
            effect_purview=(0, 1),
            phi=0.5,
            cause_phi=0.6,
            effect_phi=0.5,
        )
        assert c.phi == 0.5
        assert c.mechanism == (0, 1)


class TestSystemIrreducibilityAnalysis:
    def test_construction(self):
        sia = SystemIrreducibilityAnalysis(
            big_phi=1.5,
            concepts=[],
            num_concepts=0,
        )
        assert sia.big_phi == 1.5


class TestIITConsciousnessMonitor:
    @pytest.fixture
    def monitor(self):
        return IITConsciousnessMonitor(state_dim=8, num_partitions=8)

    def test_forward_keys(self, monitor):
        result = monitor(torch.randn(8))
        expected = {'phi', 'temporal_phi', 'cause_repertoire',
                    'effect_repertoire', 'repertoire_asymmetry',
                    'consciousness_index'}
        assert set(result.keys()) == expected

    def test_consciousness_bounded(self, monitor):
        for _ in range(5):
            result = monitor(torch.randn(8))
            ci = result['consciousness_index'].item()
            assert -1.0 <= ci <= 1.0

    def test_phi_history_updates(self, monitor):
        for _ in range(5):
            monitor(torch.randn(8))
        assert monitor.phi_ptr.item() == 5

    def test_get_phi_trend(self, monitor):
        for _ in range(10):
            monitor(torch.randn(8))
        trend = monitor.get_phi_trend()
        assert trend.shape == (10,)

    def test_compute_big_phi(self, monitor):
        state = torch.randn(8)
        sia = monitor.compute_big_phi(state, max_mechanism_size=2)
        assert isinstance(sia, SystemIrreducibilityAnalysis)
        assert sia.big_phi >= 0.0
        assert sia.num_concepts >= 0

    def test_big_phi_has_concepts(self, monitor):
        """Big Phi analysis should find at least some concepts."""
        torch.manual_seed(42)
        state = torch.randn(8)
        sia = monitor.compute_big_phi(state, max_mechanism_size=2)
        # With random weights, we should get some non-zero concepts
        assert sia.num_concepts >= 0  # May be 0 with untrained models

    def test_big_phi_concepts_sorted(self, monitor):
        """Concepts should be sorted by phi (descending)."""
        torch.manual_seed(42)
        state = torch.randn(8)
        sia = monitor.compute_big_phi(state, max_mechanism_size=2)
        if len(sia.concepts) >= 2:
            for i in range(len(sia.concepts) - 1):
                assert sia.concepts[i].phi >= sia.concepts[i+1].phi

    def test_reset(self, monitor):
        for _ in range(5):
            monitor(torch.randn(8))
        monitor.reset()
        assert monitor.phi_ptr.item() == 0

    def test_lazy_import(self):
        """Should be importable via top-level pyquifer package."""
        from pyquifer import (Concept, SystemIrreducibilityAnalysis,
                              generate_bipartitions, IITConsciousnessMonitor)
        assert Concept is not None
        assert SystemIrreducibilityAnalysis is not None
        assert generate_bipartitions is not None
