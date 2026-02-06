"""Tests for metastability module."""
import torch
import pytest
from pyquifer.metastability import WinnerlessCompetition, HeteroclinicChannel, MetastabilityIndex


class TestWinnerlessCompetition:
    def test_construction(self):
        wlc = WinnerlessCompetition(num_populations=6)
        assert wlc.activations.shape == (6,)

    def test_visits_multiple_states(self):
        torch.manual_seed(42)
        wlc = WinnerlessCompetition(num_populations=6, noise_scale=0.03)
        dominants = set()
        for _ in range(1000):
            r = wlc()
            dominants.add(r['dominant'].item())
        assert len(dominants) >= 3

    def test_activations_positive_normalized(self):
        wlc = WinnerlessCompetition(num_populations=4)
        for _ in range(100):
            r = wlc()
        assert (r['activations'] > 0).all()
        assert abs(r['activations'].sum().item() - 1.0) < 0.01

    def test_asymmetric_inhibition(self):
        wlc = WinnerlessCompetition(num_populations=6)
        assert not torch.allclose(wlc.rho, wlc.rho.t())

    def test_external_input_biases(self):
        wlc = WinnerlessCompetition(num_populations=4, noise_scale=0.01)
        bias = torch.tensor([5.0, 0.0, 0.0, 0.0])
        for _ in range(200):
            r = wlc(external_input=bias)
        assert r['activations'][0] > r['activations'][1:].mean()

    def test_reset(self):
        wlc = WinnerlessCompetition(num_populations=4)
        for _ in range(50):
            wlc()
        wlc.reset()
        assert wlc.activations.std().item() < 0.001


class TestHeteroclinicChannel:
    def test_detects_transitions(self):
        torch.manual_seed(42)
        wlc = WinnerlessCompetition(num_populations=4, noise_scale=0.03)
        hc = HeteroclinicChannel(num_states=4)
        transitions = 0
        for _ in range(1000):
            comp = wlc()
            flow = hc(comp['activations'])
            if flow['transition_occurred'].item():
                transitions += 1
        assert transitions > 0

    def test_transition_matrix(self):
        torch.manual_seed(42)
        wlc = WinnerlessCompetition(num_populations=4, noise_scale=0.03)
        hc = HeteroclinicChannel(num_states=4)
        for _ in range(1000):
            hc(wlc()['activations'])
        r = hc(wlc()['activations'])
        assert r['transition_matrix'].sum().item() > 0


class TestMetastabilityIndex:
    def test_coalition_entropy_positive(self):
        torch.manual_seed(42)
        mi = MetastabilityIndex(num_populations=6)
        for _ in range(500):
            r = mi()
        assert r['coalition_entropy'].item() > 0

    def test_output_keys(self):
        mi = MetastabilityIndex(num_populations=4)
        for _ in range(100):
            r = mi()
        assert all(k in r for k in ['activations', 'dominant', 'metastability_index',
                                      'coalition_entropy', 'chimera_index'])

    def test_reset(self):
        mi = MetastabilityIndex(num_populations=4)
        for _ in range(50):
            mi()
        mi.reset()
        assert mi.history_ptr.item() == 0
