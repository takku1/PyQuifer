"""Tests for self_model module."""
import torch
import math
import pytest
from pyquifer.self_model import MarkovBlanket, SelfModel, NarrativeIdentity


class TestMarkovBlanket:
    def test_construction(self):
        mb = MarkovBlanket(internal_dim=16, sensory_dim=8, active_dim=8)
        assert mb.internal_state.shape == (16,)

    def test_output_keys(self):
        mb = MarkovBlanket(internal_dim=16, sensory_dim=8, active_dim=8)
        r = mb(torch.randn(8))
        assert all(k in r for k in ['internal_state', 'active_output', 'blanket_state'])

    def test_blanket_dim(self):
        mb = MarkovBlanket(internal_dim=16, sensory_dim=8, active_dim=4)
        r = mb(torch.randn(8))
        assert r['blanket_state'].shape[-1] == 12  # 8 + 4

    def test_batch_support(self):
        mb = MarkovBlanket(internal_dim=16, sensory_dim=8, active_dim=8)
        r = mb(torch.randn(4, 8))
        assert r['active_output'].shape == (4, 8)

    def test_internal_state_persists(self):
        mb = MarkovBlanket(internal_dim=16, sensory_dim=8, active_dim=8)
        mb(torch.randn(8))
        assert mb.internal_state.norm().item() > 0

    def test_reset(self):
        mb = MarkovBlanket(internal_dim=16, sensory_dim=8, active_dim=8)
        mb(torch.randn(8))
        mb.reset()
        assert mb.internal_state.abs().sum().item() == 0


class TestSelfModel:
    def test_construction(self):
        sm = SelfModel(self_dim=16)
        assert sm.self_state.shape == (16,)

    def test_with_components(self):
        sm = SelfModel(self_dim=16, body_dim=4, personality_dim=3, capability_dim=2)
        r = sm(torch.randn(16), body_state=torch.randn(4),
               personality_state=torch.randn(3), capability_state=torch.randn(2))
        assert r['self_summary'].shape == (16,)

    def test_prediction_error_stabilizes(self):
        sm = SelfModel(self_dim=16, lr=0.1)
        errors = []
        for i in range(100):
            t = i * 0.1
            summary = torch.sin(torch.linspace(0, math.pi, 16) + t) * 0.3
            r = sm(summary)
            errors.append(r['self_prediction_error_magnitude'].item())
        early = sum(errors[:20]) / 20
        late = sum(errors[-20:]) / 20
        assert late < early * 2

    def test_reset(self):
        sm = SelfModel(self_dim=8)
        sm(torch.randn(8))
        sm.reset()
        assert sm.self_state.abs().sum().item() == 0


class TestNarrativeIdentity:
    def test_converges_to_stable_self(self):
        ni = NarrativeIdentity(dim=8, tau=0.01)
        for _ in range(200):
            ni(torch.ones(8) * 0.5 + torch.randn(8) * 0.01)
        r = ni(torch.ones(8) * 0.5)
        assert r['deviation'].item() < 0.5

    def test_resists_sudden_change(self):
        ni = NarrativeIdentity(dim=8, tau=0.01, consistency_weight=0.1)
        for _ in range(200):
            ni(torch.ones(8) * 0.5 + torch.randn(8) * 0.01)
        normal = ni(torch.ones(8) * 0.5)
        shifted = ni(-torch.ones(8) * 0.5)
        assert shifted['consistency_loss'] > normal['consistency_loss']

    def test_identity_strength_grows(self):
        ni = NarrativeIdentity(dim=4, tau=0.1)
        for _ in range(100):
            ni(torch.randn(4))
        r = ni(torch.randn(4))
        assert r['identity_strength'].item() > 0

    def test_reset(self):
        ni = NarrativeIdentity(dim=4)
        for _ in range(50):
            ni(torch.randn(4))
        ni.reset()
        assert ni.identity_strength.item() == 0
        assert ni.narrative.abs().sum().item() == 0


class TestTonicDrift:
    """Tests for tonic drift (developmental trajectories) in NarrativeIdentity."""

    def test_default_no_drift(self):
        """Without drift, tonic_drift buffer should be zeros."""
        ni = NarrativeIdentity(dim=4)
        assert torch.allclose(ni.tonic_drift, torch.zeros(4))

    def test_custom_drift(self):
        """Can initialize with a specific drift vector."""
        drift = torch.tensor([0.01, -0.01, 0.0, 0.005])
        ni = NarrativeIdentity(dim=4, tonic_drift=drift)
        assert torch.allclose(ni.tonic_drift, drift)

    def test_drift_moves_narrative(self):
        """With positive drift, narrative should move in that direction even with zero input."""
        drift = torch.tensor([0.01, 0.0, 0.0, 0.0])
        ni = NarrativeIdentity(dim=4, tau=0.0, tonic_drift=drift)
        # tau=0 means no EMA update from input, only drift
        initial = ni.narrative.clone()
        for _ in range(100):
            ni(torch.zeros(4))
        # Dim 0 should have drifted positively
        assert ni.narrative[0].item() > initial[0].item() + 0.5
        # Dim 1 should not have moved much
        assert abs(ni.narrative[1].item() - initial[1].item()) < 0.01

    def test_drift_direction_matters(self):
        """Negative drift should move narrative in negative direction."""
        drift = torch.tensor([0.0, -0.005, 0.0, 0.0])
        ni = NarrativeIdentity(dim=4, tau=0.0, tonic_drift=drift)
        for _ in range(100):
            ni(torch.zeros(4))
        assert ni.narrative[1].item() < -0.3

    def test_drift_plus_ema(self):
        """Drift should combine with normal EMA update."""
        drift = torch.tensor([0.01, 0.0, 0.0, 0.0])
        ni = NarrativeIdentity(dim=4, tau=0.01, tonic_drift=drift)
        # Feed input pulling toward -1 on dim 0
        for _ in range(200):
            ni(torch.tensor([-1.0, 0.0, 0.0, 0.0]))
        # EMA pulls toward -1, drift pulls toward +inf
        # Result should be somewhere in between (not purely -1)
        val = ni.narrative[0].item()
        assert val > -1.0  # drift prevented full convergence to -1

    def test_set_tonic_drift(self):
        """Can change drift after construction."""
        ni = NarrativeIdentity(dim=4)
        assert ni.tonic_drift.abs().sum().item() == 0.0
        new_drift = torch.tensor([0.002, -0.001, 0.0, 0.003])
        ni.set_tonic_drift(new_drift)
        assert torch.allclose(ni.tonic_drift, new_drift)

    def test_get_projected_identity(self):
        """Should project identity forward using drift."""
        drift = torch.tensor([0.01, -0.01, 0.0, 0.0])
        ni = NarrativeIdentity(dim=4, tonic_drift=drift)
        # Set narrative to known state
        ni.narrative.fill_(0.5)
        projected = ni.get_projected_identity(steps=100)
        assert projected[0].item() == pytest.approx(0.5 + 0.01 * 100, abs=0.01)
        assert projected[1].item() == pytest.approx(0.5 - 0.01 * 100, abs=0.01)

    def test_drift_does_not_affect_reset(self):
        """Reset should zero narrative but keep drift."""
        drift = torch.tensor([0.01, 0.0, 0.0, 0.0])
        ni = NarrativeIdentity(dim=4, tonic_drift=drift)
        for _ in range(50):
            ni(torch.randn(4))
        ni.reset()
        assert ni.narrative.abs().sum().item() == 0.0
        # Drift should persist
        assert torch.allclose(ni.tonic_drift, drift)
