"""Tests for Organ protocol and GWT wiring."""
import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class TestOrganABC:
    """Test the abstract Organ interface."""

    def test_organ_is_abstract(self):
        from pyquifer.organ import Organ
        with pytest.raises(TypeError):
            Organ("test", 64)

    def test_proposal_dataclass(self):
        from pyquifer.organ import Proposal
        p = Proposal(
            content=torch.randn(64),
            salience=0.5,
            tags={"test"},
            organ_id="dummy",
        )
        assert p.salience == 0.5
        assert "test" in p.tags
        assert p.content.shape == (64,)


class TestOscillatoryWriteGate:
    def test_gate_in_phase(self):
        from pyquifer.organ import OscillatoryWriteGate
        gate = OscillatoryWriteGate()
        # In-phase: cos(0) = 1 → high gate
        val = gate(torch.tensor(0.0), torch.tensor(0.0))
        assert val.item() > 0.5

    def test_gate_anti_phase(self):
        from pyquifer.organ import OscillatoryWriteGate
        gate = OscillatoryWriteGate()
        import math
        # Anti-phase: cos(pi) = -1 → low gate
        val = gate(torch.tensor(0.0), torch.tensor(math.pi))
        assert val.item() < 0.5

    def test_gate_differentiable(self):
        from pyquifer.organ import OscillatoryWriteGate
        gate = OscillatoryWriteGate()
        phase = torch.tensor(0.5, requires_grad=True)
        val = gate(phase, torch.tensor(0.0))
        val.backward()
        assert phase.grad is not None


class TestPreGWAdapter:
    def test_encode_decode(self):
        from pyquifer.organ import PreGWAdapter
        adapter = PreGWAdapter(organ_dim=32, workspace_dim=64)
        x = torch.randn(32)
        encoded = adapter.encode(x)
        assert encoded.shape == (64,)
        decoded = adapter.decode(encoded)
        assert decoded.shape == (32,)

    def test_cycle_consistency_loss(self):
        from pyquifer.organ import PreGWAdapter
        adapter = PreGWAdapter(organ_dim=32, workspace_dim=64)
        x = torch.randn(32)
        loss = adapter.cycle_consistency_loss(x)
        assert loss.item() >= 0
        assert loss.requires_grad


class TestHPCOrgan:
    def test_observe_propose_accept(self):
        from pyquifer.organ import HPCOrgan
        organ = HPCOrgan(latent_dim=64)
        sensory = torch.randn(64)
        organ.observe(sensory)
        proposal = organ.propose()
        assert proposal.organ_id == "hpc"
        assert proposal.content.shape == (64,)
        assert proposal.salience >= 0

        # Accept broadcast
        broadcast = torch.randn(64)
        organ.accept(broadcast)

    def test_oscillator_step(self):
        from pyquifer.organ import HPCOrgan
        organ = HPCOrgan(latent_dim=64)
        initial_phase = organ.phase.item()
        organ.step_oscillator(dt=0.01)
        assert organ.phase.item() != initial_phase


class TestMotivationOrgan:
    def test_observe_propose(self):
        from pyquifer.organ import MotivationOrgan
        organ = MotivationOrgan(latent_dim=64)
        sensory = torch.randn(64)
        organ.observe(sensory)
        proposal = organ.propose()
        assert proposal.organ_id == "motivation"
        assert "motivation" in proposal.tags


class TestSelectionOrgan:
    def test_observe_propose(self):
        from pyquifer.organ import SelectionOrgan
        organ = SelectionOrgan(latent_dim=64, num_groups=4, group_dim=16)
        sensory = torch.randn(64)
        organ.observe(sensory)
        proposal = organ.propose()
        assert proposal.organ_id == "selection"
        assert "selection" in proposal.tags


class TestGWTIntegration:
    """Test GWT wiring into CognitiveCycle."""

    def test_register_organ(self):
        from pyquifer.integration import CognitiveCycle, CycleConfig
        from pyquifer.organ import HPCOrgan
        config = CycleConfig.small()
        config.use_global_workspace = True
        config.workspace_dim = 64
        cycle = CognitiveCycle(config)
        organ = HPCOrgan(latent_dim=64)
        cycle.register_organ(organ)
        assert len(cycle._organs) == 1

    def test_tick_with_gw(self):
        from pyquifer.integration import CognitiveCycle, CycleConfig
        from pyquifer.organ import HPCOrgan, MotivationOrgan
        config = CycleConfig.small()
        config.use_global_workspace = True
        config.workspace_dim = 64
        cycle = CognitiveCycle(config)
        cycle.register_organ(HPCOrgan(latent_dim=64))
        cycle.register_organ(MotivationOrgan(latent_dim=64))
        result = cycle.tick(torch.randn(config.state_dim))
        assert 'gw_broadcast' in result['diagnostics']

    def test_tick_without_gw_unchanged(self):
        """Existing behavior unaffected when GW disabled."""
        from pyquifer.integration import CognitiveCycle, CycleConfig
        config = CycleConfig.small()
        config.use_global_workspace = False
        cycle = CognitiveCycle(config)
        result = cycle.tick(torch.randn(config.state_dim))
        assert 'gw_broadcast' not in result['diagnostics']

    def test_diversity_pressure(self):
        """Organs that haven't won get salience boost."""
        from pyquifer.global_workspace import DiversityTracker
        tracker = DiversityTracker(pressure=0.2, window_size=10)
        # Record wins for organ A but not B
        for _ in range(5):
            tracker.record_win("organ_a")
        boost_a = tracker.get_boost("organ_a")
        boost_b = tracker.get_boost("organ_b")
        assert boost_b > boost_a
