"""Tests for basal_ganglia.py — BG-thalamocortical gating loop."""

import pytest
import torch
from pyquifer.basal_ganglia import BasalGangliaLoop, GatingOutput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_neuro_levels(da=0.5):
    """Neuromodulator levels: [DA, 5HT, NE, ACh, Cortisol]."""
    return torch.tensor([da, 0.5, 0.5, 0.5, 0.0])


def _make_loop(**kw):
    return BasalGangliaLoop(**kw)


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

class TestGoPathway:
    def test_high_da_strong_activation(self):
        """High DA + high salience → strong activation."""
        loop = _make_loop()
        result = loop.step(
            saliences=torch.tensor([0.9, 0.1]),
            costs=torch.tensor([0.0, 0.0]),
            tags_list=[{'external'}, {'internal'}],
            neuro_levels=_default_neuro_levels(da=1.0),
            coherence=0.5,
        )
        assert result.selected_channel == 0
        assert result.channel_activations[0] > result.channel_activations[1]
        assert result.thalamic_gate > 0.5


class TestNoGoPathway:
    def test_low_da_high_cost_suppressed(self):
        """Low DA + high cost → suppressed activation."""
        loop = _make_loop()
        result = loop.step(
            saliences=torch.tensor([0.3, 0.3]),
            costs=torch.tensor([0.9, 0.1]),
            tags_list=[{'external'}, {'internal'}],
            neuro_levels=_default_neuro_levels(da=0.0),
            coherence=0.5,
        )
        # Channel 0 has high cost with low DA → more NoGo → suppressed
        assert result.channel_activations[0] < result.channel_activations[1]
        assert result.selected_channel == 1


class TestDAModulation:
    def test_da_bias_go_vs_nogo(self):
        """DA=1.0 boosts Go, DA=0.0 boosts NoGo."""
        loop_hi = _make_loop()
        loop_lo = _make_loop()
        sal = torch.tensor([0.5])
        cost = torch.tensor([0.5])
        tags = [set()]

        result_hi = loop_hi.step(sal, cost, tags, _default_neuro_levels(da=1.0), 0.5)
        result_lo = loop_lo.step(sal, cost, tags, _default_neuro_levels(da=0.0), 0.5)

        # High DA → stronger Go → higher activation
        assert result_hi.channel_activations[0] > result_lo.channel_activations[0]
        assert result_hi.da_bias > result_lo.da_bias


class TestSTN:
    def test_activates_on_surprise(self):
        """High novelty + low coherence → STN active."""
        loop = _make_loop(stn_surprise_threshold=0.8, stn_coherence_threshold=0.3)
        result = loop.step(
            saliences=torch.tensor([0.9]),
            costs=torch.tensor([0.0]),
            tags_list=[set()],
            neuro_levels=_default_neuro_levels(),
            coherence=0.1,  # Low coherence
            novelty=0.95,   # High novelty
        )
        assert result.stn_active is True
        assert result.selected_channel == -1
        assert result.thalamic_gate == 0.0

    def test_refractory(self):
        """STN stays active for refractory period, then releases."""
        loop = _make_loop(stn_refractory_ticks=3)
        sal = torch.tensor([0.9])
        cost = torch.tensor([0.0])
        tags = [set()]
        nl = _default_neuro_levels()

        # Trigger STN
        r1 = loop.step(sal, cost, tags, nl, coherence=0.1, novelty=0.95)
        assert r1.stn_active is True

        # Refractory ticks (ticks 2 and 3 of 3 total)
        r2 = loop.step(sal, cost, tags, nl, coherence=0.5, novelty=0.0)
        assert r2.stn_active is True

        r3 = loop.step(sal, cost, tags, nl, coherence=0.5, novelty=0.0)
        assert r3.stn_active is True

        # After refractory, should release
        r4 = loop.step(sal, cost, tags, nl, coherence=0.5, novelty=0.0)
        assert r4.stn_active is False
        assert r4.selected_channel == 0

    def test_no_trigger_high_coherence(self):
        """High novelty BUT high coherence → no STN."""
        loop = _make_loop()
        result = loop.step(
            saliences=torch.tensor([0.9]),
            costs=torch.tensor([0.0]),
            tags_list=[set()],
            neuro_levels=_default_neuro_levels(),
            coherence=0.8,  # High coherence blocks STN
            novelty=0.95,
        )
        assert result.stn_active is False


class TestHysteresis:
    def test_incumbent_bonus(self):
        """Same-salience proposals → incumbent wins."""
        loop = _make_loop(switching_penalty=0.2)
        sal = torch.tensor([0.5, 0.5])
        cost = torch.tensor([0.0, 0.0])
        tags = [set(), set()]
        nl = _default_neuro_levels()

        # First step — arbitrary winner
        r1 = loop.step(sal, cost, tags, nl, 0.5)
        winner1 = r1.selected_channel

        # Second step — same saliences, incumbent should persist
        r2 = loop.step(sal, cost, tags, nl, 0.5)
        assert r2.selected_channel == winner1

    def test_switching_on_large_difference(self):
        """Much higher salience overcomes hysteresis."""
        loop = _make_loop(switching_penalty=0.1)
        nl = _default_neuro_levels()

        # Set incumbent to channel 0
        loop.step(
            torch.tensor([0.9, 0.1]),
            torch.tensor([0.0, 0.0]),
            [set(), set()], nl, 0.5,
        )
        assert loop._prev_winner == 0

        # Now channel 1 has much higher salience
        r2 = loop.step(
            torch.tensor([0.1, 0.9]),
            torch.tensor([0.0, 0.0]),
            [set(), set()], nl, 0.5,
        )
        assert r2.selected_channel == 1


class TestThalamicGate:
    def test_gate_strength(self):
        """Strong winner → gate near 1.0, weak → closer to 0.5."""
        loop = _make_loop()
        nl = _default_neuro_levels(da=1.0)

        # Strong winner: high salience, zero cost
        r_strong = loop.step(
            torch.tensor([2.0]), torch.tensor([0.0]),
            [set()], nl, 0.5,
        )
        # Weak winner: salience barely above tonic
        loop2 = _make_loop()
        r_weak = loop2.step(
            torch.tensor([0.2]), torch.tensor([0.0]),
            [set()], _default_neuro_levels(da=0.0), 0.5,
        )
        assert r_strong.thalamic_gate > r_weak.thalamic_gate


class TestModeBias:
    def test_external_perception(self):
        """External organ wins → perception bias."""
        loop = _make_loop()
        result = loop.step(
            saliences=torch.tensor([0.9, 0.1]),
            costs=torch.tensor([0.0, 0.0]),
            tags_list=[{'external'}, {'internal'}],
            neuro_levels=_default_neuro_levels(),
            coherence=0.5,
        )
        assert result.processing_mode_bias == 0  # perception


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------

class TestGatingInCycle:
    def test_gating_in_cycle(self):
        """Full tick with use_gating_loop=True, verify bg_* diagnostics."""
        from pyquifer.integration import CognitiveCycle, CycleConfig

        config = CycleConfig.small()
        config.use_gating_loop = True
        config.use_global_workspace = True
        cycle = CognitiveCycle(config)

        # Register a minimal organ so GW competition fires
        from pyquifer.organ import HPCOrgan
        organ = HPCOrgan(latent_dim=32)
        cycle.register_organ(organ)

        sensory = torch.randn(config.state_dim)
        result = cycle.tick(sensory, return_diagnostics=True)

        # BG fields should be in diagnostics
        d = result['diagnostics']
        assert 'bg_selected_channel' in d
        assert 'bg_stn_active' in d
        assert 'bg_thalamic_gate' in d

    def test_gating_disabled_no_overhead(self):
        """use_gating_loop=False → no bg_* in output."""
        from pyquifer.integration import CognitiveCycle, CycleConfig

        config = CycleConfig.small()
        config.use_gating_loop = False
        config.use_global_workspace = True
        cycle = CognitiveCycle(config)

        from pyquifer.organ import HPCOrgan
        organ = HPCOrgan(latent_dim=32)
        cycle.register_organ(organ)

        sensory = torch.randn(config.state_dim)
        result = cycle.tick(sensory, return_diagnostics=True)
        d = result['diagnostics']

        assert 'bg_selected_channel' not in d

    def test_gating_stn_overrides_mode(self):
        """STN active → processing mode forced to balanced."""
        loop = _make_loop(stn_surprise_threshold=0.1, stn_coherence_threshold=0.9)
        result = loop.step(
            saliences=torch.tensor([0.9]),
            costs=torch.tensor([0.0]),
            tags_list=[{'external'}],
            neuro_levels=_default_neuro_levels(),
            coherence=0.5,
            novelty=0.5,  # Above the low threshold
        )
        assert result.stn_active is True
        # When STN is active, mode bias should be balanced (2)
        assert result.processing_mode_bias == 2


class TestGatingOutput:
    def test_named_tuple_fields(self):
        """GatingOutput has all expected fields."""
        loop = _make_loop()
        result = loop.step(
            torch.tensor([0.5]), torch.tensor([0.1]),
            [set()], _default_neuro_levels(), 0.5,
        )
        assert isinstance(result, GatingOutput)
        assert hasattr(result, 'selected_channel')
        assert hasattr(result, 'channel_activations')
        assert hasattr(result, 'salience_prior')
        assert hasattr(result, 'thalamic_gate')
        assert hasattr(result, 'processing_mode_bias')
        assert hasattr(result, 'stn_active')
        assert hasattr(result, 'da_bias')
        assert hasattr(result, 'switching_cost')

    def test_reset_state(self):
        """reset_state clears winner history and refractory."""
        loop = _make_loop()
        loop._prev_winner = 3
        loop._stn_refractory_counter = 5
        loop.reset_state()
        assert loop._prev_winner == -1
        assert loop._stn_refractory_counter == 0
