"""
Phase 5 Integration Tests: CognitiveCycle wiring for Phase 5 modules.

Tests that new optional modules (STP, mean-field, Stuart-Landau, Koopman,
Wilson-Cowan) wire correctly into CognitiveCycle when enabled.
"""

import torch
import pytest

from pyquifer.integration import CycleConfig, CognitiveCycle


def _make_config(**overrides):
    """Small config for fast integration tests."""
    defaults = dict(
        state_dim=32, belief_dim=16, semantic_dim=8,
        num_oscillators=16, hierarchy_dims=[32, 16, 8],
        num_populations=4, num_groups=4, group_dim=8,
        internal_dim=16, sensory_dim=8, active_dim=8,
        episodic_capacity=200, semantic_slots=50,
        volatility_base_lr=0.01, volatility_min_lr=0.001,
        volatility_max_lr=0.1,
    )
    defaults.update(overrides)
    return CycleConfig(**defaults)


class TestBaselineNoPhase5:
    """Verify CognitiveCycle still works with all Phase 5 flags off."""

    def test_tick_runs(self):
        config = _make_config()
        cycle = CognitiveCycle(config)
        result = cycle.tick(torch.randn(config.state_dim))
        assert 'modulation' in result
        assert 'consciousness' in result
        assert 'diagnostics' in result

    def test_no_phase5_keys_when_disabled(self):
        config = _make_config()
        cycle = CognitiveCycle(config)
        result = cycle.tick(torch.randn(config.state_dim))
        diag = result['diagnostics']
        # Phase 5 keys should NOT be present
        assert 'sl_order_parameter' not in diag
        assert 'mf_R' not in diag
        assert 'stability_margin' not in diag
        assert 'stp_psp' not in diag
        assert 'nm_synchronization' not in diag


class TestStuartLandauIntegration:
    """5D-1 Stuart-Landau wired into CognitiveCycle."""

    def test_stuart_landau_diagnostics(self):
        config = _make_config(use_stuart_landau=True)
        cycle = CognitiveCycle(config)
        result = cycle.tick(torch.randn(config.state_dim))
        diag = result['diagnostics']
        assert 'sl_order_parameter' in diag
        assert 'criticality_distance_sl' in diag
        assert 'amplitudes' in diag
        assert diag['amplitudes'].shape == (config.num_oscillators,)

    def test_reset_resets_stuart_landau(self):
        config = _make_config(use_stuart_landau=True)
        cycle = CognitiveCycle(config)
        cycle.tick(torch.randn(config.state_dim))
        cycle.reset()
        assert cycle._stuart_landau.step_count.item() == 0


class TestMeanFieldIntegration:
    """5B-2 Kuramoto-Daido mean-field wired into CognitiveCycle."""

    def test_mean_field_diagnostics(self):
        config = _make_config(use_mean_field=True)
        cycle = CognitiveCycle(config)
        result = cycle.tick(torch.randn(config.state_dim))
        diag = result['diagnostics']
        assert 'mf_R' in diag
        assert 'mf_Psi' in diag
        assert 'mf_synchronized' in diag


class TestKoopmanIntegration:
    """5D-2 Koopman bifurcation detection wired into CognitiveCycle."""

    def test_koopman_diagnostics(self):
        config = _make_config(use_koopman=True)
        cycle = CognitiveCycle(config)
        result = cycle.tick(torch.randn(config.state_dim))
        diag = result['diagnostics']
        assert 'stability_margin' in diag
        assert 'max_eigenvalue_mag' in diag
        assert 'approaching_bifurcation' in diag

    def test_koopman_reset(self):
        config = _make_config(use_koopman=True)
        cycle = CognitiveCycle(config)
        cycle.tick(torch.randn(config.state_dim))
        cycle.reset()
        assert cycle._koopman.hist_ptr.item() == 0


class TestSTPIntegration:
    """5B-1 Tsodyks-Markram STP wired into CognitiveCycle."""

    def test_stp_diagnostics(self):
        config = _make_config(use_stp=True)
        cycle = CognitiveCycle(config)
        result = cycle.tick(torch.randn(config.state_dim))
        diag = result['diagnostics']
        assert 'stp_psp' in diag
        assert 'mean_facilitation' in diag
        assert 'mean_depression' in diag


class TestNeuralMassIntegration:
    """5D-3 Wilson-Cowan neural mass wired into CognitiveCycle."""

    def test_neural_mass_diagnostics(self):
        config = _make_config(use_neural_mass=True)
        cycle = CognitiveCycle(config)
        result = cycle.tick(torch.randn(config.state_dim))
        diag = result['diagnostics']
        assert 'E_states' in diag
        assert 'I_states' in diag
        assert 'nm_synchronization' in diag
        assert diag['E_states'].shape == (config.num_populations,)

    def test_neural_mass_reset(self):
        config = _make_config(use_neural_mass=True)
        cycle = CognitiveCycle(config)
        cycle.tick(torch.randn(config.state_dim))
        cycle.reset()
        for pop in cycle._neural_mass.populations:
            assert pop.E.item() == pytest.approx(0.1)


class TestAllPhase5Enabled:
    """All Phase 5 modules enabled simultaneously."""

    def test_all_enabled_tick(self):
        config = _make_config(
            use_stp=True, use_mean_field=True, use_stuart_landau=True,
            use_koopman=True, use_neural_mass=True,
        )
        cycle = CognitiveCycle(config)
        result = cycle.tick(torch.randn(config.state_dim))
        diag = result['diagnostics']
        # All Phase 5 diagnostics present
        assert 'sl_order_parameter' in diag
        assert 'mf_R' in diag
        assert 'stability_margin' in diag
        assert 'stp_psp' in diag
        assert 'nm_synchronization' in diag

    def test_all_enabled_multi_tick(self):
        config = _make_config(
            use_stp=True, use_mean_field=True, use_stuart_landau=True,
            use_koopman=True, use_neural_mass=True,
        )
        cycle = CognitiveCycle(config)
        for _ in range(10):
            result = cycle.tick(torch.randn(config.state_dim), reward=0.1)
        assert result['diagnostics']['tick'] == 10
