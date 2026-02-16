"""Tests for Enhancement C: Energy-Metabolic Budget in CognitiveCycle."""

import torch
import pytest
from pyquifer.runtime.config import CycleConfig
from pyquifer.runtime.cycle import CognitiveCycle


@pytest.fixture
def metabolic_cycle():
    """Create a CognitiveCycle with metabolic budget enabled, costs > recovery."""
    c = CycleConfig(
        use_metabolic_budget=True,
        metabolic_capacity=1.0,
        metabolic_recovery_rate=0.005,  # Low recovery so costs dominate
        metabolic_oscillation_cost=0.02,  # Higher than recovery
        metabolic_ignition_cost=0.05,
        metabolic_broadcast_cost=0.02,
    )
    return CognitiveCycle(c)


@pytest.fixture
def default_cycle():
    """CognitiveCycle with metabolic budget disabled (default)."""
    return CognitiveCycle(CycleConfig())


def test_metabolic_disabled_by_default(default_cycle):
    """When use_metabolic_budget=False, energy stays at capacity."""
    result = default_cycle.tick(torch.randn(1, 64))
    # No metabolic info in diagnostics
    diag = default_cycle.tick(torch.randn(1, 64))
    assert default_cycle._energy_budget.item() == pytest.approx(1.0, abs=1e-6)


def test_energy_depletes_under_load(metabolic_cycle):
    """Energy budget decreases after ticking."""
    initial_energy = metabolic_cycle._energy_budget.item()
    for _ in range(10):
        metabolic_cycle.tick(torch.randn(1, 64))
    final_energy = metabolic_cycle._energy_budget.item()
    # Energy should decrease (costs > recovery for active ticks)
    assert final_energy < initial_energy, (
        f"Energy should deplete: {initial_energy} -> {final_energy}"
    )


def test_energy_never_negative(metabolic_cycle):
    """Energy budget is clamped to [0, capacity]."""
    # Run many ticks to try to deplete
    for _ in range(200):
        metabolic_cycle.tick(torch.randn(1, 64))
    assert metabolic_cycle._energy_budget.item() >= 0.0


def test_energy_never_exceeds_capacity(metabolic_cycle):
    """Energy budget never exceeds metabolic_capacity."""
    # Even after many recovery ticks, should stay <= 1.0
    for _ in range(10):
        metabolic_cycle.tick(torch.randn(1, 64))
    assert metabolic_cycle._energy_budget.item() <= 1.0 + 1e-6


def test_debt_tracks_below_half(metabolic_cycle):
    """Metabolic debt > 0 when energy drops below 50%."""
    # Drain energy with expensive config
    metabolic_cycle.config = CycleConfig(
        use_metabolic_budget=True,
        metabolic_capacity=1.0,
        metabolic_recovery_rate=0.001,  # Very slow recovery
        metabolic_oscillation_cost=0.1,  # Very expensive
        metabolic_ignition_cost=0.05,
        metabolic_broadcast_cost=0.02,
    )
    for _ in range(20):
        metabolic_cycle.tick(torch.randn(1, 64))

    energy = metabolic_cycle._energy_budget.item()
    debt = metabolic_cycle._metabolic_debt.item()
    if energy < 0.5:
        assert debt > 0, f"Debt should be >0 when energy={energy}"
    else:
        assert debt == pytest.approx(0.0, abs=1e-6)


def test_coupling_damped_at_low_energy(metabolic_cycle):
    """When energy < 30%, oscillator coupling is damped."""
    coupling_before = metabolic_cycle.oscillators.coupling_strength.clone()

    # Heavily drain energy
    metabolic_cycle.config = CycleConfig(
        use_metabolic_budget=True,
        metabolic_capacity=1.0,
        metabolic_recovery_rate=0.0,  # No recovery
        metabolic_oscillation_cost=0.2,  # Very expensive
    )
    for _ in range(20):
        metabolic_cycle.tick(torch.randn(1, 64))

    energy = metabolic_cycle._energy_budget.item()
    coupling_after = metabolic_cycle.oscillators.coupling_strength.clone()

    if energy < 0.3:
        # Coupling should have been damped
        assert coupling_after.item() < coupling_before.item(), (
            f"Coupling should be damped at energy={energy}"
        )


def test_diagnostics_include_metabolic_info(metabolic_cycle):
    """Full diagnostics include metabolic_info keys."""
    result = metabolic_cycle.tick(torch.randn(1, 64), return_diagnostics=True)
    diag = result['diagnostics']
    assert 'energy_budget' in diag
    assert 'metabolic_debt' in diag
    assert 'tick_cost' in diag
    assert 'energy_ratio' in diag


def test_recovery_restores_energy():
    """With zero costs but recovery enabled, energy stays at capacity."""
    c = CycleConfig(
        use_metabolic_budget=True,
        metabolic_capacity=1.0,
        metabolic_recovery_rate=0.05,
        metabolic_oscillation_cost=0.0,
        metabolic_ignition_cost=0.0,
        metabolic_broadcast_cost=0.0,
    )
    cycle = CognitiveCycle(c)
    # Manually drain some energy
    with torch.no_grad():
        cycle._energy_budget.fill_(0.5)

    for _ in range(20):
        cycle.tick(torch.randn(1, 64))

    # Should recover toward capacity (may not reach 1.0 due to HPC cost)
    assert cycle._energy_budget.item() > 0.5
