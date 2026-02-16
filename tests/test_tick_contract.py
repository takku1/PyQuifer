"""Tests for tick() input/output contract (PRs 1-2).

Validates:
- Input shape validation: 1D ok, (1,D) squeeze, (2,D) reject, 3D reject
- Minimal tick returns TickResult (tensor-only NamedTuple)
- Diagnostic tick returns full dict (backward compat)
- Bridge unpacks TickResult correctly
"""

import pytest
import torch
from pyquifer.runtime.cycle import CognitiveCycle
from pyquifer.runtime.config import CycleConfig
from pyquifer.runtime.tick_result import TickResult


@pytest.fixture
def cycle():
    """Small cycle for fast tests."""
    return CognitiveCycle(CycleConfig.small())


@pytest.fixture
def state_dim():
    return CycleConfig.small().state_dim


class TestTickInputContract:
    """PR 1: tick() input shape validation."""

    def test_1d_input_accepted(self, cycle, state_dim):
        """Standard 1D input should work."""
        x = torch.randn(state_dim)
        result = cycle.tick(x, return_diagnostics=False)
        assert isinstance(result, TickResult)

    def test_1_by_d_squeezed(self, cycle, state_dim):
        """(1, D) input should be auto-squeezed to (D,)."""
        x = torch.randn(1, state_dim)
        result = cycle.tick(x, return_diagnostics=False)
        assert isinstance(result, TickResult)

    def test_batch_2d_rejected(self, cycle, state_dim):
        """(2, D) batched input should raise ValueError."""
        x = torch.randn(2, state_dim)
        with pytest.raises(ValueError, match="does not support batched"):
            cycle.tick(x)

    def test_3d_rejected(self, cycle, state_dim):
        """(1, 1, D) 3D input should raise ValueError."""
        x = torch.randn(1, 1, state_dim)
        with pytest.raises(ValueError, match="expects 1-D input"):
            cycle.tick(x)

    def test_scalar_rejected(self, cycle):
        """0D scalar input should raise ValueError."""
        x = torch.tensor(1.0)
        with pytest.raises(ValueError, match="expects 1-D input"):
            cycle.tick(x)

    def test_4d_rejected(self, cycle, state_dim):
        """4D input should raise ValueError."""
        x = torch.randn(1, 1, 1, state_dim)
        with pytest.raises(ValueError, match="expects 1-D input"):
            cycle.tick(x)


class TestTickOutputContract:
    """PR 2: minimal tick returns TickResult (tensor-only)."""

    def test_minimal_returns_tick_result(self, cycle, state_dim):
        """Minimal tick returns TickResult NamedTuple."""
        x = torch.randn(state_dim)
        result = cycle.tick(x, return_diagnostics=False)
        assert isinstance(result, TickResult)

    def test_tick_result_all_tensors(self, cycle, state_dim):
        """All TickResult fields must be tensors (no Python strings/ints)."""
        x = torch.randn(state_dim)
        result = cycle.tick(x, return_diagnostics=False)
        for field_name in TickResult._fields:
            val = getattr(result, field_name)
            assert isinstance(val, torch.Tensor), (
                f"TickResult.{field_name} is {type(val).__name__}, expected Tensor"
            )

    def test_tick_result_field_shapes(self, cycle, state_dim):
        """Verify expected shapes of TickResult fields."""
        x = torch.randn(state_dim)
        result = cycle.tick(x, return_diagnostics=False)
        # Scalar fields
        assert result.temperature.dim() == 0
        assert result.coherence.dim() == 0
        assert result.motivation.dim() == 0
        assert result.sleep_signal.dim() == 0
        assert result.processing_mode.dim() == 0  # int tensor
        assert result.dominant_state.dim() == 0    # int tensor
        # Vector fields
        assert result.attention_bias.dim() == 1
        assert result.personality_blend.dim() == 1

    def test_diagnostic_returns_dict(self, cycle, state_dim):
        """Diagnostic tick still returns full dict for backward compat."""
        x = torch.randn(state_dim)
        result = cycle.tick(x, return_diagnostics=True)
        assert isinstance(result, dict)
        assert 'modulation' in result
        assert 'consciousness' in result
        assert 'diagnostics' in result

    def test_tick_result_deterministic(self, cycle, state_dim):
        """Two ticks with same input should produce consistent field types."""
        x = torch.randn(state_dim)
        r1 = cycle.tick(x, return_diagnostics=False)
        r2 = cycle.tick(x, return_diagnostics=False)
        assert type(r1) == type(r2)
        for field_name in TickResult._fields:
            v1 = getattr(r1, field_name)
            v2 = getattr(r2, field_name)
            assert v1.dtype == v2.dtype, f"{field_name} dtype mismatch"


class TestBridgeUnpacksTickResult:
    """PR 2-3: Bridge correctly unpacks TickResult from minimal tick."""

    def test_bridge_step_returns_modulation_state(self):
        """Bridge.step() should return ModulationState from minimal tick."""
        from pyquifer.api.bridge import PyQuiferBridge, ModulationState
        bridge = PyQuiferBridge(CycleConfig.small())
        state_dim = bridge.config.state_dim
        x = torch.randn(state_dim)
        result = bridge.step(x)
        assert isinstance(result, ModulationState)
        assert isinstance(result.temperature, float)
        assert isinstance(result.coherence, float)
        assert isinstance(result.processing_mode, str)

    def test_bridge_step_diagnostic(self):
        """Bridge.step_diagnostic() should return both ModulationState + diagnostics."""
        from pyquifer.api.bridge import PyQuiferBridge, ModulationState
        bridge = PyQuiferBridge(CycleConfig.small())
        x = torch.randn(bridge.config.state_dim)
        mod_state, diagnostics = bridge.step_diagnostic(x)
        assert isinstance(mod_state, ModulationState)
        assert isinstance(diagnostics, dict)
        assert 'consciousness' in diagnostics
        assert 'diagnostics' in diagnostics
