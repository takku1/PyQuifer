"""Tests for PhaseLockBus — multimodal phase-locking coordinator."""

import pytest
import torch
from pyquifer.phase_lock_bus import PhaseLockBus, BusConfig, BusOutput
from pyquifer.sensory_binding import MultimodalBinder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bus(modality_dims=None, **kwargs):
    dims = modality_dims or {"text": 32}
    cfg = BusConfig(modality_dims=dims, binding_dim=32, num_oscillators_per_modality=8,
                    num_sync_steps=5, **kwargs)
    return PhaseLockBus(cfg)


# ---------------------------------------------------------------------------
# Unit Tests (1-12)
# ---------------------------------------------------------------------------

class TestPhaseLockBusUnit:

    def test_single_modality_shapes(self):
        """Text-only: fused_representation = (1, binding_dim), binding_matrix = (1,1)."""
        bus = _make_bus({"text": 32})
        out = bus({"text": torch.randn(1, 32)})
        assert out.fused_representation.shape == (1, 32)
        assert out.binding_matrix.shape == (1, 1)

    def test_single_modality_neutral_coherence(self):
        """1 modality → mean_coherence = 0.5 (neutral)."""
        bus = _make_bus({"text": 32})
        out = bus({"text": torch.randn(1, 32)})
        assert out.mean_coherence.item() == pytest.approx(0.5, abs=1e-6)

    def test_two_modalities_shapes(self):
        """text + audio: binding_matrix = (2,2), fused output correct."""
        bus = _make_bus({"text": 32, "audio": 16})
        out = bus({"text": torch.randn(1, 32), "audio": torch.randn(1, 16)})
        assert out.fused_representation.shape == (1, 32)
        assert out.binding_matrix.shape == (2, 2)
        assert out.modality_count == 2

    def test_four_modalities(self):
        """text/audio/vision/tool, verify all BusOutput fields present."""
        dims = {"audio": 16, "text": 32, "tool": 8, "vision": 24}
        bus = _make_bus(dims)
        inputs = {k: torch.randn(1, v) for k, v in dims.items()}
        out = bus(inputs)
        assert isinstance(out, BusOutput)
        assert out.binding_matrix.shape == (4, 4)
        assert out.modality_count == 4
        assert len(out.per_modality_phases) == 4

    def test_missing_modality_graceful(self):
        """Register 3 modalities, pass only 2, no crash."""
        bus = _make_bus({"text": 32, "audio": 16, "vision": 24})
        out = bus({"text": torch.randn(1, 32), "audio": torch.randn(1, 16)})
        assert out.fused_representation.shape == (1, 32)
        # vision was missing → zeros injected by MultimodalBinder

    def test_coherence_ema_updates(self):
        """Run 10 ticks, verify EMA tracks smoothly (no jumps)."""
        bus = _make_bus({"text": 32, "audio": 16})
        prev = bus._coherence_ema.item()
        values = [prev]
        for _ in range(10):
            bus({"text": torch.randn(1, 32), "audio": torch.randn(1, 16)})
            values.append(bus._coherence_ema.item())
        # EMA should change smoothly — max step < 0.5
        for i in range(1, len(values)):
            assert abs(values[i] - values[i - 1]) < 0.5

    def test_coherence_range(self):
        """mean_coherence always in [0, 1]."""
        bus = _make_bus({"text": 32, "audio": 16})
        for _ in range(20):
            out = bus({"text": torch.randn(1, 32), "audio": torch.randn(1, 16)})
            assert 0.0 <= out.mean_coherence.item() <= 1.0

    def test_write_gate_modulation_range(self):
        """get_write_gate_modulation() returns value in [0, 1]."""
        bus = _make_bus({"text": 32, "audio": 16}, gate_coherence_weight=1.0)
        bus({"text": torch.randn(1, 32), "audio": torch.randn(1, 16)})
        val = bus.get_write_gate_modulation()
        assert 0.0 <= val <= 1.0

    def test_bleed_modulation_scaled(self):
        """get_bleed_modulation() scales by bleed_coherence_weight."""
        bus = _make_bus({"text": 32, "audio": 16}, bleed_coherence_weight=0.5)
        bus({"text": torch.randn(1, 32), "audio": torch.randn(1, 16)})
        val = bus.get_bleed_modulation()
        assert 0.0 <= val <= 0.5

    def test_binding_matrix_symmetric(self):
        """Off-diagonal PLV values are symmetric."""
        bus = _make_bus({"text": 32, "audio": 16, "vision": 24})
        out = bus({"text": torch.randn(1, 32), "audio": torch.randn(1, 16),
                   "vision": torch.randn(1, 24)})
        bm = out.binding_matrix
        assert torch.allclose(bm, bm.T, atol=1e-5)

    def test_reuses_multimodal_binder(self):
        """Internal binder is a MultimodalBinder instance."""
        bus = _make_bus({"text": 32})
        assert isinstance(bus.binder, MultimodalBinder)

    def test_deterministic(self):
        """Same input twice → same output."""
        bus = _make_bus({"text": 32})
        x = {"text": torch.randn(1, 32)}
        # Reset EMA to known state
        bus._coherence_ema.fill_(0.5)
        out1 = bus(x)
        bus._coherence_ema.fill_(0.5)
        out2 = bus(x)
        assert torch.allclose(out1.fused_representation, out2.fused_representation, atol=1e-6)
        assert torch.allclose(out1.mean_coherence, out2.mean_coherence, atol=1e-6)


# ---------------------------------------------------------------------------
# Integration Tests (13-15)
# ---------------------------------------------------------------------------

class TestPhaseLockBusIntegration:

    def test_bus_in_cycle_enabled(self):
        """CycleConfig(use_phase_lock_bus=True), tick() returns bus_* diagnostics."""
        from pyquifer.integration import CycleConfig, CognitiveCycle
        cfg = CycleConfig(
            state_dim=32, belief_dim=16, semantic_dim=8,
            hierarchy_dims=[32, 16, 8],
            num_oscillators=16, num_populations=4,
            num_groups=4, group_dim=8,
            use_phase_lock_bus=True,
        )
        cycle = CognitiveCycle(cfg)
        x = torch.randn(32)
        result = cycle.tick(x, return_diagnostics=True)
        diag = result['diagnostics']
        assert 'bus_mean_coherence' in diag
        assert 'bus_binding_matrix' in diag
        assert 'bus_modality_count' in diag

    def test_bus_disabled_no_overhead(self):
        """CycleConfig(use_phase_lock_bus=False), no bus_* in diagnostics."""
        from pyquifer.integration import CycleConfig, CognitiveCycle
        cfg = CycleConfig(
            state_dim=32, belief_dim=16, semantic_dim=8,
            hierarchy_dims=[32, 16, 8],
            num_oscillators=16, num_populations=4,
            num_groups=4, group_dim=8,
            use_phase_lock_bus=False,
        )
        cycle = CognitiveCycle(cfg)
        x = torch.randn(32)
        result = cycle.tick(x, return_diagnostics=True)
        diag = result['diagnostics']
        assert 'bus_mean_coherence' not in diag

    def test_bus_write_gate_scaling(self):
        """With bus enabled, write gate modulation differs from default 1.0."""
        bus = _make_bus({"text": 32, "audio": 16})
        # Run a few ticks to move EMA
        for _ in range(5):
            bus({"text": torch.randn(1, 32), "audio": torch.randn(1, 16)})
        mod = bus.get_write_gate_modulation()
        # Should be non-trivially different from 1.0 (EMA started at 0.5)
        assert mod != 1.0
