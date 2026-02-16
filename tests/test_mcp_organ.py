"""
Tests for MCP-as-Organ Protocol.

Tests the MCPOrgan class: proposal generation, salience computation,
change detection, staleness decay, workspace competition integration.
"""

import time
import torch
import pytest
from pyquifer.mcp_organ import MCPOrgan, MCPOrganConfig
from pyquifer.organ import Proposal, PreGWAdapter, OscillatoryWriteGate


@pytest.fixture
def default_config():
    return MCPOrganConfig(
        organ_id="mcp:test:resource",
        resource_uri="test://resource",
        latent_dim=32,
        base_salience=0.3,
        poll_stale_after=5.0,
        tags={"external", "mcp", "test"},
    )


@pytest.fixture
def organ(default_config):
    return MCPOrgan(default_config)


class TestMCPOrganInit:
    """Test MCPOrgan initialization."""

    def test_creates_with_config(self, organ, default_config):
        assert organ.organ_id == "mcp:test:resource"
        assert organ.latent_dim == 32
        assert organ.config is default_config

    def test_initial_state_is_zero(self, organ):
        assert not organ._has_state
        assert organ._cached_state.sum().item() == 0.0
        assert organ.standing_latent.sum().item() == 0.0

    def test_encoder_exists(self, organ):
        assert organ._encoder is not None
        # Should accept latent_dim input
        dummy = torch.randn(32)
        out = organ._encoder(dummy)
        assert out.shape == (32,)


class TestMCPOrganProposal:
    """Test proposal generation with various cache states."""

    def test_propose_no_state_zero_salience(self, organ):
        """Organ with no cached state returns salience=0."""
        proposal = organ.propose()
        assert isinstance(proposal, Proposal)
        assert proposal.salience == 0.0
        assert proposal.organ_id == "mcp:test:resource"
        assert "mcp" in proposal.tags

    def test_propose_with_state(self, organ):
        """Organ with cached state returns positive salience."""
        state = torch.randn(32)
        organ.update_cache(state, time.time())
        proposal = organ.propose()
        assert proposal.salience > 0.0

    def test_propose_large_change_high_salience(self, organ):
        """Large state change → high salience."""
        # First state
        organ.update_cache(torch.zeros(32), time.time())
        # Large change
        organ.update_cache(torch.ones(32) * 10.0, time.time())
        proposal = organ.propose()
        # Change magnitude is large → salience should be high
        assert proposal.salience > 0.5

    def test_propose_no_change_low_salience(self, organ):
        """Same state twice → low salience (base only)."""
        state = torch.ones(32)
        organ.update_cache(state.clone(), time.time())
        organ.update_cache(state.clone(), time.time())
        proposal = organ.propose()
        # Change magnitude is ~0 → salience should be low (just base * 0.3)
        assert proposal.salience < 0.2

    def test_staleness_decay(self, organ):
        """Old cached state → decaying salience."""
        state = torch.randn(32)
        # Stamp it as 10 seconds old (stale_after=5.0)
        organ.update_cache(state, time.time() - 10.0)
        proposal = organ.propose()
        # Freshness should be 0 (fully stale)
        assert proposal.salience == 0.0

    def test_fresh_state_full_salience(self, organ):
        """Just-polled state → full salience (freshness=1.0)."""
        organ.update_cache(torch.zeros(32), time.time() - 10)  # old baseline
        organ.update_cache(torch.ones(32) * 5.0, time.time())   # fresh big change
        proposal = organ.propose()
        assert proposal.salience > 0.0

    def test_propose_content_is_encoded(self, organ):
        """Proposal content should come from encoder, not raw cache."""
        state = torch.ones(32) * 3.0
        organ.update_cache(state, time.time())
        organ.observe(torch.zeros(32))  # Must observe first to encode
        proposal = organ.propose()
        # Content should be transformed (tanh bounds to [-1, 1])
        assert proposal.content.abs().max().item() <= 1.0


class TestMCPOrganCache:
    """Test cache update mechanics."""

    def test_update_cache_sets_state(self, organ):
        state = torch.randn(32)
        organ.update_cache(state, 12345.0)
        assert organ._has_state
        assert organ._state_timestamp == 12345.0
        assert torch.allclose(organ._cached_state, state)

    def test_update_cache_pads_short_tensor(self, organ):
        """Short tensor gets zero-padded to latent_dim."""
        short = torch.ones(16)
        organ.update_cache(short, time.time())
        assert organ._cached_state.shape == (32,)
        assert organ._cached_state[:16].sum().item() == 16.0
        assert organ._cached_state[16:].sum().item() == 0.0

    def test_update_cache_trims_long_tensor(self, organ):
        """Long tensor gets truncated to latent_dim."""
        long = torch.randn(64)
        organ.update_cache(long, time.time())
        assert organ._cached_state.shape == (32,)
        assert torch.allclose(organ._cached_state, long[:32])

    def test_change_magnitude_computed(self, organ):
        """Change magnitude is computed from consecutive updates."""
        organ.update_cache(torch.zeros(32), time.time())
        organ.update_cache(torch.ones(32), time.time())
        # Change = ||ones - zeros|| = sqrt(32)
        expected = torch.ones(32).norm().item()
        assert abs(organ._change_magnitude - expected) < 0.01


class TestMCPOrganObserve:
    """Test observe/accept lifecycle."""

    def test_observe_encodes_cached_state(self, organ):
        state = torch.randn(32)
        organ.update_cache(state, time.time())
        organ.observe(torch.zeros(32))
        # _latent should now be non-zero (encoded from cache)
        assert organ._latent.abs().sum().item() > 0.0

    def test_observe_without_cache_keeps_zero(self, organ):
        """If no cache, observe doesn't crash."""
        organ.observe(torch.zeros(32))
        # _latent stays zero (initialized to zeros, no cache to encode)
        assert organ._latent.abs().sum().item() == 0.0

    def test_accept_updates_standing(self, organ):
        broadcast = torch.randn(32)
        organ.accept(broadcast)
        # Standing latent should be non-zero now
        assert organ.standing_latent.abs().sum().item() > 0.0


class TestMCPOrganOscillator:
    """Test oscillator phase coupling."""

    def test_phase_coupling_to_global(self, organ):
        initial_phase = organ.phase.item()
        global_phase = torch.tensor(1.0)
        organ.step_oscillator(dt=0.01, global_phase=global_phase, coupling=0.5)
        # Phase should have changed
        assert organ.phase.item() != initial_phase

    def test_phase_stays_in_range(self, organ):
        """Phase wraps to [0, 2π)."""
        for _ in range(1000):
            organ.step_oscillator(dt=0.01)
        import math
        assert 0.0 <= organ.phase.item() < 2 * math.pi


class TestMCPOrganWorkspaceIntegration:
    """Test MCPOrgan works with workspace components."""

    def test_adapter_projects_correctly(self, organ):
        """PreGWAdapter can project MCPOrgan's latent to workspace dim."""
        adapter = PreGWAdapter(organ_dim=32, workspace_dim=64)
        state = torch.randn(32)
        organ.update_cache(state, time.time())
        organ.observe(torch.zeros(32))
        proposal = organ.propose()

        projected = adapter.encode(proposal.content.unsqueeze(0))
        assert projected.shape == (1, 64)

    def test_write_gate_computes(self, organ):
        """OscillatoryWriteGate works with MCPOrgan phase."""
        gate = OscillatoryWriteGate()
        global_phase = torch.tensor(0.5)
        gate_val = gate(organ.phase, global_phase, novelty=0.5)
        assert 0.0 <= gate_val.item() <= 1.0

    def test_diversity_compatible(self, organ):
        """MCPOrgan has organ_id suitable for DiversityTracker."""
        assert organ.organ_id.startswith("mcp:")

    def test_full_competition_round(self):
        """MCPOrgan + HPCOrgan compete in a minimal workspace."""
        from pyquifer.organ import HPCOrgan

        mcp_config = MCPOrganConfig(
            organ_id="mcp:test:alerts",
            resource_uri="test://alerts",
            latent_dim=32,
        )
        mcp_organ = MCPOrgan(mcp_config)
        hpc_organ = HPCOrgan(latent_dim=32, hierarchy_dims=[32, 16, 8])

        # Give MCP organ a big state change (should win)
        mcp_organ.update_cache(torch.zeros(32), time.time() - 1)
        mcp_organ.update_cache(torch.ones(32) * 10, time.time())

        sensory = torch.randn(32)
        global_phase = torch.tensor(0.0)

        # Run both organs
        for organ in [mcp_organ, hpc_organ]:
            organ.step_oscillator(dt=0.01, global_phase=global_phase)
            organ.observe(sensory)

        mcp_proposal = mcp_organ.propose()
        hpc_proposal = hpc_organ.propose()

        # MCP should have high salience due to large state change
        assert mcp_proposal.salience > 0.5
        assert isinstance(mcp_proposal.content, torch.Tensor)
        assert isinstance(hpc_proposal.content, torch.Tensor)
