"""
Phase 10 Tests: Spherical N*D Kuramoto + Multi-Workspace Ensemble

Tests for all Phase 10 classes:
- _params_to_skew_symmetric (so(n) algebra)
- LearnableOmega (n=2,3,4 generalized frequencies)
- SphericalKuramotoLayer (hypersphere dynamics)
- SphericalKuramotoBank (multi-band bank)
- StandingBroadcast (EMA buffer)
- CrossBleedGate (phase-coherence gating)
- WorkspaceEnsemble (parallel workspaces)
- Organ.update_standing + HPCOrgan.accept
- CognitiveCycle with use_workspace_ensemble=True
"""

import math
import pytest
import torch
import torch.nn as nn


# ============================================================
# Spherical N*D Kuramoto Tests
# ============================================================

class TestParamsToSkewSymmetric:
    """Tests for _params_to_skew_symmetric."""

    def test_n2_shape(self):
        from pyquifer.spherical import _params_to_skew_symmetric
        params = torch.tensor([0.5])
        mat = _params_to_skew_symmetric(params, n=2)
        assert mat.shape == (2, 2)

    def test_n3_shape(self):
        from pyquifer.spherical import _params_to_skew_symmetric
        params = torch.randn(3)
        mat = _params_to_skew_symmetric(params, n=3)
        assert mat.shape == (3, 3)

    def test_n4_shape(self):
        from pyquifer.spherical import _params_to_skew_symmetric
        params = torch.randn(6)
        mat = _params_to_skew_symmetric(params, n=4)
        assert mat.shape == (4, 4)

    def test_skew_symmetry_n2(self):
        from pyquifer.spherical import _params_to_skew_symmetric
        params = torch.tensor([1.7])
        mat = _params_to_skew_symmetric(params, n=2)
        assert torch.allclose(mat, -mat.T, atol=1e-7)

    def test_skew_symmetry_n3(self):
        from pyquifer.spherical import _params_to_skew_symmetric
        params = torch.randn(3)
        mat = _params_to_skew_symmetric(params, n=3)
        assert torch.allclose(mat, -mat.T, atol=1e-7)

    def test_skew_symmetry_n4(self):
        from pyquifer.spherical import _params_to_skew_symmetric
        params = torch.randn(6)
        mat = _params_to_skew_symmetric(params, n=4)
        assert torch.allclose(mat, -mat.T, atol=1e-7)

    def test_batched_shape(self):
        from pyquifer.spherical import _params_to_skew_symmetric
        params = torch.randn(8, 6)  # 8 oscillators, 6 params for n=4
        mat = _params_to_skew_symmetric(params, n=4)
        assert mat.shape == (8, 4, 4)
        # Check each is skew-symmetric
        for i in range(8):
            assert torch.allclose(mat[i], -mat[i].T, atol=1e-7)


class TestLearnableOmega:
    """Tests for LearnableOmega at different dimensions."""

    def test_n2_backward_compat(self):
        """n=2 should produce same structure as standard Kuramoto."""
        from pyquifer.spherical import LearnableOmega
        omega = LearnableOmega(num_oscillators=8, components_per_oscillator=2,
                               init_omega=0.5)
        # Input: [B, C] = [1, 16] (8 osc * 2 components)
        x = torch.randn(1, 16)
        x = x / x.reshape(1, 8, 2).norm(dim=2, keepdim=True).repeat(1, 1, 2).reshape(1, 16)
        v = omega(x)
        assert v.shape == x.shape

    def test_n4_tangent_vectors(self):
        """n=4: forward output should be tangent to state (dot ≈ 0)."""
        from pyquifer.spherical import LearnableOmega, reshape_to_groups, l2_normalize
        n_osc, n = 4, 4
        omega = LearnableOmega(num_oscillators=n_osc, components_per_oscillator=n,
                               init_omega=0.3)
        x = torch.randn(2, n_osc * n)
        # Normalize to sphere
        x_grouped = reshape_to_groups(x, n)
        x_grouped = l2_normalize(x_grouped, dim=2)
        from pyquifer.spherical import reshape_from_groups
        x_normed = reshape_from_groups(x_grouped)

        v = omega(x_normed)
        # Dot product between x and v should be ≈ 0 (tangent)
        v_grouped = reshape_to_groups(v, n)
        dots = (x_grouped * v_grouped).sum(dim=2)
        assert dots.abs().max() < 0.01, f"Tangent violation: max dot = {dots.abs().max():.6f}"

    def test_n3_gradient_flows(self):
        """n=3: gradient should flow through."""
        from pyquifer.spherical import LearnableOmega, normalize_oscillators
        omega = LearnableOmega(num_oscillators=4, components_per_oscillator=3,
                               init_omega=0.2)
        x = normalize_oscillators(torch.randn(1, 12), n=3)
        x.requires_grad_(True)
        v = omega(x)
        loss = v.sum()
        loss.backward()
        assert omega.omega_param.grad is not None
        assert omega.omega_param.grad.abs().sum() > 0

    def test_get_frequencies_n2_shape(self):
        from pyquifer.spherical import LearnableOmega
        omega = LearnableOmega(num_oscillators=8, components_per_oscillator=2)
        freqs = omega.get_frequencies()
        assert freqs.shape == (8,)

    def test_get_frequencies_n4_shape(self):
        from pyquifer.spherical import LearnableOmega
        omega = LearnableOmega(num_oscillators=6, components_per_oscillator=4)
        freqs = omega.get_frequencies()
        assert freqs.shape == (6,)

    def test_global_omega(self):
        from pyquifer.spherical import LearnableOmega
        omega = LearnableOmega(num_oscillators=8, components_per_oscillator=3,
                               global_omega=True)
        freqs = omega.get_frequencies()
        # All frequencies should be identical (single global omega)
        assert freqs.shape == (8,)
        assert torch.allclose(freqs, freqs[0].expand_as(freqs))


class TestSphericalKuramotoLayer:
    """Tests for SphericalKuramotoLayer."""

    def test_states_on_sphere_n2(self):
        from pyquifer.spherical import SphericalKuramotoLayer, reshape_to_groups
        layer = SphericalKuramotoLayer(num_channels=16, components_per_oscillator=2,
                                       coupling_type="linear")
        x = torch.randn(2, 16)
        ext = torch.randn(2, 16)
        states, energies = layer(x, ext, num_steps=5, step_size=0.1)
        # Check last state is on sphere
        grouped = reshape_to_groups(states[-1], 2)
        norms = grouped.norm(dim=2)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_states_on_sphere_n4(self):
        from pyquifer.spherical import SphericalKuramotoLayer, reshape_to_groups
        layer = SphericalKuramotoLayer(num_channels=16, components_per_oscillator=4,
                                       coupling_type="linear")
        x = torch.randn(2, 16)
        ext = torch.randn(2, 16)
        states, energies = layer(x, ext, num_steps=5, step_size=0.1)
        grouped = reshape_to_groups(states[-1], 4)
        norms = grouped.norm(dim=2)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
            f"States not on S^3: norms = {norms}"

    def test_order_parameter_range(self):
        from pyquifer.spherical import SphericalKuramotoLayer
        layer = SphericalKuramotoLayer(num_channels=8, components_per_oscillator=2,
                                       coupling_type="linear")
        x = torch.randn(1, 8)
        ext = torch.randn(1, 8)
        states, _ = layer(x, ext, num_steps=10, step_size=0.1)
        r = layer.get_order_parameter(states[-1])
        assert 0.0 <= r.item() <= 1.0 + 1e-5


class TestSphericalKuramotoBank:
    """Tests for SphericalKuramotoBank multi-band."""

    def test_step_works_n2(self):
        from pyquifer.spherical import SphericalKuramotoBank
        bank = SphericalKuramotoBank(num_bands=3, oscillators_per_band=4,
                                      components=2)
        ext = torch.randn(1, 3, 8)  # 3 bands * 4 osc * 2 components = 8 per band
        result = bank.step(external_input=ext, num_steps=5, step_size=0.1)
        assert 'states' in result
        assert 'order_parameters' in result
        assert result['states'].shape == (1, 3, 8)
        assert result['order_parameters'].shape == (1, 3)

    def test_step_works_n4(self):
        from pyquifer.spherical import SphericalKuramotoBank
        bank = SphericalKuramotoBank(num_bands=4, oscillators_per_band=4,
                                      components=4)
        ext = torch.randn(1, 4, 16)  # 4 bands * 4 osc * 4 components = 16 per band
        result = bank.step(external_input=ext, num_steps=5, step_size=0.1)
        assert result['states'].shape == (1, 4, 16)
        assert result['order_parameters'].shape == (1, 4)

    def test_global_order_scalar(self):
        from pyquifer.spherical import SphericalKuramotoBank
        bank = SphericalKuramotoBank(num_bands=2, oscillators_per_band=8, components=2)
        ext = torch.randn(1, 2, 16)
        result = bank.step(external_input=ext, num_steps=3, step_size=0.1)
        assert result['global_order'].ndim == 0  # scalar


# ============================================================
# Workspace Ensemble Tests
# ============================================================

class TestStandingBroadcast:
    """Tests for StandingBroadcast EMA buffer."""

    def test_update_accumulates(self):
        from pyquifer.global_workspace import StandingBroadcast
        sb = StandingBroadcast(dim=16, momentum=0.9)
        assert sb.content.abs().sum() == 0.0  # starts at zero

        data = torch.ones(16)
        sb.update(data)
        # After one update: content = 0.9 * 0 + 0.1 * 1 = 0.1
        assert torch.allclose(sb.content, torch.full((16,), 0.1), atol=1e-6)

        sb.update(data)
        # After second: content = 0.9 * 0.1 + 0.1 * 1 = 0.19
        assert torch.allclose(sb.content, torch.full((16,), 0.19), atol=1e-5)

    def test_get_returns_correct_dim(self):
        from pyquifer.global_workspace import StandingBroadcast
        sb = StandingBroadcast(dim=32, momentum=0.8)
        sb.update(torch.randn(32))
        out = sb.get()
        assert out.shape == (32,)


class TestCrossBleedGate:
    """Tests for CrossBleedGate phase-coherence gating."""

    def test_aligned_phases_higher_gate(self):
        from pyquifer.global_workspace import CrossBleedGate
        gate = CrossBleedGate(dim=16)
        source = torch.randn(16)

        # Aligned phases (same phase → cos(0) = 1.0 → high coherence)
        aligned = torch.zeros(8)
        result_aligned = gate(source, source_phases=aligned, target_phases=aligned)

        # Misaligned phases (opposite phase → cos(π) = -1.0 → low coherence)
        misaligned = torch.full((8,), math.pi)
        result_misaligned = gate(source, source_phases=aligned, target_phases=misaligned)

        # Aligned should produce larger magnitude output
        assert result_aligned.abs().mean() > result_misaligned.abs().mean(), \
            f"Aligned {result_aligned.abs().mean():.4f} should > misaligned {result_misaligned.abs().mean():.4f}"

    def test_output_shape(self):
        from pyquifer.global_workspace import CrossBleedGate
        gate = CrossBleedGate(dim=32)
        source = torch.randn(32)
        out = gate(source)
        assert out.shape == (32,)


class TestWorkspaceEnsemble:
    """Tests for WorkspaceEnsemble."""

    def test_forward_returns_all_workspaces(self):
        from pyquifer.global_workspace import WorkspaceEnsemble
        n_ws = 3
        ens = WorkspaceEnsemble(n_workspaces=n_ws, content_dim=16,
                                 workspace_dim=32, n_slots=4, n_winners=1)
        contents = [torch.randn(1, 2, 16) for _ in range(n_ws)]
        contexts = [torch.randn(1, 2, 16) for _ in range(n_ws)]
        result = ens(contents, contexts)

        assert len(result['workspace_results']) == n_ws
        assert 'bleed_matrix' in result
        assert result['bleed_matrix'].shape == (n_ws, n_ws)

    def test_set_active_changes_idx(self):
        from pyquifer.global_workspace import WorkspaceEnsemble
        ens = WorkspaceEnsemble(n_workspaces=3, content_dim=8, workspace_dim=16)
        assert ens.active_idx.item() == 0
        ens.set_active(2)
        assert ens.active_idx.item() == 2

    def test_bleed_matrix_diagonal_zero(self):
        from pyquifer.global_workspace import WorkspaceEnsemble
        ens = WorkspaceEnsemble(n_workspaces=2, content_dim=8, workspace_dim=16)
        # Run forward to populate standings
        contents = [torch.randn(1, 2, 8) for _ in range(2)]
        contexts = [torch.randn(1, 2, 8) for _ in range(2)]
        ens(contents, contexts)
        mat = ens.get_bleed_matrix()
        # Diagonal should be zero (no self-bleed)
        for i in range(2):
            assert mat[i, i].item() == 0.0

    def test_reset_clears_state(self):
        from pyquifer.global_workspace import WorkspaceEnsemble
        ens = WorkspaceEnsemble(n_workspaces=2, content_dim=8, workspace_dim=16)
        contents = [torch.randn(1, 2, 8) for _ in range(2)]
        contexts = [torch.randn(1, 2, 8) for _ in range(2)]
        ens(contents, contexts)
        ens.reset()
        for sb in ens.standings:
            assert sb.content.abs().sum() == 0.0


# ============================================================
# Organ Standing Latent Tests
# ============================================================

class TestOrganStandingLatent:
    """Tests for Organ.update_standing and HPCOrgan.accept."""

    def test_organ_update_standing_ema(self):
        from pyquifer.organ import HPCOrgan
        organ = HPCOrgan(latent_dim=32, hierarchy_dims=[32, 16, 8])
        assert organ.standing_latent.abs().sum() == 0.0

        broadcast = torch.randn(32)
        organ.update_standing(broadcast)
        # standing_latent should now be non-zero
        assert organ.standing_latent.abs().sum() > 0.0

    def test_hpc_organ_accept_updates_standing(self):
        from pyquifer.organ import HPCOrgan
        organ = HPCOrgan(latent_dim=32, hierarchy_dims=[32, 16, 8])
        broadcast = torch.ones(32)
        organ.accept(broadcast)
        # accept calls update_standing internally
        assert organ.standing_latent.abs().sum() > 0.0


# ============================================================
# CognitiveCycle with Workspace Ensemble
# ============================================================

class TestCognitiveCycleWorkspaceEnsemble:
    """Tests for CognitiveCycle with use_workspace_ensemble=True."""

    def test_tick_returns_ensemble_info(self):
        from pyquifer.integration import CognitiveCycle, CycleConfig
        config = CycleConfig(
            state_dim=64,
            workspace_dim=32,
            num_oscillators=16,
            hierarchy_dims=[64, 32, 16],
            use_global_workspace=True,
            use_workspace_ensemble=True,
            n_workspaces=2,
            ensemble_workspace_dim=32,
        )
        cycle = CognitiveCycle(config)
        x = torch.randn(64)  # (state_dim,)
        result = cycle.tick(x)
        # Ensemble info is nested in diagnostics
        diag = result.get('diagnostics', {})
        assert 'ensemble_bleed_matrix' in diag, \
            f"Expected ensemble_bleed_matrix in diagnostics keys: {list(diag.keys())}"
        assert 'ensemble_active_idx' in diag
        assert 'ensemble_n_workspaces' in diag
        assert diag['ensemble_n_workspaces'] == 2

    def test_ensemble_disabled_by_default(self):
        from pyquifer.integration import CognitiveCycle, CycleConfig
        config = CycleConfig(
            state_dim=64,
            workspace_dim=32,
            num_oscillators=16,
            hierarchy_dims=[64, 32, 16],
        )
        cycle = CognitiveCycle(config)
        x = torch.randn(64)  # (state_dim,)
        result = cycle.tick(x)
        assert 'ensemble_bleed_matrix' not in result
