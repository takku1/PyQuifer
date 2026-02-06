"""
Phase 5B Tests: Synapse and Oscillator Upgrades
- 5B-1: Tsodyks-Markram Short-Term Plasticity
- 5B-2: Kuramoto-Daido Mean-Field Reduction
- 5B-3: E-prop Dual Eligibility Traces
"""

import torch
import math
import pytest


# ── 5B-1: Tsodyks-Markram STP ──

class TestTsodyksMarkramSynapse:
    """Tests for short-term plasticity dynamics."""

    def test_synapse_init(self):
        """Synapse state should initialize correctly."""
        from pyquifer.short_term_plasticity import TsodyksMarkramSynapse
        syn = TsodyksMarkramSynapse(num_synapses=10, U=0.2)
        assert syn.u.shape == (10,)
        assert syn.x.shape == (10,)
        assert torch.allclose(syn.u, torch.full((10,), 0.2))
        assert torch.allclose(syn.x, torch.ones(10))

    def test_synapse_forward_shape(self):
        """Forward should return correct shapes."""
        from pyquifer.short_term_plasticity import TsodyksMarkramSynapse
        syn = TsodyksMarkramSynapse(num_synapses=10)
        spikes = (torch.rand(10) > 0.5).float()
        result = syn(spikes)
        assert result['psp'].shape == (10,)
        assert result['u'].shape == (10,)
        assert result['x'].shape == (10,)
        assert result['efficacy'].shape == (10,)

    def test_facilitation_increases_u(self):
        """Repeated spikes should increase utilization parameter u."""
        from pyquifer.short_term_plasticity import TsodyksMarkramSynapse
        syn = TsodyksMarkramSynapse(num_synapses=5, U=0.1, tau_f=100.0)
        initial_u = syn.u.clone()

        # Send repeated spikes
        for _ in range(10):
            syn(torch.ones(5))

        # u should have increased from facilitation
        assert (syn.u > initial_u).all()

    def test_depression_decreases_x(self):
        """Repeated spikes should decrease available resources x."""
        from pyquifer.short_term_plasticity import TsodyksMarkramSynapse
        syn = TsodyksMarkramSynapse(num_synapses=5, U=0.5, tau_d=100.0)

        # Send repeated spikes
        for _ in range(10):
            syn(torch.ones(5))

        # x should have decreased from depression
        assert (syn.x < 1.0).all()

    def test_recovery_during_silence(self):
        """Resources should recover during silence (no spikes)."""
        from pyquifer.short_term_plasticity import TsodyksMarkramSynapse
        syn = TsodyksMarkramSynapse(num_synapses=5, U=0.5, tau_d=50.0)

        # Depress
        for _ in range(20):
            syn(torch.ones(5))
        depressed_x = syn.x.clone()

        # Silence
        for _ in range(200):
            syn(torch.zeros(5))

        # x should have recovered toward 1.0
        assert (syn.x > depressed_x).all()

    def test_reset(self):
        """Reset should restore baseline state."""
        from pyquifer.short_term_plasticity import TsodyksMarkramSynapse
        syn = TsodyksMarkramSynapse(num_synapses=5, U=0.2)
        for _ in range(10):
            syn(torch.ones(5))
        syn.reset()
        assert torch.allclose(syn.u, torch.full((5,), 0.2))
        assert torch.allclose(syn.x, torch.ones(5))

    def test_batch_forward(self):
        """Forward should work with batched input."""
        from pyquifer.short_term_plasticity import TsodyksMarkramSynapse
        syn = TsodyksMarkramSynapse(num_synapses=10)
        spikes = (torch.rand(4, 10) > 0.5).float()
        result = syn(spikes)
        assert result['psp'].shape == (4, 10)

    def test_stp_layer_forward(self):
        """STPLayer should project and apply STP."""
        from pyquifer.short_term_plasticity import STPLayer
        layer = STPLayer(input_dim=8, output_dim=16)
        x = torch.randn(4, 8)
        result = layer(x)
        assert result['output'].shape == (4, 16)
        assert result['efficacy'].shape == (16,)

    def test_stp_layer_reset(self):
        """STPLayer reset should clear STP state."""
        from pyquifer.short_term_plasticity import STPLayer
        layer = STPLayer(input_dim=8, output_dim=16)
        layer(torch.randn(4, 8))
        layer.reset()
        assert torch.allclose(layer.stp.x, torch.ones(16))


# ── 5B-2: Kuramoto-Daido Mean-Field ──

class TestKuramotoDaidoMeanField:
    """Tests for mean-field Kuramoto reduction."""

    def test_init_state(self):
        """Initial state should have low but nonzero R."""
        from pyquifer.oscillators import KuramotoDaidoMeanField
        mf = KuramotoDaidoMeanField()
        assert mf.R.item() == pytest.approx(0.1, abs=0.01)

    def test_forward_returns_dict(self):
        """Forward should return R, Psi, Z."""
        from pyquifer.oscillators import KuramotoDaidoMeanField
        mf = KuramotoDaidoMeanField()
        result = mf(steps=10)
        assert 'R' in result
        assert 'Psi' in result
        assert 'Z' in result
        assert result['R'].shape == ()

    def test_synchronization_above_critical(self):
        """With K > Kc = 2*Delta, system should synchronize (R → 1)."""
        from pyquifer.oscillators import KuramotoDaidoMeanField
        delta = 0.1
        K = 1.0  # >> 2 * 0.1 = 0.2 (critical coupling)
        mf = KuramotoDaidoMeanField(delta=delta, coupling=K, dt=0.01)

        result = mf(steps=1000)
        # Should reach high synchronization
        assert result['R'].item() > 0.5

    def test_desynchronization_below_critical(self):
        """With K < Kc, R should decay toward 0."""
        from pyquifer.oscillators import KuramotoDaidoMeanField
        delta = 1.0
        K = 0.5  # < 2 * 1.0 = 2.0
        mf = KuramotoDaidoMeanField(delta=delta, coupling=K, dt=0.01)

        result = mf(steps=1000)
        assert result['R'].item() < 0.3

    def test_critical_coupling_formula(self):
        """Kc = 2 * Delta."""
        from pyquifer.oscillators import KuramotoDaidoMeanField
        mf = KuramotoDaidoMeanField(delta=0.5)
        Kc = mf.get_critical_coupling()
        assert Kc.item() == pytest.approx(1.0)

    def test_is_synchronized(self):
        """is_synchronized should reflect R value."""
        from pyquifer.oscillators import KuramotoDaidoMeanField
        mf = KuramotoDaidoMeanField(coupling=2.0, delta=0.1, dt=0.01)
        mf(steps=1000)
        assert mf.is_synchronized(threshold=0.5)

    def test_reset(self):
        """Reset should restore initial low-sync state."""
        from pyquifer.oscillators import KuramotoDaidoMeanField
        mf = KuramotoDaidoMeanField(coupling=2.0, delta=0.1, dt=0.01)
        mf(steps=500)
        mf.reset()
        assert mf.R.item() == pytest.approx(0.1, abs=0.01)

    def test_external_field(self):
        """External field should influence dynamics."""
        from pyquifer.oscillators import KuramotoDaidoMeanField
        mf = KuramotoDaidoMeanField(coupling=0.0, delta=0.1, dt=0.01)
        field = torch.complex(torch.tensor(0.5), torch.tensor(0.0))
        result = mf(steps=100, external_field=field)
        # External field should push R up even with zero coupling
        assert result['R'].item() > 0.1

    def test_get_order_parameter(self):
        """get_order_parameter should match R from forward."""
        from pyquifer.oscillators import KuramotoDaidoMeanField
        mf = KuramotoDaidoMeanField()
        result = mf(steps=10)
        assert mf.get_order_parameter().item() == pytest.approx(result['R'].item(), abs=1e-5)


# ── 5B-3: E-prop Dual Eligibility Traces ──

class TestEpropSTDP:
    """Tests for E-prop learning rule with dual traces."""

    def test_init_traces_zero(self):
        """All traces should start at zero."""
        from pyquifer.advanced_spiking import EpropSTDP
        ep = EpropSTDP(pre_dim=16, post_dim=8)
        assert (ep.trace_fast == 0).all()
        assert (ep.trace_slow == 0).all()
        assert (ep.eligibility == 0).all()

    def test_forward_shape(self):
        """Forward should produce correct output shape."""
        from pyquifer.advanced_spiking import EpropSTDP
        ep = EpropSTDP(pre_dim=16, post_dim=8)
        pre = (torch.rand(4, 16) > 0.8).float()
        out = ep(pre)
        assert out.shape == (4, 8)

    def test_traces_accumulate(self):
        """Traces should accumulate after update_traces calls."""
        from pyquifer.advanced_spiking import EpropSTDP
        ep = EpropSTDP(pre_dim=16, post_dim=8)

        for _ in range(10):
            pre = (torch.rand(4, 16) > 0.8).float()
            post = (torch.rand(4, 8) > 0.8).float()
            membrane = torch.randn(4, 8) * 0.5
            ep.update_traces(pre, post, membrane, threshold=1.0)

        assert ep.trace_fast.norm() > 0
        assert ep.trace_slow.norm() > 0
        assert ep.eligibility.norm() > 0

    def test_fast_trace_decays_faster(self):
        """Fast trace should decay faster than slow trace."""
        from pyquifer.advanced_spiking import EpropSTDP
        ep = EpropSTDP(pre_dim=16, post_dim=8, tau_fast=20.0, tau_slow=200.0)

        # Build up traces
        for _ in range(5):
            pre = (torch.rand(4, 16) > 0.5).float()
            post = (torch.rand(4, 8) > 0.5).float()
            membrane = torch.randn(4, 8)
            ep.update_traces(pre, post, membrane)

        fast_norm = ep.trace_fast.norm().item()
        slow_norm = ep.trace_slow.norm().item()

        # Now let traces decay (no spikes)
        for _ in range(50):
            ep.update_traces(torch.zeros(4, 16), torch.zeros(4, 8),
                             torch.zeros(4, 8))

        fast_after = ep.trace_fast.norm().item()
        slow_after = ep.trace_slow.norm().item()

        # Fast should have decayed more relative to its initial value
        fast_ratio = fast_after / (fast_norm + 1e-8)
        slow_ratio = slow_after / (slow_norm + 1e-8)
        assert fast_ratio < slow_ratio

    def test_apply_reward_updates_weights(self):
        """apply_reward should modify weights using eligibility."""
        from pyquifer.advanced_spiking import EpropSTDP
        ep = EpropSTDP(pre_dim=16, post_dim=8, learning_rate=0.1)

        initial_weights = ep.weight.data.clone()

        # Build eligibility
        for _ in range(10):
            pre = (torch.rand(4, 16) > 0.5).float()
            post = (torch.rand(4, 8) > 0.5).float()
            membrane = torch.randn(4, 8) * 0.8
            ep.update_traces(pre, post, membrane)

        # Apply reward
        ep.apply_reward(torch.tensor(1.0))

        # Weights should have changed
        assert not torch.allclose(ep.weight.data, initial_weights, atol=1e-6)

    def test_apply_reward_resets_eligibility(self):
        """After apply_reward, all traces should be zeroed."""
        from pyquifer.advanced_spiking import EpropSTDP
        ep = EpropSTDP(pre_dim=16, post_dim=8)

        for _ in range(5):
            pre = (torch.rand(4, 16) > 0.5).float()
            post = (torch.rand(4, 8) > 0.5).float()
            membrane = torch.randn(4, 8)
            ep.update_traces(pre, post, membrane)

        ep.apply_reward(torch.tensor(1.0))
        assert (ep.eligibility == 0).all()
        assert (ep.trace_fast == 0).all()
        assert (ep.trace_slow == 0).all()

    def test_refractory_masking(self):
        """Neurons in refractory period should not contribute to traces."""
        from pyquifer.advanced_spiking import EpropSTDP
        ep = EpropSTDP(pre_dim=4, post_dim=2, refractory_steps=10)

        # All post neurons spike
        pre = torch.ones(1, 4)
        post = torch.ones(1, 2)
        membrane = torch.ones(1, 2) * 1.5
        ep.update_traces(pre, post, membrane)

        # Refractory counters should be set
        assert (ep.refractory_counter > 0).all()

        # Next step: even with high membrane, refractory should block
        trace_before = ep.trace_fast.clone()
        ep.update_traces(pre, torch.zeros(1, 2), membrane)
        # Traces should barely change (refractory blocks pseudo-derivative)
        trace_diff = (ep.trace_fast - trace_before * math.exp(-1.0 / 20.0)).norm()
        # The contribution should be near-zero due to refractory masking
        assert trace_diff < 0.1

    def test_pseudo_derivative_shape(self):
        """Pseudo-derivative should match input shape."""
        from pyquifer.advanced_spiking import EpropSTDP
        ep = EpropSTDP(pre_dim=16, post_dim=8)
        v = torch.randn(8)
        psi = ep._pseudo_derivative(v)
        assert psi.shape == (8,)
        assert (psi >= 0).all()

    def test_reset(self):
        """Reset should zero all traces."""
        from pyquifer.advanced_spiking import EpropSTDP
        ep = EpropSTDP(pre_dim=16, post_dim=8)
        for _ in range(5):
            ep.update_traces(torch.rand(4, 16), torch.rand(4, 8), torch.randn(4, 8))
        ep.reset()
        assert (ep.trace_fast == 0).all()
        assert (ep.trace_slow == 0).all()
        assert (ep.eligibility == 0).all()
        assert (ep.refractory_counter == 0).all()
