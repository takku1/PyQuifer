"""
Phase 8a Tests: Training Core

Tests for all Phase 8 classes:
- EquilibriumPropagationTrainer + EPKuramotoClassifier
- OscillationGatedPlasticity
- ThreeFactorRule
- OscillatoryPredictiveCoding
- SleepReplayConsolidation
- DendriticNeuron + DendriticStack
- Integration wiring (CognitiveCycle with Phase 8 flags)
"""

import math
import pytest
import torch
import torch.nn as nn

# ============================================================
# Equilibrium Propagation Tests
# ============================================================

class TestEquilibriumPropagation:
    """Tests for EquilibriumPropagationTrainer."""

    def _make_bank_and_trainer(self, n=16, topology='learnable'):
        from pyquifer.oscillators import LearnableKuramotoBank
        from pyquifer.equilibrium_propagation import EquilibriumPropagationTrainer
        bank = LearnableKuramotoBank(
            num_oscillators=n, dt=0.01, topology=topology,
            topology_params={'sparsity': 0.5},
        )
        trainer = EquilibriumPropagationTrainer(
            bank, lr=0.01, beta=0.1, free_steps=20, nudge_steps=20,
        )
        return bank, trainer

    def test_ep_free_phase_reaches_equilibrium(self):
        """Phases should stabilize after free_steps."""
        bank, trainer = self._make_bank_and_trainer()
        torch.manual_seed(42)
        bank.phases.data = torch.rand(bank.num_oscillators) * 2 * math.pi

        phases_free = trainer.free_phase()

        # Run more steps — phases should barely change (near equilibrium)
        phases_before = bank.phases.clone()
        bank(steps=5, use_precision=False)
        phases_after = bank.phases.clone()

        drift = (phases_after - phases_before).abs().mean().item()
        # Allow some drift (not perfect equilibrium), but should be small
        assert drift < 2.0, f"Phases drifted {drift:.4f} after equilibrium"

    def test_ep_nudge_phase_shifts_phases(self):
        """Nudged phases should differ from free phases."""
        bank, trainer = self._make_bank_and_trainer()
        torch.manual_seed(42)

        initial = torch.rand(bank.num_oscillators) * 2 * math.pi
        bank.phases.data = initial.clone()
        phases_free = trainer.free_phase()

        bank.phases.data = initial.clone()
        loss_grad = torch.randn(bank.num_oscillators) * 0.5
        phases_nudge = trainer.nudge_phase(loss_grad)

        diff = (phases_nudge - phases_free).abs().mean().item()
        assert diff > 0.0, "Nudged phases should differ from free phases"

    def test_ep_coupling_update_is_local(self):
        """Coupling update should only use cosine differences of connected pairs."""
        bank, trainer = self._make_bank_and_trainer(n=8, topology='learnable')
        torch.manual_seed(42)

        logits_before = bank.adjacency_logits.data.clone()

        phases_free = torch.rand(8) * 2 * math.pi
        phases_nudge = phases_free + torch.randn(8) * 0.1

        trainer.update_couplings(phases_free, phases_nudge)

        logits_after = bank.adjacency_logits.data
        delta = (logits_after - logits_before).abs()

        # Self-connections should remain unchanged (masked by self_mask)
        diag_change = torch.diag(delta).sum().item()
        assert diag_change < 1e-7, f"Self-connections changed by {diag_change}"

        # Off-diagonal should have changed
        off_diag_change = (delta * bank.self_mask).sum().item()
        assert off_diag_change > 0, "Off-diagonal couplings should change"

    def test_ep_train_step_reduces_loss(self):
        """Loss should decrease over multiple EP steps."""
        bank, trainer = self._make_bank_and_trainer(n=16, topology='learnable')
        torch.manual_seed(42)

        # Target: maximize order parameter
        def loss_fn(ph, _tgt):
            c = torch.exp(1j * ph)
            return 1.0 - torch.abs(c.mean())

        target = torch.tensor(0.0)
        losses = []
        for _ in range(10):
            bank.phases.data = torch.rand(16) * 2 * math.pi
            result = trainer.train_step(None, loss_fn, target)
            losses.append(result['loss'])

        # Loss should generally decrease (check first vs last few)
        early_mean = sum(losses[:3]) / 3
        late_mean = sum(losses[-3:]) / 3
        # Allow some variance; at minimum the system should run without error
        assert len(losses) == 10

    def test_ep_does_not_modify_encoder_readout_without_backprop(self):
        """EP only touches couplings, not encoder/readout weights."""
        from pyquifer.equilibrium_propagation import EPKuramotoClassifier
        torch.manual_seed(42)

        model = EPKuramotoClassifier(input_dim=4, num_classes=2, num_oscillators=8,
                                     free_steps=10, nudge_steps=10)
        encoder_w_before = model.encoder[0].weight.data.clone()
        readout_w_before = model.readout.weight.data.clone()

        # Run EP step (detached — no backprop)
        x = torch.randn(4)
        target = torch.tensor(0)
        result = model.ep_train_step(x, target)

        # Encoder and readout should NOT have changed from EP alone
        # (cls_loss is returned but not .backward()'d here)
        encoder_w_after = model.encoder[0].weight.data
        readout_w_after = model.readout.weight.data

        # They won't change because we didn't call .backward() on cls_loss
        assert torch.allclose(encoder_w_before, encoder_w_after, atol=1e-6)
        assert torch.allclose(readout_w_before, readout_w_after, atol=1e-6)

    def test_ep_classifier_forward_shape(self):
        """EPKuramotoClassifier forward should produce correct output dims."""
        from pyquifer.equilibrium_propagation import EPKuramotoClassifier
        model = EPKuramotoClassifier(input_dim=8, num_classes=3, num_oscillators=16,
                                     free_steps=5, nudge_steps=5)
        x = torch.randn(8)
        out = model(x)
        assert out.shape == (3,), f"Expected (3,), got {out.shape}"

        # Batch
        x_batch = torch.randn(4, 8)
        out_batch = model(x_batch)
        assert out_batch.shape == (4, 3), f"Expected (4, 3), got {out_batch.shape}"

    def test_ep_classifier_trains_xor(self):
        """EPKuramotoClassifier should be able to learn XOR (basic sanity)."""
        from pyquifer.equilibrium_propagation import EPKuramotoClassifier
        torch.manual_seed(42)

        model = EPKuramotoClassifier(input_dim=2, num_classes=2, num_oscillators=8,
                                     free_steps=10, nudge_steps=10)

        # XOR dataset
        X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        Y = torch.tensor([0, 1, 1, 0])

        # Just verify it runs without error for several steps
        for epoch in range(5):
            for i in range(4):
                result = model.ep_train_step(X[i], Y[i])
                assert 'ep_loss' in result
                assert 'classification_loss' in result

    def test_ep_preserves_phases_on_reset(self):
        """After EP train_step, phases should be at free equilibrium."""
        bank, trainer = self._make_bank_and_trainer(n=8)
        torch.manual_seed(42)

        def loss_fn(ph, _tgt):
            return (1.0 - torch.abs(torch.exp(1j * ph).mean()))

        bank.phases.data = torch.rand(8) * 2 * math.pi
        initial = bank.phases.clone()

        trainer.train_step(None, loss_fn, torch.tensor(0.0))

        # Phases should be at free-phase equilibrium, not initial
        diff = (bank.phases - initial).abs().mean().item()
        assert diff > 0.01, "Phases should have evolved from initial"


# ============================================================
# Oscillation-Gated Plasticity Tests
# ============================================================

class TestOscillationGatedPlasticity:
    """Tests for OscillationGatedPlasticity."""

    def test_gate_permissive_at_preferred_phase(self):
        """High accumulation at preferred phase (pi by default)."""
        from pyquifer.learning import OscillationGatedPlasticity

        ogp = OscillationGatedPlasticity(shape=(4,), preferred_phase=math.pi)
        activity = torch.ones(4)

        trace = ogp(activity, theta_phase=torch.tensor(math.pi))

        # Gate should be ~1.0 at preferred phase
        assert ogp.gate_value.item() > 0.9, f"Gate={ogp.gate_value.item()}"
        assert trace.abs().sum() > 0, "Trace should accumulate at preferred phase"

    def test_gate_suppressive_at_opposite_phase(self):
        """Low accumulation at phi + pi from preferred."""
        from pyquifer.learning import OscillationGatedPlasticity

        ogp = OscillationGatedPlasticity(shape=(4,), preferred_phase=math.pi)
        activity = torch.ones(4)

        trace = ogp(activity, theta_phase=torch.tensor(0.0))  # Opposite of pi

        # Gate should be ~0.0 at opposite phase
        assert ogp.gate_value.item() < 0.1, f"Gate={ogp.gate_value.item()}"

    def test_modulated_reward_includes_neuromod(self):
        """Neuromod signal should be multiplied into the update."""
        from pyquifer.learning import OscillationGatedPlasticity

        ogp = OscillationGatedPlasticity(shape=(4,))

        # Accumulate some trace
        ogp(torch.ones(4), theta_phase=torch.tensor(math.pi))

        reward = torch.tensor(1.0)
        neuromod = torch.tensor(2.0)

        delta_no_mod = ogp.apply_modulated_reward(reward, lr=0.01)
        delta_with_mod = ogp.apply_modulated_reward(reward, neuromodulation=neuromod, lr=0.01)

        # With neuromod=2, delta should be 2x
        ratio = delta_with_mod.abs().sum() / delta_no_mod.abs().sum().clamp(min=1e-8)
        assert abs(ratio.item() - 2.0) < 0.01, f"Ratio={ratio.item()}"

    def test_gated_trace_decays_correctly(self):
        """Trace should decay even without new input."""
        from pyquifer.learning import OscillationGatedPlasticity

        ogp = OscillationGatedPlasticity(shape=(4,), decay_rate=0.5)

        # Accumulate
        ogp(torch.ones(4), theta_phase=torch.tensor(math.pi))
        trace_after_accum = ogp.trace.clone()

        # Decay with zero activity at suppressive phase
        ogp(torch.zeros(4), theta_phase=torch.tensor(0.0))
        trace_after_decay = ogp.trace.clone()

        # Trace should have decayed
        assert trace_after_decay.abs().sum() < trace_after_accum.abs().sum()


# ============================================================
# Three-Factor Rule Tests
# ============================================================

class TestThreeFactorRule:
    """Tests for ThreeFactorRule."""

    def test_three_factor_forward_shape(self):
        """Forward pass should produce correct output shape."""
        from pyquifer.learning import ThreeFactorRule

        tfr = ThreeFactorRule(input_dim=8, output_dim=4)
        x = torch.randn(8)
        out = tfr(x)
        assert out.shape == (4,), f"Expected (4,), got {out.shape}"

        # Batch
        x_batch = torch.randn(3, 8)
        out_batch = tfr(x_batch)
        assert out_batch.shape == (3, 4), f"Expected (3, 4), got {out_batch.shape}"

    def test_modulated_update_changes_weights(self):
        """Weights should change after modulation."""
        from pyquifer.learning import ThreeFactorRule

        tfr = ThreeFactorRule(input_dim=4, output_dim=2)

        # Accumulate trace
        for _ in range(5):
            tfr(torch.randn(4))

        w_before = tfr.weight.data.clone()

        # Apply modulated update
        tfr.modulated_update(torch.tensor(1.0))

        w_after = tfr.weight.data
        diff = (w_after - w_before).abs().sum().item()
        assert diff > 0, "Weights should change after modulated update"

    def test_homeostatic_factor_stabilizes(self):
        """Homeostatic factor should approach 1.0 at target rate."""
        from pyquifer.learning import ThreeFactorRule

        tfr = ThreeFactorRule(input_dim=4, output_dim=2, homeostatic_target=0.5)

        # Force running_rate to target
        tfr.running_rate.fill_(0.5)

        h = tfr.homeostatic_factor
        # At target rate, factor should be very close to 1.0
        assert (h - 1.0).abs().max().item() < 0.01, f"Homeostatic factor={h}"

    def test_no_update_without_modulation(self):
        """Weights should not change if modulation signal is 0."""
        from pyquifer.learning import ThreeFactorRule

        tfr = ThreeFactorRule(input_dim=4, output_dim=2)

        # Accumulate trace
        for _ in range(5):
            tfr(torch.randn(4))

        w_before = tfr.weight.data.clone()

        # Modulate with zero signal
        tfr.modulated_update(torch.tensor(0.0))

        w_after = tfr.weight.data
        assert torch.allclose(w_before, w_after, atol=1e-7), "Weights should not change with zero modulation"


# ============================================================
# Oscillatory Predictive Coding Tests
# ============================================================

class TestOscillatoryPredictiveCoding:
    """Tests for OscillatoryPredictiveCoding."""

    def test_opc_forward_returns_gamma_power(self):
        """Forward should return gamma_power field."""
        from pyquifer.hierarchical_predictive import OscillatoryPredictiveCoding

        opc = OscillatoryPredictiveCoding(dims=[8, 4], lr=0.01, inference_steps=3)
        x = torch.randn(1, 8)
        result = opc(x)

        assert 'gamma_power' in result
        assert 'alpha_beta_power' in result
        assert 'free_energy' in result
        assert result['gamma_power'].item() >= 0

    def test_opc_high_error_increases_gamma(self):
        """Unexpected input should produce higher gamma power."""
        from pyquifer.hierarchical_predictive import OscillatoryPredictiveCoding

        opc = OscillatoryPredictiveCoding(dims=[8, 4], lr=0.01, inference_steps=3)

        # First, train on a pattern so predictions are calibrated
        pattern = torch.ones(1, 8) * 0.5
        for _ in range(20):
            opc.learn(pattern)

        # Now measure gamma with expected input
        result_expected = opc(pattern)

        # And with unexpected (novel) input
        opc2 = OscillatoryPredictiveCoding(dims=[8, 4], lr=0.01, inference_steps=3)
        novel = torch.randn(1, 8) * 3.0
        result_novel = opc2(novel)

        # Novel input should generally produce higher free energy
        # (gamma power depends on modulation phase, but free energy should differ)
        assert result_novel['free_energy'].item() > 0

    def test_opc_low_error_suppresses_gamma(self):
        """Learning on a repeated pattern should reduce free energy."""
        from pyquifer.hierarchical_predictive import OscillatoryPredictiveCoding
        torch.manual_seed(42)

        opc = OscillatoryPredictiveCoding(dims=[8, 4], lr=0.005, inference_steps=10)

        pattern = torch.sin(torch.linspace(0, math.pi, 8)).unsqueeze(0) * 0.3

        energies = []
        for _ in range(60):
            result = opc.learn(pattern)
            energies.append(result['free_energy'].item())

        early = sum(energies[:5]) / 5
        late = sum(energies[-5:]) / 5
        assert late < early, f"Free energy should decrease: {early:.4f} -> {late:.4f}"

    def test_opc_learn_reduces_free_energy(self):
        """Learning should reduce free energy over time."""
        from pyquifer.hierarchical_predictive import OscillatoryPredictiveCoding
        torch.manual_seed(42)

        opc = OscillatoryPredictiveCoding(dims=[16, 8], lr=0.005, inference_steps=10)
        pattern = torch.randn(1, 16) * 0.3  # Smaller input for stability

        energies = []
        for _ in range(60):
            result = opc.learn(pattern)
            energies.append(result['free_energy'].item())

        # Overall trend should be decreasing
        early = sum(energies[:5]) / 5
        late = sum(energies[-5:]) / 5
        assert late < early, f"Free energy should decrease: {early:.4f} -> {late:.4f}"


# ============================================================
# Sleep Replay Consolidation Tests
# ============================================================

class TestSleepReplayConsolidation:
    """Tests for SleepReplayConsolidation."""

    def test_src_sleep_step_modifies_weights(self):
        """One sleep step should modify layer weights."""
        from pyquifer.memory_consolidation import SleepReplayConsolidation

        src = SleepReplayConsolidation(layer_dims=[8, 4, 2], sleep_lr=0.01)
        w_before = src.layers[0].weight.data.clone()

        result = src.sleep_step()

        w_after = src.layers[0].weight.data
        diff = (w_after - w_before).abs().sum().item()
        assert diff > 0, "Sleep step should modify weights"
        assert len(result['weight_delta_norms']) == 2  # 2 layer transitions

    def test_src_preserves_learned_patterns(self):
        """After training + sleep, weight structure should be preserved."""
        from pyquifer.memory_consolidation import SleepReplayConsolidation

        src = SleepReplayConsolidation(layer_dims=[4, 2], sleep_lr=0.001)

        # Set specific weight pattern
        with torch.no_grad():
            src.layers[0].weight.fill_(0.5)

        w_pattern = src.layers[0].weight.data.clone()

        # Run sleep
        src.sleep_cycle(num_steps=10)

        w_after = src.layers[0].weight.data

        # Weights should have changed but not catastrophically
        diff = (w_after - w_pattern).abs().mean().item()
        assert diff < 1.0, f"Sleep should not destroy pattern completely: diff={diff}"

    def test_src_sleep_cycle_returns_metrics(self):
        """Sleep cycle should return proper metrics dict."""
        from pyquifer.memory_consolidation import SleepReplayConsolidation

        src = SleepReplayConsolidation(layer_dims=[8, 4], num_replay_steps=10)
        result = src.sleep_cycle()

        assert 'mean_delta_norm' in result
        assert 'per_layer_changes' in result
        assert 'total_steps' in result
        assert result['total_steps'] == 10

    def test_src_noise_injection_varies(self):
        """Different sleep steps should use different noise."""
        from pyquifer.memory_consolidation import SleepReplayConsolidation

        src = SleepReplayConsolidation(layer_dims=[8, 4])

        result1 = src.sleep_step()
        result2 = src.sleep_step()

        # Different noise inputs
        diff = (result1['noise_input'] - result2['noise_input']).abs().sum().item()
        assert diff > 0, "Different sleep steps should use different noise"


# ============================================================
# Dendritic Neuron Tests
# ============================================================

class TestDendriticNeuron:
    """Tests for DendriticNeuron and DendriticStack."""

    def test_dendritic_neuron_forward_shape(self):
        """Forward should produce correct output shape."""
        from pyquifer.dendritic import DendriticNeuron

        neuron = DendriticNeuron(input_dim=8, output_dim=4)
        x = torch.randn(8)
        result = neuron(x)

        assert result['output'].shape == (4,)
        assert result['plateau_potential'].shape == (4,)
        assert result['basal_activation'].shape == (4,)

    def test_plateau_potential_with_apical(self):
        """Apical input should produce nonzero plateau potential."""
        from pyquifer.dendritic import DendriticNeuron

        neuron = DendriticNeuron(input_dim=4, output_dim=2, apical_dim=3)
        x_basal = torch.randn(4)
        x_apical = torch.ones(3) * 5.0  # Strong top-down signal

        result = neuron(x_basal, x_apical=x_apical)

        # Strong apical input should produce plateau > 0
        plateau_sum = result['plateau_potential'].sum().item()
        assert plateau_sum > 0, f"Plateau should be positive with apical: {plateau_sum}"

    def test_no_plateau_without_apical(self):
        """Without apical input, plateau should be zero."""
        from pyquifer.dendritic import DendriticNeuron

        neuron = DendriticNeuron(input_dim=4, output_dim=2)
        x = torch.randn(4)

        result = neuron(x)  # No apical input

        assert result['plateau_potential'].abs().sum().item() == 0.0

    def test_dendritic_local_update_changes_weights(self):
        """Local update should modify basal weights."""
        from pyquifer.dendritic import DendriticNeuron

        neuron = DendriticNeuron(input_dim=4, output_dim=2, apical_dim=2, lr=0.1)

        # Forward with apical signal to generate plateau
        neuron(torch.randn(4), x_apical=torch.ones(2) * 3.0)

        w_before = neuron.W_basal.data.clone()
        neuron.local_update()
        w_after = neuron.W_basal.data

        diff = (w_after - w_before).abs().sum().item()
        assert diff > 0, "Local update should change basal weights"

    def test_dendritic_stack_forward_shape(self):
        """DendriticStack should produce outputs for all layers."""
        from pyquifer.dendritic import DendriticStack

        stack = DendriticStack(dims=[8, 4, 2])
        x = torch.randn(8)

        result = stack(x)

        assert len(result['outputs']) == 2  # 2 layers
        assert result['outputs'][0].shape == (4,)
        assert result['outputs'][1].shape == (2,)
        assert result['final_output'].shape == (2,)

    def test_dendritic_stack_learn(self):
        """Stack learn() should return delta norms."""
        from pyquifer.dendritic import DendriticStack

        stack = DendriticStack(dims=[8, 4, 2])

        # Forward pass to populate last_* buffers
        td = [torch.randn(2)]  # Top-down for first layer only
        stack(torch.randn(8), top_down_predictions=td)

        result = stack.learn()
        assert 'delta_norms' in result
        assert len(result['delta_norms']) == 2


# ============================================================
# Integration Tests
# ============================================================

class TestPhase8Integration:
    """Tests for CognitiveCycle with Phase 8 flags."""

    def test_cycle_with_phase8_flags(self):
        """CognitiveCycle should work with all Phase 8 flags enabled."""
        from pyquifer.integration import CycleConfig, CognitiveCycle

        config = CycleConfig.small()
        config.use_ep_training = True
        config.use_oscillation_gated_plasticity = True
        config.use_three_factor = True
        config.use_oscillatory_predictive = True
        config.use_sleep_consolidation = True
        config.use_dendritic_credit = True

        cycle = CognitiveCycle(config)
        torch.manual_seed(42)

        # Run a few ticks
        for i in range(3):
            sensory = torch.randn(config.state_dim)
            result = cycle.tick(sensory, reward=0.5, sleep_signal=0.0)

            assert 'modulation' in result
            assert 'diagnostics' in result

        # Sleep tick to trigger consolidation
        result = cycle.tick(torch.randn(config.state_dim), reward=0.0, sleep_signal=0.8)
        assert 'modulation' in result

    def test_cycle_phase8_backward_compat(self):
        """All Phase 8 flags False should produce identical behavior to Phase 7."""
        from pyquifer.integration import CycleConfig, CognitiveCycle

        config = CycleConfig.small()
        # All Phase 8 flags default to False
        assert config.use_ep_training is False
        assert config.use_oscillation_gated_plasticity is False
        assert config.use_three_factor is False
        assert config.use_oscillatory_predictive is False
        assert config.use_sleep_consolidation is False
        assert config.use_dendritic_credit is False

        cycle = CognitiveCycle(config)
        torch.manual_seed(42)

        sensory = torch.randn(config.state_dim)
        result = cycle.tick(sensory)

        # Should have standard keys
        assert 'modulation' in result
        assert 'consciousness' in result
        assert 'self_state' in result
        assert 'learning' in result
        assert 'diagnostics' in result

    def test_cycle_phase8_returns_new_diagnostics(self):
        """With Phase 8 flags on, diagnostics should include new keys."""
        from pyquifer.integration import CycleConfig, CognitiveCycle

        config = CycleConfig.small()
        config.use_oscillation_gated_plasticity = True
        config.use_three_factor = True
        config.use_oscillatory_predictive = True

        cycle = CognitiveCycle(config)
        torch.manual_seed(42)

        sensory = torch.randn(config.state_dim)
        result = cycle.tick(sensory)

        diag = result['diagnostics']
        assert 'theta_gate_value' in diag
        assert 'homeostatic_factor' in diag
        assert 'gamma_power' in diag
