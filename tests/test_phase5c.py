"""
Phase 5C Tests: Plasticity and Evolution
- 5C-1: Differentiable Plasticity + Learnable Eligibility Trace
- 5C-2: Speciated Neural Darwinism
- 5C-3: AdEx Neuron Model
"""

import torch
import math
import pytest


# ── 5C-1: Differentiable Plasticity ──

class TestDifferentiablePlasticity:
    """Tests for Miconi-style differentiable Hebbian plasticity."""

    def test_init(self):
        """Should initialize with correct shapes."""
        from pyquifer.learning import DifferentiablePlasticity
        dp = DifferentiablePlasticity(input_dim=8, output_dim=4)
        assert dp.W.shape == (4, 8)
        assert dp.H.shape == (4, 8)
        assert dp.alpha.shape == ()
        assert dp.eta.shape == ()

    def test_forward_shape(self):
        """Forward should produce correct output shape."""
        from pyquifer.learning import DifferentiablePlasticity
        dp = DifferentiablePlasticity(input_dim=8, output_dim=4)
        x = torch.randn(3, 8)
        y = dp(x)
        assert y.shape == (3, 4)

    def test_forward_1d(self):
        """Should work with unbatched input."""
        from pyquifer.learning import DifferentiablePlasticity
        dp = DifferentiablePlasticity(input_dim=8, output_dim=4)
        y = dp(torch.randn(8))
        assert y.shape == (4,)

    def test_hebbian_trace_updates(self):
        """Hebbian trace H should update after forward pass."""
        from pyquifer.learning import DifferentiablePlasticity
        dp = DifferentiablePlasticity(input_dim=8, output_dim=4, eta_init=0.1)
        assert (dp.H == 0).all()

        dp(torch.randn(8))
        assert dp.H.norm() > 0

    def test_hebbian_trace_clamped(self):
        """H should be clamped to [-1, 1]."""
        from pyquifer.learning import DifferentiablePlasticity
        dp = DifferentiablePlasticity(input_dim=8, output_dim=4, eta_init=1.0)

        # Many forward passes to build up H
        for _ in range(100):
            dp(torch.ones(8))

        assert dp.H.max() <= 1.0
        assert dp.H.min() >= -1.0

    def test_output_bounded(self):
        """Output should be in [-1, 1] due to tanh."""
        from pyquifer.learning import DifferentiablePlasticity
        dp = DifferentiablePlasticity(input_dim=8, output_dim=4)
        y = dp(torch.randn(10, 8) * 10)
        assert y.max() <= 1.0
        assert y.min() >= -1.0

    def test_reset_clears_trace(self):
        """Reset should zero Hebbian trace."""
        from pyquifer.learning import DifferentiablePlasticity
        dp = DifferentiablePlasticity(input_dim=8, output_dim=4)
        dp(torch.randn(8))
        dp.reset()
        assert (dp.H == 0).all()

    def test_alpha_eta_are_parameters(self):
        """Alpha and eta should be nn.Parameters (learnable by backprop)."""
        from pyquifer.learning import DifferentiablePlasticity
        dp = DifferentiablePlasticity(input_dim=8, output_dim=4)
        param_names = [name for name, _ in dp.named_parameters()]
        assert 'alpha' in param_names
        assert 'eta' in param_names

    def test_plasticity_affects_output(self):
        """With plasticity, repeated inputs should change behavior."""
        from pyquifer.learning import DifferentiablePlasticity
        dp = DifferentiablePlasticity(input_dim=4, output_dim=2, alpha_init=0.5, eta_init=0.1)

        x = torch.ones(4)
        y1 = dp(x).clone()

        # Run more forward passes to build Hebbian trace
        for _ in range(20):
            dp(x)

        y2 = dp(x)
        # Output should differ due to Hebbian contribution
        assert not torch.allclose(y1, y2, atol=1e-3)


class TestLearnableEligibilityTrace:
    """Tests for learnable eligibility trace."""

    def test_fixed_decay(self):
        """Non-learnable trace should use fixed decay."""
        from pyquifer.learning import LearnableEligibilityTrace
        trace = LearnableEligibilityTrace((5, 3), decay_rate=0.9, learnable=False)
        assert trace.decay_rate.item() == pytest.approx(0.9)

    def test_learnable_decay(self):
        """Learnable trace should have decay as parameter."""
        from pyquifer.learning import LearnableEligibilityTrace
        trace = LearnableEligibilityTrace((5, 3), decay_rate=0.9, learnable=True)
        assert hasattr(trace, 'decay_logit')
        assert trace.decay_rate.item() == pytest.approx(0.9, abs=0.01)

    def test_forward_updates_trace(self):
        """Forward should update internal trace."""
        from pyquifer.learning import LearnableEligibilityTrace
        trace = LearnableEligibilityTrace((5,), decay_rate=0.9)
        trace(torch.ones(5))
        assert trace.trace.norm() > 0

    def test_apply_reward(self):
        """apply_reward should return correct shape."""
        from pyquifer.learning import LearnableEligibilityTrace
        trace = LearnableEligibilityTrace((5, 3), decay_rate=0.9)
        trace(torch.randn(5), torch.randn(3))
        delta = trace.apply_reward(torch.tensor(1.0))
        assert delta.shape == (5, 3)

    def test_reset(self):
        """Reset should zero trace."""
        from pyquifer.learning import LearnableEligibilityTrace
        trace = LearnableEligibilityTrace((5,), decay_rate=0.9)
        trace(torch.ones(5))
        trace.reset()
        assert (trace.trace == 0).all()


# ── 5C-2: Speciated Neural Darwinism ──

class TestSpeciatedSelectionArena:
    """Tests for speciated selection arena."""

    def test_init(self):
        """Should initialize with correct number of groups."""
        from pyquifer.neural_darwinism import SpeciatedSelectionArena
        arena = SpeciatedSelectionArena(num_groups=6, group_dim=8)
        assert len(arena.groups) == 6
        assert arena.species_ids.shape == (6,)

    def test_forward_output_shape(self):
        """Forward should return correctly shaped output."""
        from pyquifer.neural_darwinism import SpeciatedSelectionArena
        arena = SpeciatedSelectionArena(num_groups=4, group_dim=8)
        result = arena(torch.randn(8))
        assert result['output'].shape == (8,)
        assert result['fitnesses'].shape == (4,)
        assert 'num_species' in result
        assert 'species_ids' in result

    def test_species_assignment(self):
        """Species should be assigned based on weight similarity."""
        from pyquifer.neural_darwinism import SpeciatedSelectionArena
        arena = SpeciatedSelectionArena(num_groups=6, group_dim=8,
                                       compatibility_threshold=0.5)
        arena._assign_species()
        # Should have at least 1 species
        assert arena.species_ids.unique().numel() >= 1

    def test_fitness_sharing(self):
        """Fitness sharing should divide by species size."""
        from pyquifer.neural_darwinism import SpeciatedSelectionArena
        arena = SpeciatedSelectionArena(num_groups=4, group_dim=8)
        # All same species
        arena.species_ids.fill_(0)
        raw = torch.tensor([0.8, 0.6, 0.4, 0.2])
        adjusted = arena._fitness_sharing(raw)
        # Each should be divided by 4
        assert adjusted[0].item() == pytest.approx(0.2)

    def test_stagnation_reduces_resources(self):
        """Stagnant species should have resources reduced."""
        from pyquifer.neural_darwinism import SpeciatedSelectionArena
        arena = SpeciatedSelectionArena(num_groups=4, group_dim=8,
                                       stagnation_limit=5)
        arena.species_ids.fill_(0)
        arena.species_best_fitness[0] = 0.9  # Set high best

        # Record initial resources
        initial_resources = [g.resources.item() for g in arena.groups]

        # Report low fitness many times (below the recorded best of 0.9)
        low_fitness = torch.tensor([0.1, 0.1, 0.1, 0.1])
        for _ in range(10):
            arena._stagnation_check(low_fitness)

        # After stagnation triggers, resources should be reduced
        # and species_best_fitness reset to 0.0
        final_resources = [g.resources.item() for g in arena.groups]
        # At least some groups should have had resources reduced
        assert any(f < i for f, i in zip(final_resources, initial_resources))

    def test_same_interface_as_selection_arena(self):
        """SpeciatedSelectionArena should be a drop-in for SelectionArena."""
        from pyquifer.neural_darwinism import SpeciatedSelectionArena
        arena = SpeciatedSelectionArena(num_groups=4, group_dim=8)
        result = arena(torch.randn(8))
        # Should have same keys as SelectionArena plus species info
        assert 'output' in result
        assert 'fitnesses' in result
        assert 'resources' in result
        assert 'mean_fitness' in result

    def test_reset(self):
        """Reset should clear all state."""
        from pyquifer.neural_darwinism import SpeciatedSelectionArena
        arena = SpeciatedSelectionArena(num_groups=4, group_dim=8)
        arena(torch.randn(8))
        arena.reset()
        assert arena.step_count.item() == 0
        assert (arena.species_ids == 0).all()


# ── 5C-3: AdEx Neuron Model ──

class TestAdExNeuron:
    """Tests for Adaptive Exponential Integrate-and-Fire neuron."""

    def test_init(self):
        """Should initialize with correct parameters."""
        from pyquifer.spiking import AdExNeuron
        adex = AdExNeuron(a=0.1, b=0.05)
        assert adex.a == 0.1
        assert adex.b == 0.05

    def test_init_state(self):
        """init_state should return V at EL and w at 0."""
        from pyquifer.spiking import AdExNeuron
        adex = AdExNeuron(EL=-0.7)
        V, w = adex.init_state((4, 10))
        assert V.shape == (4, 10)
        assert w.shape == (4, 10)
        assert torch.allclose(V, torch.full((4, 10), -0.7))
        assert (w == 0).all()

    def test_forward_shape(self):
        """Forward should return correctly shaped outputs."""
        from pyquifer.spiking import AdExNeuron
        adex = AdExNeuron()
        V, w = adex.init_state((2, 8))
        current = torch.randn(2, 8) * 0.5
        spikes, V_new, w_new = adex(current, V, w)
        assert spikes.shape == (2, 8)
        assert V_new.shape == (2, 8)
        assert w_new.shape == (2, 8)

    def test_spikes_on_strong_input(self):
        """Strong input should produce spikes."""
        from pyquifer.spiking import AdExNeuron
        adex = AdExNeuron(dt=1.0)
        V, w = adex.init_state((1, 10))

        total_spikes = 0
        for _ in range(100):
            current = torch.ones(1, 10) * 2.0  # Strong input
            spikes, V, w = adex(current, V, w)
            total_spikes += spikes.sum().item()

        assert total_spikes > 0

    def test_adaptation_reduces_firing(self):
        """With b > 0, firing rate should decrease over time (adaptation)."""
        from pyquifer.spiking import AdExNeuron
        adex = AdExNeuron(a=0.0, b=0.1, tau_w=50.0, dt=1.0)
        V, w = adex.init_state((1, 10))

        # Count spikes in early vs late window
        early_spikes = 0
        late_spikes = 0
        for t in range(200):
            current = torch.ones(1, 10) * 1.5
            spikes, V, w = adex(current, V, w)
            if t < 100:
                early_spikes += spikes.sum().item()
            else:
                late_spikes += spikes.sum().item()

        # Late should have fewer spikes due to adaptation
        assert late_spikes <= early_spikes

    def test_no_spikes_weak_input(self):
        """Weak input should not produce spikes."""
        from pyquifer.spiking import AdExNeuron
        adex = AdExNeuron(dt=1.0)
        adex.eval()  # Disable surrogate gradient
        V, w = adex.init_state((1, 10))

        total_spikes = 0
        for _ in range(50):
            current = torch.zeros(1, 10)  # No input at all
            spikes, V, w = adex(current, V, w)
            total_spikes += spikes.sum().item()

        assert total_spikes == 0

    def test_reset_after_spike(self):
        """V should reset to V_reset after spike."""
        from pyquifer.spiking import AdExNeuron
        adex = AdExNeuron(V_reset=-0.7, V_cutoff=0.5, dt=1.0)
        adex.eval()  # disable surrogate grad

        V, w = adex.init_state((1, 5))

        # Drive to spike
        for _ in range(100):
            spikes, V, w = adex(torch.ones(1, 5) * 3.0, V, w)
            # Where spikes occurred, V should be near V_reset
            if spikes.sum() > 0:
                spiked_V = V[spikes > 0.5]
                assert (spiked_V <= adex.V_reset + 0.1).all()
                break

    def test_adaptation_current_grows_with_spikes(self):
        """w should increase with each spike (b > 0)."""
        from pyquifer.spiking import AdExNeuron
        adex = AdExNeuron(a=0.0, b=0.1, dt=1.0)
        V, w = adex.init_state((1, 5))
        initial_w = w.clone()

        for _ in range(100):
            spikes, V, w = adex(torch.ones(1, 5) * 2.0, V, w)

        # w should have grown from spike-triggered increments
        assert w.mean() > initial_w.mean()

    def test_exponential_term_accelerates_spikes(self):
        """AdEx with large DT should produce spikes with moderate input."""
        from pyquifer.spiking import AdExNeuron
        adex = AdExNeuron(DT=0.2, dt=1.0)
        adex.eval()
        V, w = adex.init_state((1, 10))

        total_spikes = 0
        for t in range(100):
            spikes, V, w = adex(torch.ones(1, 10) * 0.5, V, w)
            total_spikes += spikes.sum().item()

        # Should produce spikes with moderate input
        assert total_spikes > 0
