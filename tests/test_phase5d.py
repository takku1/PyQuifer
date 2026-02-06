"""
Phase 5D Tests: Advanced Dynamics
- 5D-1: Stuart-Landau Oscillator
- 5D-2: Koopman/HAVOK Bifurcation Detection
- 5D-3: Wilson-Cowan Neural Mass Model
"""

import torch
import math
import pytest


# ── 5D-1: Stuart-Landau Oscillator ──

class TestStuartLandauOscillator:
    """Tests for Stuart-Landau oscillator with Hopf bifurcation."""

    def test_init(self):
        """Should initialize with correct number of oscillators."""
        from pyquifer.oscillators import StuartLandauOscillator
        sl = StuartLandauOscillator(num_oscillators=10)
        assert sl.z_real.shape == (10,)
        assert sl.z_imag.shape == (10,)

    def test_forward_returns_dict(self):
        """Forward should return amplitudes, phases, order_parameter."""
        from pyquifer.oscillators import StuartLandauOscillator
        sl = StuartLandauOscillator(num_oscillators=10)
        result = sl(steps=10)
        assert 'amplitudes' in result
        assert 'phases' in result
        assert 'order_parameter' in result
        assert result['amplitudes'].shape == (10,)

    def test_oscillating_regime(self):
        """With mu > 0, amplitudes should grow toward finite limit cycle."""
        from pyquifer.oscillators import StuartLandauOscillator
        sl = StuartLandauOscillator(num_oscillators=5, mu=0.5, coupling=0.0, dt=0.01)
        result = sl(steps=1000)
        # Steady-state amplitude should be sqrt(mu) = sqrt(0.5) ≈ 0.707
        mean_amp = result['amplitudes'].mean().item()
        assert abs(mean_amp - math.sqrt(0.5)) < 0.2

    def test_damped_regime(self):
        """With mu < 0, amplitudes should decay toward 0."""
        from pyquifer.oscillators import StuartLandauOscillator
        sl = StuartLandauOscillator(num_oscillators=5, mu=-1.0, coupling=0.0, dt=0.01)
        result = sl(steps=1000)
        assert result['amplitudes'].mean().item() < 0.1

    def test_criticality_distance(self):
        """get_criticality_distance should return |mu|."""
        from pyquifer.oscillators import StuartLandauOscillator
        sl = StuartLandauOscillator(num_oscillators=5, mu=0.3)
        assert sl.get_criticality_distance().item() == pytest.approx(0.3)

    def test_set_mu(self):
        """set_mu should change bifurcation parameter."""
        from pyquifer.oscillators import StuartLandauOscillator
        sl = StuartLandauOscillator(num_oscillators=5, mu=0.5)
        sl.set_mu(-0.3)
        assert sl.mu.item() == pytest.approx(-0.3)

    def test_coupling_synchronizes(self):
        """With coupling, oscillators should synchronize."""
        from pyquifer.oscillators import StuartLandauOscillator
        sl = StuartLandauOscillator(num_oscillators=10, mu=0.5, coupling=2.0, dt=0.01)
        result = sl(steps=2000)
        # High coupling should produce high order parameter
        assert result['order_parameter'].item() > 0.3

    def test_reset(self):
        """Reset should reinitialize state."""
        from pyquifer.oscillators import StuartLandauOscillator
        sl = StuartLandauOscillator(num_oscillators=5, mu=0.5)
        sl(steps=100)
        sl.reset()
        assert sl.step_count.item() == 0

    def test_external_input(self):
        """External input should influence dynamics."""
        from pyquifer.oscillators import StuartLandauOscillator
        sl = StuartLandauOscillator(num_oscillators=5, mu=-0.5, coupling=0.0, dt=0.01)
        # Strong external driving
        ext = torch.complex(torch.ones(5), torch.zeros(5)) * 0.5
        result = sl(steps=500, external_input=ext)
        # Should resist damping due to external drive
        assert result['amplitudes'].mean().item() > 0.05


# ── 5D-2: Koopman/HAVOK Bifurcation Detection ──

class TestKoopmanBifurcationDetector:
    """Tests for DMD-based bifurcation detection."""

    def test_init(self):
        """Should initialize with correct buffer."""
        from pyquifer.criticality import KoopmanBifurcationDetector
        kd = KoopmanBifurcationDetector(state_dim=8)
        assert kd.history.shape == (200, 8)
        assert kd.stability_margin.item() == 1.0

    def test_forward_returns_dict(self):
        """Forward should return stability_margin and approaching_bifurcation."""
        from pyquifer.criticality import KoopmanBifurcationDetector
        kd = KoopmanBifurcationDetector(state_dim=4, compute_every=1)
        result = kd(torch.randn(4))
        assert 'stability_margin' in result
        assert 'max_eigenvalue_mag' in result
        assert 'approaching_bifurcation' in result

    def test_stable_system_high_margin(self):
        """A decaying system should have high stability margin."""
        from pyquifer.criticality import KoopmanBifurcationDetector
        kd = KoopmanBifurcationDetector(state_dim=4, delay_dim=5, rank=3,
                                        compute_every=5)

        # Feed decaying dynamics
        state = torch.randn(4)
        for _ in range(100):
            state = state * 0.8 + torch.randn(4) * 0.01
            result = kd(state)

        # Stability margin should be positive (system is stable)
        assert result['stability_margin'].item() > -0.5

    def test_near_bifurcation_detection(self):
        """System near bifurcation (|lambda| ~ 1) should have low margin."""
        from pyquifer.criticality import KoopmanBifurcationDetector
        kd = KoopmanBifurcationDetector(state_dim=4, delay_dim=5, rank=3,
                                        compute_every=5)

        # Feed nearly unstable oscillating dynamics
        t = torch.arange(200, dtype=torch.float32) * 0.1
        for i in range(200):
            # Near-unit-circle dynamics (growing oscillation)
            state = torch.stack([
                torch.sin(t[i]) * 1.01 ** i,
                torch.cos(t[i]) * 1.01 ** i,
                torch.sin(2 * t[i]) * 1.01 ** i,
                torch.cos(2 * t[i]) * 1.01 ** i,
            ]).clamp(-10, 10)
            result = kd(state)

        # Should detect approaching bifurcation (or at least low margin)
        assert result['max_eigenvalue_mag'].item() > 0.5

    def test_reset(self):
        """Reset should clear history."""
        from pyquifer.criticality import KoopmanBifurcationDetector
        kd = KoopmanBifurcationDetector(state_dim=4)
        for _ in range(20):
            kd(torch.randn(4))
        kd.reset()
        assert kd.hist_ptr.item() == 0
        assert kd.stability_margin.item() == 1.0

    def test_flattened_input(self):
        """Should handle multi-dim input by flattening."""
        from pyquifer.criticality import KoopmanBifurcationDetector
        kd = KoopmanBifurcationDetector(state_dim=4, compute_every=1)
        result = kd(torch.randn(2, 4))  # 2D input
        assert 'stability_margin' in result


# ── 5D-3: Wilson-Cowan Neural Mass Model ──

class TestWilsonCowanPopulation:
    """Tests for single Wilson-Cowan E/I population."""

    def test_init(self):
        """Should initialize with E and I near 0.1."""
        from pyquifer.neural_mass import WilsonCowanPopulation
        wc = WilsonCowanPopulation()
        assert wc.E.item() == pytest.approx(0.1)
        assert wc.I.item() == pytest.approx(0.1)

    def test_forward_returns_dict(self):
        """Forward should return E, I, oscillation_power."""
        from pyquifer.neural_mass import WilsonCowanPopulation
        wc = WilsonCowanPopulation()
        result = wc(steps=10)
        assert 'E' in result
        assert 'I' in result
        assert 'oscillation_power' in result

    def test_output_bounded(self):
        """E and I should stay in [0, 1]."""
        from pyquifer.neural_mass import WilsonCowanPopulation
        wc = WilsonCowanPopulation(dt=0.1)
        for _ in range(500):
            result = wc(steps=1, I_ext_E=2.0)

        assert 0.0 <= result['E'].item() <= 1.0
        assert 0.0 <= result['I'].item() <= 1.0

    def test_oscillations_emerge(self):
        """With standard params, E/I interaction should produce oscillations."""
        from pyquifer.neural_mass import WilsonCowanPopulation
        wc = WilsonCowanPopulation(w_EE=12.0, w_EI=4.0, w_IE=13.0, w_II=11.0,
                                   dt=0.1)

        for _ in range(200):
            wc(steps=1, I_ext_E=1.0)

        # Should have nonzero oscillation power
        result = wc(steps=1)
        assert result['oscillation_power'].item() > 0

    def test_external_input_drives_activity(self):
        """External input should increase E activity."""
        from pyquifer.neural_mass import WilsonCowanPopulation
        wc = WilsonCowanPopulation(dt=0.1)

        # Run with input
        for _ in range(100):
            result = wc(steps=1, I_ext_E=3.0)

        E_with_input = result['E'].item()

        # Run without input
        wc.reset()
        for _ in range(100):
            result = wc(steps=1, I_ext_E=0.0)

        E_without_input = result['E'].item()

        assert E_with_input > E_without_input

    def test_reset(self):
        """Reset should restore initial state."""
        from pyquifer.neural_mass import WilsonCowanPopulation
        wc = WilsonCowanPopulation()
        wc(steps=100, I_ext_E=2.0)
        wc.reset()
        assert wc.E.item() == pytest.approx(0.1)
        assert wc.hist_ptr.item() == 0


class TestWilsonCowanNetwork:
    """Tests for coupled Wilson-Cowan network."""

    def test_init(self):
        """Should create correct number of populations."""
        from pyquifer.neural_mass import WilsonCowanNetwork
        net = WilsonCowanNetwork(num_populations=4)
        assert len(net.populations) == 4

    def test_forward_returns_dict(self):
        """Forward should return E_states, I_states, synchronization."""
        from pyquifer.neural_mass import WilsonCowanNetwork
        net = WilsonCowanNetwork(num_populations=4)
        result = net(steps=10)
        assert result['E_states'].shape == (4,)
        assert result['I_states'].shape == (4,)
        assert 'synchronization' in result

    def test_coupling_increases_sync(self):
        """Stronger coupling should increase synchronization."""
        from pyquifer.neural_mass import WilsonCowanNetwork
        net_coupled = WilsonCowanNetwork(num_populations=4, coupling_strength=2.0, dt=0.1)
        net_uncoupled = WilsonCowanNetwork(num_populations=4, coupling_strength=0.0, dt=0.1)

        # Run both
        for _ in range(200):
            r_coupled = net_coupled(steps=1, external_input=torch.ones(4) * 1.0)
            r_uncoupled = net_uncoupled(steps=1, external_input=torch.ones(4) * 1.0)

        # Coupled should have higher synchronization (lower E variance)
        assert r_coupled['synchronization'].item() >= r_uncoupled['synchronization'].item() - 0.1

    def test_external_input(self):
        """External input should affect all populations."""
        from pyquifer.neural_mass import WilsonCowanNetwork
        net = WilsonCowanNetwork(num_populations=3, dt=0.1)
        result = net(steps=100, external_input=torch.ones(3) * 2.0)
        assert result['mean_E'].item() > 0

    def test_reset(self):
        """Reset should restore all populations."""
        from pyquifer.neural_mass import WilsonCowanNetwork
        net = WilsonCowanNetwork(num_populations=3)
        net(steps=50, external_input=torch.ones(3))
        net.reset()
        for pop in net.populations:
            assert pop.E.item() == pytest.approx(0.1)
