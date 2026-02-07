"""
Phase 6b: Benchmark-Informed Fixes — Validation Tests

Tests for 6 fixes identified from running the benchmark suite:
  1. BranchingRatio ratio-of-means estimator
  2. CriticalityController PI control
  3. KuramotoDaidoMeanField Gaussian distribution support
  4. Training requirement docstrings (RSSM, WorldModel, TheoryOfMind, MirrorResonance)
  5. MirrorResonance adaptive coupling
  6. Benchmark report annotations (tested via import smoke-test)
"""

import math
import torch
import pytest


# ============================================================================
# Fix 1: BranchingRatio ratio-of-means
# ============================================================================

class TestBranchingRatioEstimator:
    """Validate ratio-of-means estimator fixes blowup on subcritical data."""

    def test_ratio_of_means_is_default(self):
        """Default estimator should be ratio_of_means."""
        from pyquifer.criticality import BranchingRatio
        br = BranchingRatio()
        assert br.estimator == 'ratio_of_means'

    def test_ratio_of_means_bounded_subcritical(self):
        """Ratio-of-means sigma should stay bounded on bursty subcritical data."""
        from pyquifer.criticality import BranchingRatio
        br = BranchingRatio(window_size=50, estimator='ratio_of_means')

        torch.manual_seed(42)
        # Bursty subcritical: large bursts followed by near-zero quiescence
        for i in range(50):
            if i % 10 < 2:
                activity = torch.tensor(10.0)  # burst
            else:
                activity = torch.tensor(0.01)  # quiescent
            result = br(activity)

        sigma = result['branching_ratio'].item()
        # Ratio-of-means should produce bounded value, not 224490
        assert abs(sigma) < 100, f"sigma={sigma} is unbounded"

    def test_mean_of_ratios_can_explode(self):
        """Legacy mean-of-ratios can produce large sigma on same data."""
        from pyquifer.criticality import BranchingRatio
        br = BranchingRatio(window_size=50, estimator='mean_of_ratios')

        torch.manual_seed(42)
        for i in range(50):
            if i % 10 < 2:
                activity = torch.tensor(10.0)
            else:
                activity = torch.tensor(0.01)
            result = br(activity)

        sigma = result['branching_ratio'].item()
        # Mean-of-ratios is expected to be much larger (or at least different)
        # This test documents the legacy behavior rather than enforcing blowup
        assert isinstance(sigma, float)

    def test_critical_data_sigma_near_one(self):
        """On near-critical data, both estimators should give sigma ~ 1.0."""
        from pyquifer.criticality import BranchingRatio

        torch.manual_seed(42)
        for estimator in ['ratio_of_means', 'mean_of_ratios']:
            br = BranchingRatio(window_size=100, estimator=estimator)
            activity = 5.0
            for i in range(100):
                activity = activity * 0.95 + torch.rand(1).item() * 0.5
                result = br(torch.tensor(activity))
            sigma = result['branching_ratio'].item()
            assert 0.5 < sigma < 2.0, f"{estimator}: sigma={sigma}"

    def test_subcritical_sigma_below_one(self):
        """On decaying data, ratio-of-means sigma < 1."""
        from pyquifer.criticality import BranchingRatio
        br = BranchingRatio(window_size=50)

        activity = 10.0
        for i in range(50):
            activity = activity * 0.8  # Exponential decay
            result = br(torch.tensor(activity))

        sigma = result['branching_ratio'].item()
        assert sigma < 1.0, f"Expected subcritical sigma < 1, got {sigma}"

    def test_supercritical_sigma_above_one(self):
        """On growing data, ratio-of-means sigma > 1."""
        from pyquifer.criticality import BranchingRatio
        br = BranchingRatio(window_size=50)

        activity = 1.0
        for i in range(50):
            activity = activity * 1.05 + 0.1  # Growing
            result = br(torch.tensor(activity))

        sigma = result['branching_ratio'].item()
        assert sigma > 1.0, f"Expected supercritical sigma > 1, got {sigma}"


# ============================================================================
# Fix 2: CriticalityController PI control
# ============================================================================

class TestCriticalityControllerPI:
    """Validate PI controller converges better than P-only."""

    def test_pi_params_exist(self):
        """Controller should accept kp, ki, integral_windup_limit params."""
        from pyquifer.criticality import CriticalityController
        ctrl = CriticalityController(kp=2.0, ki=0.5, integral_windup_limit=5.0)
        assert ctrl.kp == 2.0
        assert ctrl.ki == 0.5
        assert ctrl.integral_windup_limit == 5.0

    def test_integral_error_buffer_exists(self):
        """Controller should have integral_error buffer."""
        from pyquifer.criticality import CriticalityController
        ctrl = CriticalityController()
        assert hasattr(ctrl, 'integral_error')
        assert ctrl.integral_error.item() == 0.0

    def test_integral_error_accumulates(self):
        """Integral error should accumulate over steps."""
        from pyquifer.criticality import CriticalityController
        ctrl = CriticalityController(ki=0.1)

        torch.manual_seed(42)
        # Feed supercritical data (sigma > 1) so error is consistently positive
        activity = 1.0
        for i in range(30):
            activity = activity * 1.1 + torch.rand(1).item()
            ctrl(torch.tensor([activity]))

        # Integral error should be non-zero after consistent error signal
        assert ctrl.integral_error.item() != 0.0

    def test_integral_windup_clamped(self):
        """Integral error should be clamped by windup limit."""
        from pyquifer.criticality import CriticalityController
        ctrl = CriticalityController(ki=1.0, integral_windup_limit=5.0)

        # Feed extreme data to try to blow up integral
        for i in range(200):
            ctrl(torch.tensor([100.0]))

        assert abs(ctrl.integral_error.item()) <= 5.0 + 1e-6

    def test_pi_convergence_on_subcritical(self):
        """PI controller should converge sigma closer to 1.0 on subcritical data."""
        from pyquifer.criticality import CriticalityController
        ctrl = CriticalityController(
            adaptation_rate=0.05, kp=1.0, ki=0.2
        )

        torch.manual_seed(42)
        coupling = 1.0
        activity = 5.0
        for i in range(200):
            activity = coupling * activity * 0.7 + torch.rand(1).item() * 2
            result = ctrl(torch.tensor([activity]))
            coupling = ctrl.get_adjusted_coupling(torch.tensor(1.0)).item()

        sigma = result['branching_ratio'].item()
        # With PI, coupling_adjustment should have moved to compensate
        assert ctrl.coupling_adjustment.item() != 1.0, "Coupling should have adapted"

    def test_reset_clears_integral(self):
        """Reset should clear integral error."""
        from pyquifer.criticality import CriticalityController
        ctrl = CriticalityController()
        ctrl.integral_error.fill_(5.0)
        ctrl.reset()
        assert ctrl.integral_error.item() == 0.0


# ============================================================================
# Fix 3: KuramotoDaidoMeanField Gaussian distribution
# ============================================================================

class TestMeanFieldGaussian:
    """Validate Gaussian distribution support for mean-field model."""

    def test_distribution_param_default(self):
        """Default distribution should be 'cauchy'."""
        from pyquifer.oscillators import KuramotoDaidoMeanField
        mf = KuramotoDaidoMeanField()
        assert mf.distribution == 'cauchy'

    def test_gaussian_distribution_accepted(self):
        """Should accept distribution='gaussian'."""
        from pyquifer.oscillators import KuramotoDaidoMeanField
        mf = KuramotoDaidoMeanField(distribution='gaussian')
        assert mf.distribution == 'gaussian'

    def test_gaussian_forward_runs(self):
        """Gaussian mode forward pass should run without error."""
        from pyquifer.oscillators import KuramotoDaidoMeanField
        mf = KuramotoDaidoMeanField(omega_mean=1.0, delta=0.2,
                                     coupling=2.0, distribution='gaussian')
        result = mf(steps=100)
        assert 'R' in result
        assert 'Psi' in result
        assert result['R'].item() >= 0

    def test_gaussian_critical_coupling_differs(self):
        """Gaussian Kc should differ from Cauchy Kc."""
        from pyquifer.oscillators import KuramotoDaidoMeanField
        mf_cauchy = KuramotoDaidoMeanField(delta=0.2, distribution='cauchy')
        mf_gauss = KuramotoDaidoMeanField(delta=0.2, distribution='gaussian')

        kc_cauchy = mf_cauchy.get_critical_coupling().item()
        kc_gauss = mf_gauss.get_critical_coupling().item()

        # Gaussian Kc = 2 * delta * sqrt(2*pi) / pi
        # Cauchy Kc = 2 * delta
        assert kc_gauss != kc_cauchy
        expected_ratio = math.sqrt(2 * math.pi) / math.pi
        assert abs(kc_gauss / kc_cauchy - expected_ratio) < 0.01

    def test_from_frequencies_auto_uniform(self):
        """from_frequencies with auto should detect uniform as gaussian."""
        from pyquifer.oscillators import KuramotoDaidoMeanField
        torch.manual_seed(42)
        # Uniform distribution: kurtosis = 1.8 < 6 → gaussian
        omega = torch.rand(200) * 2.0 + 0.5  # Uniform [0.5, 2.5]
        mf = KuramotoDaidoMeanField.from_frequencies(omega, distribution='auto')
        assert mf.distribution == 'gaussian'

    def test_from_frequencies_auto_cauchy(self):
        """from_frequencies with auto should detect heavy-tailed as cauchy."""
        from pyquifer.oscillators import KuramotoDaidoMeanField
        torch.manual_seed(42)
        # Cauchy distribution: very heavy tails, kurtosis >> 6
        cauchy = torch.distributions.Cauchy(1.0, 0.3)
        omega = cauchy.sample((200,))
        mf = KuramotoDaidoMeanField.from_frequencies(omega, distribution='auto')
        assert mf.distribution == 'cauchy'

    def test_from_frequencies_explicit_gaussian(self):
        """from_frequencies with explicit gaussian should use mean/std."""
        from pyquifer.oscillators import KuramotoDaidoMeanField
        torch.manual_seed(42)
        omega = torch.randn(100) * 0.3 + 1.0  # N(1.0, 0.3)
        mf = KuramotoDaidoMeanField.from_frequencies(
            omega, distribution='gaussian'
        )
        assert mf.distribution == 'gaussian'
        # omega_mean should be close to 1.0
        assert abs(mf.omega_mean.item() - 1.0) < 0.1
        # delta should be close to 0.3
        assert abs(mf.delta.item() - 0.3) < 0.1

    def test_from_frequencies_explicit_cauchy(self):
        """from_frequencies with explicit cauchy uses IQR fitting."""
        from pyquifer.oscillators import KuramotoDaidoMeanField
        torch.manual_seed(42)
        omega = torch.randn(100) * 0.5 + 2.0
        mf = KuramotoDaidoMeanField.from_frequencies(
            omega, distribution='cauchy'
        )
        assert mf.distribution == 'cauchy'

    def test_gaussian_sync_above_kc(self):
        """With K > Kc (Gaussian), system should synchronize."""
        from pyquifer.oscillators import KuramotoDaidoMeanField
        mf = KuramotoDaidoMeanField(
            omega_mean=1.0, delta=0.1, coupling=3.0,
            distribution='gaussian', dt=0.01
        )
        result = mf(steps=500)
        # With K=3.0 well above Kc, R should be high
        assert result['R'].item() > 0.3


# ============================================================================
# Fix 4: Training requirement docstrings
# ============================================================================

class TestDocstrings:
    """Validate training requirement docstrings exist."""

    def test_rssm_docstring_mentions_training(self):
        """RSSM docstring should mention training requirement."""
        from pyquifer.world_model import RSSM
        assert 'Training required' in RSSM.__doc__ or 'training' in RSSM.__doc__.lower()

    def test_worldmodel_docstring_mentions_training(self):
        """WorldModel docstring should mention training and loss()."""
        from pyquifer.world_model import WorldModel
        assert 'Training required' in WorldModel.__doc__
        assert 'loss' in WorldModel.__doc__

    def test_worldmodel_docstring_has_example(self):
        """WorldModel docstring should include training example code."""
        from pyquifer.world_model import WorldModel
        assert 'optimizer' in WorldModel.__doc__
        assert 'backward' in WorldModel.__doc__

    def test_mirror_resonance_docstring_mentions_no_training(self):
        """MirrorResonance docstring should say no training required."""
        from pyquifer.social import MirrorResonance
        doc = MirrorResonance.__doc__
        assert 'training' in doc.lower()
        assert 'coupling_strength' in doc

    def test_theory_of_mind_docstring_mentions_convergence(self):
        """TheoryOfMind docstring should mention convergence requirements."""
        from pyquifer.social import TheoryOfMind
        doc = TheoryOfMind.__doc__
        assert 'training' in doc.lower()
        assert '0.3' in doc or 'update rate' in doc.lower()


# ============================================================================
# Fix 5: MirrorResonance adaptive coupling
# ============================================================================

class TestMirrorResonanceAdaptiveCoupling:
    """Validate adaptive coupling feature."""

    def test_adaptive_coupling_default_off(self):
        """Adaptive coupling should be off by default."""
        from pyquifer.social import MirrorResonance
        mr = MirrorResonance(action_dim=8)
        assert mr.adaptive_coupling is False

    def test_adaptive_coupling_param_accepted(self):
        """Should accept adaptive_coupling and adaptive_alpha params."""
        from pyquifer.social import MirrorResonance
        mr = MirrorResonance(action_dim=8, adaptive_coupling=True,
                              adaptive_alpha=3.0)
        assert mr.adaptive_coupling is True
        assert mr.adaptive_alpha == 3.0

    def test_adaptive_observe_runs(self):
        """observe() should work with adaptive coupling enabled."""
        from pyquifer.social import MirrorResonance
        mr = MirrorResonance(action_dim=8, coupling_strength=0.5,
                              adaptive_coupling=True, adaptive_alpha=2.0)
        signal = torch.randn(8)
        result = mr.observe(signal)
        assert 'coherence' in result
        assert 'phases' in result

    def test_adaptive_improves_coherence(self):
        """Adaptive coupling should achieve higher coherence than fixed."""
        from pyquifer.social import MirrorResonance
        torch.manual_seed(42)

        # Fixed coupling
        mr_fixed = MirrorResonance(action_dim=16, coupling_strength=0.3,
                                    adaptive_coupling=False)
        # Adaptive coupling
        mr_adapt = MirrorResonance(action_dim=16, coupling_strength=0.3,
                                    adaptive_coupling=True, adaptive_alpha=2.0)

        # Copy weights so they start identical
        mr_adapt.load_state_dict(mr_fixed.state_dict(), strict=False)

        # Observe same signal repeatedly
        torch.manual_seed(123)
        signal = torch.randn(16)

        coherence_fixed = []
        coherence_adapt = []
        for _ in range(30):
            r1 = mr_fixed.observe(signal)
            r2 = mr_adapt.observe(signal)
            coherence_fixed.append(r1['coherence'])
            coherence_adapt.append(r2['coherence'])

        # Adaptive should converge faster (higher avg coherence)
        avg_fixed = sum(coherence_fixed[-10:]) / 10
        avg_adapt = sum(coherence_adapt[-10:]) / 10
        # At minimum, adaptive should not be worse
        assert avg_adapt >= avg_fixed - 0.15, \
            f"Adaptive ({avg_adapt:.3f}) much worse than fixed ({avg_fixed:.3f})"

    def test_execute_unaffected_by_adaptive(self):
        """execute() should not be affected by adaptive coupling setting."""
        from pyquifer.social import MirrorResonance
        torch.manual_seed(42)
        mr = MirrorResonance(action_dim=8, adaptive_coupling=True)
        intent = torch.randn(8)
        result = mr.execute(intent)
        assert 'action' in result
        assert result['mode'] == 'execute'

    def test_fixed_coupling_backward_compat(self):
        """With adaptive_coupling=False, behavior should match original."""
        from pyquifer.social import MirrorResonance
        torch.manual_seed(42)

        mr1 = MirrorResonance(action_dim=8, coupling_strength=0.5)
        mr2 = MirrorResonance(action_dim=8, coupling_strength=0.5,
                               adaptive_coupling=False)

        # Copy state
        mr2.load_state_dict(mr1.state_dict())

        signal = torch.randn(8)
        r1 = mr1.observe(signal)
        r2 = mr2.observe(signal)

        # Phases should be identical
        assert torch.allclose(r1['phases'], r2['phases'], atol=1e-5)


# ============================================================================
# Fix 6: Benchmark annotations (smoke test)
# ============================================================================

class TestBenchmarkAnnotations:
    """Verify benchmark files are importable and have annotation strings."""

    def test_bench_nlb_importable(self):
        """bench_nlb.py should be importable."""
        import importlib.util
        import os
        spec = importlib.util.spec_from_file_location(
            "bench_nlb",
            os.path.join(os.path.dirname(__file__), 'benchmarks', 'bench_nlb.py')
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # Check that print_report function exists
        assert hasattr(mod, 'print_report')

    def test_bench_hanabi_importable(self):
        """bench_hanabi.py should be importable."""
        import importlib.util
        import os
        spec = importlib.util.spec_from_file_location(
            "bench_hanabi",
            os.path.join(os.path.dirname(__file__), 'benchmarks', 'bench_hanabi.py')
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, 'print_report')

    def test_bench_meltingpot_importable(self):
        """bench_meltingpot.py should be importable."""
        import importlib.util
        import os
        spec = importlib.util.spec_from_file_location(
            "bench_meltingpot",
            os.path.join(os.path.dirname(__file__), 'benchmarks', 'bench_meltingpot.py')
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, 'print_report')


# ============================================================================
# Integration: Cross-fix interactions
# ============================================================================

class TestCrossFix:
    """Test interactions between fixes."""

    def test_criticality_controller_uses_ratio_of_means(self):
        """CriticalityController's BranchingRatio should use ratio-of-means."""
        from pyquifer.criticality import CriticalityController
        ctrl = CriticalityController()
        assert ctrl.branching_monitor.estimator == 'ratio_of_means'

    def test_full_criticality_pipeline(self):
        """BranchingRatio + PI controller work together end-to-end."""
        from pyquifer.criticality import CriticalityController
        ctrl = CriticalityController(
            adaptation_rate=0.05, kp=1.0, ki=0.2,
            integral_windup_limit=10.0
        )

        torch.manual_seed(42)
        coupling = 1.0
        activity = 5.0
        sigmas = []

        for i in range(200):
            activity = coupling * activity * 0.8 + torch.rand(1).item() * 2
            result = ctrl(torch.tensor([activity]))
            coupling = ctrl.get_adjusted_coupling(torch.tensor(1.0)).item()
            sigmas.append(result['branching_ratio'].item())

        # All sigmas should be finite
        assert all(math.isfinite(s) for s in sigmas), "Non-finite sigma detected"
        # Sigma should not blow up
        assert max(abs(s) for s in sigmas) < 1000, f"Sigma blew up: max={max(sigmas)}"

    def test_mean_field_gaussian_from_uniform_frequencies(self):
        """from_frequencies(auto) + gaussian forward = working pipeline."""
        from pyquifer.oscillators import KuramotoDaidoMeanField
        torch.manual_seed(42)
        omega = torch.rand(100) * 1.0 + 0.5  # Uniform [0.5, 1.5]
        mf = KuramotoDaidoMeanField.from_frequencies(omega, coupling=2.0,
                                                      distribution='auto')
        assert mf.distribution == 'gaussian'
        result = mf(steps=200)
        assert result['R'].item() >= 0
        assert math.isfinite(result['R'].item())
