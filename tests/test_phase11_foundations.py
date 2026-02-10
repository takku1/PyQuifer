"""Tests for Phase 11 foundation modules: ODE solvers, complex oscillators, CUDA kernels."""
import pytest
import torch
import math


# ── ODE Solvers ──────────────────────────────────────────────────────────────

class TestODESolvers:
    def test_euler_solver_instantiate(self):
        from pyquifer.ode_solvers import EulerSolver
        solver = EulerSolver()
        assert solver is not None

    def test_rk4_solver_instantiate(self):
        from pyquifer.ode_solvers import RK4Solver
        solver = RK4Solver()
        assert solver is not None

    def test_dopri_solver_instantiate(self):
        from pyquifer.ode_solvers import DopriSolver
        solver = DopriSolver()
        assert solver is not None

    def test_euler_exponential_decay(self):
        from pyquifer.ode_solvers import EulerSolver
        solver = EulerSolver()
        y0 = torch.tensor([1.0])
        def f(t, y): return -y
        result = solver(f, y0, t_span=(0.0, 1.0), dt=0.001)
        expected = math.exp(-1.0)
        # Result is a tensor (last state)
        final = result[-1] if result.dim() > 0 and result.shape[0] > 1 else result
        assert abs(final.item() - expected) < 0.01

    def test_rk4_more_accurate_than_euler(self):
        from pyquifer.ode_solvers import EulerSolver, RK4Solver
        y0 = torch.tensor([1.0])
        def f(t, y): return -y
        euler_result = EulerSolver()(f, y0, t_span=(0.0, 1.0), dt=0.01)
        rk4_result = RK4Solver()(f, y0, t_span=(0.0, 1.0), dt=0.01)
        expected = math.exp(-1.0)
        euler_final = euler_result[-1] if euler_result.dim() > 0 and euler_result.shape[0] > 1 else euler_result
        rk4_final = rk4_result[-1] if rk4_result.dim() > 0 and rk4_result.shape[0] > 1 else rk4_result
        assert abs(rk4_final.item() - expected) < abs(euler_final.item() - expected) + 1e-8

    def test_dopri_adaptive_stepping(self):
        from pyquifer.ode_solvers import DopriSolver
        solver = DopriSolver()
        y0 = torch.tensor([1.0])
        def f(t, y): return -y
        result = solver(f, y0, t_span=(0.0, 1.0), dt=0.1)
        final = result[-1] if result.dim() > 0 and result.shape[0] > 1 else result
        expected = math.exp(-1.0)
        assert abs(final.item() - expected) < 0.01

    def test_solve_ivp_functional_api(self):
        from pyquifer.ode_solvers import solve_ivp
        y0 = torch.tensor([1.0])
        def f(t, y): return -y
        result = solve_ivp(f, y0, t_span=(0.0, 0.5), dt=0.001)
        assert result is not None

    def test_create_solver_factory(self):
        from pyquifer.ode_solvers import create_solver, SolverConfig
        for method in ['euler', 'rk4', 'dopri']:
            solver = create_solver(SolverConfig(method=method))
            assert solver is not None

    def test_solver_config_presets(self):
        from pyquifer.ode_solvers import SolverConfig
        cfg = SolverConfig()
        assert cfg.atol > 0
        assert cfg.rtol > 0

    def test_batched_solver(self):
        from pyquifer.ode_solvers import RK4Solver
        solver = RK4Solver()
        y0 = torch.randn(4, 3)  # batch of 4, dim 3
        def f(t, y): return -0.5 * y
        result = solver(f, y0, t_span=(0.0, 0.1), dt=0.01)
        assert result is not None


# ── Complex Oscillators ──────────────────────────────────────────────────────

class TestComplexOscillators:
    def test_complex_kuramoto_instantiate(self):
        from pyquifer.complex_oscillators import ComplexKuramotoBank
        bank = ComplexKuramotoBank(num_oscillators=16)
        assert bank is not None

    def test_complex_kuramoto_forward(self):
        from pyquifer.complex_oscillators import ComplexKuramotoBank
        bank = ComplexKuramotoBank(num_oscillators=8)
        result = bank()
        assert result is not None

    def test_complex_kuramoto_methods(self):
        from pyquifer.complex_oscillators import ComplexKuramotoBank
        bank = ComplexKuramotoBank(num_oscillators=8)
        phases = bank.get_phases()
        assert phases.shape[0] == 8
        amps = bank.get_amplitudes()
        assert amps.shape[0] == 8
        R = bank.get_order_parameter()
        assert R.numel() >= 1

    def test_complex_coupling(self):
        from pyquifer.complex_oscillators import ComplexCoupling
        coupling = ComplexCoupling(num_oscillators=8)
        z = torch.randn(8) + 1j * torch.randn(8)
        z = z.to(torch.complex64)
        result = coupling(z)
        assert result.shape == z.shape

    def test_mod_relu(self):
        from pyquifer.complex_oscillators import ModReLU
        act = ModReLU(16)
        z = torch.randn(4, 16, dtype=torch.complex64)
        out = act(z)
        assert out.shape == z.shape
        assert out.dtype == torch.complex64

    def test_complex_linear(self):
        from pyquifer.complex_oscillators import ComplexLinear
        layer = ComplexLinear(16, 8)
        z = torch.randn(4, 16, dtype=torch.complex64)
        out = layer(z)
        assert out.shape == (4, 8)

    def test_complex_batchnorm(self):
        from pyquifer.complex_oscillators import ComplexBatchNorm
        bn = ComplexBatchNorm(16)
        z = torch.randn(8, 16, dtype=torch.complex64)
        out = bn(z)
        assert out.shape == (8, 16)

    def test_complex_order_parameter(self):
        from pyquifer.complex_oscillators import complex_order_parameter
        # All-ones phasors = perfectly synchronized
        z = torch.ones(16, dtype=torch.complex64)
        R = complex_order_parameter(z)
        assert R.abs().item() > 0.99

    def test_complex_kuramoto_synchronization(self):
        from pyquifer.complex_oscillators import ComplexKuramotoBank
        bank = ComplexKuramotoBank(num_oscillators=8)
        for _ in range(200):
            bank()
        R = bank.get_order_parameter()
        # Just verify it runs without error and R is a valid number
        assert R.numel() >= 1


# ── CUDA Kernels ─────────────────────────────────────────────────────────────

class TestCUDAKernels:
    def test_kernel_module_imports(self):
        from pyquifer._cuda.kuramoto_kernel import KuramotoCUDAKernel, TensorDiagnostics
        assert KuramotoCUDAKernel is not None
        assert TensorDiagnostics is not None

    def test_kuramoto_kernel_instantiate(self):
        from pyquifer._cuda.kuramoto_kernel import KuramotoCUDAKernel
        kernel = KuramotoCUDAKernel(use_triton=False)
        assert kernel is not None

    def test_kernel_cpu_fallback_coupling(self):
        from pyquifer._cuda.kuramoto_kernel import KuramotoCUDAKernel
        kernel = KuramotoCUDAKernel(use_triton=False)
        phases = torch.rand(8) * 2 * math.pi
        coupling_matrix = torch.ones(8, 8)
        coupling = kernel.all_to_all_coupling(phases, coupling_matrix, 2.0)
        assert coupling.shape == (8,)

    def test_kernel_fused_order_parameter(self):
        from pyquifer._cuda.kuramoto_kernel import KuramotoCUDAKernel
        kernel = KuramotoCUDAKernel(use_triton=False)
        phases = torch.zeros(16)  # all synchronized
        R, Psi = kernel.fused_order_parameter(phases)
        assert R.item() > 0.99

    def test_kernel_fused_step(self):
        from pyquifer._cuda.kuramoto_kernel import KuramotoCUDAKernel
        kernel = KuramotoCUDAKernel(use_triton=False)
        phases = torch.rand(8) * 2 * math.pi
        omega = torch.ones(8)
        coupling_matrix = torch.ones(8, 8)
        new_phases = kernel.fused_step(phases, omega, coupling_matrix, K=1.0, dt=0.01)
        assert new_phases.shape == (8,)

    def test_tensor_diagnostics_roundtrip(self):
        from pyquifer._cuda.kuramoto_kernel import TensorDiagnostics
        td = TensorDiagnostics()
        data = {'coherence': torch.tensor(0.75), 'complexity': torch.tensor(3.14)}
        sdict = td.to_scalar_dict(data)
        assert abs(sdict['coherence'] - 0.75) < 1e-5

    def test_tensor_diagnostics_extract(self):
        from pyquifer._cuda.kuramoto_kernel import TensorDiagnostics
        td = TensorDiagnostics()
        data = {'val_0': torch.tensor(0.0), 'val_1': torch.tensor(1.0)}
        scalars = td.extract_scalars(data, ['val_0', 'val_1'])
        assert len(scalars) == 2
