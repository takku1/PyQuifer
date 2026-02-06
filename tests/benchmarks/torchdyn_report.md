# Benchmark Report: PyQuifer Continuous Dynamics vs torchdyn

**Date:** 2026-02-06
**PyQuifer version:** Phase 5 (479 tests, 12 enhancements)
**Benchmark repo:** `tests/benchmarks/torchdyn/` -- Poli et al. (2020), DiffEqML
**Script:** `tests/benchmarks/bench_torchdyn.py`

---

## Executive Summary

PyQuifer's ODE-evolving dynamics (Kuramoto, Wilson-Cowan, Stuart-Landau) can all be **wrapped as NeuralODE vector fields** and integrated with higher-order solvers. This validates that PyQuifer's dynamics are well-formed ODEs suitable for the torchdyn ecosystem. The key difference: torchdyn provides **framework abstractions** (NeuralODE/SDE wrappers, adjoint sensitivity, Hamiltonian NNs, Galerkin layers, continuous normalizing flows) while PyQuifer provides **domain-specific neuroscience models** (oscillators, spiking neurons, criticality detection, consciousness metrics). Autograd and adjoint-style backprop produce identical gradients, with adjoint using 63x less memory. No new correctness gaps were found -- torchdyn validates PyQuifer's ODE formulations and highlights the memory cost of PyQuifer's current store-all-states backprop approach.

## What torchdyn Is

torchdyn (DiffEqML) is a higher-level framework built on top of torchdiffeq, providing:

- **NeuralODE/NeuralSDE wrappers**: Wrap any `(t, x) -> dx/dt` function as a differentiable layer
- **Sensitivity methods**: Autograd (store all), adjoint (O(1) memory), interpolated adjoint (dense output)
- **Hamiltonian/Lagrangian NNs**: Energy-preserving dynamics with automatic symplectic structure
- **Galerkin basis layers**: Spectral decomposition of continuous-depth networks
- **Continuous normalizing flows**: Density estimation via ODE-defined transforms
- **Multiple shooting**: Parallel-in-time ODE solving for faster training

**Key difference**: torchdyn is a *framework for building continuous-depth models*. PyQuifer uses continuous dynamics *within a specific cognitive architecture*. torchdyn's question: "How do we parameterize and train continuous dynamics?" PyQuifer's question: "How do we model biological oscillations and neural populations for cognitive AI?"

## Results

### 1. NeuralODE Wrapping

Can PyQuifer's ODE dynamics be expressed as NeuralODE vector fields?

| System | Wrappable | Trajectory Valid | Steps | Time |
|--------|:---------:|:----------------:|:-----:|:----:|
| Kuramoto (N=20) | YES | YES | 201 | 20.9 ms |
| Wilson-Cowan | YES | YES | 501 | 155.2 ms |
| Stuart-Landau (N=20) | YES | YES | 201 | 47.6 ms |

**Analysis:**

- **All three systems wrappable**: Kuramoto phase coupling, Wilson-Cowan population dynamics, and Stuart-Landau complex oscillators all implement the standard `(t, x) -> dx/dt` interface. This means they could use torchdyn's adaptive solvers, adjoint backprop, and other features as drop-in improvements.

- **RK4 integration used**: The NeuralODE wrapper here uses RK4 (order 4) rather than Euler (order 1). All trajectories remain finite and stable, confirming that PyQuifer's dynamics are well-conditioned for higher-order integration -- consistent with the torchdiffeq benchmark findings.

- **Wilson-Cowan slower**: 155 ms for 501 steps due to the sigmoid nonlinearity in each RK4 sub-step (4 function evaluations per step). PyQuifer's Euler at dt=0.1 would do the same work in ~5 ms. The speed-accuracy tradeoff favors Euler for online simulation.

### 2. SDE vs Stochastic Resonance

Comparison of torchdyn-style Neural SDE (Euler-Maruyama) with PyQuifer's AdaptiveStochasticResonance.

| System | Noise Effect (std) | SNR |
|--------|:-----------------:|:---:|
| Neural SDE (Euler-Maruyama) | 0.367 | 1.197 |
| PyQuifer StochasticResonance | 0.250 | 1.000 |

**Analysis:**

- **Different noise philosophies**: Neural SDE adds noise as part of the dynamics (`dx = f dt + g dW`), creating continuous stochastic perturbations. PyQuifer's SR adds noise to find the *optimal noise level* for signal detection (hill-climbing on SNR). The SDE approach is more general; the SR approach is task-specific.

- **Both produce stochastic variation**: Neural SDE shows std=0.367 across 10 runs (diffusion-driven), while SR shows std=0.250 (noise-level adaptation varies with seed). The higher SDE variation reflects its additive noise at every timestep.

- **SNR comparison**: Neural SDE achieves SNR=1.20 (drift exceeds noise at equilibrium since dx/dt = -0.1x attractors). PyQuifer SR achieves SNR=1.00 (binary detection at threshold, so signal = 0 or full strength).

- **Complementary approaches**: For PyQuifer's cognitive architecture, SR's adaptive noise optimization is more appropriate than raw SDE noise. SR asks "what noise level maximizes signal detection?" while SDE asks "how does noise affect trajectories?"

### 3. Sensitivity/Memory Comparison

Autograd (store all states) vs adjoint-style (recompute during backward).

| Method | Forward (ms) | Backward (ms) | Memory (KB) | Grad Norm |
|--------|:----------:|:-----------:|:----------:|:---------:|
| Autograd (store all) | 1.75 | 15.31 | 12.6 | 33.493 |
| Adjoint-style (recompute) | 1.15 | 3.74 | 0.2 | 33.493 |

**Analysis:**

- **Identical gradients**: Both methods produce gradient norm = 33.493, confirming mathematical equivalence. The adjoint method computes the same gradients via a different computational strategy (recompute forward during backward pass).

- **63x memory reduction**: Adjoint stores only start and end states (0.2 KB) vs autograd storing all 100 intermediate states (12.6 KB). For long-time integration (thousands of steps), this becomes critical: O(1) vs O(T) memory scaling.

- **Backward 4x faster for adjoint**: The adjoint backward pass (3.74 ms) is faster than autograd backward (15.31 ms) because it avoids traversing a deep computation graph. However, adjoint includes a forward recompute, so total compute is similar.

- **Implication for PyQuifer**: PyQuifer's current approach stores all states during forward (autograd-style). For long simulations, adding an adjoint option via torchdiffeq/torchdyn would reduce memory. However, PyQuifer typically runs 1-10 steps per LLM token (not 100+), so the memory benefit is modest for the primary use case.

### 4. Energy / Attractor Dynamics

Comparing Hamiltonian energy preservation (torchdyn HNN concept) with PyQuifer oscillator dynamics.

| System | E_initial | E_final | Drift | Bounded |
|--------|:---------:|:-------:|:-----:|:-------:|
| Hamiltonian (HNN-style) | 0.5000 | 0.5000 | 0.000001 | YES |
| PyQuifer Stuart-Landau | 1.0000 | 0.1216 | 0.878 | YES |
| PyQuifer Kuramoto (R) | 0.0382 | 0.9477 | 23.84 | YES |

**Analysis:**

- **Hamiltonian near-perfect conservation (drift=0.000001)**: The simple harmonic oscillator H = 0.5(q^2 + p^2) integrated with RK4 preserves energy to 6 decimal places over T=10.0. This validates the HamiltonianDynamics implementation and shows the benefit of symplectic-aware integration.

- **Stuart-Landau converges to limit cycle**: Starting at |z|=1.0, the amplitude converges to |z|~0.35 (E=0.12). This is expected -- the limit cycle radius depends on lambda/coupling balance, not the initial radius. The "energy drift" is actually *convergence*, not error.

- **Kuramoto synchronizes**: R increases from 0.04 (near-random phases) to 0.95 (near-synchronized). The large "drift" (23.84) reflects the system evolving toward synchronization, which is the intended behavior, not energy violation.

- **Different conservation properties**: Hamiltonian systems conserve energy by construction (symplectic structure). Kuramoto and Stuart-Landau are *dissipative* systems that converge to attractors (synchronized state, limit cycle). PyQuifer's systems aren't designed to conserve energy -- they're designed to find synchronization and criticality.

### 5. Architecture Feature Comparison

| Feature | torchdyn | PyQuifer |
|---------|:--------:|:--------:|
| adjoint_sensitivity | YES | no |
| cnf | YES | no |
| consciousness_metrics | no | YES |
| criticality_detection | no | YES |
| event_handling | no | no |
| galerkin_layers | YES | no |
| hamiltonian_nn | YES | no |
| interpolated_adjoint | YES | no |
| lagrangian_nn | YES | no |
| multiple_shooting | YES | no |
| neural_cde | YES | no |
| neural_ode | YES | no |
| neural_sde | YES | no |
| oscillator_models | no | YES |
| precision_weighting | no | YES |
| spiking_neurons | no | YES |
| **Total** | **10/16** | **5/16** |

**Analysis:**

- **torchdyn dominates in ODE/SDE framework features (10/16)**: NeuralODE/SDE/CDE wrappers, adjoint methods, Hamiltonian/Lagrangian NNs, Galerkin layers, CNF, multiple shooting. These are general-purpose tools for continuous-depth models.

- **PyQuifer dominates in neuroscience domain features (5/16)**: Oscillator models, spiking neurons, criticality detection, precision weighting, consciousness metrics. These are specific to cognitive architecture.

- **Zero overlap**: Neither framework provides any of the other's features. This confirms they serve completely different use cases -- torchdyn is infrastructure, PyQuifer is application.

## Comparative Assessment

### Where torchdyn exceeds PyQuifer

| Dimension | torchdyn | PyQuifer |
|-----------|:--------:|:--------:|
| ODE/SDE abstraction | NeuralODE wrapper | Raw Euler loops |
| Memory-efficient backprop | Adjoint O(1) | Autograd O(T) |
| Energy preservation | HNN/LNN | No energy constraint |
| Solver variety | RK4, dopri5, adjoint | Euler only |
| Spectral methods | Galerkin layers | None |
| Density estimation | CNF | None |
| Parallel-in-time | Multiple shooting | Sequential only |

### Where PyQuifer has advantages

| Dimension | PyQuifer | torchdyn |
|-----------|:--------:|:--------:|
| Oscillator models | Kuramoto, Stuart-Landau, KuramotoDaido | Not designed for this |
| Spiking networks | LIF, AdEx, STP, STDP | Not designed for this |
| Criticality | Avalanche detection, bifurcation | Not designed for this |
| Consciousness metrics | Coherence, complexity, PCI | Not designed for this |
| Precision weighting | Hierarchical predictive coding | Not designed for this |
| Online simulation | Step-by-step, per-token | Requires time span |
| Speed | 13x faster Euler | Adaptive overhead |

### Complementarity

torchdyn and PyQuifer operate at different abstraction levels:

1. **torchdyn** (infrastructure): Provides the mathematical machinery to solve, differentiate, and optimize continuous dynamics. Any ODE/SDE system can use its tools.

2. **PyQuifer** (application): Defines specific neuroscience-inspired dynamics (oscillators, populations, spiking neurons) that happen to be ODEs. Could benefit from torchdyn's infrastructure (adjoint backprop, adaptive solvers) but doesn't need it for the primary use case (online simulation).

The wrapping test confirms that PyQuifer's dynamics are well-formed NeuralODE vector fields. A future integration path: wrap PyQuifer dynamics in torchdyn's NeuralODE for offline calibration (adjoint backprop for memory efficiency), then run with PyQuifer's Euler for online inference.

## Gaps Identified

No new correctness or functionality gaps were found. torchdyn validates PyQuifer's ODE formulations and confirms that the dynamics are well-conditioned for higher-order integration. The two potential enhancements (adjoint backprop, energy conservation) are already captured by G-16/G-17 from the torchdiffeq benchmark.

**Note**: The sensitivity comparison shows that adjoint-style backprop uses 63x less memory with identical gradients. This reinforces G-17 (adaptive step-size / torchdiffeq backend) -- adding optional torchdyn/torchdiffeq integration for offline training would improve memory efficiency for long simulations.

## Pytest Results

```
9/9 passed (9.43s)

TestNeuralODEWrapping::test_all_systems_wrappable         PASSED
TestNeuralODEWrapping::test_trajectories_finite           PASSED
TestSDEComparison::test_sde_produces_noise                PASSED
TestSDEComparison::test_sr_produces_noise                 PASSED
TestSensitivity::test_gradients_match                     PASSED
TestSensitivity::test_adjoint_less_memory                 PASSED
TestEnergyPreservation::test_hamiltonian_preserves_energy PASSED
TestEnergyPreservation::test_stuart_landau_converges      PASSED
TestEnergyPreservation::test_kuramoto_bounded             PASSED
```

Existing test suite: **479/479 passed** (unaffected).

## Verdict

**PyQuifer continuous dynamics: PASS -- all ODE systems wrap as NeuralODE, well-conditioned for higher-order integration.**

The benchmark confirms that PyQuifer's Kuramoto, Wilson-Cowan, and Stuart-Landau dynamics are well-formed ODE vector fields that integrate correctly with RK4. Hamiltonian dynamics (torchdyn's HNN concept) achieves near-perfect energy conservation, while PyQuifer's dissipative systems correctly converge to their attractors (synchronization, limit cycles). Autograd and adjoint backprop produce identical gradients with 63x memory difference. The two frameworks are fully complementary: torchdyn provides ODE/SDE infrastructure (10/16 features), PyQuifer provides neuroscience domain models (5/16 features), with zero overlap. No new gaps identified beyond the existing G-16/G-17 (higher-order integration / adaptive stepping).
