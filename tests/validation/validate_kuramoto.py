"""
Validation: PyQuifer Kuramoto vs Neurolib Reference Implementation

Compares:
1. Phase evolution equation (same Kuramoto dynamics)
2. Order parameter (synchronization index) trajectory
3. Coupling-dependence of synchronization (critical coupling)

Reference: neurolib (Cakan et al. 2021) — Numba-compiled Kuramoto model.
neurolib equation:
  theta[i,t] = theta[i,t-1] + dt * (omega[i] + (k/N)*sum(C[j]*sin(theta[j]-theta[i])) + noise + ext)

PyQuifer equation (oscillators.py:238):
  d_theta_dt = omega + K * (1/N_conn) * sum(adj * sin(theta_j - theta_i))
  phases = (phases + d_theta_dt * dt) % (2*pi)

They are the same equation (Euler integration of Kuramoto).
"""

import numpy as np
import torch
import math
import sys
import os

# Add PyQuifer to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


def reference_kuramoto_numpy(N, omega, K, dt, T, theta_init, seed=42):
    """
    Reference Kuramoto implementation matching neurolib's equation exactly.
    Pure NumPy, no dependencies.
    """
    np.random.seed(seed)
    steps = int(T / dt)
    theta = np.zeros((N, steps + 1))
    theta[:, 0] = theta_init

    order_params = np.zeros(steps + 1)
    z = np.mean(np.exp(1j * theta[:, 0]))
    order_params[0] = np.abs(z)

    for t in range(steps):
        for i in range(N):
            coupling = 0.0
            for j in range(N):
                if i != j:
                    coupling += np.sin(theta[j, t] - theta[i, t])
            coupling *= K / (N - 1)  # normalize by connections (not N)
            theta[i, t + 1] = theta[i, t] + dt * (omega[i] + coupling)
            theta[i, t + 1] = theta[i, t + 1] % (2 * np.pi)

        z = np.mean(np.exp(1j * theta[:, t + 1]))
        order_params[t + 1] = np.abs(z)

    return theta, order_params


def pyquifer_kuramoto(N, omega, K, dt, T, theta_init):
    """Run PyQuifer's LearnableKuramotoBank with matching parameters."""
    from pyquifer.oscillators import LearnableKuramotoBank

    steps = int(T / dt)

    bank = LearnableKuramotoBank(
        num_oscillators=N,
        dt=dt,
        topology='global',
    )

    # Override parameters to match reference
    with torch.no_grad():
        bank.natural_frequencies.copy_(torch.tensor(omega, dtype=torch.float32))
        bank.coupling_strength.fill_(K)
        bank.phases.copy_(torch.tensor(theta_init, dtype=torch.float32))

    order_params = [bank.get_order_parameter().item()]

    for _ in range(steps):
        bank(steps=1)
        order_params.append(bank.get_order_parameter().item())

    return bank.phases.numpy(), np.array(order_params)


def test_order_parameter_agreement():
    """
    Test 1: Order parameter trajectory should match between implementations.

    With identical initial conditions and parameters, the order parameter
    should follow the same trajectory (within numerical precision of
    float32 vs float64).
    """
    N = 8
    dt = 0.01
    T = 5.0  # 500 steps
    K = 2.0
    np.random.seed(42)
    omega = np.random.uniform(0.5, 1.5, N)
    theta_init = np.random.uniform(0, 2 * np.pi, N)

    _, ref_order = reference_kuramoto_numpy(N, omega, K, dt, T, theta_init, seed=42)
    _, pyq_order = pyquifer_kuramoto(N, omega, K, dt, T, theta_init)

    # Compare order parameter trajectories
    max_diff = np.max(np.abs(ref_order - pyq_order))
    mean_diff = np.mean(np.abs(ref_order - pyq_order))

    # Allow float32 vs float64 numerical drift
    # Over 500 steps, accumulated error should be small
    assert max_diff < 0.05, f"Max order parameter difference {max_diff:.6f} exceeds tolerance"
    assert mean_diff < 0.01, f"Mean order parameter difference {mean_diff:.6f} exceeds tolerance"
    print(f"  Order parameter max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")


def test_synchronization_with_strong_coupling():
    """
    Test 2: Strong coupling should synchronize oscillators.

    Both implementations should show order parameter → 1.0 with strong coupling.
    This is the fundamental Kuramoto result.
    """
    N = 16
    dt = 0.01
    T = 20.0
    K = 5.0  # Strong coupling
    np.random.seed(123)
    omega = np.ones(N) * 1.0  # Identical frequencies (easy sync)
    theta_init = np.random.uniform(0, 2 * np.pi, N)

    _, ref_order = reference_kuramoto_numpy(N, omega, K, dt, T, theta_init, seed=123)
    _, pyq_order = pyquifer_kuramoto(N, omega, K, dt, T, theta_init)

    # Both should synchronize (order param near 1)
    assert ref_order[-1] > 0.9, f"Reference failed to sync: R={ref_order[-1]:.3f}"
    assert pyq_order[-1] > 0.9, f"PyQuifer failed to sync: R={pyq_order[-1]:.3f}"
    print(f"  Ref final R={ref_order[-1]:.4f}, PyQ final R={pyq_order[-1]:.4f}")


def test_desynchronization_with_weak_coupling():
    """
    Test 3: Weak coupling + spread frequencies should NOT synchronize.
    """
    N = 16
    dt = 0.01
    T = 20.0
    K = 0.01  # Very weak coupling
    np.random.seed(456)
    omega = np.random.uniform(0.1, 5.0, N)  # Wide frequency spread
    theta_init = np.random.uniform(0, 2 * np.pi, N)

    _, ref_order = reference_kuramoto_numpy(N, omega, K, dt, T, theta_init, seed=456)
    _, pyq_order = pyquifer_kuramoto(N, omega, K, dt, T, theta_init)

    # Neither should synchronize
    assert ref_order[-1] < 0.6, f"Reference synced unexpectedly: R={ref_order[-1]:.3f}"
    assert pyq_order[-1] < 0.6, f"PyQuifer synced unexpectedly: R={pyq_order[-1]:.3f}"
    print(f"  Ref final R={ref_order[-1]:.4f}, PyQ final R={pyq_order[-1]:.4f}")


def test_critical_coupling_transition():
    """
    Test 4: Kuramoto transition — order parameter should increase with coupling.

    The Kuramoto model has a phase transition at critical coupling K_c.
    Both implementations should show the same qualitative behavior:
    R increases monotonically with K (for fixed frequency distribution).
    """
    N = 32
    dt = 0.01
    T = 30.0
    np.random.seed(789)
    omega = np.random.uniform(0.5, 1.5, N)
    theta_init = np.random.uniform(0, 2 * np.pi, N)

    coupling_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    ref_finals = []
    pyq_finals = []

    for K in coupling_values:
        _, ref_order = reference_kuramoto_numpy(N, omega, K, dt, T, theta_init.copy(), seed=789)
        _, pyq_order = pyquifer_kuramoto(N, omega, K, dt, T, theta_init.copy())
        ref_finals.append(ref_order[-1])
        pyq_finals.append(pyq_order[-1])

    # Both should show monotonically increasing R with K
    for i in range(len(coupling_values) - 1):
        assert ref_finals[i] <= ref_finals[i + 1] + 0.1, \
            f"Reference non-monotonic at K={coupling_values[i]}"
        assert pyq_finals[i] <= pyq_finals[i + 1] + 0.1, \
            f"PyQuifer non-monotonic at K={coupling_values[i]}"

    # Final R values should be correlated
    correlation = np.corrcoef(ref_finals, pyq_finals)[0, 1]
    assert correlation > 0.9, f"Coupling sweep correlation {correlation:.3f} too low"
    print(f"  Coupling sweep correlation: {correlation:.4f}")
    for K, r, p in zip(coupling_values, ref_finals, pyq_finals):
        print(f"    K={K:.1f}: ref R={r:.3f}, pyq R={p:.3f}")


if __name__ == '__main__':
    print("=== Kuramoto Validation: PyQuifer vs Reference ===\n")

    print("Test 1: Order parameter trajectory agreement")
    test_order_parameter_agreement()

    print("\nTest 2: Synchronization with strong coupling")
    test_synchronization_with_strong_coupling()

    print("\nTest 3: Desynchronization with weak coupling")
    test_desynchronization_with_weak_coupling()

    print("\nTest 4: Critical coupling transition")
    test_critical_coupling_transition()

    print("\n[PASS] All Kuramoto validation tests passed!")
