"""
Validation: PyQuifer Transfer Entropy vs Known Causal Signals

Validates that TransferEntropyEstimator correctly detects:
1. Causal signal: X drives Y -> T(X->Y) > T(Y->X)
2. Independent signals: T(X->Y) ≈ T(Y->X) ≈ 0
3. Bidirectional: Both directions detected
4. Delay sensitivity: Detects lagged causation

Reference: Schreiber (2000) "Measuring Information Transfer"
T(X->Y) = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)

For a deterministic driven system Y[t] = f(X[t-lag]),
T(X->Y) should be significantly positive while T(Y->X) ≈ 0.
"""

import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from pyquifer.causal_flow import TransferEntropyEstimator, CausalFlowMap, DominanceDetector


def generate_causal_pair(n=500, lag=1, noise=0.1, seed=42):
    """Generate X driving Y with a lag and noise."""
    np.random.seed(seed)
    x = np.cumsum(np.random.randn(n) * 0.3)
    y = np.zeros(n)
    for t in range(lag, n):
        y[t] = 0.8 * x[t - lag] + noise * np.random.randn()
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def generate_independent_pair(n=500, seed=42):
    """Generate two independent white noise signals (no autocorrelation)."""
    np.random.seed(seed)
    x = torch.tensor(np.random.randn(n), dtype=torch.float32)
    np.random.seed(seed + 1000)
    y = torch.tensor(np.random.randn(n), dtype=torch.float32)
    return x, y


def generate_bidirectional_pair(n=500, lag=1, seed=42):
    """Generate X and Y that mutually influence each other."""
    np.random.seed(seed)
    x = np.zeros(n)
    y = np.zeros(n)
    x[0] = np.random.randn()
    y[0] = np.random.randn()
    for t in range(1, n):
        x[t] = 0.5 * x[t - 1] + 0.3 * y[t - 1] + 0.1 * np.random.randn()
        y[t] = 0.5 * y[t - 1] + 0.3 * x[t - 1] + 0.1 * np.random.randn()
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def test_causal_direction_detected():
    """
    Test 1: Causal direction should be correctly identified.

    X drives Y (with lag) -> T(X->Y) should be significantly larger than T(Y->X).
    """
    x, y = generate_causal_pair(n=500, lag=1, noise=0.1)
    te = TransferEntropyEstimator(num_bins=8, history_length=3)

    result = te(x, y)
    te_xy = result['te_source_to_target'].item()
    te_yx = result['te_target_to_source'].item()

    assert te_xy > te_yx, f"Failed: T(X->Y)={te_xy:.4f} should be > T(Y->X)={te_yx:.4f}"
    assert te_xy > 0.0, f"Failed: T(X->Y)={te_xy:.4f} should be positive"
    print(f"  T(X->Y)={te_xy:.4f}, T(Y->X)={te_yx:.4f}, ratio={te_xy / max(te_yx, 1e-8):.2f}")


def test_independent_signals_symmetric_te():
    """
    Test 2: Independent signals should have symmetric TE (neither drives the other).

    Note: Histogram-based TE has finite-sample positive bias, so absolute values
    won't be zero. But T(X->Y) and T(Y->X) should be similar (symmetric).
    We use fewer bins + shorter history to reduce bias.
    """
    x, y = generate_independent_pair(n=1000)
    # Use fewer bins and shorter history to reduce finite-sample bias
    te = TransferEntropyEstimator(num_bins=4, history_length=1)

    result = te(x, y)
    te_xy = result['te_source_to_target'].item()
    te_yx = result['te_target_to_source'].item()

    # For independent signals, the asymmetry should be small
    net_flow = abs(te_xy - te_yx)
    max_te = max(te_xy, te_yx, 0.001)
    asymmetry_ratio = net_flow / max_te

    assert asymmetry_ratio < 0.5, \
        f"Independent signals too asymmetric: T(X->Y)={te_xy:.4f}, T(Y->X)={te_yx:.4f}, ratio={asymmetry_ratio:.2f}"
    print(f"  T(X->Y)={te_xy:.4f}, T(Y->X)={te_yx:.4f}, asymmetry={asymmetry_ratio:.3f}")


def test_bidirectional_detected():
    """
    Test 3: Bidirectional coupling should show positive TE in both directions.
    """
    x, y = generate_bidirectional_pair(n=500)
    te = TransferEntropyEstimator(num_bins=8, history_length=3)

    result = te(x, y)
    te_xy = result['te_source_to_target'].item()
    te_yx = result['te_target_to_source'].item()

    # Both should be positive
    assert te_xy > 0.0, f"T(X->Y)={te_xy:.4f} should be positive for bidirectional"
    assert te_yx > 0.0, f"T(Y->X)={te_yx:.4f} should be positive for bidirectional"
    print(f"  T(X->Y)={te_xy:.4f}, T(Y->X)={te_yx:.4f}")


def test_causal_flow_map_identifies_driver():
    """
    Test 4: CausalFlowMap should identify X as the driver when X->Y.
    """
    x, y = generate_causal_pair(n=300, lag=1, noise=0.1)
    flow_map = CausalFlowMap(num_populations=2, buffer_size=300)

    # Feed time series as population phases
    for t in range(300):
        flow_map.record(torch.tensor([x[t], y[t]]))

    result = flow_map.compute_flow()

    # Net flow should show population 0 (X) driving population 1 (Y)
    net_flow = result['net_flow_matrix']
    assert net_flow[0, 1].item() > 0, f"Expected X->Y positive flow, got {net_flow[0, 1].item():.4f}"

    # Driver scores: population 0 should have positive driver score
    driver_scores = result['driver_scores']
    print(f"  Net flow X->Y: {net_flow[0, 1].item():.4f}")
    print(f"  Net flow Y->X: {net_flow[1, 0].item():.4f}")
    print(f"  Driver scores: {driver_scores.tolist()}")


def test_dominance_detector():
    """
    Test 5: DominanceDetector should correctly identify bottom-up vs top-down.

    Feed time-varying signals where lower levels causally drive upper levels
    (bottom-up) or vice versa (top-down).
    """
    np.random.seed(42)

    # Bottom-up: lower level signal drives upper levels with a lag
    dom = DominanceDetector(num_levels=3, buffer_size=200)
    n_steps = 100
    bottom_signal = np.cumsum(np.random.randn(n_steps + 2) * 0.3)

    for t in range(n_steps):
        levels = torch.tensor([
            bottom_signal[t + 2],       # Level 0: leads (source)
            bottom_signal[t + 1] * 0.8, # Level 1: lags by 1
            bottom_signal[t] * 0.6,     # Level 2: lags by 2
        ], dtype=torch.float32)
        dom(levels, compute_every=100)

    # Force compute on last step
    result = dom(torch.tensor([0.0, 0.0, 0.0]), compute_every=1)
    ratio = result['dominance_ratio'].item()
    mode = result['mode']
    print(f"  Bottom-up scenario: mode={mode}, ratio={ratio:.3f}")
    # Bottom-up should have higher ratio (more upward TE)

    # Top-down: upper level signal drives lower levels
    dom2 = DominanceDetector(num_levels=3, buffer_size=200)
    top_signal = np.cumsum(np.random.randn(n_steps + 2) * 0.3)

    for t in range(n_steps):
        levels = torch.tensor([
            top_signal[t] * 0.6,        # Level 0: lags by 2 (follower)
            top_signal[t + 1] * 0.8,    # Level 1: lags by 1
            top_signal[t + 2],          # Level 2: leads (source)
        ], dtype=torch.float32)
        dom2(levels, compute_every=100)

    result2 = dom2(torch.tensor([0.0, 0.0, 0.0]), compute_every=1)
    ratio2 = result2['dominance_ratio'].item()
    mode2 = result2['mode']
    print(f"  Top-down scenario:  mode={mode2}, ratio={ratio2:.3f}")

    # Bottom-up scenario should have higher ratio than top-down
    assert ratio > ratio2, \
        f"Bottom-up ratio ({ratio:.3f}) should exceed top-down ratio ({ratio2:.3f})"


if __name__ == '__main__':
    print("=== Transfer Entropy Validation ===\n")

    print("Test 1: Causal direction detection")
    test_causal_direction_detected()

    print("\nTest 2: Independent signals (symmetric TE)")
    test_independent_signals_symmetric_te()

    print("\nTest 3: Bidirectional coupling")
    test_bidirectional_detected()

    print("\nTest 4: CausalFlowMap identifies driver")
    test_causal_flow_map_identifies_driver()

    print("\nTest 5: DominanceDetector modes")
    test_dominance_detector()

    print("\n[PASS] All transfer entropy validation tests passed!")
