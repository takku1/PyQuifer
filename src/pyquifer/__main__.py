"""
Runtime smoke test for PyQuifer.

Usage:
    python -m pyquifer          # Quick sanity check
    python -m pyquifer --tick 5 # Run N ticks

Verifies the library works when installed, without needing pytest.
"""
import argparse
import sys
import time


def smoke_test(num_ticks: int = 3) -> bool:
    """Run a minimal PyQuifer smoke test."""
    import torch

    print(f"PyQuifer smoke test (torch {torch.__version__}, "
          f"CUDA {'available' if torch.cuda.is_available() else 'unavailable'})")
    print("-" * 50)

    # 1. Import bridge
    t0 = time.perf_counter()
    from pyquifer.bridge import PyQuiferBridge, ModulationState
    print(f"[OK] Bridge import: {(time.perf_counter() - t0)*1000:.0f}ms")

    # 2. Create bridge
    t0 = time.perf_counter()
    bridge = PyQuiferBridge.small()
    print(f"[OK] Bridge.small() created: {(time.perf_counter() - t0)*1000:.0f}ms")

    # 3. Run ticks
    for i in range(num_ticks):
        sensory = torch.randn(bridge.config.state_dim)
        state = bridge.step(sensory)
        assert isinstance(state, ModulationState)
        print(f"[OK] tick {i+1}: temp={state.temperature:.3f}, "
              f"coherence={state.coherence:.3f}, "
              f"latency={state.step_latency_ms:.1f}ms")

    # 4. Verify key exports
    from pyquifer import (
        CognitiveCycle, CycleConfig, LearnableKuramotoBank,
        HierarchicalPredictiveCoding, CriticalityController,
    )
    print(f"[OK] Key exports: CognitiveCycle, CycleConfig, "
          f"LearnableKuramotoBank, HPC, CriticalityController")

    print("-" * 50)
    print("All checks passed.")
    return True


def main():
    parser = argparse.ArgumentParser(description="PyQuifer runtime smoke test")
    parser.add_argument("--tick", type=int, default=3,
                        help="Number of cognitive ticks to run (default: 3)")
    args = parser.parse_args()

    try:
        ok = smoke_test(args.tick)
        sys.exit(0 if ok else 1)
    except Exception as e:
        print(f"[FAIL] {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
