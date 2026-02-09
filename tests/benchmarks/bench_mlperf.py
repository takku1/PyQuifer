"""Benchmark: MLPerf — ML Performance Benchmarks (Training + Inference).

SKIPPED: MLPerf measures hardware/system-level throughput, not model quality.
Requires specialized hardware configuration and submission infrastructure.
"""
import pytest
pytestmark = pytest.mark.skip(reason="MLPerf measures system throughput, not model quality")


def run_full_suite():
    print("SKIPPED: MLPerf requires specialized hardware benchmarking setup")
    return {"status": "skipped", "reason": "system-level benchmark"}


if __name__ == "__main__":
    run_full_suite()
