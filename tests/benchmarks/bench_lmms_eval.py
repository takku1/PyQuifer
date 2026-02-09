"""Benchmark: lmms-eval — Large Multimodal Model Evaluation.

SKIPPED: Requires a multimodal model.  Set PYQUIFER_VLM_MODEL to enable.
"""
import pytest
pytestmark = pytest.mark.skip(reason="Requires multimodal model")


def run_full_suite():
    print("SKIPPED: lmms-eval requires a multimodal model")
    print("Set PYQUIFER_VLM_MODEL to enable")
    return {"status": "skipped", "reason": "requires multimodal model"}


if __name__ == "__main__":
    run_full_suite()
