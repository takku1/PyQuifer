"""Benchmark: VLMEvalKit — Vision-Language Model Evaluation.

SKIPPED: Requires a vision-language model (VLM).  Set PYQUIFER_VLM_MODEL to enable.
"""
import pytest
pytestmark = pytest.mark.skip(reason="Requires vision-language model (VLM)")


def run_full_suite():
    print("SKIPPED: VLMEvalKit requires a vision-language model")
    print("Set PYQUIFER_VLM_MODEL to enable")
    return {"status": "skipped", "reason": "requires VLM"}


if __name__ == "__main__":
    run_full_suite()
