"""Benchmark: MMMU — Massive Multi-discipline Multimodal Understanding.

SKIPPED: Requires a vision-language model (VLM).  Phi-4-mini-instruct is
text-only.  Set PYQUIFER_VLM_MODEL to enable.
"""
import pytest
pytestmark = pytest.mark.skip(reason="Requires vision-language model (VLM)")


def run_full_suite():
    print("SKIPPED: MMMU requires a vision-language model")
    print("Set PYQUIFER_VLM_MODEL to enable")
    return {"status": "skipped", "reason": "requires VLM"}


if __name__ == "__main__":
    run_full_suite()
