"""Benchmark: Perception Test — Video Understanding.

SKIPPED: Requires a video-language model.  Set PYQUIFER_VLM_MODEL to enable.
"""
import pytest
pytestmark = pytest.mark.skip(reason="Requires video-language model")


def run_full_suite():
    print("SKIPPED: Perception Test requires a video-language model")
    print("Set PYQUIFER_VLM_MODEL to enable")
    return {"status": "skipped", "reason": "requires video-language model"}


if __name__ == "__main__":
    run_full_suite()
