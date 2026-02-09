"""Benchmark: BEIR — Information Retrieval Benchmark.

SKIPPED: Requires embedding/retrieval model, not generative LLM.
BEIR evaluates encode() → retrieval, not text generation.
"""
import pytest
pytestmark = pytest.mark.skip(reason="Requires embedding/retrieval model, not generative LLM")


def run_full_suite():
    print("SKIPPED: BEIR requires an embedding/retrieval model")
    return {"status": "skipped", "reason": "requires retrieval model"}


if __name__ == "__main__":
    run_full_suite()
