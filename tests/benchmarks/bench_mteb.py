"""Benchmark: MTEB — Massive Text Embedding Benchmark.

SKIPPED: Requires embedding model with encode() interface, not generative LLM.
"""
import pytest
pytestmark = pytest.mark.skip(reason="Requires embedding model, not generative LLM")


def run_full_suite():
    print("SKIPPED: MTEB requires an embedding model")
    return {"status": "skipped", "reason": "requires embedding model"}


if __name__ == "__main__":
    run_full_suite()
