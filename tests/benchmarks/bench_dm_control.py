"""Benchmark: dm_control — DeepMind Control Suite.

SKIPPED: Requires a reinforcement learning agent, not an LLM.
The RL evaluation paradigm is fundamentally different from text generation.
"""
import pytest
pytestmark = pytest.mark.skip(reason="Requires RL agent, not LLM")


def run_full_suite():
    print("SKIPPED: dm_control requires a reinforcement learning agent")
    return {"status": "skipped", "reason": "requires RL agent"}


if __name__ == "__main__":
    run_full_suite()
