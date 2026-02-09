"""
Benchmark conftest.py — Prevent pytest from crawling vendored third-party repos
and provide shared fixtures for data paths.

All directories listed in collect_ignore_glob are third-party repos cloned
for comparison benchmarking. They are NOT part of PyQuifer's test suite.
"""

import pytest
from pathlib import Path

# collect_ignore prevents pytest from importing conftest.py files in these
# directories (collect_ignore_glob does NOT prevent conftest.py discovery).
_BENCH_DIR = Path(__file__).parent
_VENDORED_DIRS = [
    "alphagenome",
    "bbeh",
    "beir",
    "bifurcation",
    "BIG-bench",
    "bigcode-evaluation-harness",
    "chessqa-benchmark",
    "code_contests",
    "dm_control",
    "EGG",
    "emergent_communication_at_scale",
    "FactBench",
    "FactReasoner",
    "FActScore",
    "formal-conjectures",
    "gpqa",
    "grade-school-math",
    "Gymnasium",
    "hanabi-learning-environment",
    "HarmBench",
    "helm",
    "human-eval",
    "jailbreakbench",
    "KuraNet",
    "lab",
    "lava",
    "lm-evaluation-harness",
    "lmms-eval",
    "MD-Bench",
    "meltingpot",
    "mlperf-inference",
    "mlperf-training",
    "MMMU",
    "mteb",
    "neurobench",
    "neurodiscoverybench",
    "nlb_tools",
    "open_spiel",
    "openai-evals",
    "opencompass",
    "osbenchmarks",
    "perception_test",
    "physics-IQ-benchmark",
    "Plaskett_puzzle",
    "PredBench",
    "SciMLBenchmarks.jl",
    "searchless_chess",
    "SWE-bench",
    "Torch2PC",
    "torchdiffeq",
    "torchdyn",
    "TOXIGEN",
    "TruthfulQA",
    "Video-MME",
    "VLMEvalKit",
    "data",
    "results",
]

# This list stops pytest from entering directories AND importing their conftest
collect_ignore = [
    str(_BENCH_DIR / d)
    for d in _VENDORED_DIRS
    if (_BENCH_DIR / d).exists()
]

# Also add glob patterns for arXiv- prefixed dirs
collect_ignore_glob = [
    "arXiv-*/**",
]


# ── Data path fixtures ──

BENCH_DIR = Path(__file__).parent


@pytest.fixture
def chessqa_data_dir():
    """Path to chessqa-benchmark/benchmark/ JSONL files."""
    return BENCH_DIR / "chessqa-benchmark" / "benchmark"


@pytest.fixture
def searchless_chess_dir():
    """Path to searchless_chess/ repo root."""
    return BENCH_DIR / "searchless_chess"


# ── Skip markers ──

def _has_chessqa():
    d = BENCH_DIR / "chessqa-benchmark" / "benchmark"
    return d.exists() and any(d.glob("*.jsonl"))


def _has_searchless():
    d = BENCH_DIR / "searchless_chess" / "src"
    return d.exists()


skip_if_no_chessqa = pytest.mark.skipif(
    not _has_chessqa(),
    reason="ChessQA benchmark data not available",
)

skip_if_no_searchless = pytest.mark.skipif(
    not _has_searchless(),
    reason="Searchless chess repo not available",
)
