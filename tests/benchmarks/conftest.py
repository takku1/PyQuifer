"""
Benchmark conftest.py — Prevent pytest from crawling vendored third-party repos
and provide shared fixtures for data paths.

All directories listed in collect_ignore_glob are third-party repos cloned
for comparison benchmarking. They are NOT part of PyQuifer's test suite.
"""

import pytest
from pathlib import Path

collect_ignore_glob = [
    # Third-party vendored repos (alphabetical)
    "arXiv-*/**",
    "bifurcation/**",
    "chessqa-benchmark/**",
    "EGG/**",
    "emergent_communication_at_scale/**",
    "Gymnasium/**",
    "hanabi-learning-environment/**",
    "KuraNet/**",
    "lava/**",
    "MD-Bench/**",
    "meltingpot/**",
    "neurobench/**",
    "neurodiscoverybench/**",
    "nlb_tools/**",
    "open_spiel/**",
    "osbenchmarks/**",
    "Plaskett_puzzle/**",
    "PredBench/**",
    "SciMLBenchmarks.jl/**",
    "searchless_chess/**",
    "Torch2PC/**",
    "torchdiffeq/**",
    "torchdyn/**",
    # Runtime artifacts
    "data/**",
    "results/**",
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
