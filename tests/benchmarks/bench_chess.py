"""Chess & Strategic Reasoning Benchmark — Three-Column Evaluation.

Thin wrapper around bench_penrose_chess.py. All chess logic lives there;
this file provides the standardized harness interface for the benchmark suite.

Dual-mode:
  pytest:  python -m pytest tests/benchmarks/bench_chess.py -v --timeout=60
  CLI:     python tests/benchmarks/bench_chess.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from bench_penrose_chess import (
    DRAW, BLACK_WINS, WHITE_WINS, PENROSE_FEN, FORTRESS_FENS,
    run_material_analysis, run_minimax_baseline,
    run_pyquifer_pattern, run_engine_override, run_generalization,
    run_three_column_suite,
)


# ---------------------------------------------------------------------------
# Pytest Classes
# ---------------------------------------------------------------------------

class TestFortressDetection:
    def test_penrose_pattern_recognizes_draw(self):
        result = run_pyquifer_pattern(PENROSE_FEN)
        assert result.verdict == DRAW, f"Expected DRAW, got {result.verdict}"

    def test_minimax_fails_on_fortress(self):
        result = run_minimax_baseline(PENROSE_FEN, depth=2)
        # Minimax should NOT find DRAW — that's the whole point
        assert result.verdict != DRAW


class TestEngineOverride:
    def test_override_finds_draw(self):
        result = run_engine_override(PENROSE_FEN, engine_eval=-28.0)
        assert result.final_verdict == DRAW

    def test_override_confidence_above_threshold(self):
        result = run_engine_override(PENROSE_FEN, engine_eval=-28.0)
        assert result.override_confidence > 0.5


class TestGeneralization:
    def test_generalization_accuracy(self):
        gen = run_generalization()
        assert gen.accuracy >= 0.5, f"Expected >= 50% fortress detection, got {gen.accuracy:.0%}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_three_column_suite()
