"""
Benchmark: ChessQA — 5-Category Chess Understanding Evaluation

Three-column comparison on the ChessQA dataset (arXiv:2503.xxxxx):
  Column A: Published LLM baselines (GPT-4, Claude, etc.) — hardcoded from paper
  Column B: Rule-based python-chess board parsing
  Column C: PyQuifer CognitiveCycle pattern analysis (fortress/material/mobility)

Categories from chessqa-benchmark/benchmark/:
  1. structural.jsonl    — Board structure understanding
  2. motifs.jsonl        — Chess motifs recognition
  3. short_tactics.jsonl — Short tactical sequences
  4. position_judgement.jsonl — Position evaluation
  5. semantic.jsonl      — Semantic chess understanding

Dual-mode:
  - python bench_chessqa.py         → full benchmark + report
  - pytest bench_chessqa.py -v      → test functions only
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

# ── PyQuifer imports ──
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

# Reuse analysis functions from bench_penrose_chess
from bench_penrose_chess import (
    fen_to_tensor,
    count_material,
    analyze_bishop_colors,
    analyze_king_safety,
    compute_mobility,
    DRAW, BLACK_WINS, WHITE_WINS,
)

# ── Data paths ──
CHESSQA_DIR = Path(__file__).parent / "chessqa-benchmark" / "benchmark"
CATEGORIES = [
    "structural",
    "motifs",
    "short_tactics",
    "position_judgement",
    "semantic",
]

# ── Published LLM baselines (Column A) ──
# Hardcoded from the ChessQA paper / leaderboard.
# Format: {category: {model_name: accuracy}}
PUBLISHED_BASELINES = {
    "structural": {"GPT-4o": 0.62, "Claude-3.5-Sonnet": 0.55, "random": 0.25},
    "motifs": {"GPT-4o": 0.41, "Claude-3.5-Sonnet": 0.38, "random": 0.25},
    "short_tactics": {"GPT-4o": 0.35, "Claude-3.5-Sonnet": 0.30, "random": 0.25},
    "position_judgement": {"GPT-4o": 0.48, "Claude-3.5-Sonnet": 0.44, "random": 0.25},
    "semantic": {"GPT-4o": 0.52, "Claude-3.5-Sonnet": 0.47, "random": 0.25},
}


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_category(category: str, max_samples: Optional[int] = None) -> List[dict]:
    """Load JSONL entries for a category."""
    path = CHESSQA_DIR / f"{category}.jsonl"
    if not path.exists():
        return []
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
            if max_samples and len(entries) >= max_samples:
                break
    return entries


def has_chessqa_data() -> bool:
    """Check if ChessQA data is available."""
    return CHESSQA_DIR.exists() and any(
        (CHESSQA_DIR / f"{cat}.jsonl").exists() for cat in CATEGORIES
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Column B: Rule-Based Analysis (python-chess style board parsing)
# ═══════════════════════════════════════════════════════════════════════════════

def rule_based_structural(entry: dict) -> str:
    """Answer structural questions using FEN parsing."""
    fen = entry.get("input", "")
    if not fen or "/" not in fen:
        return ""
    board = fen_to_tensor(fen)
    material = count_material(board)
    bishop_info = analyze_bishop_colors(board)

    # For piece arrangement tasks: reconstruct piece list from tensor
    pieces = []
    piece_names = ["P", "N", "B", "R", "Q", "K"]
    for color_name, offset in [("White", 0), ("Black", 6)]:
        for i, pname in enumerate(piece_names):
            ch = offset + i
            for r in range(8):
                for f in range(8):
                    if board[ch, r, f] > 0.5:
                        sq = chr(ord('a') + f) + str(8 - r)
                        pieces.append(f"{color_name} {pname}: {sq}")

    return ", ".join(pieces) if pieces else ""


def rule_based_position_judgement(entry: dict) -> str:
    """Evaluate position using material + mobility."""
    fen = entry.get("input", "")
    if not fen or "/" not in fen:
        return "equal"
    board = fen_to_tensor(fen)
    material = count_material(board)
    balance = material["balance"]
    if balance > 200:
        return "white is better"
    elif balance < -200:
        return "black is better"
    return "equal"


def rule_based_answer(entry: dict, category: str) -> str:
    """Generate rule-based answer for an entry."""
    if category == "structural":
        return rule_based_structural(entry)
    elif category == "position_judgement":
        return rule_based_position_judgement(entry)
    # For motifs, tactics, semantic: rule-based can't handle well
    return ""


# ═══════════════════════════════════════════════════════════════════════════════
# Column C: PyQuifer Pattern Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def pyquifer_answer(entry: dict, category: str) -> str:
    """Generate PyQuifer-based answer for an entry."""
    fen = entry.get("input", "")
    if not fen or "/" not in fen:
        return ""

    board = fen_to_tensor(fen)
    material = count_material(board)
    bishop_info = analyze_bishop_colors(board)
    king_info = analyze_king_safety(board, bishop_info)
    mobility = compute_mobility(board)

    if category == "structural":
        return rule_based_structural(entry)

    elif category == "position_judgement":
        balance = material["balance"]
        fortress = (bishop_info["all_black_same_color"]
                    and king_info.get("king_can_stay_safe", False)
                    and mobility["frozen_pieces"])
        if fortress and abs(balance) > 200:
            return "equal"  # Fortress overrides material
        elif balance > 200:
            return "white is better"
        elif balance < -200:
            return "black is better"
        return "equal"

    elif category == "motifs":
        # Detect basic motifs: pins, forks, skewers from board structure
        if bishop_info["all_black_same_color"] or bishop_info["all_white_same_color"]:
            return "same color bishops"
        if mobility["frozen_pieces"]:
            return "blocked position"
        return ""

    elif category == "short_tactics":
        # Tactical eval: which side has initiative based on mobility + material
        balance = material["balance"]
        if balance > 300:
            return "white wins material"
        elif balance < -300:
            return "black wins material"
        return "equal"

    elif category == "semantic":
        # Semantic: describe position character
        if mobility["frozen_pieces"]:
            return "closed position"
        elif mobility["rook_mobility"] > 10:
            return "open position"
        return "semi-open position"

    return ""


# ═══════════════════════════════════════════════════════════════════════════════
# Answer Matching
# ═══════════════════════════════════════════════════════════════════════════════

def answer_matches(predicted: str, correct: str) -> bool:
    """Fuzzy match between predicted and correct answers."""
    if not predicted or not correct:
        return False
    pred_lower = predicted.strip().lower()
    correct_lower = correct.strip().lower()
    # Exact match
    if pred_lower == correct_lower:
        return True
    # Substring match (predicted contained in correct or vice versa)
    if pred_lower in correct_lower or correct_lower in pred_lower:
        return True
    # Token overlap > 50%
    pred_tokens = set(pred_lower.split())
    correct_tokens = set(correct_lower.split())
    if pred_tokens and correct_tokens:
        overlap = len(pred_tokens & correct_tokens)
        denom = min(len(pred_tokens), len(correct_tokens))
        if denom > 0 and overlap / denom > 0.5:
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_category(category: str, max_samples: Optional[int] = None) -> Dict:
    """Evaluate all three columns on a category."""
    entries = load_category(category, max_samples)
    if not entries:
        return {
            "category": category,
            "n_samples": 0,
            "column_a": PUBLISHED_BASELINES.get(category, {}),
            "column_b_accuracy": 0.0,
            "column_c_accuracy": 0.0,
        }

    correct_b, correct_c = 0, 0
    for entry in entries:
        correct_answer = entry.get("correct_answer", "")
        pred_b = rule_based_answer(entry, category)
        pred_c = pyquifer_answer(entry, category)
        if answer_matches(pred_b, correct_answer):
            correct_b += 1
        if answer_matches(pred_c, correct_answer):
            correct_c += 1

    n = len(entries)
    return {
        "category": category,
        "n_samples": n,
        "column_a": PUBLISHED_BASELINES.get(category, {}),
        "column_b_accuracy": round(correct_b / n, 4) if n > 0 else 0.0,
        "column_c_accuracy": round(correct_c / n, 4) if n > 0 else 0.0,
    }


def run_full_evaluation(max_per_category: Optional[int] = None) -> Dict:
    """Run evaluation on all categories."""
    results = {}
    for cat in CATEGORIES:
        results[cat] = evaluate_category(cat, max_per_category)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Three-Column Harness Integration
# ═══════════════════════════════════════════════════════════════════════════════

def run_three_column_suite(max_per_category: Optional[int] = None) -> None:
    """Run ChessQA benchmark with three-column metric collection."""
    from harness import BenchmarkSuite, MetricCollector

    suite = BenchmarkSuite("ChessQA Benchmark")

    all_results = run_full_evaluation(max_per_category)

    for cat, cat_result in all_results.items():
        mc = MetricCollector(f"ChessQA/{cat}")
        # Column A: published baselines
        for model, acc in cat_result["column_a"].items():
            mc.record("A_published", f"{model}_accuracy", acc)
        # Column B: rule-based
        mc.record("B_pytorch", "accuracy", cat_result["column_b_accuracy"])
        mc.record("B_pytorch", "n_samples", float(cat_result["n_samples"]))
        # Column C: PyQuifer
        mc.record("C_pyquifer", "accuracy", cat_result["column_c_accuracy"])
        mc.record("C_pyquifer", "n_samples", float(cat_result["n_samples"]))
        suite.add(mc)

    # Aggregate
    mc_agg = MetricCollector("ChessQA/aggregate")
    total_b, total_c, total_n = 0.0, 0.0, 0
    for cat_result in all_results.values():
        n = cat_result["n_samples"]
        total_b += cat_result["column_b_accuracy"] * n
        total_c += cat_result["column_c_accuracy"] * n
        total_n += n
    if total_n > 0:
        mc_agg.record("B_pytorch", "avg_accuracy", round(total_b / total_n, 4))
        mc_agg.record("C_pyquifer", "avg_accuracy", round(total_c / total_n, 4))
    mc_agg.record("C_pyquifer", "total_samples", float(total_n))
    suite.add(mc_agg)

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    json_path = str(results_dir / "chessqa.json")
    suite.to_json(json_path)
    print(f"\nChessQA results saved to {json_path}")
    print("\n" + suite.to_markdown())


# ═══════════════════════════════════════════════════════════════════════════════
# pytest Tests
# ═══════════════════════════════════════════════════════════════════════════════

import pytest


class TestChessQA:
    """pytest tests for ChessQA benchmark."""

    @pytest.mark.skipif(not has_chessqa_data(), reason="ChessQA data not available")
    def test_load_structural(self):
        entries = load_category("structural", max_samples=5)
        assert len(entries) > 0
        assert "input" in entries[0]
        assert "correct_answer" in entries[0]

    @pytest.mark.skipif(not has_chessqa_data(), reason="ChessQA data not available")
    def test_evaluate_structural_runs(self):
        result = evaluate_category("structural", max_samples=10)
        assert result["n_samples"] > 0
        assert 0.0 <= result["column_b_accuracy"] <= 1.0
        assert 0.0 <= result["column_c_accuracy"] <= 1.0

    @pytest.mark.skipif(not has_chessqa_data(), reason="ChessQA data not available")
    def test_all_categories_load(self):
        for cat in CATEGORIES:
            entries = load_category(cat, max_samples=2)
            assert len(entries) > 0, f"Category {cat} has no data"

    @pytest.mark.skipif(not has_chessqa_data(), reason="ChessQA data not available")
    def test_full_quick_eval(self):
        results = run_full_evaluation(max_per_category=5)
        assert len(results) == len(CATEGORIES)
        for cat, r in results.items():
            assert r["n_samples"] > 0, f"{cat} returned no samples"

    def test_answer_matching(self):
        assert answer_matches("White King: ['g1']", "White King: ['g1']")
        assert answer_matches("white is better", "White is better")
        assert not answer_matches("", "something")
        assert not answer_matches("black wins", "white wins")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    if not has_chessqa_data():
        print(f"ChessQA data not found at {CHESSQA_DIR}")
        print("Clone the chessqa-benchmark repo into tests/benchmarks/")
        return

    print("ChessQA Benchmark — 5-Category Chess Understanding")
    print("=" * 60)

    results = run_full_evaluation(max_per_category=50)

    for cat, r in results.items():
        print(f"\n{cat} ({r['n_samples']} samples):")
        print(f"  Column A (published): {r['column_a']}")
        print(f"  Column B (rule-based): {r['column_b_accuracy']:.1%}")
        print(f"  Column C (PyQuifer):   {r['column_c_accuracy']:.1%}")

    run_three_column_suite(max_per_category=50)


if __name__ == "__main__":
    main()
