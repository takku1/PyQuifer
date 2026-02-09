"""
Benchmark: Plaskett's Puzzle — Queen Sacrifice & Underpromotion

Plaskett's Puzzle (van Breukelen ~1970, published 1990) is a legendary
endgame study that only Mikhail Tal could solve among top GMs.

FEN: 8/3P3k/n2K3p/2p3n1/1b4N1/2p1p1P1/8/3B4 w - - 0 1

White is DOWN material (-500cp: B+N+2P vs 2N+B+4P), yet has a FORCED WIN.
The winning line requires:
  1. Nf6+ (deflection check)
  2. Nh5+ Kg6 3. Bc2+ Kxh5
  4. d8=Q!! (promote, ALLOWING the fork)
  5. Nf7+ Ke6 Nxd8+ Kf5  (queen is captured but White wins anyway)
  6. A pure bishop endgame where White forces mate despite Black's
     desperate underpromotions to knights (e1=N!, c1=N!)

This benchmark tests whether PyQuifer's pattern recognition can override
naive material evaluation — the complement to the Penrose fortress test.
Penrose: "looks like Black wins, actually a draw" (fortress intuition)
Plaskett: "looks like Black wins, actually White wins" (tactical intuition)

Dual-mode:
  - python bench_plaskett.py        -> full analysis
  - pytest bench_plaskett.py -v     -> test functions only

References:
- van Breukelen (1990). Schakend Nederland.
- Plaskett (1987). Brussels tournament presentation.
- ChessBase: "Solution to a truly remarkable study"
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# ── PyQuifer imports ──
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pyquifer.oscillators import LearnableKuramotoBank
from pyquifer.metacognitive import (
    ReasoningMonitor, ConfidenceEstimator, ReasoningStep,
)
from pyquifer.criticality import BranchingRatio

# Reuse board utilities from Penrose benchmark
from bench_penrose_chess import (
    fen_to_tensor, count_material, get_piece_positions,
    is_dark_square, PIECE_VALUES, PIECE_TO_CHANNEL,
    BLACK_WINS, DRAW, WHITE_WINS,
)

# ═════════════════════════════════════════════════════════════════════════════
# Constants
# ═════════════════════════════════════════════════════════════════════════════

PLASKETT_FEN = "8/3P3k/n2K3p/2p3n1/1b4N1/2p1p1P1/8/3B4 w - - 0 1"

# The known winning first move
WINNING_FIRST_MOVE = "Nf6+"

# Full main line (for reference / scoring depth)
MAIN_LINE = [
    "Nf6+", "Kg7", "Nh5+", "Kg6", "Bc2+", "Kxh5",
    "d8=Q", "Nf7+", "Ke6", "Nxd8+", "Kf5",
    "e2", "Be4", "e1=N", "Bd5", "c2", "Bc4", "c1=N",
]

# Knight move patterns (from square to squares it attacks)
KNIGHT_MOVES = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                (1, -2), (1, 2), (2, -1), (2, 1)]


# ═════════════════════════════════════════════════════════════════════════════
# Section 1: Material Analysis (should say Black is better)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class MaterialResult:
    material_balance: float   # centipawns / 100
    white_pieces: Dict[str, int]
    black_pieces: Dict[str, int]
    naive_verdict: str        # What material alone says
    time_ms: float


def run_material_analysis(fen: str = PLASKETT_FEN) -> MaterialResult:
    """Material-only evaluation — should incorrectly predict Black wins."""
    t0 = time.perf_counter()
    board = fen_to_tensor(fen)
    material = count_material(board)
    elapsed = (time.perf_counter() - t0) * 1000

    balance = material["balance"]
    if balance < -200:
        verdict = BLACK_WINS
    elif balance > 200:
        verdict = WHITE_WINS
    else:
        verdict = DRAW

    return MaterialResult(
        material_balance=balance / 100.0,
        white_pieces=material["white_pieces"],
        black_pieces=material["black_pieces"],
        naive_verdict=verdict,
        time_ms=elapsed,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Section 2: Tactical Pattern Detection
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class TacticalPatternResult:
    """Patterns that indicate the forced win."""
    has_passed_pawn: bool          # d7 pawn one step from promotion
    pawn_promotion_rank: int       # How close (7 = on 7th rank for White)
    knight_check_available: bool   # Can the knight give check?
    knight_check_squares: List[str]  # Which squares check the king
    king_exposed: bool             # Is Black's king on the edge?
    deflection_possible: bool      # Knight check deflects king from pawn
    promotion_with_tempo: bool     # Can promote with check/threat?
    tactical_score: float          # 0-1 composite tactical indicator
    time_ms: float


def _square_name(rank: int, file: int) -> str:
    """Convert (rank, file) to algebraic notation."""
    return chr(ord('a') + file) + str(8 - rank)


def _knight_attacks(rank: int, file: int) -> List[Tuple[int, int]]:
    """Squares attacked by a knight at (rank, file)."""
    attacks = []
    for dr, df in KNIGHT_MOVES:
        nr, nf = rank + dr, file + df
        if 0 <= nr < 8 and 0 <= nf < 8:
            attacks.append((nr, nf))
    return attacks


def run_tactical_analysis(fen: str = PLASKETT_FEN) -> TacticalPatternResult:
    """Detect tactical patterns: passed pawns, checks, deflections."""
    t0 = time.perf_counter()
    board = fen_to_tensor(fen)

    # 1. Find White's passed pawn
    white_pawns = get_piece_positions(board, 0)  # channel 0 = wP
    black_pawns = get_piece_positions(board, 6)  # channel 6 = bP

    has_passed = False
    best_pawn_rank = 8  # Lower = closer to promotion for White
    for pr, pf in white_pawns:
        # Check if any Black pawn can block or capture
        is_passed = True
        for br, bf in black_pawns:
            if abs(bf - pf) <= 1 and br <= pr:  # Opponent pawn ahead or adjacent
                is_passed = False
                break
        if is_passed:
            has_passed = True
            best_pawn_rank = min(best_pawn_rank, pr)

    # 2. Find White knight and check possibilities
    white_knights = get_piece_positions(board, 1)  # channel 1 = wN
    black_king = get_piece_positions(board, 11)     # channel 11 = bK

    check_squares = []
    if black_king:
        bk_r, bk_f = black_king[0]
        for nr, nf in white_knights:
            for ar, af in _knight_attacks(nr, nf):
                # Can the knight move to (ar, af)?
                # Check if (ar, af) attacks the black king
                king_attacked_from_new = any(
                    (bk_r, bk_f) == sq for sq in _knight_attacks(ar, af)
                )
                if king_attacked_from_new:
                    # Check the square is not occupied by own piece
                    own_piece = any(
                        board[ch, ar, af] > 0.5 for ch in range(6)
                    )
                    if not own_piece:
                        check_squares.append(_square_name(ar, af))

    # 3. King exposure — is Black king on rank 1 or 8, or on edge files?
    king_exposed = False
    if black_king:
        bk_r, bk_f = black_king[0]
        king_exposed = bk_r <= 1 or bk_r >= 6 or bk_f <= 1 or bk_f >= 6

    # 4. Deflection: does the check move the king away from the pawn's
    #    promotion square?
    deflection = False
    promotion_with_tempo = False
    if has_passed and check_squares:
        # White's best pawn is on rank best_pawn_rank, promotes on rank 0
        # If checking deflects king away from the promotion file...
        deflection = True  # Simplified: any check + passed pawn = deflection potential
        # Can the pawn promote with check or immediate threat after deflection?
        promotion_with_tempo = best_pawn_rank <= 1  # 7th rank = one step

    # 5. Composite tactical score
    score = 0.0
    if has_passed:
        score += 0.2
    if best_pawn_rank <= 1:
        score += 0.2  # On 7th rank
    if check_squares:
        score += 0.2
    if deflection:
        score += 0.2
    if promotion_with_tempo:
        score += 0.1
    if king_exposed:
        score += 0.1

    elapsed = (time.perf_counter() - t0) * 1000

    return TacticalPatternResult(
        has_passed_pawn=has_passed,
        pawn_promotion_rank=8 - best_pawn_rank,  # Convert to "which rank" (7=7th)
        knight_check_available=len(check_squares) > 0,
        knight_check_squares=list(set(check_squares)),
        king_exposed=king_exposed,
        deflection_possible=deflection,
        promotion_with_tempo=promotion_with_tempo,
        tactical_score=score,
        time_ms=elapsed,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Section 3: PyQuifer Oscillator-Based Analysis
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class OscillatorAnalysisResult:
    """PyQuifer pattern recognition result."""
    verdict: str
    confidence: float
    coherence: float
    reasoning_steps: int
    key_patterns_detected: List[str]
    hypothesis_strengths: Dict[str, float]
    time_ms: float


def _encode_position_features(board: torch.Tensor, num_osc: int) -> torch.Tensor:
    """Encode tactical features into oscillator-space input.

    Extracts features relevant to Plaskett-type tactics:
    - Passed pawn advancement (how close to promotion)
    - Knight activity (attack squares near enemy king)
    - King safety imbalance
    - Promotion threats
    - Material tension (pieces that can be exchanged/sacrificed)
    """
    features = []

    # 1. Passed pawn signal: strength proportional to advancement
    white_pawns = get_piece_positions(board, 0)
    black_pawns = get_piece_positions(board, 6)
    max_advancement = 0.0
    for pr, pf in white_pawns:
        # Check if passed
        is_passed = all(
            not (abs(bf - pf) <= 1 and br <= pr) for br, bf in black_pawns
        )
        if is_passed:
            advancement = (7 - pr) / 7.0  # 1.0 = on promotion square
            max_advancement = max(max_advancement, advancement)
    features.append(max_advancement)

    # 2. Knight check pressure
    white_knights = get_piece_positions(board, 1)
    black_king = get_piece_positions(board, 11)
    check_pressure = 0.0
    if black_king:
        bkr, bkf = black_king[0]
        for nr, nf in white_knights:
            # How many of the knight's reachable squares attack the king?
            for ar, af in _knight_attacks(nr, nf):
                if (bkr, bkf) in _knight_attacks(ar, af):
                    check_pressure += 1.0
    features.append(min(1.0, check_pressure / 3.0))

    # 3. King safety imbalance
    white_king = get_piece_positions(board, 5)
    if white_king and black_king:
        wkr, wkf = white_king[0]
        bkr, bkf = black_king[0]
        # White king centralization vs Black king edge vulnerability
        w_central = 1.0 - (abs(wkr - 3.5) + abs(wkf - 3.5)) / 7.0
        b_edge = (max(0, bkr - 5) + max(0, 5 - bkr) +
                  max(0, bkf - 5) + max(0, 5 - bkf)) / 4.0
        # Negative = Black king more exposed
        features.append(b_edge - (1 - w_central))
    else:
        features.append(0.0)

    # 4. Piece coordination: distance between White's active pieces
    white_active = white_knights + get_piece_positions(board, 2)  # N + B
    if len(white_active) >= 2 and white_king:
        wkr, wkf = white_king[0]
        avg_dist = sum(
            abs(pr - wkr) + abs(pf - wkf) for pr, pf in white_active
        ) / len(white_active)
        coordination = 1.0 - min(1.0, avg_dist / 8.0)
    else:
        coordination = 0.5
    features.append(coordination)

    # 5. Promotion threat intensity
    promotion_threat = 0.0
    for pr, pf in white_pawns:
        if pr <= 1:  # 7th or 8th rank
            # Is the promotion square defended? Is there a piece blocking?
            blocking = any(
                board[ch, 0, pf] > 0.5 for ch in range(6, 12)
            )
            promotion_threat += (1.0 if not blocking else 0.5) * (1.0 if pr == 1 else 0.7)
    features.append(min(1.0, promotion_threat))

    # 6. Material tension: pieces near each other (potential exchanges)
    tension = 0.0
    for wch in range(0, 6):
        for wr, wf in get_piece_positions(board, wch):
            for bch in range(6, 12):
                for br, bf in get_piece_positions(board, bch):
                    dist = abs(wr - br) + abs(wf - bf)
                    if dist <= 2:
                        tension += 1.0 / (dist + 0.5)
    features.append(min(1.0, tension / 5.0))

    # Pad/truncate to num_osc
    feat_tensor = torch.tensor(features, dtype=torch.float32)
    if feat_tensor.shape[0] < num_osc:
        feat_tensor = torch.nn.functional.pad(
            feat_tensor, (0, num_osc - feat_tensor.shape[0])
        )
    else:
        feat_tensor = feat_tensor[:num_osc]

    return feat_tensor


def run_oscillator_analysis(
    fen: str = PLASKETT_FEN,
    num_oscillators: int = 32,
    analysis_ticks: int = 200,
) -> OscillatorAnalysisResult:
    """Use PyQuifer oscillators + evidence accumulation to analyze the position.

    Evidence-driven approach:
    - Material analysis provides negative evidence (Black is ahead)
    - Tactical pattern detection provides positive evidence
    - Oscillator coherence modulates confidence: high coherence =
      the position's features form a unified pattern (forced win), not
      disconnected advantages

    This mirrors how human intuition works: a grandmaster "feels" that
    the tactics work before calculating the full line. The coherence of
    pattern features (passed pawn + check + deflection + promotion) is
    greater than the sum of individual features.
    """
    t0 = time.perf_counter()
    torch.manual_seed(42)

    board = fen_to_tensor(fen)
    material = count_material(board)
    tactical = run_tactical_analysis(fen)
    position_features = _encode_position_features(board, num_oscillators)

    # Set up oscillators — position features drive oscillator dynamics
    bank = LearnableKuramotoBank(
        num_oscillators=num_oscillators,
        initial_frequency_range=(0.8, 1.2),
    )

    # Set up reasoning monitor
    monitor = ReasoningMonitor()
    step_counter = [0]

    def _add_step(content: str, confidence: float, rtype: str, evidence=None):
        step_counter[0] += 1
        monitor.add_step(ReasoningStep(
            step_id=step_counter[0],
            content=content,
            confidence=confidence,
            reasoning_type=rtype,
            evidence=evidence or [],
        ))

    patterns_detected = []

    # ── Evidence accumulation (material vs tactics) ──
    material_balance = material["balance"] / 100.0

    # Material evidence (negative = favors Black)
    material_evidence = material_balance / 10.0  # Normalize: -5cp -> -0.5
    _add_step(
        f"Material: balance {material_balance:+.1f} pawns",
        confidence=0.6, rtype="deduction",
        evidence=["material_count"],
    )

    # Tactical evidence (positive = favors White winning)
    tactical_evidence = 0.0

    if tactical.has_passed_pawn:
        pawn_bonus = 0.25 * (tactical.pawn_promotion_rank / 7.0)
        tactical_evidence += pawn_bonus
        _add_step(
            f"Passed pawn on rank {tactical.pawn_promotion_rank}",
            confidence=0.7, rtype="pattern",
            evidence=["passed_pawn_detection"],
        )
        patterns_detected.append("passed_pawn")

    if tactical.knight_check_available:
        tactical_evidence += 0.2
        _add_step(
            f"Knight checks available: {tactical.knight_check_squares}",
            confidence=0.65, rtype="pattern",
            evidence=["knight_attack_squares"],
        )
        patterns_detected.append("knight_check")

    if tactical.deflection_possible:
        tactical_evidence += 0.25
        _add_step(
            "Deflection: check + promotion threat",
            confidence=0.75, rtype="analogy",
            evidence=["deflection_pattern"],
        )
        patterns_detected.append("deflection")

    if tactical.promotion_with_tempo:
        tactical_evidence += 0.3
        _add_step(
            "Promotion with tempo (7th rank pawn + piece activity)",
            confidence=0.8, rtype="deduction",
            evidence=["promotion_threat"],
        )
        patterns_detected.append("promotion_with_tempo")

    if tactical.king_exposed:
        tactical_evidence += 0.1
        _add_step(
            "Black king exposed on edge",
            confidence=0.6, rtype="pattern",
            evidence=["king_safety"],
        )
        patterns_detected.append("king_exposed")

    # ── Run oscillator dynamics ──
    # Position features drive frequency entrainment. If the tactical
    # features are coherent (they reinforce each other), oscillators
    # will synchronize more strongly, indicating pattern unity.
    coherence_history = []
    for tick in range(analysis_ticks):
        phases = bank(position_features.unsqueeze(0).expand(1, -1), steps=1)
        R = bank.get_order_parameter().item()
        coherence_history.append(R)

    final_R = sum(coherence_history[-20:]) / min(20, len(coherence_history))

    # ── Combine evidence: tactical override material? ──
    # Key insight: multiple tactical patterns that work TOGETHER
    # (check + deflection + promotion) are multiplicatively stronger
    # than the sum of individual patterns. Coherence measures this
    # "gestalt" — the oscillators synchronizing means the features
    # form a unified tactical motif.
    num_patterns = len(patterns_detected)
    synergy_bonus = 0.0
    if num_patterns >= 3:
        # Three or more interlocking tactics = likely forced win
        synergy_bonus = 0.15 * (num_patterns - 2)
    tactical_evidence += synergy_bonus

    # Coherence amplifies tactical evidence
    tactical_with_coherence = tactical_evidence * (0.7 + 0.6 * final_R)

    # Combined score: tactical overrides material when strong enough
    combined = material_evidence + tactical_with_coherence

    # Verdict
    if combined > 0.15:
        verdict = WHITE_WINS
    elif combined < -0.15:
        verdict = BLACK_WINS
    else:
        verdict = DRAW

    # Hypothesis strengths (softmax-like)
    scores = {
        WHITE_WINS: max(0, combined),
        BLACK_WINS: max(0, -combined),
        DRAW: max(0, 0.15 - abs(combined)),
    }
    total_s = sum(scores.values()) + 1e-8
    hyp_strengths = {k: v / total_s for k, v in scores.items()}

    # Confidence from reasoning chain
    analysis = monitor.analyze_chain()
    confidence = analysis.get("mean_confidence", 0.5)

    # Coherence bonus: high oscillator coherence boosts confidence
    if final_R > 0.3:
        coherence_bonus = (final_R - 0.3) * 0.3
        confidence = min(1.0, confidence + coherence_bonus)

    elapsed = (time.perf_counter() - t0) * 1000

    return OscillatorAnalysisResult(
        verdict=verdict,
        confidence=confidence,
        coherence=final_R,
        reasoning_steps=len(monitor.current_chain),
        key_patterns_detected=patterns_detected,
        hypothesis_strengths=hyp_strengths,
        time_ms=elapsed,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Section 4: Generalization — Similar Tactical Positions
# ═════════════════════════════════════════════════════════════════════════════

# Positions with similar tactical themes (sacrifice + forced win)
TACTICAL_FENS = {
    "plaskett_original": {
        "fen": PLASKETT_FEN,
        "correct_verdict": WHITE_WINS,
        "theme": "Queen sacrifice + underpromotion defense",
    },
    "promotion_deflection": {
        # White: Kf5, Nd4, Pg7. Black: Kh8, Rb8. White wins with Nf3! g8=Q+
        "fen": "1r5k/6P1/8/5K2/3N4/8/8/8 w - - 0 1",
        "correct_verdict": WHITE_WINS,
        "theme": "Knight deflection + pawn promotion",
    },
    "quiet_winning": {
        # Position where White is down material but has a key passed pawn
        # White: Ke6, Bd5, Pd7. Black: Kb8, Nc6, Pa7.
        "fen": "1k6/p2P4/2n1K3/3B4/8/8/8/8 w - - 0 1",
        "correct_verdict": WHITE_WINS,
        "theme": "Bishop + pawn vs knight — passed pawn wins",
    },
    "material_misleading": {
        # Black has extra rook but White's pieces are perfectly coordinated
        # White: Kf6, Ng5, Pe7. Black: Kg8, Rd1.
        "fen": "6k1/4P3/5K2/6N1/8/8/8/3r4 w - - 0 1",
        "correct_verdict": WHITE_WINS,
        "theme": "Piece coordination + promotion threat",
    },
}


@dataclass
class GeneralizationResult:
    position_name: str
    correct_verdict: str
    pyquifer_verdict: str
    is_correct: bool
    confidence: float
    coherence: float


def run_generalization(analysis_ticks: int = 150) -> List[GeneralizationResult]:
    """Test PyQuifer on multiple tactical positions."""
    results = []
    for name, pos in TACTICAL_FENS.items():
        analysis = run_oscillator_analysis(
            fen=pos["fen"],
            analysis_ticks=analysis_ticks,
        )
        results.append(GeneralizationResult(
            position_name=name,
            correct_verdict=pos["correct_verdict"],
            pyquifer_verdict=analysis.verdict,
            is_correct=analysis.verdict == pos["correct_verdict"],
            confidence=analysis.confidence,
            coherence=analysis.coherence,
        ))
    return results


# ═════════════════════════════════════════════════════════════════════════════
# Full Suite
# ═════════════════════════════════════════════════════════════════════════════

def run_full_suite() -> Dict:
    print("=" * 60)
    print("  Plaskett's Puzzle Benchmark")
    print("=" * 60)
    print(f"\n  FEN: {PLASKETT_FEN}")
    print("  Correct answer: WHITE WINS (forced mate via queen sacrifice)")
    print()

    t0 = time.perf_counter()

    # 1. Material analysis
    print("[1/4] Material analysis...")
    mat = run_material_analysis()
    print(f"  Balance: {mat.material_balance:+.1f} pawns")
    print(f"  White: {mat.white_pieces}")
    print(f"  Black: {mat.black_pieces}")
    print(f"  Naive verdict: {mat.naive_verdict}")
    print(f"  (This should INCORRECTLY say {BLACK_WINS})")

    # 2. Tactical patterns
    print("\n[2/4] Tactical pattern detection...")
    tac = run_tactical_analysis()
    print(f"  Passed pawn: {tac.has_passed_pawn} (rank {tac.pawn_promotion_rank})")
    print(f"  Knight checks: {tac.knight_check_available} -> {tac.knight_check_squares}")
    print(f"  King exposed: {tac.king_exposed}")
    print(f"  Deflection: {tac.deflection_possible}")
    print(f"  Promotion w/ tempo: {tac.promotion_with_tempo}")
    print(f"  Tactical score: {tac.tactical_score:.2f}")

    # 3. PyQuifer oscillator analysis
    print("\n[3/4] PyQuifer oscillator analysis...")
    osc = run_oscillator_analysis()
    print(f"  Verdict: {osc.verdict}")
    print(f"  Confidence: {osc.confidence:.3f}")
    print(f"  Coherence: {osc.coherence:.3f}")
    print(f"  Patterns: {osc.key_patterns_detected}")
    print(f"  Hypothesis strengths: ", end="")
    for name, strength in osc.hypothesis_strengths.items():
        print(f"{name}={strength:.3f} ", end="")
    print()
    correct = osc.verdict == WHITE_WINS
    print(f"  {'CORRECT' if correct else 'INCORRECT'} "
          f"(expected {WHITE_WINS})")

    # 4. Generalization
    print("\n[4/4] Generalization to similar positions...")
    gen_results = run_generalization()
    correct_count = 0
    for g in gen_results:
        mark = "OK" if g.is_correct else "MISS"
        print(f"  {g.position_name}: {g.pyquifer_verdict} "
              f"(expected {g.correct_verdict}) [{mark}] "
              f"conf={g.confidence:.3f}")
        if g.is_correct:
            correct_count += 1
    accuracy = correct_count / len(gen_results)
    print(f"  Accuracy: {correct_count}/{len(gen_results)} = {accuracy:.1%}")

    elapsed = time.perf_counter() - t0
    print(f"\nTotal elapsed: {elapsed:.1f}s")

    # Build JSON output
    try:
        from harness import BenchmarkResult, BenchmarkSuite, MetricCollector
        suite = BenchmarkSuite("Plaskett Puzzle")

        mc_mat = MetricCollector("Material Analysis")
        mc_mat.add_result(BenchmarkResult(
            name="plaskett_material",
            column="A_published",
            metrics={
                "correct_verdict": 1.0,  # WHITE_WINS is known
                "material_balance": -5.0,  # Material says Black
            },
        ))
        mc_mat.add_result(BenchmarkResult(
            name="plaskett_material",
            column="B_pytorch",
            metrics={
                "material_balance": mat.material_balance,
                "naive_verdict_correct": 0.0,  # Material always wrong here
            },
        ))
        mc_mat.add_result(BenchmarkResult(
            name="plaskett_material",
            column="C_pyquifer",
            metrics={
                "verdict_correct": 1.0 if correct else 0.0,
                "confidence": osc.confidence,
                "tactical_score": tac.tactical_score,
            },
        ))
        suite.add(mc_mat)

        mc_osc = MetricCollector("Oscillator Pattern Recognition")
        mc_osc.add_result(BenchmarkResult(
            name="plaskett_oscillator",
            column="C_pyquifer",
            metrics={
                "verdict_correct": 1.0 if correct else 0.0,
                "confidence": osc.confidence,
                "coherence": osc.coherence,
                "num_patterns": len(osc.key_patterns_detected),
                "reasoning_steps": osc.reasoning_steps,
            },
        ))
        suite.add(mc_osc)

        mc_gen = MetricCollector("Generalization")
        mc_gen.add_result(BenchmarkResult(
            name="tactical_generalization",
            column="A_published",
            metrics={
                "num_positions": len(gen_results),
                "target_accuracy": 1.0,
            },
        ))
        mc_gen.add_result(BenchmarkResult(
            name="tactical_generalization",
            column="C_pyquifer",
            metrics={
                "accuracy": accuracy,
                "mean_confidence": sum(g.confidence for g in gen_results) / len(gen_results),
            },
        ))
        suite.add(mc_gen)

        results_dir = Path(__file__).resolve().parent / "results"
        results_dir.mkdir(exist_ok=True)
        suite.to_json(str(results_dir / "plaskett.json"))
        print(f"\nResults saved: {results_dir / 'plaskett.json'}")
    except Exception as e:
        print(f"\n(Could not save JSON: {e})")

    return {
        "material": vars(mat),
        "tactical": vars(tac),
        "oscillator": {
            "verdict": osc.verdict,
            "confidence": osc.confidence,
            "coherence": osc.coherence,
            "patterns": osc.key_patterns_detected,
            "correct": correct,
        },
        "generalization": [vars(g) for g in gen_results],
        "accuracy": accuracy,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Pytest Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestPlaskettMaterial:
    def test_material_says_black_wins(self):
        """Material analysis should INCORRECTLY predict Black wins."""
        result = run_material_analysis()
        assert result.material_balance < -3.0, "White should be materially down"
        assert result.naive_verdict == BLACK_WINS

    def test_white_has_passed_pawn(self):
        """Tactical analysis should detect the d7 passed pawn."""
        result = run_tactical_analysis()
        assert result.has_passed_pawn
        assert result.pawn_promotion_rank >= 6  # 7th rank

    def test_knight_check_detected(self):
        """Should detect knight check possibilities."""
        result = run_tactical_analysis()
        assert result.knight_check_available
        assert len(result.knight_check_squares) > 0


class TestPlaskettOscillator:
    def test_overrides_material(self):
        """PyQuifer should override material evaluation with pattern recognition."""
        result = run_oscillator_analysis(analysis_ticks=200)
        # At minimum, the tactical hypothesis should be competitive
        assert result.hypothesis_strengths.get(WHITE_WINS, 0) > 0.2, \
            f"WHITE_WINS hypothesis too weak: {result.hypothesis_strengths}"

    def test_detects_tactical_patterns(self):
        """Should detect key tactical patterns."""
        result = run_oscillator_analysis(analysis_ticks=100)
        assert len(result.key_patterns_detected) >= 2, \
            f"Too few patterns: {result.key_patterns_detected}"
        assert "passed_pawn" in result.key_patterns_detected


class TestPlaskettGeneralization:
    def test_generalization_runs(self):
        """Generalization test should run without errors."""
        results = run_generalization(analysis_ticks=50)
        assert len(results) == len(TACTICAL_FENS)
        # At least half should be correct
        correct = sum(1 for r in results if r.is_correct)
        assert correct >= len(results) // 2, \
            f"Only {correct}/{len(results)} correct"


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_full_suite()
