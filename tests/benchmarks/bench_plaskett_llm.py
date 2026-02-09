"""
Plaskett's Puzzle: LLM vs PyQuifer Comparison

Loads Phi-4-multimodal-instruct and asks it to solve Plaskett's Puzzle,
then compares its reasoning to PyQuifer's pattern recognition.

This is a head-to-head comparison:
- Column B (vanilla LLM): Just the model, no oscillator modulation
- Column C (PyQuifer): Oscillator-based pattern recognition

Usage:
    python bench_plaskett_llm.py
"""

import sys
import time
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

MODEL_PATH = r"Z:\mizukiai\models\unified\Phi-4-multimodal-instruct"
PLASKETT_FEN = "8/3P3k/n2K3p/2p3n1/1b4N1/2p1p1P1/8/3B4 w - - 0 1"

PROMPT_TEXT = (
    "You are a chess grandmaster. Analyze the following chess position "
    "and find the best move for White.\n\n"
    "Position (FEN): 8/3P3k/n2K3p/2p3n1/1b4N1/2p1p1P1/8/3B4 w - - 0 1\n\n"
    "Board layout:\n"
    "- White: King d6, Knight g4, Bishop d1, Pawns d7 and g3\n"
    "- Black: King h7, Knights a6 and g5, Bishop b4, Pawns c5, c3, e3, h6\n\n"
    "White is down material (-5 pawns equivalent) but has a passed pawn on d7.\n\n"
    "Questions:\n"
    "1. What is the material balance? Who seems to be winning?\n"
    "2. What is the best first move for White and why?\n"
    "3. What happens after that move? Give the key continuation.\n"
    "4. What is the final verdict: does White win, lose, or draw?\n\n"
    "Think step by step."
)


def load_model():
    """Load Phi-4 in 4-bit quantization to fit on RTX 4090.

    Matches the exact loading pattern from brain_unified.py.
    """
    from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

    print(f"Loading model from {MODEL_PATH}...")
    t0 = time.time()

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
    )
    model.eval()

    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")
    print(f"VRAM used: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    return model, processor


def run_llm_analysis(model, processor):
    """Ask the LLM to solve Plaskett's Puzzle using official Phi-4 API."""
    print("\n" + "=" * 60)
    print("LLM Analysis (Phi-4, no PyQuifer)")
    print("=" * 60)

    t0 = time.time()

    # Use the official Phi-4-multimodal prompt format for text-only
    prompt = f"<|user|>{PROMPT_TEXT}<|end|><|assistant|>"

    # processor(..., images=None) sets input_mode=LANGUAGE automatically
    inputs = processor(prompt, images=None, return_tensors="pt").to("cuda:0")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.1,
            do_sample=True,
        )

    # Decode only the new tokens
    generate_ids = outputs[:, inputs["input_ids"].shape[1]:]
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    elapsed = time.time() - t0

    print(f"\nLLM Response ({elapsed:.1f}s):")
    print("-" * 40)
    print(response)
    print("-" * 40)

    # Score the response
    response_lower = response.lower()

    scores = {
        "mentions_nf6": "nf6" in response_lower or "n f6" in response_lower,
        "mentions_promotion": "d8" in response_lower or "promot" in response_lower,
        "mentions_queen_sac": ("sacrifice" in response_lower or "sac" in response_lower) and "queen" in response_lower,
        "says_white_wins": "white wins" in response_lower or "white is winning" in response_lower,
        "mentions_check": "check" in response_lower,
        "mentions_deflection": "deflect" in response_lower,
        "mentions_underpromotion": "underpromot" in response_lower or "knight promot" in response_lower or "=n" in response_lower,
    }

    correct_count = sum(scores.values())
    print(f"\nScoring ({correct_count}/7):")
    for key, val in scores.items():
        print(f"  {key}: {'YES' if val else 'no'}")

    return {
        "response": response,
        "scores": scores,
        "correct_count": correct_count,
        "time_s": elapsed,
    }


def run_pyquifer_analysis():
    """Run PyQuifer's pattern recognition for comparison."""
    print("\n" + "=" * 60)
    print("PyQuifer Analysis (oscillator pattern recognition)")
    print("=" * 60)

    from bench_plaskett import run_oscillator_analysis, run_tactical_analysis

    t0 = time.time()
    tactical = run_tactical_analysis()
    osc = run_oscillator_analysis()
    elapsed = time.time() - t0

    print(f"\nPyQuifer Result ({elapsed:.3f}s):")
    print(f"  Verdict: {osc.verdict}")
    print(f"  Confidence: {osc.confidence:.3f}")
    print(f"  Coherence: {osc.coherence:.3f}")
    print(f"  Patterns detected: {osc.key_patterns_detected}")
    print(f"  Tactical score: {tactical.tactical_score:.2f}")

    scores = {
        "mentions_nf6": "knight_check" in osc.key_patterns_detected,
        "mentions_promotion": "promotion_with_tempo" in osc.key_patterns_detected,
        "mentions_queen_sac": False,  # PyQuifer doesn't reason about specific moves
        "says_white_wins": osc.verdict == "WHITE_WINS",
        "mentions_check": "knight_check" in osc.key_patterns_detected,
        "mentions_deflection": "deflection" in osc.key_patterns_detected,
        "mentions_underpromotion": False,  # Would require deeper analysis
    }

    correct_count = sum(scores.values())
    print(f"\nScoring ({correct_count}/7):")
    for key, val in scores.items():
        print(f"  {key}: {'YES' if val else 'no'}")

    return {
        "verdict": osc.verdict,
        "confidence": osc.confidence,
        "patterns": osc.key_patterns_detected,
        "scores": scores,
        "correct_count": correct_count,
        "time_s": elapsed,
    }


def main():
    print("=" * 60)
    print("  Plaskett's Puzzle: LLM vs PyQuifer Head-to-Head")
    print("=" * 60)
    print(f"\n  FEN: {PLASKETT_FEN}")
    print("  Correct: White wins (forced mate via Nf6+ -> d8=Q!! sacrifice)")
    print()

    # PyQuifer first (fast, no GPU needed for this)
    pq_result = run_pyquifer_analysis()

    # Then LLM
    try:
        model, processor = load_model()
        llm_result = run_llm_analysis(model, processor)

        # Comparison
        print("\n" + "=" * 60)
        print("  HEAD-TO-HEAD COMPARISON")
        print("=" * 60)
        print(f"{'Metric':<30s} {'LLM':>10s} {'PyQuifer':>10s}")
        print("-" * 50)
        print(f"{'Correct verdict?':<30s} {'YES' if llm_result['scores']['says_white_wins'] else 'no':>10s} {'YES' if pq_result['scores']['says_white_wins'] else 'no':>10s}")
        print(f"{'Score (out of 7)':<30s} {llm_result['correct_count']:>10d} {pq_result['correct_count']:>10d}")
        print(f"{'Time':<30s} {llm_result['time_s']:>9.1f}s {pq_result['time_s']:>9.3f}s")
        print(f"{'Speedup':<30s} {'':>10s} {llm_result['time_s'] / max(pq_result['time_s'], 0.001):>9.0f}x")

        # Cleanup GPU memory
        del model, processor
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"\nLLM loading failed: {e}")
        import traceback
        traceback.print_exc()
        print("\n(PyQuifer result above is still valid)")


if __name__ == "__main__":
    main()
