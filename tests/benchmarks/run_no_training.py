"""
Run all benchmarks EXCEPT training-heavy ones.
Skips: EP training, local rules, continual, predictive coding.
"""
import sys, os, time, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import torch
torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print()

os.makedirs(os.path.join(os.path.dirname(__file__), 'results'), exist_ok=True)
results_dir = os.path.join(os.path.dirname(__file__), 'results')

timings = {}

# ── 1. Module throughput ──
print("=" * 60)
print("1/11  bench_modules (throughput)")
print("=" * 60)
t0 = time.time()
try:
    from bench_modules import run_full_suite as run_modules
    run_modules()
    timings['modules'] = time.time() - t0
    print(f"  Done in {timings['modules']:.1f}s")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback; traceback.print_exc()
    timings['modules'] = -1

# ── 2. Consciousness ──
print("\n" + "=" * 60)
print("2/11  bench_consciousness (PCI, metastability)")
print("=" * 60)
t0 = time.time()
try:
    from bench_consciousness import run_full_suite as run_consciousness
    run_consciousness()
    timings['consciousness'] = time.time() - t0
    print(f"  Done in {timings['consciousness']:.1f}s")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback; traceback.print_exc()
    timings['consciousness'] = -1

# ── 3. Efficiency ──
print("\n" + "=" * 60)
print("3/11  bench_efficiency (dtype, compile, scaling)")
print("=" * 60)
t0 = time.time()
try:
    from bench_efficiency import run_full_suite as run_efficiency
    run_efficiency()
    timings['efficiency'] = time.time() - t0
    print(f"  Done in {timings['efficiency']:.1f}s")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback; traceback.print_exc()
    timings['efficiency'] = -1

# ── 4. Robustness ──
print("\n" + "=" * 60)
print("4/11  bench_robustness (noise sweep)")
print("=" * 60)
t0 = time.time()
try:
    from bench_robustness import run_full_suite as run_robustness
    run_robustness()
    timings['robustness'] = time.time() - t0
    print(f"  Done in {timings['robustness']:.1f}s")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback; traceback.print_exc()
    timings['robustness'] = -1

# ── 5. Chess (bench_chess.py is pytest-only, run its tests directly) ──
print("\n" + "=" * 60)
print("5/11  bench_chess (fortress tests via pytest classes)")
print("=" * 60)
t0 = time.time()
try:
    from bench_chess import TestFortressDetection, TestEngineOverride, TestGeneralization
    t = TestFortressDetection()
    t.test_penrose_pattern_recognizes_draw()
    t.test_minimax_fails_on_fortress()
    print("  FortressDetection: 2/2 passed")
    t2 = TestEngineOverride()
    t2.test_override_finds_draw()
    t2.test_override_confidence_above_threshold()
    print("  EngineOverride: 2/2 passed")
    t3 = TestGeneralization()
    t3.test_generalization_accuracy()
    print("  Generalization: 1/1 passed")
    timings['chess'] = time.time() - t0
    print(f"  Done in {timings['chess']:.1f}s")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback; traceback.print_exc()
    timings['chess'] = -1

# ── 6. ChessQA ──
print("\n" + "=" * 60)
print("6/11  bench_chessqa (5 categories)")
print("=" * 60)
t0 = time.time()
try:
    from bench_chessqa import has_chessqa_data, run_three_column_suite as run_chessqa
    if has_chessqa_data():
        run_chessqa(max_per_category=50)
        timings['chessqa'] = time.time() - t0
        print(f"  Done in {timings['chessqa']:.1f}s")
    else:
        print("  SKIPPED: ChessQA data not available")
        timings['chessqa'] = 0
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback; traceback.print_exc()
    timings['chessqa'] = -1

# ── 7. Searchless Chess ──
print("\n" + "=" * 60)
print("7/11  bench_searchless_chess (position eval)")
print("=" * 60)
t0 = time.time()
try:
    from bench_searchless_chess import run_three_column_suite as run_searchless
    run_searchless()
    timings['searchless_chess'] = time.time() - t0
    print(f"  Done in {timings['searchless_chess']:.1f}s")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback; traceback.print_exc()
    timings['searchless_chess'] = -1

# ── 8. Cycle ──
print("\n" + "=" * 60)
print("8/11  bench_cycle (CognitiveCycle throughput)")
print("=" * 60)
t0 = time.time()
try:
    from bench_cycle import run_full_suite as run_cycle
    run_cycle()
    timings['cycle'] = time.time() - t0
    print(f"  Done in {timings['cycle']:.1f}s")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback; traceback.print_exc()
    timings['cycle'] = -1

# ── 9. LLM A/B ──
print("\n" + "=" * 60)
print("9/11  bench_llm_ab (dummy model, no real LLM)")
print("=" * 60)
t0 = time.time()
try:
    from bench_llm_ab import run_full_suite as run_llm_ab
    run_llm_ab()
    timings['llm_ab'] = time.time() - t0
    print(f"  Done in {timings['llm_ab']:.1f}s")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback; traceback.print_exc()
    timings['llm_ab'] = -1

# ── 10. Penrose Chess ──
print("\n" + "=" * 60)
print("10/11  bench_penrose_chess (Penrose fortress)")
print("=" * 60)
t0 = time.time()
try:
    from bench_penrose_chess import run_three_column_suite as run_penrose
    run_penrose()
    timings['penrose_chess'] = time.time() - t0
    print(f"  Done in {timings['penrose_chess']:.1f}s")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback; traceback.print_exc()
    timings['penrose_chess'] = -1

# ── 11. Plaskett Puzzle ──
print("\n" + "=" * 60)
print("11/11  bench_plaskett (Plaskett tactical puzzle)")
print("=" * 60)
t0 = time.time()
try:
    from bench_plaskett import run_full_suite as run_plaskett
    run_plaskett()
    timings['plaskett'] = time.time() - t0
    print(f"  Done in {timings['plaskett']:.1f}s")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback; traceback.print_exc()
    timings['plaskett'] = -1

# ── Summary ──
num_benchmarks = len(timings)
print("\n" + "=" * 60)
print("TIMING SUMMARY (no-training run)")
print("=" * 60)
total = 0
passed = 0
failed = 0
skipped = 0
for name, t in timings.items():
    if t > 0:
        status = f"{t:.1f}s"
        total += t
        passed += 1
    elif t == 0:
        status = "SKIPPED"
        skipped += 1
    else:
        status = "FAILED"
        failed += 1
    print(f"  {name:24s}: {status}")
print(f"  {'TOTAL':24s}: {total:.1f}s")
print(f"  Passed: {passed}/{num_benchmarks}  Failed: {failed}  Skipped: {skipped}")

with open(os.path.join(results_dir, 'no_training_timings.json'), 'w') as f:
    json.dump(timings, f, indent=2)

# ── Generate report ──
print("\n" + "=" * 60)
print("GENERATING REPORT")
print("=" * 60)
try:
    from generate_report import enhanced_report
    report_path = enhanced_report(results_dir, os.path.join(results_dir, 'BENCHMARK_REPORT.md'))
    print(f"Report: {report_path}")
except Exception as e:
    print(f"Report generation failed: {e}")
    import traceback; traceback.print_exc()

print("\nResults saved to tests/benchmarks/results/")
print("Done!")
