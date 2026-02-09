"""
Run all benchmarks EXCEPT training-heavy ones.
Skips: EP training, local rules, continual, predictive coding.

Stubs (framework not installed) are reported as SKIPPED, not PASSED.
"""
import sys, os, time, json, traceback

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

# Clean known result stems to prevent stale artifacts from prior runs
# inflating report coverage (audit item #4).
_KNOWN_STEMS = [
    'modules', 'consciousness', 'efficiency', 'robustness',
    'chess', 'chessqa', 'searchless_chess', 'cycle', 'llm_ab',
    'penrose_chess', 'plaskett', 'lm_eval',
    'harmbench', 'jailbreakbench', 'toxigen',
    'factscore', 'factbench', 'factreasoner',
    'opencompass', 'helm', 'realtime',
    'no_training_timings',
]
_cleaned = 0
for stem in _KNOWN_STEMS:
    p = os.path.join(results_dir, f'{stem}.json')
    if os.path.exists(p):
        os.remove(p)
        _cleaned += 1
if _cleaned:
    print(f"Cleaned {_cleaned} stale result files from results/")

timings = {}  # name -> seconds (>0 = passed, 0 = skipped, <0 = failed)


def _is_stub_result(result):
    """Check if a run_full_suite() return value indicates a stub/skipped run."""
    if not isinstance(result, dict):
        return False
    if result.get("status") == "skipped":
        return True
    if result.get("stub"):
        return True
    return False


def _run_bench(name, label, runner, *args, **kwargs):
    """Run a benchmark, detect stubs, and record timing."""
    print("\n" + "=" * 60)
    print(f"{label}")
    print("=" * 60)
    t0 = time.time()
    try:
        result = runner(*args, **kwargs)
        if _is_stub_result(result):
            timings[name] = 0  # SKIPPED
            print(f"  SKIPPED (stub)")
        else:
            timings[name] = time.time() - t0
            print(f"  Done in {timings[name]:.1f}s")
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        timings[name] = -1


# ── 1. Module throughput ──
from bench_modules import run_full_suite as run_modules
_run_bench('modules', '1/21  bench_modules (throughput)', run_modules)

# ── 2. Consciousness ──
from bench_consciousness import run_full_suite as run_consciousness
_run_bench('consciousness', '2/21  bench_consciousness (PCI, metastability)', run_consciousness)

# ── 3. Efficiency ──
from bench_efficiency import run_full_suite as run_efficiency
_run_bench('efficiency', '3/21  bench_efficiency (dtype, compile, scaling)', run_efficiency)

# ── 4. Robustness ──
from bench_robustness import run_full_suite as run_robustness
_run_bench('robustness', '4/21  bench_robustness (noise sweep)', run_robustness)

# ── 5. Chess (pytest-only, run its tests directly) ──
print("\n" + "=" * 60)
print("5/21  bench_chess (fortress tests via pytest classes)")
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
    traceback.print_exc()
    timings['chess'] = -1

# ── 6. ChessQA ──
print("\n" + "=" * 60)
print("6/21  bench_chessqa (5 categories)")
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
    traceback.print_exc()
    timings['chessqa'] = -1

# ── 7. Searchless Chess ──
from bench_searchless_chess import run_three_column_suite as run_searchless
_run_bench('searchless_chess', '7/21  bench_searchless_chess (position eval)', run_searchless)

# ── 8. Cycle ──
from bench_cycle import run_full_suite as run_cycle
_run_bench('cycle', '8/21  bench_cycle (CognitiveCycle throughput)', run_cycle)

# ── 9. LLM A/B ──
from bench_llm_ab import run_full_suite as run_llm_ab
_run_bench('llm_ab', '9/21  bench_llm_ab (dummy model, no real LLM)', run_llm_ab)

# ── 10. Penrose Chess ──
from bench_penrose_chess import run_three_column_suite as run_penrose
_run_bench('penrose_chess', '10/21  bench_penrose_chess (Penrose fortress)', run_penrose)

# ── 11. Plaskett Puzzle ──
from bench_plaskett import run_full_suite as run_plaskett
_run_bench('plaskett', '11/21  bench_plaskett (Plaskett tactical puzzle)', run_plaskett)

# ── 12. lm-evaluation-harness ──
from bench_lm_eval import run_full_suite as run_lm_eval
_run_bench('lm_eval', '12/21  bench_lm_eval (lm-evaluation-harness, 8 benchmarks)', run_lm_eval)

# ── 13. HarmBench ──
from bench_harmbench import run_full_suite as run_harmbench
_run_bench('harmbench', '13/21  bench_harmbench (safety refusal rate)', run_harmbench)

# ── 14. JailbreakBench ──
from bench_jailbreakbench import run_full_suite as run_jailbreakbench
_run_bench('jailbreakbench', '14/21  bench_jailbreakbench (jailbreak resistance)', run_jailbreakbench)

# ── 15. TOXIGEN ──
from bench_toxigen import run_full_suite as run_toxigen
_run_bench('toxigen', '15/21  bench_toxigen (toxicity detection)', run_toxigen)

# ── 16. FActScore ──
from bench_factscore import run_full_suite as run_factscore
_run_bench('factscore', '16/21  bench_factscore (factuality scoring)', run_factscore)

# ── 17. FactBench ──
from bench_factbench import run_full_suite as run_factbench
_run_bench('factbench', '17/21  bench_factbench (claim verification)', run_factbench)

# ── 18. FactReasoner ──
from bench_factreasoner import run_full_suite as run_factreasoner
_run_bench('factreasoner', '18/21  bench_factreasoner (factual reasoning)', run_factreasoner)

# ── 19. OpenCompass (stub) ──
from bench_opencompass import run_full_suite as run_opencompass
_run_bench('opencompass', '19/21  bench_opencompass (comprehensive eval, stub)', run_opencompass)

# ── 20. HELM (stub) ──
from bench_helm_eval import run_full_suite as run_helm
_run_bench('helm', '20/21  bench_helm_eval (holistic eval, stub)', run_helm)

# ── 21. Real-Time Latency ──
from bench_realtime import run_full_suite as run_realtime
_run_bench('realtime', '21/21  bench_realtime (bridge latency product targets)', run_realtime)

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
    traceback.print_exc()

print("\nResults saved to tests/benchmarks/results/")
print("Done!")
