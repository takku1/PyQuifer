"""
Quick benchmark run — small datasets, few epochs.
Produces JSON results + unified report without waiting an hour.
"""
import sys, os, time, json

# Ensure pyquifer is importable
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
total_sections = 14

# ── 1/14. Module throughput (no training, fast) ──
print("=" * 60)
print("1/14  bench_modules (throughput)")
print("=" * 60)
t0 = time.time()
try:
    from bench_modules import run_full_suite as run_modules
    run_modules()
    timings['modules'] = time.time() - t0
    print(f"  Done in {timings['modules']:.1f}s")
except Exception as e:
    print(f"  FAILED: {e}")
    timings['modules'] = -1

# ── 2/14. Consciousness metrics (no training, fast) ──
print("\n" + "=" * 60)
print("2/14  bench_consciousness (PCI, metastability)")
print("=" * 60)
t0 = time.time()
try:
    from bench_consciousness import run_full_suite as run_consciousness
    run_consciousness()
    timings['consciousness'] = time.time() - t0
    print(f"  Done in {timings['consciousness']:.1f}s")
except Exception as e:
    print(f"  FAILED: {e}")
    timings['consciousness'] = -1

# ── 3/14. Efficiency sweeps (no training, fast) ──
print("\n" + "=" * 60)
print("3/14  bench_efficiency (dtype, compile, scaling)")
print("=" * 60)
t0 = time.time()
try:
    from bench_efficiency import run_full_suite as run_efficiency
    run_efficiency()
    timings['efficiency'] = time.time() - t0
    print(f"  Done in {timings['efficiency']:.1f}s")
except Exception as e:
    print(f"  FAILED: {e}")
    timings['efficiency'] = -1

# ── 4/14. EP training (tiny: 500 samples, 3 epochs) ──
print("\n" + "=" * 60)
print("4/14  bench_ep_training (MNIST 500 samples, 3 epochs)")
print("=" * 60)
t0 = time.time()
try:
    from bench_ep_training import bench_classification, bench_xor, bench_convergence
    from harness import MetricCollector

    # XOR first (fast, should reach 100%)
    mc_xor = bench_xor(num_epochs=100, seed=42)
    print(f"  XOR: {mc_xor.results[-1].metrics}")

    # MNIST with tiny settings
    mc_mnist = bench_classification(
        dataset="mnist", num_epochs=3, seed=42,
        max_train=500, max_test=200, num_osc=32, ep_steps=20,
    )
    print(f"  MNIST (500 train, 3 ep): {mc_mnist.results[-1].metrics}")

    # Fashion-MNIST tiny
    mc_fmnist = bench_classification(
        dataset="fashion_mnist", num_epochs=3, seed=42,
        max_train=500, max_test=200, num_osc=32, ep_steps=20,
    )
    print(f"  Fashion-MNIST (500 train, 3 ep): {mc_fmnist.results[-1].metrics}")

    # Convergence check
    mc_conv = bench_convergence(
        num_epochs=3, ep_steps=20,
        max_train=300, max_test=100,
    )
    print(f"  Convergence: {mc_conv.results[-1].metrics}")

    # Save results
    ep_results = {
        'xor': mc_xor.results[-1].metrics if mc_xor.results else {},
        'mnist': mc_mnist.results[-1].metrics if mc_mnist.results else {},
        'fashion_mnist': mc_fmnist.results[-1].metrics if mc_fmnist.results else {},
        'convergence': mc_conv.results[-1].metrics if mc_conv.results else {},
        'settings': {'max_train': 500, 'max_test': 200, 'num_epochs': 3, 'ep_steps': 20},
    }
    with open(os.path.join(results_dir, 'ep_training_quick.json'), 'w') as f:
        json.dump(ep_results, f, indent=2, default=str)

    timings['ep_training'] = time.time() - t0
    print(f"  Done in {timings['ep_training']:.1f}s")
except Exception as e:
    import traceback; traceback.print_exc()
    timings['ep_training'] = -1

# ── 5/14. Local rules (tiny: 500 samples, 3 epochs) ──
print("\n" + "=" * 60)
print("5/14  bench_local_rules (500 samples, 3 epochs)")
print("=" * 60)
t0 = time.time()
try:
    from bench_local_rules import bench_three_factor, bench_dendritic, bench_osc_gated

    mc_3f = bench_three_factor(
        num_epochs=3, seed=42, max_train=500, max_test=200,
    )
    print(f"  ThreeFactorRule: {mc_3f.results[-1].metrics}")

    mc_dend = bench_dendritic(
        num_epochs=3, seed=42, max_train=500, max_test=200,
    )
    print(f"  DendriticStack: {mc_dend.results[-1].metrics}")

    mc_osg = bench_osc_gated(
        num_epochs=3, seed=42, max_train=500, max_test=200,
    )
    print(f"  OscGatedPlasticity: {mc_osg.results[-1].metrics}")

    local_results = {
        'three_factor': mc_3f.results[-1].metrics if mc_3f.results else {},
        'dendritic': mc_dend.results[-1].metrics if mc_dend.results else {},
        'osc_gated': mc_osg.results[-1].metrics if mc_osg.results else {},
        'settings': {'max_train': 500, 'max_test': 200, 'num_epochs': 3},
    }
    with open(os.path.join(results_dir, 'local_rules_quick.json'), 'w') as f:
        json.dump(local_results, f, indent=2, default=str)

    timings['local_rules'] = time.time() - t0
    print(f"  Done in {timings['local_rules']:.1f}s")
except Exception as e:
    import traceback; traceback.print_exc()
    timings['local_rules'] = -1

# ── 6/14. Continual learning (tiny: 300 samples, 2 epochs/task) ──
print("\n" + "=" * 60)
print("6/14  bench_continual (300 samples, 2 epochs/task)")
print("=" * 60)
t0 = time.time()
try:
    from bench_continual import bench_split_mnist

    mc_cont = bench_split_mnist(
        epochs_per_task=2, seed=42,
        max_train=300, max_test=100,
    )
    print(f"  Split-MNIST continual: {mc_cont.results[-1].metrics}")

    cont_results = {
        'split_mnist': mc_cont.results[-1].metrics if mc_cont.results else {},
        'settings': {'max_train': 300, 'max_test': 100, 'epochs_per_task': 2},
    }
    with open(os.path.join(results_dir, 'continual_quick.json'), 'w') as f:
        json.dump(cont_results, f, indent=2, default=str)

    timings['continual'] = time.time() - t0
    print(f"  Done in {timings['continual']:.1f}s")
except Exception as e:
    import traceback; traceback.print_exc()
    timings['continual'] = -1

# ── 7/14. Robustness (tiny noise sweep) ──
print("\n" + "=" * 60)
print("7/14  bench_robustness (noise sweep)")
print("=" * 60)
t0 = time.time()
try:
    from bench_robustness import run_full_suite as run_robustness
    run_robustness()
    timings['robustness'] = time.time() - t0
    print(f"  Done in {timings['robustness']:.1f}s")
except Exception as e:
    print(f"  FAILED: {e}")
    timings['robustness'] = -1

# ── 8/14. Chess (fortress detection) ──
print("\n" + "=" * 60)
print("8/14  bench_chess (fortress tests)")
print("=" * 60)
t0 = time.time()
try:
    from bench_chess import run_full_suite as run_chess
    run_chess()
    timings['chess'] = time.time() - t0
    print(f"  Done in {timings['chess']:.1f}s")
except Exception as e:
    print(f"  FAILED: {e}")
    timings['chess'] = -1

# ── 9/14. ChessQA (5-category quick eval, 10 samples each) ──
print("\n" + "=" * 60)
print("9/14  bench_chessqa (5 categories, 10 samples each)")
print("=" * 60)
t0 = time.time()
try:
    from bench_chessqa import has_chessqa_data, run_three_column_suite as run_chessqa
    if has_chessqa_data():
        run_chessqa(max_per_category=10)
        timings['chessqa'] = time.time() - t0
        print(f"  Done in {timings['chessqa']:.1f}s")
    else:
        print("  SKIPPED: ChessQA data not available")
        timings['chessqa'] = 0
except Exception as e:
    print(f"  FAILED: {e}")
    timings['chessqa'] = -1

# ── 10/14. Searchless Chess (position eval, small set) ──
print("\n" + "=" * 60)
print("10/14  bench_searchless_chess (position eval)")
print("=" * 60)
t0 = time.time()
try:
    from bench_searchless_chess import run_three_column_suite as run_searchless
    run_searchless()
    timings['searchless_chess'] = time.time() - t0
    print(f"  Done in {timings['searchless_chess']:.1f}s")
except Exception as e:
    print(f"  FAILED: {e}")
    timings['searchless_chess'] = -1

# ── 11/14. Cycle (CognitiveCycle throughput) ──
print("\n" + "=" * 60)
print("11/14  bench_cycle (CognitiveCycle throughput)")
print("=" * 60)
t0 = time.time()
try:
    from bench_cycle import run_full_suite as run_cycle
    run_cycle()
    timings['cycle'] = time.time() - t0
    print(f"  Done in {timings['cycle']:.1f}s")
except Exception as e:
    print(f"  FAILED: {e}")
    timings['cycle'] = -1

# ── 12/14. LLM A/B (dummy model only, no real LLM) ──
print("\n" + "=" * 60)
print("12/14  bench_llm_ab (dummy model, no real LLM)")
print("=" * 60)
t0 = time.time()
try:
    from bench_llm_ab import run_full_suite as run_llm_ab
    run_llm_ab()
    timings['llm_ab'] = time.time() - t0
    print(f"  Done in {timings['llm_ab']:.1f}s")
except Exception as e:
    print(f"  FAILED: {e}")
    timings['llm_ab'] = -1

# ── 13/14. Predictive Coding (if Torch2PC available, else skip) ──
print("\n" + "=" * 60)
print("13/14  bench_predictive_coding (Torch2PC comparison)")
print("=" * 60)
t0 = time.time()
try:
    from bench_predictive_coding import run_full_suite as run_predcoding
    run_predcoding()
    timings['predictive_coding'] = time.time() - t0
    print(f"  Done in {timings['predictive_coding']:.1f}s")
except Exception as e:
    print(f"  FAILED: {e}")
    timings['predictive_coding'] = -1

# ── 14/14. Penrose Chess (fortress detection) ──
print("\n" + "=" * 60)
print("14/14  bench_penrose_chess (Penrose fortress)")
print("=" * 60)
t0 = time.time()
try:
    from bench_penrose_chess import run_three_column_suite as run_penrose
    run_penrose()
    timings['penrose_chess'] = time.time() - t0
    print(f"  Done in {timings['penrose_chess']:.1f}s")
except Exception as e:
    print(f"  FAILED: {e}")
    timings['penrose_chess'] = -1

# ── Summary ──
print("\n" + "=" * 60)
print("TIMING SUMMARY")
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
print(f"  Passed: {passed}/{total_sections}  Failed: {failed}  Skipped: {skipped}")

with open(os.path.join(results_dir, 'quick_timings.json'), 'w') as f:
    json.dump(timings, f, indent=2)

print("\nResults saved to tests/benchmarks/results/")
print("Done!")
