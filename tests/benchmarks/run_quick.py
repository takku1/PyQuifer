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

# ── 1. Module throughput (no training, fast) ──
print("=" * 60)
print("1/6  bench_modules (throughput)")
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

# ── 2. Consciousness metrics (no training, fast) ──
print("\n" + "=" * 60)
print("2/6  bench_consciousness (PCI, metastability)")
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

# ── 3. Efficiency sweeps (no training, fast) ──
print("\n" + "=" * 60)
print("3/6  bench_efficiency (dtype, compile, scaling)")
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

# ── 4. EP training (tiny: 500 samples, 3 epochs) ──
print("\n" + "=" * 60)
print("4/6  bench_ep_training (MNIST 500 samples, 3 epochs)")
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

# ── 5. Local rules (tiny: 500 samples, 3 epochs) ──
print("\n" + "=" * 60)
print("5/6  bench_local_rules (500 samples, 3 epochs)")
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

# ── 6. Continual learning (tiny: 300 samples, 2 epochs/task) ──
print("\n" + "=" * 60)
print("6/6  bench_continual (300 samples, 2 epochs/task)")
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

# ── Summary ──
print("\n" + "=" * 60)
print("TIMING SUMMARY")
print("=" * 60)
total = 0
for name, t in timings.items():
    status = f"{t:.1f}s" if t >= 0 else "FAILED"
    print(f"  {name:20s}: {status}")
    if t > 0:
        total += t
print(f"  {'TOTAL':20s}: {total:.1f}s")

with open(os.path.join(results_dir, 'quick_timings.json'), 'w') as f:
    json.dump(timings, f, indent=2)

print("\nResults saved to tests/benchmarks/results/")
print("Done!")
