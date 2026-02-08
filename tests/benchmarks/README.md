# PyQuifer Benchmark Suite

Benchmarks comparing PyQuifer modules against published baselines and third-party libraries.

## Structure

```
benchmarks/
  bench_*.py          PyQuifer-owned benchmark scripts
  harness.py          Shared timing, memory tracking, metric collection, reporting
  generate_report.py  Aggregates JSON results into unified markdown report
  conftest.py         Prevents pytest from crawling vendor directories
  results/            JSON output from benchmark runs (gitignored)
  data/               Cached datasets (gitignored)
  <vendor dirs>/      Third-party repos cloned for comparison (gitignored)
```

## Quick Start

```bash
# Run all library tests (does NOT run benchmarks)
python -m pytest tests/ -v

# Run benchmark smoke tests only (fast, pytest-compatible)
python -m pytest tests/benchmarks/bench_efficiency.py -v
python -m pytest tests/benchmarks/bench_robustness.py -v

# Run a full benchmark suite (writes JSON to results/)
python tests/benchmarks/bench_modules.py
python tests/benchmarks/bench_ep_training.py

# Generate unified report from all results
python tests/benchmarks/generate_report.py
```

## Benchmark Categories

### PyQuifer Internal (no vendor deps)
| Script | What it tests |
|--------|--------------|
| `bench_modules.py` | Throughput of core modules (Kuramoto, HPC, CriticalityController, CFC, CognitiveCycle) |
| `bench_consciousness.py` | PCI (Lempel-Ziv), metastability, coherence-complexity sweep |
| `bench_cycle.py` | CognitiveCycle tick timing |
| `bench_efficiency.py` | dtype sweep (fp32/bf16/fp16), torch.compile, oscillator scaling, batch scaling |
| `bench_robustness.py` | Perturbation stability, phase recovery, input consistency, noise resilience |
| `bench_ep_training.py` | EP-trained Kuramoto vs MLP backprop (MNIST/Fashion-MNIST) |
| `bench_local_rules.py` | ThreeFactorRule, DendriticStack, OscGatedPlasticity vs backprop |
| `bench_continual.py` | EP + SleepReplayConsolidation vs naive fine-tuning + EWC |
| `bench_llm_ab.py` | CognitiveCycle modulation pipeline A/B testing |
| `bench_penrose_chess.py` | Penrose gap-closure: fortress detection, position eval |
| `bench_chess.py` | Wrapper for chess benchmarks |
| `bench_predictive_coding.py` | OscillatoryPredictiveCoding vs Torch2PC |

### Third-Party Comparisons
| Script | Vendor | Comparison |
|--------|--------|-----------|
| `bench_oscillators.py` | KuraNet | Kuramoto/Daido/Stuart-Landau dynamics |
| `bench_torchdiffeq.py` | torchdiffeq | ODE integration (Kuramoto, Wilson-Cowan) |
| `bench_torchdyn.py` | torchdyn | Continuous dynamics / Neural ODEs |
| `bench_spiking.py` | neurobench | Spiking neuron operations |
| `bench_lava.py` | lava | Neuromorphic spiking framework |
| `bench_bifurcation.py` | bifurcation | Bifurcation detection |
| `bench_predbench.py` | PredBench | Temporal prediction (15 datasets) |
| `bench_gymnasium.py` | Gymnasium | RL motivation/learning |
| `bench_hanabi.py` | hanabi | Multi-agent world model |
| `bench_emergent_comm.py` | EGG | Emergent communication |
| `bench_meltingpot.py` | meltingpot | Multi-agent social mechanisms |
| `bench_open_spiel.py` | open_spiel | Game theory / neural darwinism |
| `bench_scientific_ml.py` | SciMLBenchmarks.jl, MD-Bench | Scientific ML / molecular dynamics |
| `bench_neurodiscovery.py` | neurodiscoverybench | Neural dynamics metadata |
| `bench_nlb.py` | nlb_tools | Neural Latents Benchmark |

## Configuration

Environment variables:
- `PYQUIFER_BENCH_DEVICE` — Force device (`cpu`, `cuda`, `cuda:0`). Default: auto-detect.
- `PYQUIFER_BENCH_SEEDS` — Comma-separated seeds for reproducibility. Default: `42`.

## Output

Each `bench_*.py` writes a JSON file to `results/` when run as `__main__`.
`generate_report.py` reads all JSON results and produces `BENCHMARK_REPORT.md`.

See `VENDOR_MANIFEST.md` for details on third-party repositories.
