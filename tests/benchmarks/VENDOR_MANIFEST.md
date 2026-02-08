# Vendor Manifest

Third-party repositories cloned into this directory for comparison benchmarking.
These are NOT part of PyQuifer. They are gitignored and used only as reference implementations.

| Directory | Upstream | License | Used By | Purpose |
|-----------|----------|---------|---------|---------|
| `EGG/` | facebook/EGG | MIT | `bench_emergent_comm.py` | Emergent communication framework |
| `emergent_communication_at_scale/` | google-deepmind/emergent_communication_at_scale | Apache-2.0 | `bench_emergent_comm.py` | Scaled emergent communication |
| `Gymnasium/` | Farama-Foundation/Gymnasium | MIT | `bench_gymnasium.py` | RL environment framework |
| `hanabi-learning-environment/` | google-deepmind/hanabi-learning-environment | Apache-2.0 | `bench_hanabi.py` | Multi-agent card game |
| `KuraNet/` | KuraNet | MIT | `bench_oscillators.py` | Kuramoto oscillator networks |
| `lava/` | lava-nc/lava | BSD-3 | `bench_lava.py` | Intel neuromorphic framework |
| `MD-Bench/` | RRZE-HPC/MD-Bench | LGPL-3.0 | `bench_scientific_ml.py` | Molecular dynamics benchmark |
| `meltingpot/` | google-deepmind/meltingpot | Apache-2.0 | `bench_meltingpot.py` | Multi-agent social scenarios |
| `neurobench/` | NeuroBench | BSD-3 | `bench_spiking.py` | Neuromorphic benchmark suite |
| `neurodiscoverybench/` | neurodiscoverybench | MIT | `bench_neurodiscovery.py` | Neural dynamics discovery |
| `nlb_tools/` | neurallatents/nlb_tools | MIT | `bench_nlb.py` | Neural Latents Benchmark tools |
| `open_spiel/` | google-deepmind/open_spiel | Apache-2.0 | `bench_open_spiel.py` | Game theory framework |
| `PredBench/` | PredBench | MIT | `bench_predbench.py` | Temporal prediction benchmark |
| `SciMLBenchmarks.jl/` | SciML/SciMLBenchmarks.jl | MIT | `bench_scientific_ml.py` | Scientific ML ODE/SDE/PDE solvers |
| `Torch2PC/` | Torch2PC | MIT | `bench_predictive_coding.py` | Predictive coding gradients |
| `torchdiffeq/` | rtqichen/torchdiffeq | MIT | `bench_torchdiffeq.py` | Differentiable ODE solvers |
| `torchdyn/` | DiffEqML/torchdyn | Apache-2.0 | `bench_torchdyn.py` | Neural ODE framework |

## Reference-only directories (no bench script needed)

| Directory | Content | Why it's here |
|-----------|---------|---------------|
| `arXiv-2512.01992v1/` | LaTeX paper (ICLR 2026) | Academic reference for AKOrN architecture |
| `bifurcation/` | Julia continuation library | Reference for bifurcation detection (Julia, not Python) |
| `chessqa-benchmark/` | Chess QA dataset + eval | Potential future benchmark for reasoning tasks |
| `osbenchmarks/` | OS-level benchmark research notes | System-level reference docs |
| `Plaskett_puzzle/` | Astronomy problem setup | Domain puzzle reference |
| `searchless_chess/` | Chess engine + checkpoints | Potential future complement to penrose chess |

## Update Procedure

To update a vendored repo:
```bash
cd tests/benchmarks/<dir>
git pull origin main
```

To add a new vendor:
1. Clone into `tests/benchmarks/`
2. Add entry to this manifest
3. Add directory to `.gitignore` and `conftest.py`
4. Create corresponding `bench_*.py` if applicable
