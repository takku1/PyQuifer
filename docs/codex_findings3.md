# codex_findings3: nom_nom Research Notes (akorn, flashrnn, multilayer-cudamoto)

Date: 2026-02-09

Scope: read-only skim of:
- `Z:\mizukiai\nom_nom\akorn`
- `Z:\mizukiai\nom_nom\flashrnn`
- `Z:\mizukiai\nom_nom\multilayer-cudamoto`

Goal: identify "major" implementation ideas that could plausibly improve PyQuifer (especially latency and oscillator dynamics), without assuming we should vendor/copy these projects.

## Executive Summary (What Looks Worth Stealing)

1. **FlashRNN (NXAI)** demonstrates how to make *recurrent/stateful* computation fast on GPUs by:
- choosing a **multi-head/block-diagonal** formulation to increase parallelism,
- using **fused CUDA kernels** (matmul + pointwise + state update) with **WMMA** and custom **register/shared-memory caching**,
- adding a simple **autotuning constraint solver** (`autotune/constrint.py`) to pick tiling parameters that satisfy real register/shared-mem limits.

Applied to PyQuifer:
- If you ever need oscillator banks in the *thousands* (or run many cycles in parallel), a fused kernel + autotuned tiling could be the difference between "toy" and "production".
- For today's PyQuifer, the recent Cat 11 results show `CognitiveCycle.tick()` is generally a poor GPU workload (many small ops). But **oscillator-only kernels** can still be a good GPU target if they are large enough and fused.

2. **multilayer-cudamoto** provides a concrete CUDA implementation for Kuramoto-like dynamics on **multilayer networks**, including:
- **CSR/adjacency-list** based neighbor iteration (avoids dense O(N^2) phase diffs),
- local order parameter estimation and interaction types (interdependent/competitive),
- warp-level reductions + atomic accumulation for global stats.

Applied to PyQuifer:
- PyQuifer’s non-global topologies in `LearnableKuramotoBank` currently form dense `phase_diffs` (`O(N^2)` memory+compute). A CSR adjacency-list path is a major scalability win.
- Also provides a design template for "inter-layer coupling" that maps nicely to PyQuifer’s "populations/bands" concept.

3. **AKOrN (ICLR 2025 Oral)** shows a *learnable Kuramoto-layer* abstraction used inside deep nets:
- connectivity `J` can be **Conv2D** or **Attention** ("attn" backend),
- update projects onto the **tangent space** of the normalized state manifold,
- repeated update steps `T` with normalization every step.

Applied to PyQuifer:
- This is a usable pattern for “oscillators as a neural layer” beyond classic scalar-phase Kuramoto.
- A practical takeaway is the **tangent-space projection + renormalization** approach: it tends to keep iterative dynamics stable (no exploding norms) while still letting the update field be expressive (conv/attn).

## Repo 1: FlashRNN (`nom_nom/flashrnn`)

### What it is
FlashRNN implements traditional RNNs (LSTM/GRU/Elman) and sLSTM with CUDA and Triton backends.
Key pitch: **state tracking** on modern hardware with fused kernels and caching.

### Notable engineering details (relevant to PyQuifer)
- Backends are explicit (`vanilla`, `cuda`, `cuda_fused`, `triton_fused`) in `flashrnn/flashrnn/flashrnn/flashrnn.py`.
- The CUDA fused kernel uses WMMA fragments and dynamic shared memory:
  - recurrent matrices are cached into registers/shared memory (see `flashrnn_fused_forward.cu` patterns: `extern __shared__`, `wmma::load_matrix_sync`, register caching loops).
- There is an explicit *hardware resource model*:
  - `flashrnn.py` constructs constraints around `REGISTERS_PER_BLOCK_*`, `SHARED_MEMORY_PER_BLOCK`, `multiProcessorCount`, etc.
  - `autotune/constrint.py` is a lightweight integer CSP solver optimized for multiplicative/divisibility constraints (tiling sizes, padding, looping counts).

### What we could implement in PyQuifer
High-leverage ideas (not necessarily GPU-first):
- **Backend strategy** for oscillators:
  - Provide `LearnableKuramotoBank` backends similar to FlashRNN: `torch` (baseline), `torch.compile` (CPU), `triton` (GPU), `cuda_ext` (GPU).
  - Do not force GPU: use heuristics (N, steps, batch) to pick CPU vs GPU.
- **Autotuned fused kernels** if oscillators become large:
  - If PyQuifer grows to large oscillator counts or batched simulation, FlashRNN-style “compile+autotune” is the blueprint.
- **Multi-head decomposition**:
  - FlashRNN uses multi-head structure as block-diagonal recurrence (hidden split into heads).
  - PyQuifer analog: split oscillators into independent/weakly-coupled bands (“heads”) so each head can be simulated with higher occupancy and predictable memory footprints.

### Cautions
- FlashRNN fused kernels assume enough work per launch to amortize launch overhead; PyQuifer’s Cat 11 showed the opposite problem for `tick()` on CUDA.
- Stealing the whole “GPU everything” approach is likely wrong for the full `CognitiveCycle`. It may be right for specific hotspots (oscillators, coupling, reductions) if fused.

## Repo 2: multilayer-cudamoto (`nom_nom/multilayer-cudamoto`)

### What it is
GPU Kuramoto simulation for two-layer multilayer networks (+ Qt viewer).
CUDA kernels operate on adjacency lists (`adjList`, `offsetList`) and support interaction types described in Nature Physics 2019.

### Notable engineering details (relevant to PyQuifer)
From `cudamoto2/src/cudamoto2_kernels.cuh`:
- **Warp-level reductions** using `__shfl_down_sync` and atomic adds for global aggregates (sum cos/sin, L1 norms).
- **CSR neighbor iteration**:
  - For each node, neighbors are `adjList[offsetList[i]..offsetList[i+1])`, avoiding dense `N x N`.
- **Local order parameter / neighbor phase** computed via sums of cos/sin, then `r_local = sqrt(x*x + y*y)/k`.
- **Inter-layer coupling / interaction types**:
  - Modulate coupling term by `r_local` or `(1 - r_local)` depending on interdependent vs competitive coupling.

### What we could implement in PyQuifer
This is the most directly transferable to PyQuifer’s existing code:
- **Sparse topology path for `LearnableKuramotoBank`**:
  - Today, non-global topologies compute `phase_diffs = phases.unsqueeze(1) - phases.unsqueeze(0)` and then multiply by `adj` (dense).
  - A CSR/adj-list implementation makes sparse topologies scale far better (memory and compute).
  - Even on CPU, adjacency iteration can beat dense O(N^2) for sparse graphs (and it removes the `phase_diffs` tensor).
- **Multilayer oscillator networks**:
  - Add explicit layer-to-layer coupling to support “populations” or “frequency bands” with different coupling regimes.
  - This maps to PyQuifer conceptually (metastability populations, global workspace bands), even if you implement it first in pure PyTorch.
- **Fast global stat reductions**:
  - Cat 11 and the bridge use order parameters and coherence frequently. A fused reduction (sin/cos sum, plus weighted variants) is an easy win if it’s called often.

### Cautions
- Their kernels are written for a specific simulation setting (two-layer, specific interaction types). Use the patterns, not the exact physics.
- Adding a C++/CUDA extension increases build complexity; only worth it if you have a clear performance wall that PyTorch can’t cross.

## Repo 3: AKOrN (`nom_nom/akorn`)

### What it is
“Artificial Kuramoto Oscillatory Neurons (AKOrN)” (ICLR 2025 Oral). The repo is mostly an experiments codebase (CLEVR-Tex, Sudoku, etc.), but it contains a reusable Kuramoto-style layer.

### Notable engineering details (relevant to PyQuifer)
From `source/layers/klayer.py`:
- `KLayer` is a “Kuramoto layer” where connectivity `J` is either:
  - `conv`: `nn.Conv2d(ch, ch, ksize, padding=ksize//2)`
  - `attn`: an Attention module (multi-head) used as the coupling operator
- Update field:
  - compute `y = connectivity(x) + c`
  - reshape channels into `(groups, n)` where `n` is the Kuramoto dimension and enforce `ch % n == 0`
  - project update onto tangent space: `y - (sum(x*y)*x)`
  - normalize each `(…, n)` vector to unit norm after each update step
- Optional “omega” term (`OmegaLayer`) behaves like a learnable rotational field applied to the `(…, n)` representation.

### What we could implement in PyQuifer
AKOrN is a good design reference for stability and “oscillatory layer” semantics:
- **Tangent-space projection + renormalization** can stabilize iterative dynamics:
  - A PyQuifer analog could be used when you run iterative “inner loops” (multiple micro-steps) for attention biasing or state refinement.
- **Attention as coupling**:
  - For LLM integration, the closest match is not “Conv2D coupling” but “Attention coupling”:
    - Use a low-rank attention-derived coupling to generate an attention bias vector (or low-rank logit bias) from state.
  - This can be implemented without touching the full transformer internals (logits/attention bias interfaces).

### Cautions
- AKOrN uses “Kuramoto-like” dynamics on normalized vector groups, not classic scalar phase oscillators. It’s conceptually adjacent, not a drop-in replacement for `LearnableKuramotoBank`.

## Concrete “Major” Implementation Candidates for PyQuifer

If you want a shortlist of changes that are likely to matter:

1. **Sparse adjacency / CSR topologies in `LearnableKuramotoBank`**
- Replace dense `phase_diffs` path for sparse topologies with adjacency-list neighbor iteration.
- Optional step 2: provide both CPU and CUDA implementations once the algorithmic shape is fixed.

2. **A dedicated “oscillator fast path” (possibly fused)**
- A single function that computes:
  - phase update
  - order parameter (coherence)
  - optionally local stats needed by `CognitiveCycle`
in one pass to minimize intermediate tensors.

3. **Multi-band / multilayer oscillators**
- Model “bands” (theta/alpha/beta/gamma analogs) as layers with different coupling rules, plus a small inter-layer coupling matrix.
- The multilayer-cudamoto interaction-type pattern is a good starting point.

4. **Backend selection strategy**
- Similar to FlashRNN, make the “fast” backend opt-in and conditional:
  - if `num_oscillators` small, CPU wins (Cat 11 suggests this strongly)
  - if very large or highly batched, GPU fused kernels may win

## Practical Next Step (If You Want To Turn This Into Work Items)

- For performance:
  - start with **CSR/sparse topology path** (algorithmic win first).
  - then benchmark at different oscillator counts and choose CPU-vs-GPU thresholds.
- For modeling:
  - experiment with an AKOrN-style “attention as coupling” at the bridge layer (logits/attention-bias modulation), since it plugs into LLM integration more directly than changing the oscillator math.

