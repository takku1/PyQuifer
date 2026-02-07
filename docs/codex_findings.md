# Codex Findings (Read-Only Audit)

This document captures a **read-only** static audit of `src/pyquifer/`.
It focuses on correctness traps (runtime errors), device/dtype safety (CPU vs CUDA),
gradient semantics, and low-effort optimization wins.

Audit date: 2026-02-06

## Quick Summary (Highest Impact)

1. **Device mismatch / CUDA failures**: several modules create CPU tensors inside `forward()` and/or build tensors from Python lists of tensors; this can error on CUDA or silently degrade performance.
2. **Gradient semantics mismatch**: some code/comments imply gradients for oscillator-derived metrics, but oscillator state is explicitly detached.
3. **Dependency compatibility**: preprocessing uses `OneHotEncoder(..., sparse_output=False)` without a scikit-learn minimum version guarantee.
4. **Performance traps**: frequent `.item()` in potentially-hot paths (bridge) can cause GPU synchronization; many `print()` calls across library.

## Project Understanding (What PyQuifer Is Doing)

PyQuifer is primarily a **research-style PyTorch library** organized as a set of cognitive/dynamical modules (oscillators, criticality, predictive coding, global workspace, etc.) plus an integration layer:

- `integration.py` (`CognitiveCycle`) is the main “wiring harness” that composes many modules into a tick loop.
- `bridge.py` provides a clean interface to feed *signals* from an LLM into PyQuifer and read out modulation knobs (temperature/top_p, attention bias, hidden-state perturbation).
- Many modules intentionally implement “physics-first” / non-backprop dynamics (e.g., buffers updated with `no_grad` or `.detach()`), which is totally valid, but it means **gradient semantics must be explicit** in comments and APIs.

Optimization stance for this repo (based on the code + docs):

- Prefer **vectorized torch ops** over Python loops in any per-tick/per-step code.
- Avoid GPU sync points (`.item()`, accidental CPU tensor creation) in hot paths.
- Keep “demo” code in `__main__` blocks safe for both CPU and CUDA.

## Findings (Detailed)

### 1) Device / dtype safety (CPU vs CUDA)

Several functions allocate tensors on CPU by default inside `forward()` / computation paths.
When the rest of the pipeline is on GPU, this can raise errors such as:
`RuntimeError: Expected all tensors to be on the same device`.

#### 1.1 `criticality.AvalancheDetector` returns CPU tensors

- `src/pyquifer/criticality.py:73` creates `last_size = torch.tensor(0)` (CPU).
- `src/pyquifer/criticality.py:102` returns `torch.tensor(avalanche_ended)` (CPU).
- `src/pyquifer/criticality.py:116` returns `torch.tensor([]), torch.tensor([])` (CPU) when empty.

Impact:
- If `activity` is on CUDA, these CPU tensors can poison downstream computations.

Recommendation:
- Construct scalars/empties on `activity.device` (or `self.avalanche_sizes.device`) and a stable dtype.

#### 1.2 Building tensors from Python lists of tensors (breaks on CUDA)

These patterns are both slow and frequently **invalid on CUDA** because Python tries to coerce CUDA tensors to CPU scalars.

- `src/pyquifer/criticality.py:119`:
  - `counts = torch.tensor([(valid == s).sum() for s in sizes])`
- `src/pyquifer/metastability.py:304`:
  - `counts = torch.tensor([(patterns == p).sum().float() for p in unique_patterns], device=history.device)`

Impact:
- On CPU: performance hit (Python loop + tensor construction).
- On CUDA: likely runtime error (`TypeError` converting CUDA tensor to Python number), or sync + CPU fallback.

Recommendation:
- Replace list-based tensor builds with vectorized operations (e.g., `torch.bincount` / histogramming patterns).

#### 1.4 Buffer re-assignment hazards (breaks `.to(device)` expectations)

Multiple modules `register_buffer(...)` a tensor and later do `self.some_buffer = torch.tensor(...)`.
This **replaces** the registered buffer with a brand new tensor (often CPU), which can:

- break device consistency after `.to('cuda')`
- break expectations for `state_dict()` / checkpointing (the original buffer name still exists, but now points to a new tensor)

Examples:

- `src/pyquifer/ecology.py:264` sets `self.memory_filled = torch.tensor(True)`
- `src/pyquifer/motivation.py:169` sets `self.memory_filled = torch.tensor(True, device=x.device)` (still replaces the buffer)
- `src/pyquifer/developmental.py:101` sets `self.history_filled = torch.tensor(True)`

Recommendation:
- Use `self.memory_filled.fill_(True)` / `self.history_filled.fill_(True)` (or `copy_`) instead of reassigning.

#### 1.3 More CPU scalar initializations in compute paths

- `src/pyquifer/causal_flow.py:370` initializes `bottom_up_te = torch.tensor(0.0)` (CPU).
- `src/pyquifer/causal_flow.py:371` initializes `top_down_te = torch.tensor(0.0)` (CPU).

Recommendation:
- Initialize these on an existing device (e.g., `device=flow_matrix.device` or the input device).

### 2) Gradient-flow semantics (oscillators / order parameter)

There’s a conceptual mismatch between:

- `LearnableKuramotoBank.forward()` explicitly detaching internal state, and
- comments suggesting some metrics preserve gradients.

#### 2.1 Oscillator state buffer is detached

- `src/pyquifer/oscillators.py:362`:
  - `self.phases.data = phases.detach()`

This makes `self.phases` a **read-only diagnostic buffer** from an autograd perspective.

#### 2.2 `FrequencyBank.get_aggregated_order_parameter()` comment suggests gradient flow

- `src/pyquifer/frequency_bank.py:94` says aggregated order parameter returns a tensor “to preserve gradient flow”.
- `src/pyquifer/frequency_bank.py:99` aggregates `torch.stack(all_rs).mean()`.

But:
- `all_rs` comes from `bank.get_order_parameter()` which (by default) uses `self.phases` (detached buffer), so gradients are typically not meaningful for training oscillators via `R`.

Recommendation:
- Decide which design you want:
  - If oscillators are intentionally non-differentiable relative to LLM/training, update the comment/docs to reflect that `R` is diagnostic.
  - If you want differentiable `R` for some training scenario, compute it from the returned `phases` tensor (not the detached buffer), and avoid `.detach()` or provide a differentiable path.

### 3) Robustness / correctness traps

#### 3.1 Potential divide-by-zero for 1-oscillator case

- `src/pyquifer/oscillators.py:277`:
  - `normalized_interaction = interaction_sum / (self.num_oscillators - 1)`

If `num_oscillators == 1`, this divides by zero.

Recommendation:
- Assert `num_oscillators >= 2` for Kuramoto coupling, or handle the degenerate case explicitly.

#### 3.2 scikit-learn version compatibility: `sparse_output`

- `src/pyquifer/core.py:129`:
  - `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`

Risk:
- Older scikit-learn versions don’t support `sparse_output=` (they use `sparse=`), causing a runtime `TypeError`.

Recommendation:
- Either pin `scikit-learn` minimum version in `pyproject.toml`, or implement a small compatibility shim.

### 4) Performance / maintainability

#### 4.1 `.item()` in hot paths can synchronize GPU

If `PyQuiferBridge.step()` / modulation are called per token, avoid `.item()` where possible.

- `src/pyquifer/bridge.py:280`: `state.neuromodulator_levels.mean().item()`
- `src/pyquifer/bridge.py:305`: `phase_signal.mean().item()`

Impact:
- On CUDA: forces device synchronization (hard performance cliff).

Recommendation:
- Keep computations on-tensor where feasible; if a Python float is needed, consider doing it less frequently or only on CPU.

#### 4.2 Heavy `print()` usage across library

`src/pyquifer/*` contains a large number of `print()` calls (currently ~683 matches).

Impact:
- Noisy stdout for library consumers.
- Can significantly slow training/inference loops.

Recommendation:
- Replace with `logging` (or gate behind a verbosity flag).

#### 4.3 `.numpy()` calls without `.cpu()` in examples

Some `__main__` example blocks call `.numpy()` on tensors without ensuring they are on CPU.
If a user runs those examples on CUDA, this will raise:
`TypeError: can't convert CUDA tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.`

- `src/pyquifer/diffusion.py:221` uses `initial_specimen_states[0].numpy()`.
- `src/pyquifer/diffusion.py:222` uses `archetype_vector.numpy()`.

Note:
- Several other modules correctly use `.detach().cpu().numpy()` in their examples (good pattern).

### 5) Integration-cycle specific device/performance issues

`CognitiveCycle.tick()` (the main “wiring” loop) contains a few patterns worth hardening.

#### 5.1 `.item()` in the tick loop

`tick()` frequently converts tensors to Python floats via `.item()`. If any of those tensors
live on CUDA, this can cause GPU synchronization.

- `src/pyquifer/integration.py:339` computes `coherence = order_param.item()`.

#### 5.2 Python-list scalarization for level activations

`tick()` builds a tensor using a Python list comprehension of `.item()` values:

- `src/pyquifer/integration.py:450` / `src/pyquifer/integration.py:451`:
  - `level_activations = torch.tensor([err.norm().item() for err in hpc_result['errors']], device=device)`

Impact:
- Forces scalarization of each level’s error norm.
- Potential GPU sync if errors are CUDA tensors.

Recommendation:
- Prefer a fully-tensor path (e.g., stack norms) to avoid `.item()` and Python loops.

#### 5.3 Additional Python-list tensor construction patterns

Similar patterns appear elsewhere and will have the same CUDA/performance implications:

- `src/pyquifer/neural_mass.py:278`, `src/pyquifer/neural_mass.py:291`, `src/pyquifer/neural_mass.py:292`:
  - `torch.tensor([p.E.item() for p in self.populations])`
- `src/pyquifer/neural_darwinism.py:240`, `src/pyquifer/neural_darwinism.py:263`, `src/pyquifer/neural_darwinism.py:606`, `src/pyquifer/neural_darwinism.py:626`:
  - `torch.tensor([g.activation_level.item() for g in self.groups], ...)`
  - `torch.tensor([g.resources.item() for g in self.groups])`

Recommendation:
- Prefer collecting values into an existing tensor buffer, or storing state in tensors from the start.

### 6) Things that look good / validated by inspection

#### 5.1 Lazy import map consistency

`src/pyquifer/__init__.py` maintains a large `__all__` and `_LAZY_IMPORTS` mapping.
A static AST check confirmed they match 1:1 (no missing and no extras).

### 7) Additional robustness note: deprecated pandas dtype checks

- `src/pyquifer/core.py:125` uses `pd.api.types.is_categorical_dtype(df[col])`.

This pandas API has been deprecated in some versions in favor of `isinstance(dtype, pd.CategoricalDtype)`
or checking `pd.api.types.is_categorical_dtype(df[col].dtype)`.
It’s not necessarily broken today, but it’s a forward-compat warning source.

## Module-by-Module Checklist (Selected Hotspots)

This is not an exhaustive review of every algorithm, but a practical checklist of the most likely
runtime/performance footguns per module.

### `criticality.py`

- CPU tensors created in `forward()` outputs (`src/pyquifer/criticality.py:73`, `src/pyquifer/criticality.py:102`, `src/pyquifer/criticality.py:116`).
- List-of-tensors construction (`src/pyquifer/criticality.py:119`) likely breaks on CUDA.

### `metastability.py`

- List-of-tensors construction for entropy (`src/pyquifer/metastability.py:304`) likely breaks on CUDA.
- Several branches return `torch.tensor(0.0)` without device; harmless on CPU but inconsistent for GPU pipelines (`src/pyquifer/metastability.py:311`).

### `causal_flow.py`

- Initializes CPU tensors in compute path (`src/pyquifer/causal_flow.py:370`, `src/pyquifer/causal_flow.py:371`).

### `oscillators.py` / `frequency_bank.py`

- Phase buffer is detached (`src/pyquifer/oscillators.py:362`), so “order parameter” metrics are typically diagnostic, not differentiable.
- Potential divide-by-zero for `num_oscillators == 1` (`src/pyquifer/oscillators.py:277`).
- `FrequencyBank.get_aggregated_order_parameter()` comment about preserving gradients conflicts with detached state (`src/pyquifer/frequency_bank.py:94`).

### `integration.py`

- `.item()` in tick loop and summary methods (`src/pyquifer/integration.py:339`).
- Tensor built from Python list of `.item()` values (`src/pyquifer/integration.py:450`, `src/pyquifer/integration.py:451`).

### `bridge.py`

- `.item()` used to derive modulation scalars (`src/pyquifer/bridge.py:280`, `src/pyquifer/bridge.py:305`); avoid in hot loops on CUDA.

### `core.py`

- `OneHotEncoder(..., sparse_output=False)` requires a scikit-learn version that supports `sparse_output` (`src/pyquifer/core.py:129`).
- Pandas dtype check forward-compat risk (`src/pyquifer/core.py:125`).

### `diffusion.py`

- Example `.numpy()` calls without `.cpu()` will break on CUDA (`src/pyquifer/diffusion.py:221`, `src/pyquifer/diffusion.py:222`).

### `neural_mass.py`

- Python list `.item()` collection into `torch.tensor(...)` (`src/pyquifer/neural_mass.py:278`) can sync GPU and loses device/dtype intent.

### `neural_darwinism.py`

- Python list `.item()` collection into `torch.tensor(...)` (multiple places: `src/pyquifer/neural_darwinism.py:240`, `src/pyquifer/neural_darwinism.py:263`).

### `ecology.py`

- Buffer device hazard: reassigning a registered buffer with a fresh CPU tensor:
  - `src/pyquifer/ecology.py:264` sets `self.memory_filled = torch.tensor(True)`.
- CPU scalar allocation in runtime path:
  - `src/pyquifer/ecology.py:296` sets `nn_similarity = torch.tensor(0.5)` (CPU) when no memory exists.
- Potential device mismatch in tensor creation:
  - `src/pyquifer/ecology.py:576` uses `torch.tensor([frequency])` (CPU unless device specified).
- Many `.item()` conversions in `recognize()` for return dict fields (fine for logging, but will sync GPU if tensors are CUDA).

Recommendation:
- Prefer `self.memory_filled.fill_(True)` (or `copy_`) rather than reassigning the buffer.
- Create fallback scalars on `pattern.device` / buffer device.

### `developmental.py`

- Buffer re-assignment hazard:
  - `src/pyquifer/developmental.py:101` sets `self.history_filled = torch.tensor(True)`.
- CPU fallback tensors inside compute methods:
  - `src/pyquifer/developmental.py:114` sets `historical_variance = torch.tensor(0.5)`.
  - `src/pyquifer/developmental.py:136` sets `velocity = torch.tensor(0.5)`.

Impact:
- If the module is moved to CUDA, these fallbacks can device-mismatch.

### `memory_consolidation.py`

- Several `torch.zeros(...)` / `torch.tensor(...)` fallbacks omit device, so “empty buffer” paths can return CPU tensors:
  - `src/pyquifer/memory_consolidation.py:128` (in the `valid == 0` branch) returns `torch.zeros(...)` without device.
  - `src/pyquifer/memory_consolidation.py:154` / `src/pyquifer/memory_consolidation.py:155` use `else torch.tensor(0.0)` (CPU).
- `SharpWaveRipple.forward()` returns `replay_active` as a CPU tensor:
  - `src/pyquifer/memory_consolidation.py:222` returns `'replay_active': torch.tensor(False)`.
  - `src/pyquifer/memory_consolidation.py:242` returns `'replay_active': torch.tensor(True)`.
- Consolidation return dict uses CPU tensors:
  - `src/pyquifer/memory_consolidation.py:345` returns `'consolidated': torch.tensor(True)`.
  - `src/pyquifer/memory_consolidation.py:346` returns `'num_consolidated': torch.tensor(consolidated_count)`.
- Reconsolidation returns `'lability': torch.tensor(lability)` (CPU):
  - `src/pyquifer/memory_consolidation.py:453`.

Impact:
- If the buffer/engine is moved to CUDA, these paths can throw device mismatch errors.

### `motivation.py`

- Float input path creates a CPU tensor:
  - `src/pyquifer/motivation.py:354` uses `order_parameter = torch.tensor(order_parameter)`.

Impact:
- If the module buffers are on CUDA, the next line that mixes `order_parameter` with them can device-error.

Recommendation:
- Use `torch.as_tensor(order_parameter, device=self.coherence_history.device)` or similar.

### `neuromodulation.py`

- `compute_snr()` uses `.item()` inside loops (`nl.item()`, and `.item()` when appending SNRs) and returns `torch.tensor(snrs)` (CPU):
  - `src/pyquifer/neuromodulation.py:327`.

Impact:
- Probably intended as an offline diagnostic; in a hot loop it will be very slow and will sync GPU.

Recommendation:
- Treat this as analysis-only or rewrite it to stay fully on tensors and return on `signal.device`.

### `voice_dynamics.py`

This module is particularly likely to device-error if used on CUDA because `step()` constructs many CPU tensors.

- `src/pyquifer/voice_dynamics.py:105` creates `d_phase = torch.tensor(...)` (CPU) during stepping.
- `src/pyquifer/voice_dynamics.py:366` creates `neuro_tensor = torch.tensor([...])` without device.
- `src/pyquifer/voice_dynamics.py:385` / `src/pyquifer/voice_dynamics.py:386` / `src/pyquifer/voice_dynamics.py:388` use `torch.tensor(0.0)` as fallbacks (CPU).
- `src/pyquifer/voice_dynamics.py:395` creates `band_frequencies=torch.tensor([6.0, 10.0, 18.0, 40.0])` (CPU).
- `src/pyquifer/voice_dynamics.py:397` creates `global_coherence=torch.tensor(global_coherence)` (CPU).

Recommendation:
- Ensure tensors are created on a known device (e.g., `device=neuro_tensor.device` or the phase tensor device).
- Avoid mixing Python floats and tensors in per-step logic; keep it tensor-native.

### `global_workspace.py`

- The implementation is mostly device-safe (buffers and parameters), but uses `.item()` for control flow:
  - `src/pyquifer/global_workspace.py:500` passes `self.current_time.item()`.
  - `src/pyquifer/global_workspace.py:518` uses `did_ignite.any().item()`.

Impact:
- On CUDA this can introduce synchronization, though global-workspace logic is inherently “bottleneck-y”, so this may be acceptable.

### `iit_metrics.py`

The main risk here is not “wrong results” as much as **computational cost** and hot-loop suitability.

- `_to_distribution()` bins values using a Python `for` loop over bins:
  - `src/pyquifer/iit_metrics.py:343` / `src/pyquifer/iit_metrics.py:358`.
- `PartitionedInformation.forward()` loops over every stored partition:
  - `src/pyquifer/iit_metrics.py:410`.

Cost profile:
- For `state_dim <= 12` it enumerates all bipartitions (up to 2047 partitions) plus per-partition distribution work.
- For `state_dim > 12` it samples at least 64 partitions, still using a Python loop.

Recommendation:
- Use Phi/IIT metrics as **diagnostics** (e.g., compute every N steps), not per-token.
- Consider vectorizing partitions/bins if performance becomes important.

### 8) Suggested next actions (if you want to fix)

1. **CUDA-hardening pass**: make tensor creation device-aware and remove list-of-tensors constructions (`criticality.py`, `metastability.py`, `causal_flow.py`).
2. **Clarify oscillator gradient design**: align comments/docs with `.detach()` behavior, or provide an explicit differentiable API.
3. **Pin or shim scikit-learn** for `OneHotEncoder` arguments.
4. **Replace `print()` with `logging`** (at least in non-`__main__` code paths).
5. **Example safety**: ensure all `.numpy()` conversions in examples go through `.detach().cpu().numpy()`.
