# Codex Findings 2 (Read-Only Audit)

This document captures a read-only static audit of `PyQuifer/src/pyquifer`.
No code was executed; files were only listed/searched/read.

Audit date: 2026-02-08

## Context: How This Repo Tests/Benchmarks

Before interpreting the findings below, it helps to know how you are currently validating PyQuifer:

- `PyQuifer/tests/validation/validate_*.py` are deterministic, assert-driven validation scripts (not pytest-style unit tests).
- `PyQuifer/tests/benchmarks/bench_*.py` are PyQuifer-specific benchmark scripts.
  - They are dual-mode: runnable as CLI and also contain pytest-compatible `Test*` classes in some files (example: `PyQuifer/tests/benchmarks/bench_cycle.py`).
  - They write JSON results to `PyQuifer/tests/benchmarks/results/*.json`.
  - `PyQuifer/tests/benchmarks/generate_report.py` aggregates JSON into `PyQuifer/tests/benchmarks/results/BENCHMARK_REPORT.md`.
- `PyQuifer/tests/benchmarks/` also contains many vendored external projects (Gymnasium, OpenSpiel, etc.). Grepping for `pytest` in that tree will hit a lot of third-party code; the PyQuifer-owned benchmark surface is primarily the `bench_*.py` scripts plus `harness.py` and `generate_report.py`.

This matters because several “risky” patterns in `src/pyquifer` are likely intentional for your design goal (“local/no-backprop dynamics”), but still worth tightening to prevent aliasing and optimizer/autograd footguns.

## Delta From `codex_findings.md`

`PyQuifer/docs/codex_findings.md` already covers several important issues (device placement, buffer reassignment, `.item()` GPU sync risks, etc.).
This follow-up focuses on a narrower set of higher-risk patterns found during a fresh scan:

- `.data` / `.grad.data` usage in non-demo code (autograd/optimizer footguns)
- one concrete aliasing bug (`target_vector.data = archetype_vector.data`)
- training/inference semantics mismatch in EP module
- minor robustness issues (`except Exception` masking)
- text encoding corruption in comments/docstrings and benchmark output

## High Impact Findings

### 1) Tensor aliasing bug in `models.py`

File: `PyQuifer/src/pyquifer/models.py:109`

```python
self.mind_eye_actualization.target_vector.data = self.archetype_vector.data
```

Why this is risky:
- This can make `target_vector` share storage with `archetype_vector` (aliasing), so later updates to one silently change the other.
- It also bypasses autograd bookkeeping and can confuse optimizers/state.

Safer pattern:
- `with torch.no_grad(): self.mind_eye_actualization.target_vector.copy_(self.archetype_vector)`
- If shapes differ, explicitly validate and slice/pad.

### 2) `.data` usage hotspots in core code paths

Counts of `.data` occurrences per file (non-`__pycache__`):
- `PyQuifer/src/pyquifer/continual_learning.py`: 10
- `PyQuifer/src/pyquifer/equilibrium_propagation.py`: 9
- `PyQuifer/src/pyquifer/criticality.py`: 4
- `PyQuifer/src/pyquifer/ecology.py`: 3
- `PyQuifer/src/pyquifer/potentials.py`: 3 (demo prints in `__main__`)
- `PyQuifer/src/pyquifer/oscillators.py`: 2
- `PyQuifer/src/pyquifer/models.py`: 2
- `PyQuifer/src/pyquifer/social.py`: 2
- `PyQuifer/src/pyquifer/frequency_bank.py`: 1 (demo in `__main__`)
- `PyQuifer/src/pyquifer/dendritic.py`: 1
- `PyQuifer/src/pyquifer/multiplexing.py`: 1

Most uses appear intended as “local/no-backprop” updates, but `.data` is still a sharp edge.
If the goal is “update state without building a graph”, prefer:
- `with torch.no_grad(): tensor.copy_(...) / add_(...) / fill_(...)`
- `tensor = tensor.detach()` only when you really want to sever gradients on a returned value

Notable sites:
- `PyQuifer/src/pyquifer/oscillators.py:386`: `self.phases.data = _store` (buffer update)
- `PyQuifer/src/pyquifer/equilibrium_propagation.py:143,153,160,274,291` and others: `self.bank.phases.data = ...`
- `PyQuifer/src/pyquifer/criticality.py:505-506,597-598`: `.data.fill_` in `reset()`
- `PyQuifer/src/pyquifer/dendritic.py:133`: `self.W_basal.data.add_(...)`
- `PyQuifer/src/pyquifer/multiplexing.py:251`: `gate.phase_preferences.data.fill_(...)`
- `PyQuifer/src/pyquifer/social.py:262,801`: `.data` writes to coupling/phase buffers
- `PyQuifer/src/pyquifer/ecology.py:426,461,477`: `.data` used for norms

### 3) Equilibrium propagation: `forward()` detaches encoder output

File: `PyQuifer/src/pyquifer/equilibrium_propagation.py:278`

```python
self.bank(external_input=ext_input.detach(), ...)
```

This is semantically OK if:
- `forward()` is inference-only, and
- training uses `ep_train_step()` (which passes `external_input` without detaching).

It becomes a trap if callers expect `EPKuramotoClassifier(x)` to be differentiable w.r.t. encoder in a standard training loop.

Recommendation:
- Clarify in docs/typing/comments whether `forward()` is inference-only.
- Optionally rename to make intent explicit (e.g., `forward_inference`) or add a `forward_train` wrapper.

### 4) Continual learning: `.grad.data` and `.data` in training-time logic

Files/lines:
- `PyQuifer/src/pyquifer/continual_learning.py:285` (SynapticIntelligence): `-param.grad.data * (param.data - self.W_old[name])`
- `PyQuifer/src/pyquifer/continual_learning.py:520` (MESU): `grad = param.grad.data`
- `PyQuifer/src/pyquifer/continual_learning.py:556`: `param.grad.data *= lr_mods[name]`

Risks:
- `.grad.data` bypasses safety checks and can lead to silent incorrectness.
- Modifying `param.grad` in-place is fine, but should be done as `param.grad.mul_(...)` under `torch.no_grad()`.

Safer patterns:
- Replace `param.grad.data` with `param.grad.detach()` (read-only usage).
- For scaling gradients: `with torch.no_grad(): param.grad.mul_(scale)`.

### 5) Broad exception masking in somatic sensor

File: `PyQuifer/src/pyquifer/somatic.py:86,99`

```python
except Exception:
    metrics["vram_stress"] = 0.0
```

This is probably intentional (“best effort” hardware metrics), but it also hides genuine bugs.
Recommendation:
- Narrow to expected exceptions, e.g. `ImportError`, `RuntimeError`, `AttributeError`.
- Optionally log at debug level once per session.

## Medium/Low Impact Notes

### 6) Text encoding corruption (source + generated benchmark output)

Multiple files contain mojibake sequences such as `"â†’"`, `"â€”"`, `"â”€"`, or `"UexkÃ¼ll"`.

Examples in source:
- `PyQuifer/src/pyquifer/integration.py:8` and many lines in `tick()` separators
- `PyQuifer/src/pyquifer/advanced_spiking.py:653` docstring/comment
- `PyQuifer/src/pyquifer/ecology.py:484` ("von UexkÃ¼ll")

Examples in generated output:
- `PyQuifer/tests/benchmarks/results/BENCHMARK_REPORT.md` contains mojibake placeholders like `â€”`.
  - `PyQuifer/tests/benchmarks/harness.py` currently emits a mojibake dash in markdown tables for missing values, which will propagate into reports.

Recommendation:
- Normalize `.py` sources to UTF-8 and re-save the affected strings.
- Consider switching missing-value placeholder to plain ASCII (e.g., `--`) to avoid encoding issues in generated reports.

### 7) Repo hygiene: build artifacts in `src/`

`PyQuifer/src` contains:
- `pyquifer.egg-info/`
- many `.pyc` files under `pyquifer/__pycache__/`

These are typically generated artifacts and are usually excluded from source control and distributions.
They can also confuse packaging/layout assumptions.

## Optimization Opportunities (Low Risk)

- `PyQuifer/src/pyquifer/linoss.py`: inner Python loop over oscillators (`for i, osc in enumerate(self.oscillators)`) is a clear vectorization candidate.
- `PyQuifer/src/pyquifer/social.py`: avoid cloning and re-zeroing diagonal each step; keep a precomputed diagonal mask buffer and apply it with multiplication.
- Across modules: prefer `logging` over `print()` outside `__main__`.

## Suggested Next Steps (If You Want Fixes)

1. Replace all non-demo `.data` and `.grad.data` usage with `torch.no_grad()` + in-place ops (`copy_`, `add_`, `mul_`, `fill_`).
2. Fix the `models.py` aliasing line by using `copy_` (and add a shape/device assertion).
3. Clarify EP module semantics: is `forward()` differentiable or explicitly inference-only.
4. Normalize encoding on the handful of corrupted comment/docstring strings, and fix the report placeholder in `harness.py`.
