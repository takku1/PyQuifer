# Codex Findings 5: Tests vs Src Alignment (Quick Audit)

Date: 2026-02-10
Scope: `PyQuifer/tests` and key runtime contracts in `PyQuifer/src/pyquifer`

Goal: sanity-check whether recent test changes are merely "making things pass" (masking src problems), or whether src and tests are genuinely aligned.

## Executive Summary

Overall, the tests look like they validate real behavior (no heavy mocking, minimal skipping). The two remaining areas where tests currently *fail to protect you* are:

1. **Contract enforcement not covered:** `CognitiveCycle.tick()` documents single-sample only, but the implementation still does not actually reject 2D inputs. Tests only pass 1D inputs, so they do not catch the mismatch.
2. **Autograd footgun not covered:** `integration.py` still uses `coupling_strength.data` in the criticality feedback call path; tests do not detect this class of issue.

If you fix those two, the suite will better reflect reality and reduce the risk that tests are "propping up" incorrect semantics.

## What Tests Are Doing (And Not Doing)

### 1) Import / API Surface

- `PyQuifer/tests/test_imports.py`:
  - enumerates modules by listing files under `src/pyquifer` and imports each `pyquifer.<module>`.
  - checks lazy import surface (`getattr(pyquifer, class_name)` for a set of classes).

Assessment:
- This is a strong guard against "tests pass but package is broken".
- One caveat: it relies on repo layout via file listing rather than installed-package metadata.

### 2) Path Handling

- `PyQuifer/tests/conftest.py` adds `../src` to `sys.path`.

Assessment:
- Normal for in-repo testing.
- Not a substitute for an install test (`pip install -e .` style). If you want industry-standard confidence, add a CI job that runs tests against an installed wheel or editable install.

### 3) Skips / Xfails / Mocking

- Skip/xfail usage is minimal.
  - Only seen: `test_bridge.py` contains a `skipif` for CUDA availability.
- Mocking/monkeypatching appears essentially absent in unit tests (good signal).

Assessment:
- Tests are not "faking" outcomes via mocks.

### 4) Integration Contract Coverage

From `PyQuifer/tests/test_integration_cycle.py` and related integration tests:
- The tests assert that `tick()` returns full dict outputs (modulation/consciousness/self_state/learning/diagnostics).

This interacts with recent src changes:
- `integration.py` introduced `return_diagnostics` fast path that returns only `{'modulation': ...}`.

Gaps:
- There is no unit test that exercises `cycle.tick(..., return_diagnostics=False)`.
  - Without that, regressions in the fast path could slip in undetected.

Recommendation:
- Add a test for `return_diagnostics=False` that asserts:
  - it returns only the modulation dict
  - its fields are present and within expected ranges

### 5) Batch Semantics

- Tests currently only feed 1D tensors to `CognitiveCycle.tick()`.
- In src, tick docstring says: "Single-sample only; batched inputs are not supported."
- But the current enforcement block does not reject 2D inputs.

Assessment:
- This is exactly the kind of thing where tests can pass while an API contract is false.

Recommendation:
- Add a test that passing a 2D tensor raises.
- Alternatively, if you want to allow `(1, state_dim)`, add test(s) for that specific accepted shape.

### 6) Duplicate / Overlapping Coverage

- `PyQuifer/tests/test_integration_cycle.py` content is extremely similar to other integration tests (appears duplicated).

Assessment:
- Duplication is not harmful for correctness, but it increases maintenance cost.

Recommendation:
- Consolidate overlapping tests to reduce churn when API changes.

## Benchmarks vs Tests

- Benchmarks are large and vendor-heavy; `PyQuifer/tests/benchmarks/conftest.py` correctly prevents pytest from crawling vendored repos.
- Benchmark report generation (`tests/benchmarks/generate_report.py`) has explicit logic to exclude stub/dummy/proxy suites from scoring.

Assessment:
- The harness design is reasonable.
- For "industry standard" credibility, ensure that scored scenarios are reproducible from a clean checkout without manual vendor setup, or clearly separate "core" vs "vendor" benchmark tiers.

## Concrete Src Issues That Tests Won't Catch Yet

1. `PyQuifer/src/pyquifer/integration.py`:
   - `tick()` claims single-sample but does not actually reject 2D inputs.
2. `PyQuifer/src/pyquifer/integration.py`:
   - remaining `.data` usage: `coupling_strength.data` is still passed into criticality feedback.
3. `PyQuifer/src/pyquifer/__pycache__`:
   - still present in-tree; tests won’t fail because of it, but it’s a packaging/repo hygiene hazard.

## "Are We Ready To Move On?" Checklist

If your intent is: "PyQuifer is stable enough to be a dependency of Mizuki":

Minimum bar:
- fix single-sample enforcement in `tick()` (and add a unit test for it)
- remove remaining `.data` usage in the tick hot path
- add unit coverage for `return_diagnostics=False`
- remove `src/pyquifer/__pycache__` from the repo and ignore it

After that, it’s reasonable to shift main attention to Mizuki and treat further benchmark expansion (AKOrN/CLEVRTex, NeuroBench tasks, LLM eval wiring) as backlog.
