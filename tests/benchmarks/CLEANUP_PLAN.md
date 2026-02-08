# PyQuifer Benchmarks Cleanup Plan

This is a plan to clean up and harden `PyQuifer/tests/benchmarks/` without changing benchmark semantics.

Important constraint:
- The large third-party directories under `PyQuifer/tests/benchmarks/` are *comparators only*.
  - They are not incorporated into PyQuifer.
  - They exist so PyQuifer benchmark code can compare outputs/behavior against reference ecosystems.

## Snapshot (Current State)

- Directory: `PyQuifer/tests/benchmarks/`
- Size: ~1.053 GB
- Files/dirs: ~5559 files / ~1223 directories
- Contains:
  - PyQuifer-owned benchmark code: `bench_*.py`, `harness.py`, `generate_report.py`, `mcp_stockfish_server.py`
  - Runtime artifacts: `results/`, `data/`, `__pycache__/`
  - Comparator source trees (vendored): `Gymnasium/`, `open_spiel/`, `meltingpot/`, etc.

## Goals

1. Make it obvious what is "PyQuifer benchmark code" vs "comparator source".
2. Ensure comparator trees never participate in test discovery unless explicitly requested.
3. Standardize how benchmarks are run, configured (seeds/device), and reported.
4. Make results reproducible and CI-friendly while keeping local runs fast.

## Non-Goals

- Deleting comparator repos.
- Making comparator repos pass their own upstream test suites inside this repo.

## Issues / Risks

1. Tree mixing makes review and search noisy
- Simple greps and IDE search hit thousands of comparator files.
- It is easy to mistake comparator tests/configs for PyQuifer-owned tests.

2. Test discovery hazards (real)
- Many PyQuifer `bench_*.py` files embed `class Test*` pytest tests.
- Comparator repos also contain pytest tests/configs/workflows.
- Without strict ignores, `pytest` can collect unintended tests and/or spend minutes scanning.

3. Benchmark tests can be heavy by default
- Some `bench_*.py` import large stacks, may download datasets (via torchvision), and may be slow on CPU.
- A clean separation between "smoke tests" and "full benchmark runs" helps a lot.

4. Inconsistent entrypoints / import strategy
- Many scripts do `sys.path.insert(...)` to import from `src/`.
- Some scripts are CLI-first, some are pytest-first.

5. Data + results lifecycle
- `data/` and `results/` are runtime artifacts; they should be cleanly managed and ignored in git.
- Result schema consistency matters because `generate_report.py` aggregates across suites.

6. Encoding / report hygiene
- Generated markdown currently contains mojibake placeholders (missing-value dash), which propagates into reports.

## Proposed Target Layout

Keep the PyQuifer-owned surface small and predictable:

- `PyQuifer/tests/benchmarks/README.md`
  - How to run benches, env vars, output locations, minimal deps.
- `PyQuifer/tests/benchmarks/core/`
  - `harness.py`, report tooling, shared utilities.
- `PyQuifer/tests/benchmarks/suites/`
  - PyQuifer-owned suites only: `bench_cycle.py`, `bench_modules.py`, etc.
- `PyQuifer/tests/benchmarks/comparators/` (or `vendor/`, `third_party/`)
  - Comparator repos with a manifest file and license notes.
- `PyQuifer/tests/benchmarks/data/` (gitignored)
- `PyQuifer/tests/benchmarks/results/` (gitignored)

If moving folders is too disruptive short-term, start with docs + pytest ignore rules first.

## Phase 0 (No Behavior Changes)

1. Add docs
- [ ] Add `PyQuifer/tests/benchmarks/README.md` explaining:
  - which files are PyQuifer-owned (`bench_*.py`, `harness.py`, `generate_report.py`)
  - which directories are comparators and "comparison only"
  - how JSON results are produced and how to regenerate `results/BENCHMARK_REPORT.md`
  - supported env vars (e.g. `PYQUIFER_BENCH_SEEDS`)

2. Add a comparator manifest
- [ ] Add `PyQuifer/tests/benchmarks/COMPARATORS.md` (or `VENDOR_MANIFEST.md`):
  - repo name, upstream URL/commit (if known), license, why it exists here, update procedure

3. Tighten pytest collection (must-have)
- [ ] Add `PyQuifer/tests/benchmarks/conftest.py` to ignore comparator trees.
  - Use `collect_ignore` / `collect_ignore_glob` for `comparators/**`, `Gymnasium/**`, `open_spiel/**`, etc.
- [ ] Ensure root `pytest.ini` (or `pyproject.toml`) sets `testpaths` so default `pytest` does not walk comparator trees.

4. Git hygiene
- [ ] Ensure `PyQuifer/tests/benchmarks/results/`, `PyQuifer/tests/benchmarks/data/`, and `__pycache__/` are ignored.

5. Fix report placeholder encoding
- [ ] In `PyQuifer/tests/benchmarks/harness.py`, replace the missing-value placeholder with ASCII, e.g. `--`.
  - This will stop mojibake in `BENCHMARK_REPORT.md`.

## Phase 1 (Benchmark/Test Separation)

1. Mark embedded pytest tests as smoke
- [ ] Add a consistent convention so benchmark scripts can include pytest tests without becoming CI hazards.
  - Option A: put pytest tests behind an env var (e.g. `PYQUIFER_BENCH_SMOKE=1`).
  - Option B: mark with `@pytest.mark.slow` / `@pytest.mark.benchmark` and configure CI to exclude by default.

2. Make smoke tests deterministic and cheap
- [ ] Ensure all `Test*` methods use tiny sizes (few steps/samples) and avoid downloads where possible.
- [ ] Add clear skips when optional deps are missing (gymnasium, lava, etc.).

3. Standardize import strategy
- [ ] Prefer running benchmarks with `PYTHONPATH=PyQuifer/src` or editable install rather than per-script `sys.path.insert`.
- [ ] If `sys.path.insert` stays, centralize it in one helper imported by all suites.

## Phase 2 (Bigger Cleanup)

1. Isolate comparator trees physically
- [ ] Relocate comparator repos under `comparators/`.
- [ ] Update any paths in benchmark suites that reference comparator content.

2. Add a simple runner
- [ ] Add `PyQuifer/tests/benchmarks/run.py`:
  - list suites
  - run selected suites
  - optionally run all
  - call `generate_report.py`

3. Add minimal schema validation
- [ ] Define a JSON schema for suite output and validate before report generation.

## Notes About `bench_*.py` pytest tests

A lot of `bench_*.py` include embedded `Test*` classes (example suites: bifurcation, consciousness, continual, EP training, lava, LLM A/B, local rules, NLB, OpenSpiel, oscillators, torchdiffeq/torchdyn, etc.).
That is useful, but it also means:
- running `pytest` can trigger heavy benchmark logic unless carefully bounded
- accidental collection of comparator tests becomes very expensive

## Acceptance Criteria

- `pytest` from repo root collects only intended tests quickly.
- Running a full bench run produces:
  - JSON files in `results/`
  - `results/BENCHMARK_REPORT.md`
- Comparator directories are clearly documented as comparison-only and do not participate in test discovery.
- Benchmark report no longer contains mojibake placeholders.
