<!--
  Sync Impact Report
  Version change: 0.0.0 → 1.0.0
  Modified principles: N/A (initial constitution)
  Added sections: 7 principles, Architecture Constraints, Development Workflow, Governance
  Removed sections: none
  Templates requiring updates:
    ✅ plan-template.md — Constitution Check gates align with principles
    ✅ spec-template.md — no changes needed
    ✅ tasks-template.md — no changes needed
  Follow-up TODOs: none
-->

# PyQuifer Constitution

## Core Principles

### I. Oscillator Autonomy

Oscillators MUST evolve through their own dynamics (Kuramoto coupling, PLL entrainment,
Stuart-Landau bifurcation) — never through backpropagation from the LLM.
The gradient flow from LLM to oscillator state is severed by design.
The oscillatory system represents the "soul" — an independent dynamical substrate
that modulates cognition but is not subordinate to it.

**Core equation**: `Modified_Hidden = Original + A*sin(ωt+φ) * Trait_Vector`

### II. Emergence Over Forcing

All confidence, decision, and evaluation functions MUST be pure emergent measures.
No artificial scaling, no hard thresholds, no hand-tuned gates.
Evidence speaks for itself: `get_confidence = hypothesis_weighted / total_weighted`.
Arena competition is one evidence source among many, not a gatekeeper.
This principle applies to all confidence/decision code in PyQuifer.

### III. Zero Regression

All existing tests MUST pass before any new code merges.
New modules MUST include tests covering core functionality.
No feature is complete until `pytest tests/ --ignore=tests/benchmarks` is fully green.
The test suite is the ground truth for correctness — never weaken tests to accommodate code.

### IV. Module Independence

Each PyQuifer module MUST be standalone and independently importable.
All optional modules are gated by `CycleConfig` boolean flags (default `False`).
Lazy imports MUST be used for optional modules — importing `pyquifer` must not
pull in the full dependency tree.
Modules MUST NOT have circular dependencies.

### V. PyTorch Idioms

Non-learnable state MUST use `register_buffer()`, not `nn.Parameter`.
In-place buffer updates MUST use `torch.no_grad()` with `.add_()`, `.mul_()`, `.copy_()`.
Never use `.data` to bypass autograd — it is unnecessary inside `no_grad()` blocks.
All dynamically-created tensors in `forward()` MUST pass `device=input.device`.
Phase updates follow the pattern: `self.phase.add_(delta).remainder_(2 * math.pi)`.

### VI. Performance Discipline

The `tick()` hot path MUST target <=2 ms latency (CPU, realtime config).
No `.item()` calls in the tick hot path on GPU — use tensor-only diagnostics.
Hierarchical timestepping: not every module runs every tick.
`torch.compile` compatibility MUST be maintained — no graph-breaking operations
in compilable submodules.

### VII. Single-Sample Contract

`CognitiveCycle.tick()` processes exactly one moment of consciousness.
Batched inputs (2D with batch > 1) MUST be rejected with `ValueError`.
A `(1, state_dim)` input is squeezed automatically.
This contract is load-bearing — HPC, binding, and diagnostics all assume 1D flow.

## Architecture Constraints

- **CognitiveCycle** is the single integration point — all modules wire through it
- **PyQuiferBridge** is the LLM-facing API (streaming via `SteppedModulator`)
- **PyQuifer (core.py)** is the data-science API (`ingest`, `actualize`, `train`)
- State dimension, oscillator count, and hierarchy dims are set at construction time
- The tick pipeline has a fixed stage order (oscillators → criticality → HPC → precision → workspace → metastability → causal → darwinism → self-model → memory → output)
- New modules MUST be inserted at the appropriate pipeline stage, not appended arbitrarily

## Development Workflow

- Phase-based development: each phase adds a coherent capability set
- Every phase completion requires: tests green, exports in `__init__.py`, CycleConfig flag wired, benchmark on toy problem
- Two git repositories: Mizuki parent (`Z:\mizukiai\`) and PyQuifer nested (`Z:\mizukiai\PyQuifer\`) — push both after changes
- Documentation lives in `docs/current/`; stale docs archived to `docs/archive/YYYY-MM-DD-description/`
- Benchmark results are reproducible from clean checkout without manual vendor setup

## Governance

This constitution supersedes ad-hoc conventions. All code review and AI-assisted
development MUST verify compliance with the principles above.

Amendments require:
1. Documentation of what changed and why
2. Version bump (MAJOR for principle removal/redefinition, MINOR for additions, PATCH for clarifications)
3. Sync Impact Report updating dependent templates

Complexity MUST be justified — three similar lines of code is better than a premature abstraction.

**Version**: 1.0.0 | **Ratified**: 2026-02-10 | **Last Amended**: 2026-02-10
