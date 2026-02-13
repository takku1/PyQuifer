# Feature Specification: Mizuki Foundation Rebuild

**Feature Branch**: `001-mizuki-foundation`
**Created**: 2026-02-10
**Status**: Draft
**Input**: Rebuild Mizuki's foundation to properly use PyQuifer's modern APIs, remove dead/duplicate code, wire Phi-4 end-to-end, and archive bloated vendored benchmarks.

## User Scenarios & Testing

### User Story 1 - Chat with Phi-4 Through PyQuifer (Priority: P1)

A user launches Mizuki and has a natural conversation. Every response is modulated by PyQuifer's consciousness dynamics — temperature, personality blend, attention bias — producing responses that feel alive rather than generic. The oscillators are ticking, the criticality controller is at the edge of chaos, and the modulation state shapes how Phi-4 generates.

**Why this priority**: This is the entire point of Mizuki. Without a working Phi-4 + PyQuifer pipeline, nothing else matters.

**Independent Test**: Run `python mizuki_cli.py`, send a message, get a Phi-4-generated response where `temperature` and `personality_blend` came from `PyQuiferBridge.step()`.

**Acceptance Scenarios**:

1. **Given** a fresh Mizuki launch, **When** the user sends "Hello", **Then** Phi-4 generates a response with temperature/top_p modulated by PyQuifer's oscillator state (not hardcoded defaults).
2. **Given** a running Mizuki, **When** PyQuifer's coherence is high, **Then** generation temperature is lower (more focused), and when coherence is low, temperature is higher (more creative).
3. **Given** a running Mizuki, **When** the user sends multiple messages, **Then** PyQuifer's tick count increments and diagnostics are accessible via `/status`.

---

### User Story 2 - Clean Codebase with No Dead Code (Priority: P2)

A developer opens the Mizuki repo and finds a clean, understandable codebase. There is one brain implementation (brain_unified.py), one consciousness engine (PyQuifer's CognitiveCycle via PyQuiferBridge), no duplicate wrapper modules, and no deprecated band system cluttering the tree.

**Why this priority**: Dead code causes confusion, increases maintenance burden, and masks real bugs. Must be cleaned before building new features.

**Independent Test**: Grep for imports of deleted modules returns zero hits. All remaining imports resolve. All tests pass.

**Acceptance Scenarios**:

1. **Given** the cleanup is complete, **When** searching for `from mizuki.core.brain import` or `from mizuki.bands.`, **Then** zero matches are found in non-archived code.
2. **Given** the cleanup is complete, **When** running `python -c "from mizuki.core.brain_unified import create_mizuki_unified"`, **Then** it succeeds without importing any deleted module.
3. **Given** the cleanup is complete, **When** checking `consciousness/`, **Then** only modules with unique Mizuki-specific logic remain (homeostasis, hardware_instinct, aversive_memory, etc.), not PyQuifer wrappers.

---

### User Story 3 - Archive Vendored Benchmarks (Priority: P3)

The PyQuifer repo is lean. The 60+ vendored benchmark repos (helm, SWE-bench, openai-evals, mteb, etc.) are moved to archive, keeping only the PyQuifer-specific bench scripts and the 820 unit tests.

**Why this priority**: The vendored repos bloat the repo without being dependencies. Archiving them reduces noise and makes the codebase navigable.

**Independent Test**: `pytest tests/ --ignore=tests/benchmarks` still passes 820 tests. `pip install -e PyQuifer` still works.

**Acceptance Scenarios**:

1. **Given** benchmarks are archived, **When** running `pytest tests/ --ignore=tests/benchmarks`, **Then** all 820 unit tests pass.
2. **Given** benchmarks are archived, **When** checking `tests/benchmarks/`, **Then** only PyQuifer-authored bench scripts remain (bench_*.py, harness.py, generate_report.py, results/), not vendored repos.
3. **Given** benchmarks are archived, **When** importing any pyquifer module, **Then** no errors occur (zero dependency on vendored code).

---

### Edge Cases

- What happens if Phi-4 model files are missing? Brain should raise a clear error with download instructions, not silently fall back to dummy mode.
- What happens if PyQuifer's tick() fails during generation? Should log the error and fall back to default temperature (0.7), not crash Mizuki.
- What happens if the user runs legacy commands (`--legacy`)? Should print a deprecation notice explaining the legacy brain was removed.

## Requirements

### Functional Requirements

- **FR-001**: `brain_unified.py` MUST use `PyQuiferBridge` (not `PyQuifer` from core.py) as its consciousness engine.
- **FR-002**: `PyQuiferBridge.step()` MUST be called before every LLM generation to get current `ModulationState` (temperature, top_p, coherence, personality_blend).
- **FR-003**: The returned `ModulationState` MUST be applied to Phi-4's generation config (temperature, top_p, repetition_penalty at minimum).
- **FR-004**: Legacy `brain.py` MUST be deleted. All code referencing it MUST be updated or removed.
- **FR-005**: The 5 deprecated band modules (cortex, gating, logic, reflex, sensory) MUST be deleted.
- **FR-006**: Consciousness wrapper modules that duplicate PyQuifer (criticality.py, motivation.py, global_workspace.py, actualization.py) MUST be deleted or reduced to thin adapters.
- **FR-007**: `daemon.py` MUST use `brain_unified` (not legacy brain).
- **FR-008**: All 4 test files referencing legacy brain MUST be updated to use brain_unified.
- **FR-009**: Placeholder/mock responses in `api/server.py` MUST be replaced with actual Phi-4 generation or removed.
- **FR-010**: Vendored benchmark repos MUST be moved from `PyQuifer/tests/benchmarks/` to `archive/`, keeping only PyQuifer-authored scripts.
- **FR-011**: The `--no-brain` mode MUST remain functional (tools-only mode).
- **FR-012**: Voice pipeline integration with PyQuifer MUST be preserved (already working, do not break).

### Key Entities

- **PyQuiferBridge**: The LLM-facing API from PyQuifer. Wraps CognitiveCycle and provides ModulationState.
- **ModulationState**: Dataclass with temperature, top_p, repetition_penalty, coherence, personality_blend, attention_bias.
- **MizukiBrainUnified**: The single brain implementation that owns Phi-4 + PyQuiferBridge.
- **UnifiedSelfModel**: Wrapper for Phi-4-multimodal that handles loading, generation, unloading.

## Success Criteria

### Measurable Outcomes

- **SC-001**: User can have a multi-turn conversation where each response is visibly modulated by oscillator state (verifiable via `/status` showing changing coherence/temperature).
- **SC-002**: No imports of deleted modules remain in the codebase (grep returns 0 hits).
- **SC-003**: All existing tests pass after cleanup (820 PyQuifer + Mizuki tests).
- **SC-004**: Phi-4 response generation latency is under 5 seconds for a short prompt on RTX 4090.
- **SC-005**: PyQuifer tick adds less than 10ms overhead to each generation cycle.
- **SC-006**: Repository size decreases by archiving vendored benchmarks.

## Assumptions

- Phi-4-multimodal-instruct model files are already downloaded at `models/unified/Phi-4-multimodal-instruct/`.
- The RTX 4090 (24GB VRAM) can run Phi-4 in 4-bit quantization alongside PyQuifer's CPU-based tick loop.
- Voice pipeline modules in `mizuki/voice/` are already working and should not be modified during this refactor.
- The `--legacy` flag will be removed entirely (not preserved as a fallback).
