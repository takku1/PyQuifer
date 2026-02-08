# Superposition: Cross-Domain Transfer via Oscillatory Workspaces (Concept Note)

Last updated: 2026-02-08

This document is a **concise research-oriented concept note**. It describes a way to get cross-domain “cognitive bleed” (analogical transfer without explicit retrieval) by coupling multiple specialized workspaces through oscillatory dynamics and a limited-capacity global workspace.

It is **not** a claim that all mechanisms below are implemented today. Treat it as a design hypothesis that can be tested.

## 1) Problem Statement

Tool-using AI systems often feel fragmented: capabilities act like isolated “calls” rather than a unified, persistent mind. A companion OS-style system needs:
- parallel specialist processes,
- stable internal state,
- controlled integration (what becomes globally available),
- and long-horizon coherence.

## 2) Core Idea: Oscillatory “Superposition” of Workspaces

Use multiple specialist workspaces running in parallel (e.g., “Chess”, “Law”, “Code”), each with its own internal dynamics.

“Superposition” here does **not** mean quantum computing. It means:
- multiple candidate interpretations/strategies can coexist in parallel modules,
- partial membership/activation can be graded,
- and interaction is mediated by **phase/coherence** rather than explicit symbolic mapping.

### 2.1 Architecture Sketch (Abstract)

- **Workspace_i**: specialist module producing a latent proposal `z_i` and a salience signal `s_i`.
- **Global Workspace (GW)**: limited-capacity broadcast state `z_gw` selected from competing proposals (winner or top-k coalition).
- **Oscillatory Coupling**: phase alignment / coherence determines effective communication bandwidth between modules.

Practical interpretation:
- A background workspace is not “idle”; it continues metastable dynamics near criticality.
- When coupling is enabled, latent structure from a background workspace can bias an active workspace.
- The GW enforces a bottleneck: only some content becomes globally available and persists.

## 3) Theory Links (Why This Is Plausible)

### 3.1 Global Workspace Theory (GWT) and Global Neuronal Workspace (GNW)

GWT/GNW-style models motivate:
- many specialized processes operating in parallel,
- a bottlenecked broadcast mechanism (“ignition” / access),
- and integration through recurrent/global availability.

### 3.2 Communication Through Coherence (CTC)

CTC-style hypotheses motivate:
- oscillatory coherence as a routing/gating mechanism,
- dynamic reconfiguration of functional connectivity,
- and timing-based selection of what influences what.

### 3.3 Metastability / Criticality

Operating near the edge of order and chaos can:
- preserve stability while enabling rapid reconfiguration,
- support exploration (escaping local minima),
- and produce transient coalitions (useful for “thought-like” dynamics).

## 4) What This Should Enable (Testable Claims)

1. **Cross-domain analogical transfer without explicit prompting**.
   - Example: a “Chess” workspace running in the background changes the strategy selection in “Law” reasoning, even without an explicit “think like chess” instruction.

2. **Content-specific influence, not just load/parallelism**.
   - The influence depends on *which* background structure is active (e.g., sacrifice vs endgame conversion), not merely that something else is running.

3. **Coalition stability and coherence**.
   - Under appropriate coupling and workspace selection, the system converges to a stable global interpretation faster (or with fewer reversals) than a non-coupled baseline.

## 5) Evaluation Sketches (Benchmarks)

These are *not* standard LLM benchmarks. They are designed to detect dynamical cross-talk.

### 5.1 Cross-Domain Analogical Transfer (CAD)

Measure whether performance in Domain B improves when Domain A is “spinning”.

- Baseline: Domain B task with only B active.
- Control: Domain B task with unrelated Domain A' active.
- Experimental: Domain B task with intended Domain A active.

Primary metric: improvement vs baseline **and** separation vs control.

### 5.2 Parallel Constraint Satisfaction (PCS)

Give tasks with conflicting constraints (ethics, cost, precedent, time, user preference) and measure:
- convergence time to a stable solution,
- constraint satisfaction rate,
- and stability (how often the solution reverses under small perturbations).

### 5.3 Global Workspace Broadcast Priming

Prime Workspace A with a structured stimulus, then shortly after ask Workspace B a task. Measure:
- whether the *content* of A predicts strategy choices in B,
- and whether effects persist across a short delay (broadcast persistence).

## 6) Scope Boundary: Library vs Application

This concept splits cleanly into:

- **Library primitives** (fit for PyQuifer): oscillators, coupling operators, metastability/criticality controllers, workspace competition/broadcast modules, and instrumentation.
- **Application orchestration** (fit for a companion OS project): domain definitions (“Chess/Law/Code”), tool adapters, scheduling, data collection, and human-facing evaluation.

## 7) References (Starting Points)

These are the canonical anchors for the ideas above:

- GWT/GNW (broadcast, ignition, access)
  - Dehaene, Kerszberg, Changeux (1998)
  - Mashour, Roelfsema, Changeux, Dehaene (2020)

- Communication-through-coherence (oscillatory routing/gating)
  - Fries (2005)
  - Fries (2015)

- Metastability / criticality (edge-of-chaos computation)
  - (General criticality literature; choose a few target papers for the exact controller/metric you use.)

---

Previous content was a chat-style transcript and has been backed up to:
- `PyQuifer/docs/superposition.chat_transcript.bak.md`

