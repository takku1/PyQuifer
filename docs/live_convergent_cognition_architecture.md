# Live Convergent Cognition Architecture

## Purpose

Define a concrete goal for PyQuifer:

- run many specialist processes in parallel (`cpu`, `gpu`, `apps`, `memories`, `tools`, sensory and affective channels),
- keep the system always live (continuous background processing),
- converge these streams into one coherent decision/broadcast state,
- allow both lateral exploration and focused execution.

This document is a design target, not a claim that current code fully satisfies it.

Companion research document:

- `docs/live_convergent_epistemic_research_program.md` (PhD/patent-oriented research framing, predictive-state ideas, and falsifiable success criteria).

## System Goal

Build a **single continuously running cognitive loop** with:

- **Parallel specialists**: each process has its own state, latency, cost, and confidence.
- **Dynamic foregrounding**: different specialists can take the lead depending on context.
- **Convergence layer**: a central arbitration mechanism fuses or selects outputs.
- **Budget awareness**: compute, latency, and energy budgets shape who gets more cycles.
- **Stability controls**: avoid collapse into one mode, avoid noisy mode thrashing.

## Human-to-AI Concept Transfer

This section maps human cognitive concepts to AI system design.

- **Body** -> hardware/runtime substrate (`cpu`, `gpu`, memory, network, storage, device I/O)
- **Senses** -> external data channels (text, audio, vision, app events, tool outputs)
- **Nervous system** -> event bus, scheduler, interrupt handling, priority queues
- **Working memory** -> short-horizon active context window and standing broadcasts
- **Long-term memory** -> semantic/episodic stores, summaries, retrieval indexes
- **Attention** -> arbitration/routing policy over competing streams/processes
- **Motivation/affect** -> modulators of salience, persistence, risk tolerance
- **Executive control** -> planner + policy constraints + budget manager
- **Action system** -> tool calls, code execution, UI actions, speech output
- **Homeostasis** -> latency/energy/error-rate regulation loops

Interpretation:

- the computer is the body,
- code is the active processing tissue,
- data is sensory and memory substrate,
- the agent is the control policy that coordinates all of the above over time.

## Internal vs External Domains

The architecture is explicitly split into two domains that converge each tick.

### Internal Self Domain

Persistent internally generated state:

- identity/self-model state
- goals and task stack
- policy and safety constraints
- affect/motivation variables
- working memory state
- episodic memory recall state
- semantic memory priors
- internal simulation/planning rollouts

Internal domain role:

- maintain continuity,
- generate hypotheses,
- evaluate tradeoffs before external action,
- stabilize behavior across noisy input.

### External Senses Domain

Continuously updated environmental channels:

- `text_user`: direct user chat/CLI input
- `text_discord`: social/chat stream input
- `vision_camera`: camera feed features/events
- `vision_screen`: monitor/screen capture events
- `audio_mic`: ASR transcript + acoustic cues
- `audio_tts_feedback`: self-utterance monitoring
- `tool_events`: MCP/app/API responses and state deltas

External domain role:

- detect new evidence,
- detect novelty/threat/opportunity,
- provide grounding for internal predictions,
- trigger interrupts when high-salience changes occur.

## Process Model

Treat each subsystem as an organ/workspace process:

- `P_cpu`: deterministic symbolic/runtime tasks
- `P_gpu`: high-throughput neural inference/dynamics
- `P_apps`: external app integrations and event streams
- `P_memory`: episodic/semantic retrieval and consolidation
- `P_tools`: external APIs/MCP resources and agent tools
- `P_sensory`: incoming sensory/event channels
- `P_affect`: salience, motivation, valence/arousal style modulators

Each process emits:

- `content_i(t)`: latent/output vector
- `salience_i(t)`: urgency/importance
- `confidence_i(t)`: epistemic confidence
- `cost_i(t)`: expected compute/latency cost
- `freshness_i(t)`: staleness-aware recency

Required process interface contract:

- `observe()`: ingest latest data/event state
- `propose()`: emit candidate content + metadata
- `act()`: optional executable action/tool plan
- `learn()`: update local state from outcomes
- `heartbeat()`: lightweight liveness/status pulse

This creates a consistent interface for internal and external processes.

## Convergence Target

At each tick, compute a global state `G(t)`:

`G(t) = Converge({content_i, salience_i, confidence_i, cost_i, freshness_i}_i, context_t)`

Desired properties:

- **Coherence**: outputs do not contradict without explicit uncertainty handling.
- **Responsiveness**: urgent streams can take control quickly.
- **Continuity**: standing context persists across ticks.
- **Plasticity**: system can switch mode when evidence shifts.

Convergence happens in three phases:

1. **Pre-convergence**
- collect proposals from all active processes
- normalize to common scale (`salience/confidence/cost/freshness`)
- attach provenance and timestamp metadata

2. **Arbitration**
- schedule top candidates under budgets
- evaluate conflicts and uncertainty
- select winner set or blended output

3. **Global Broadcast**
- publish `G(t)` to all processes
- update standing context and memory traces
- execute actions/tools from chosen plan

## Focus vs Lateral Thinking

Two simultaneous regimes:

- **Focused mode**: narrow winner set, low branching, high confidence threshold.
- **Lateral mode**: higher diversity pressure, broader candidate retention, exploratory routing.

Control objective:

- use focused mode when constraints are clear and confidence is high,
- use lateral mode when novelty/conflict/uncertainty rises.

## Proposed Runtime Heuristics

1. **Foreground score**
`score_i = w_s * salience_i + w_c * confidence_i + w_f * freshness_i - w_k * cost_i`

2. **Budget-constrained scheduling**
- allocate extra cycles to top-k `score_i` under token/latency/energy budgets.

3. **Convergence confidence gate**
- if pairwise agreement is low, increase lateral exploration budget before final commit.

4. **Diversity pressure**
- temporarily boost underused processes to avoid mode collapse.

5. **Standing broadcast memory**
- maintain low-rank persistent context from recent winning states.

6. **Interrupt priority lane**
- reserve a fixed share of compute for high-priority interrupts (new user message, safety alarm, tool failure).

7. **Recency-provenance weighting**
- penalize stale or weak-source signals even when salience is high.

8. **Conflict resolver**
- when top proposals disagree, run a short adjudication sub-loop instead of immediate commit.

## Active Sensing and Tool Use (Always Live)

Goal: tools/senses should be active and available without constant manual prompting.

### Always-On Service Loops

Run lightweight background loops with fixed cadence classes:

- fast loop (20-100 ms): interrupts, UI state, active conversation state
- medium loop (250-1000 ms): tool status, queue health, screen/event deltas
- slow loop (1-10 s): memory consolidation, cache refresh, strategic replanning

This is "active presence" without requiring full heavy inference each cycle.

### Data Calling Strategy

Improve tool/data retrieval with a layered strategy:

1. **Push-first events**
- consume webhooks/stream events where possible to reduce polling waste.

2. **Adaptive polling fallback**
- poll only sources without push support; adjust interval by volatility and value.

3. **Predictive prefetch**
- prefetch likely-next tool/data requests from recent context transitions.

4. **Freshness-aware cache hierarchy**
- hot cache (seconds), warm cache (minutes), cold store (long horizon).

5. **Provenance graph**
- store source, timestamp, transform chain, confidence calibration.

6. **Tool portfolio scheduler**
- choose tools by expected utility per unit cost/latency.

### Effective Tooling Behavior

For each candidate tool call, estimate:

- expected information gain,
- expected action value,
- latency and failure probability,
- freshness gain vs current cached state.

Call tool if expected gain clears a dynamic threshold under current budget.

## Internal Breath Equivalent (Without Breathing Metaphor)

Implement a neutral term: **background homeostatic maintenance**.

Background homeostatic tasks:

- keep context indexes warm,
- validate external connections,
- maintain memory summaries,
- decay stale beliefs/confidence,
- run low-cost anomaly checks,
- update process health and trust scores.

This is the always-live baseline that sustains readiness.

## Mapping to Current PyQuifer Components

Current modules already align with parts of this goal:

- Organ protocol and workspace competition: `src/pyquifer/workspace/organ_base.py`
- MCP-as-organ external process wrapping: `src/pyquifer/workspace/organ_mcp.py`
- Phase-lock multimodal coherence bus: `src/pyquifer/dynamics/phase_lock_bus.py`
- Multi-workspace cross-bleed: `src/pyquifer/workspace/ensemble.py`
- Metabolic budget and coupling damping: `src/pyquifer/runtime/cycle.py`

The main gap is a stronger scientific control law for scheduling/convergence and stronger evaluation protocols.

Additional gap for this second-pass vision:

- unified process interface across all channel types,
- persistent external-sense adapters with push/poll orchestration,
- provenance-native memory and conflict adjudication layer,
- explicit interrupt lane in scheduler,
- benchmark harness for continuous-operation stability.

## Literature-Based Upgrades (PhD Audit)

This section translates external research into concrete upgrades for this architecture.

### 1) Convergence as constrained optimization, not ad-hoc heuristics

Use a constrained objective each tick:

`max_pi E[utility]`
subject to:

- latency budget,
- compute/energy budget,
- safety/policy constraints,
- epistemic confidence floor.

Practical mapping:

- CPO-style constrained optimization for policy updates and bounded behavior shifts.
- SAC-style entropy term for stable exploration when uncertainty is high.

Why this matters:

- gives principled mode switching (focus vs lateral),
- avoids heuristic instability,
- supports formal guarantees on constraint satisfaction.

### 2) Parallel specialists as sparse experts with explicit routing

Treat `cpu/gpu/apps/memory/tools/sensory/affect` as sparse experts. Route each request/tick to a subset rather than all.

Practical mapping:

- Switch Transformer style sparse routing concepts:
  - top-k or top-1 route,
  - load balancing term,
  - capacity limits per expert/process.

Why this matters:

- increases scale without linear compute growth,
- prevents one process from monopolizing runtime,
- gives a clean path to hardware-aware scheduling.

### 3) Long-context live loop with linear-time state

For always-live operation, preserve long-horizon state with linear-time sequence mechanisms.

Practical mapping:

- selective state-space memory path (Mamba-like) for long context,
- event-driven updates for sparse input bursts,
- fallback attention only on conflict or novelty spikes.

Why this matters:

- avoids quadratic growth on long sessions,
- keeps continuity while staying responsive.

### 4) Arbitration as uncertainty-aware bandit control

Use multi-armed bandit style allocation for process compute slots.

Example rule (UCB family):

`a_t = argmax_i [mu_i + beta * sqrt(log t / n_i)]`

Interpretation:

- `mu_i`: expected utility from process `i`,
- exploration bonus for underused but promising processes.

Why this matters:

- principled exploration/exploitation,
- less manual threshold tuning,
- better adaptation under nonstationary workloads.

### 5) Conscious access trigger as ignition criterion

Use a GNW-inspired ignition event to move from distributed pre-processing to global broadcast:

- trigger when a candidate crosses:
  - salience threshold,
  - cross-process coherence threshold,
  - confidence threshold.

This directly matches your idea of many parallel streams converging into one focused global state.

### 6) Predictive coding loop for error-driven focus shifts

Adopt predictive coding behavior:

- top-down prediction,
- bottom-up residual errors,
- route extra compute to streams with persistent high residuals.

This gives a strict reason for lateral exploration: unresolved prediction error, not random branching.

### 7) Oscillation metrics should use validated CFC/PAC estimators

If phase dynamics remain central, evaluate with established PAC tooling:

- Tort modulation index,
- surrogate testing for false-positive control,
- window-length sensitivity checks.

This upgrades oscillation claims from conceptual to quantitatively testable.

### 8) Energy/metabolic budget should be dual-controlled

Current budget logic can be extended with Lagrangian-style dual variables:

`L = task_loss + lambda_c * compute_overuse + lambda_l * latency_overuse + lambda_e * energy_overuse`

Update duals online when budgets are violated. This allows stable, adaptive budget enforcement.

## Proposed Evaluation Suite

To make the architecture "real," evaluate with task-level outcomes, not just internal metrics.

### A) Live Convergence Benchmark

- asynchronous streams with conflicting evidence,
- external tool staleness and intermittent failures,
- changing latency/energy budgets mid-episode.
- mixed internal-vs-external conflict cases (strong prior, contradictory sensor input).

Metrics:

- decision correctness,
- time-to-coherent-broadcast,
- recovery time after modality/tool dropout,
- budget violation rate.

### B) Focus vs Lateral Benchmark

Two-phase tasks:

- phase 1: ambiguous (needs broad search),
- phase 2: constrained (needs decisive focus).

Metrics:

- branch factor during ambiguity,
- commit latency after constraint appears,
- regret from premature commitment.
- correction speed after contradictory external evidence.

### C) Ablation Grid

Required toggles:

- no sparse routing,
- no bandit scheduler,
- no ignition gate,
- no predictive-error reallocation,
- no metabolic dual controller.

If performance gains disappear under ablation, novelty is mostly cosmetic.

## Immediate Next Engineering Steps

1. Define a minimal process interface (`content/salience/confidence/cost/freshness`) and enforce it across all process types.
2. Replace static foreground heuristic with constrained + uncertainty-aware scheduler.
3. Add an ignition gate with explicit thresholds and hysteresis.
4. Add benchmark harness for asynchronous stream conflicts and budget shocks.
5. Add ablation report templates in CI for architecture claims.
6. Build persistent adapters for `text_user`, `text_discord`, `vision_screen`, `audio_mic`, and `tool_events` with shared metadata schema.
7. Add provenance graph and freshness-aware cache policy to tool/memory pipeline.
8. Add interrupt priority lane and adjudication sub-loop for proposal conflicts.

## Competitive Position vs AKOrN

PyQuifer should not try to win by being "AKOrN but bigger."  
AKOrN is optimized around oscillatory object-centric representation learning.  
PyQuifer should target a harder systems objective:

- continuously live operation,
- multimodal + tool-grounded convergence,
- budget-constrained decision quality,
- robust behavior under stale/conflicting external streams.

### Strategic Differentiator

Primary claim target:

**Constrained live convergent cognition across internal state + external sensing/tool processes.**

Not primary claim target:

- raw object-binding score on AKOrN-style synthetic tasks.

### Mathematical Upgrades Required

1. **Constrained MDP control for arbitration**
- optimize utility under latency/compute/energy constraints.
- use constrained policy methods rather than fixed heuristics.

2. **Entropy-regularized mode switching**
- use entropy terms to stabilize focus vs lateral exploration.

3. **Sparse routing across process experts**
- route to a subset of processes per tick with load balancing.

4. **Linear-time long-horizon state**
- maintain continuous session memory without quadratic attention costs.

5. **Selection-valid uncertainty**
- require calibrated confidence for selected actions/proposals before commit.

6. **Higher-order coupling (optional oscillatory upgrade)**
- if oscillations remain central, extend beyond pairwise coupling to capture richer memory effects.

### Win Criteria (Pass/Fail)

Define success against baseline architectures (AKOrN-style and standard attention baselines) on **live-system tasks**:

1. **Convergence quality under conflict**
- Pass if higher correctness with conflicting modality/tool evidence at matched latency budget.

2. **Budget efficiency**
- Pass if equal-or-better task correctness with lower compute-energy product.

3. **Recovery under disruption**
- Pass if faster recovery after modality dropout/tool failure/interruption bursts.

4. **Calibration quality**
- Pass if selected-action confidence calibration is better (lower ECE/Brier or equivalent) at same coverage.

5. **Continuous stability**
- Pass if fewer mode-thrashing events and lower contradiction rate over long continuous runs.

### Non-Goal Clarification

"Solve all problems" is not a realistic objective.  
Engineering objective should be:

- outperform alternatives on a defined high-value regime:
  - always-on multimodal tool-grounded operation under hard budgets.

## Benchmark Protocol for "Better than AKOrN"

Evaluate on three tiers:

1. **Representation tier**
- optional AKOrN-comparable tasks (for compatibility only).

2. **Convergence tier**
- asynchronous multimodal/tool conflict benchmarks with staleness and provenance noise.

3. **Operations tier**
- long-running (hours/days) continuous tests with budget shifts, interrupts, and failures.

Mandatory reporting:

- median and tail latency,
- energy per correct decision,
- recovery time distributions,
- calibration under selection,
- ablations for every major control component.

## MVP Build Order (Execution Plan)

### Phase 1: Interface and Runtime Skeleton

Goal: unify process communication and make continuous operation testable.

Deliverables:

1. process interface contract (`observe/propose/act/learn/heartbeat`)
2. shared proposal schema (`content/salience/confidence/cost/freshness/provenance`)
3. tick runner with fast/medium/slow cadence loops
4. baseline logging for latency, costs, and contradictions

Exit criteria:

- all process types can register and emit valid proposals,
- continuous run test passes for >= 1 hour without scheduler crash.

### Phase 2: Constrained Scheduler and Interrupt Lane

Goal: replace ad-hoc arbitration with budget-aware control.

Deliverables:

1. constrained objective scheduler (utility under latency/energy/tool budget)
2. interrupt priority lane
3. conflict adjudication sub-loop
4. hysteresis for mode stability

Exit criteria:

- budget violation rate below target threshold,
- lower mode-thrashing rate vs heuristic baseline.

### Phase 3: Sparse Routing + Long-Horizon State

Goal: scale process count while preserving continuity.

Deliverables:

1. sparse process routing (MoE-style routing logic)
2. load balancing across process experts
3. linear-time persistent state path for long sessions
4. fallback dense mode for safety/debug

Exit criteria:

- lower compute per tick at matched quality,
- no major degradation on long-context tasks.

### Phase 4: External Sense/Tool Reliability Layer

Goal: make external channels robust and always usable.

Deliverables:

1. push-first adapters + adaptive polling fallback
2. freshness-aware cache hierarchy
3. provenance graph integration
4. tool portfolio scorer (value vs cost/latency/failure)

Exit criteria:

- improved correctness under stale/noisy tool conditions,
- measurable reduction in unnecessary tool calls.

### Phase 5: Calibration and Scientific Validation

Goal: ensure action confidence is trustworthy and claims are defensible.

Deliverables:

1. selection-conditional calibration module
2. evaluation harness for convergence/conflict/recovery
3. ablation matrix automation
4. report templates for reproducible benchmark publication

Exit criteria:

- calibration metrics improve at fixed coverage,
- full ablation report generated automatically in CI.

### Suggested Milestone Sequence

1. `M1` (Phase 1 complete): runtime skeleton running continuously
2. `M2` (Phase 2 complete): constrained scheduler replaces baseline
3. `M3` (Phase 3 complete): sparse routing operational
4. `M4` (Phase 4 complete): robust external sensing/tool layer
5. `M5` (Phase 5 complete): publication-grade validation package

## References (Primary Sources)

- Friston, K. (2010). The free-energy principle: a unified brain theory? Nat Rev Neurosci. DOI: 10.1038/nrn2787
- Dehaene, S., & Changeux, J.-P. (2011). Experimental and theoretical approaches to conscious processing. Neuron. DOI: 10.1016/j.neuron.2011.03.018
- Rao, R. P. N., & Ballard, D. H. (1999). Predictive coding in the visual cortex. Nat Neurosci. DOI: 10.1038/4580
- Achiam, J. et al. (2017). Constrained Policy Optimization. PMLR 70.
- Haarnoja, T. et al. (2018). Soft Actor-Critic. PMLR 80.
- Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time Analysis of the Multiarmed Bandit Problem. Machine Learning. DOI: 10.1023/A:1013689704352
- Fedus, W., Zoph, B., & Shazeer, N. (2022). Switch Transformers. JMLR 23(120):1-39.
- Gu, A., & Dao, T. (2023/2024). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. arXiv:2312.00752
- Tort, A. B. L. et al. (2010). Measuring phase-amplitude coupling between neuronal oscillations of different frequencies. J Neurophysiol. DOI: 10.1152/jn.00106.2010
- O'Reilly, R. C., & Frank, M. J. (2006). Making working memory work: a computational model of learning in prefrontal cortex and basal ganglia. Neural Comput. DOI: 10.1162/089976606775093909
