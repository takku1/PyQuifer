# Live Convergent Epistemic Research Program

## Purpose

Define a research path that is:

- PhD-defensible,
- potentially patentable at the systems level,
- clearly not "just adding to AKOrN."

This document is a critical research program, not marketing copy.

## Core Research Thesis

Build an AI control plane that performs **epistemic decision-making under uncertainty and resource constraints** across:

- internal state (self-model, memory, goals, appraisal),
- external streams (user text, voice, camera, screen, tools/apps),
- continuous operation (always live).

Primary claim target:

**Risk-calibrated convergence of parallel internal/external processes into one coherent action state under hard budgets.**

## Why This Is Not an AKOrN Add-On

AKOrN-like methods focus on oscillatory representation/binding.  
This program focuses on:

- uncertainty calibration for action,
- freshness/provenance-aware tool use,
- constrained scheduling and arbitration,
- long-horizon continuous operation with failure recovery.

Oscillators may remain one substrate, but they are not the thesis center.

## PhD Program Structure (Three-Paper Arc)

### Paper 1: Theory

Topic:

- constrained epistemic arbitration with uncertainty guarantees.

Deliverables:

- formal objective under latency/energy/tool budgets,
- selective action policy with statistical confidence control,
- proof sketch or finite-sample bounds for risk control.

### Paper 2: System

Topic:

- live convergent runtime implementing the theory.

Deliverables:

- internal + external process interface,
- scheduler + interrupt lane + conflict adjudication loop,
- provenance-aware memory/tool layer,
- continuous loop with stability controls.

### Paper 3: Evaluation

Topic:

- benchmark suite for epistemic reliability in real operation.

Deliverables:

- asynchronous conflict tasks,
- stale tool/data and interruption stress tests,
- ablation study with pass/fail thresholds.

## Patentable System Angles (Non-Legal Framing)

Potential claim families:

1. **Calibrated tool arbitration**
- selecting tool calls/actions by calibrated uncertainty and budget dual variables.

2. **Freshness-provenance convergence**
- weighting process proposals by recency, source trust, and transform chain.

3. **Interrupt-aware bounded scheduler**
- guaranteed budget/latency constraint control in always-live multi-process loops.

4. **Convergence gate with conflict loop**
- requiring adjudication under disagreement before global commit.

Defensibility requirement:

- each claim must map to measurable outcomes and ablation evidence.

## Prediction Beyond Token Semantics

Your "do you like cake" idea is useful if framed as state prediction, not word lookup.

### Proposed Predictive State Model

Let:

- `x_t` = latent cognitive state,
- `m_t` = retrieved episodic/semantic memory context,
- `e_t` = appraisal/affect vector (valence, arousal, stance),
- `s_t` = sensory/tool evidence state,
- `g_t` = goals/constraints,
- `u_t` = uncertainty summary.

Then prediction is:

`x_{t+1} = F(x_t, m_t, e_t, s_t, g_t, u_t)`

Answer/action is selected from `x_{t+1}` under calibration and budget constraints, not from token statistics alone.

### Emotion/Appraisal as Weighting, Not Anthropomorphic Feeling

Critical stance:

- "emotion" should be implemented as computational appraisal signals (priority/risk/value modulation),
- not as a claim of subjective feeling.

For prompts like "do you like cake":

- retrieve relevant memory traces,
- apply appraisal weights and social/context policy,
- emit stance with confidence and provenance.

This yields richer behavior than dictionary-style output while staying technically grounded.

## Pattern-Matching Prediction Dimensions (Beyond Emotion)

Use multiple weighted channels:

- episodic similarity,
- semantic schema match,
- social/pragmatic context,
- utility/value estimate,
- safety/risk estimate,
- temporal freshness,
- source reliability,
- novelty/conflict signal.

Prediction quality should come from combining these channels, not a single lexical path.

## Predicting Kuramoto Convergence (Research Direction)

This is a valid and interesting sub-problem.

### Problem

Given initial phases, frequencies, and coupling structure, predict:

- convergence time,
- lock state class (global lock/chimera/partial lock),
- stability margin.

### Practical Approaches

1. **Surrogate predictor**
- train a meta-model on simulation traces to estimate convergence outcomes quickly.

2. **Control-oriented estimation**
- predict near-term order parameter trajectory and adjust coupling proactively.

3. **Stability diagnostics**
- use Lyapunov-style or empirical local stability indicators as features.

4. **Topology-aware predictors**
- include graph/topology statistics as explanatory variables.

### Why It Matters

- reduces expensive simulation steps,
- enables adaptive scheduler decisions,
- improves runtime predictability in continuous operation.

## Critical Risks and Failure Modes

1. **Anthropomorphic overreach**
- risk: claiming "emotion" or "sentience" without operational definition.
- mitigation: define machine-appraisal variables and measurable behavior only.

2. **Heuristic pile-up**
- risk: many knobs, weak generalization.
- mitigation: constrained objective + ablations + calibration checks.

3. **Uncertainty theater**
- risk: confidence scores not statistically meaningful.
- mitigation: selection-conditional calibration with coverage/error guarantees.

4. **Tool-grounding brittleness**
- risk: stale/low-trust sources dominate decisions.
- mitigation: freshness/provenance-weighted arbitration.

5. **Mode thrashing**
- risk: unstable switching between lateral and focused regimes.
- mitigation: hysteresis + interrupt quotas + bounded switching penalties.

## Falsifiable Success Criteria

The program succeeds only if all hold:

1. better decision correctness under stale/conflicting external evidence at matched budgets,
2. lower budget-violation rate under long continuous runs,
3. improved calibration for selected actions/tool calls,
4. faster recovery after stream/tool failure,
5. gains persist in full ablation matrix.

If not, novelty is likely architectural complexity without real epistemic advantage.

## Immediate Research Backlog

1. Formalize appraisal state variables and update equations.
2. Define proposal provenance schema and trust scoring.
3. Implement calibrated action-selection gate.
4. Build Kuramoto convergence surrogate dataset/task.
5. Add conflict-heavy continuous benchmark harness.
6. Publish reproducible ablation protocol and failure taxonomy.

## Execution Matrix (From Research to Real System)

Use this as the build-and-validate backbone. Every row needs a measurable pass condition.

| ID | Hypothesis | Implementation Target | Evaluation Setup | Primary Metrics | Pass Threshold | Required Ablations |
|---|---|---|---|---|---|---|
| E1 | Calibrated selection reduces unsafe/wrong actions at fixed utility | Selection-conditional calibration gate in scheduler | Tool-agent tasks with ambiguous outcomes (`tau-bench` style) | ECE, Brier, selective risk, task success | >=15% selective-risk reduction at matched success | No calibration gate |
| E2 | Freshness-aware arbitration reduces temporal errors | Timestamp + validity window + freshness penalty | Time-shifted QA/tool tasks (`Set the Clock` style) | Temporal accuracy, stale-action rate | >=20% stale-action reduction | No freshness weighting |
| E3 | Provenance gating reduces hallucinated tool-grounded claims | Provenance graph + factuality validator | RAG + tool outputs with injected source conflicts | Hallucination rate, correction latency | >=20% hallucination reduction | No provenance graph; no validator |
| E4 | Constrained scheduler improves reliability under budgets | CMDP-like scheduler with compute/latency/energy constraints | Long-run mixed workload with budget shocks | Budget violation rate, success under constraint | <=5% budget violations with <=5% success drop | Heuristic scheduler baseline |
| E5 | Interrupt lane improves responsiveness without mode thrash | Priority lane + hysteresis + quota | Interruption storm scenarios (user/tool/safety events) | p95 reaction latency, mode-switch count | >=30% better p95 reaction; <=10% extra thrash | No interrupt lane; no hysteresis |
| E6 | Sparse routing lowers cost without quality collapse | MoE-style process routing + load balance | Multi-process live loop (internal + external channels) | Compute/tick, energy/correct action, task quality | >=25% compute reduction at <=3% quality drop | Dense-all-process baseline |
| E7 | Long-horizon state path improves continuous coherence | Linear-time persistent state path | Multi-hour sessions with context drift | Contradiction rate, long-context task accuracy | >=20% contradiction reduction | No persistent state path |
| E8 | Kuramoto convergence surrogate improves runtime control | Convergence-time predictor + adaptive coupling | Oscillator simulation corpus + online control loop | Prediction MAE, control stability, runtime savings | <=10% convergence-time MAPE + runtime gain | No surrogate predictor |

## Minimal Real System Definition (MRS-v1)

System is "real" only when all are true:

1. Runs continuously for >=24h without scheduler deadlock/crash.
2. Maintains bounded budgets (latency/energy/compute) with measurable guarantees.
3. Demonstrates calibrated action/tool uncertainty.
4. Uses provenance + freshness in every high-impact action path.
5. Shows improved reliability vs baseline on repeated trials (`pass^k` style reporting).

If any item fails, system remains prototype/research-only.

## 90-Day Build Plan

### Days 1-30: Foundations

Deliver:

- unified process schema and runtime loop,
- telemetry for latency/energy/success/contradiction,
- baseline heuristic scheduler + benchmark harness.

Gate:

- all E-metrics collect reliably in CI and local runs.

### Days 31-60: Core Control

Deliver:

- constrained scheduler (E4),
- interrupt lane + hysteresis (E5),
- freshness-provenance pipeline (E2/E3).

Gate:

- pass E2, E4, E5 thresholds against baseline.

### Days 61-90: Scale + Reliability

Deliver:

- sparse routing (E6),
- long-horizon persistent state path (E7),
- calibrated selection gate (E1),
- Kuramoto surrogate experiment (E8, optional for v1 release but required for oscillator thesis claims).

Gate:

- pass E1, E6, E7; publish first ablation report.

## Reporting Standard (Required)

Every milestone report must include:

1. exact model/runtime version and config,
2. dataset/benchmark version and contamination controls,
3. median + tail metrics (p50/p95/p99),
4. ablation deltas with confidence intervals,
5. failure taxonomy with concrete traces.

No report is considered valid without ablation evidence and tail-latency statistics.

## 2026 Research Update (Primary Evidence)

This section upgrades the program with recent evidence from primary sources.

### A) Uncertainty is still a core failure mode

Recent NLP evidence shows persistent miscalibration:

- ICL settings remain miscalibrated even when task performance improves (ACL Findings 2025).
- LM probabilities are not calibrated even in simple numeric contexts (ACL 2025).
- faithful uncertainty expression is weak and needs dedicated calibration methods (EMNLP 2025 MetaFaith).

Design implications:

1. confidence must be treated as a first-class output with calibration metrics in CI,
2. action/tool selection must require calibrated uncertainty thresholds,
3. natural-language hedging alone is insufficient without statistical calibration.

### B) Tool-agent reliability remains low and inconsistent

TAU-bench results indicate that strong models still fail many real-world tool-agent tasks and show low reliability across repeated trials (`pass^k` behavior).

Design implications:

1. optimize for reliability under repetition, not only single-run success,
2. add explicit policy-rule checking and post-condition verification per tool call,
3. include trajectory-level consistency metrics, not only final answer metrics.

### C) Benchmark contamination and realism are active risks

Recent SWE evaluation work argues that benchmark results can be inflated by memorization/contamination and mismatch with real assistant usage patterns.

Design implications:

1. include freshness-controlled, continuously renewed evaluation tasks,
2. use mutation-based benchmark variants to avoid overfitting to known phrasing,
3. separate "in-distribution benchmark score" from "live operational score."

### D) Temporal alignment is not optional

Set the Clock (ACL Findings 2024) shows that models can answer with older internal knowledge despite newer pretraining cutoffs.

Design implications:

1. every belief/proposal needs explicit timestamp and validity window,
2. arbitration must penalize stale but high-confidence proposals,
3. temporal alignment modules should be evaluated as separate components.

### E) Provenance and factuality need explicit runtime checks

RAGTruth (ACL 2024) and Provenance fact-checking (EMNLP Industry 2024) support lightweight, practical factuality checking and hallucination detection in RAG pipelines.

Design implications:

1. add lightweight factuality validator on generated/tool-grounded outputs,
2. preserve chunk-level provenance to enable correction tracing,
3. gate high-impact actions on factuality thresholds.

### F) Long-context + sparse routing remains a practical scaling path

Switch Transformers (JMLR 2022) and selective SSM/Mamba work support sparse compute and linear-time sequence handling as practical scale mechanisms.

Design implications:

1. keep MoE-style routing as scheduler primitive for process experts,
2. maintain linear-time persistent state for always-live sessions,
3. use dense fallback path for safety-critical or high-conflict ticks.

## Updated "Defensible Novelty" Criteria

To claim meaningful novelty, the system must show all of:

1. calibrated action confidence under selection (not just token probabilities),
2. improved `pass^k` reliability in multi-step tool-agent tasks,
3. temporal correctness under time-shifted queries,
4. provenance-aware correction of hallucinated/tool-conflicted outputs,
5. sustained long-run stability under dynamic compute/latency budgets.

If any one is missing, the work risks being interpreted as an architectural remix rather than a substantive epistemic systems advance.

## Sources for This Update

- Li et al., "Large Language Models are Miscalibrated In-Context Learners" (Findings ACL 2025): https://aclanthology.org/2025.findings-acl.603/
- Lovering et al., "Language Model Probabilities are Not Calibrated in Numeric Contexts" (ACL 2025): https://aclanthology.org/2025.acl-long.1417/
- Liu et al., "MetaFaith" (EMNLP 2025): https://aclanthology.org/2025.emnlp-main.1505/
- Yao et al., "tau-bench" (arXiv 2406.12045): https://arxiv.org/abs/2406.12045
- TAU-bench leaderboard: https://www.tau-bench.com/
- Zhao et al., "Set the Clock" (Findings ACL 2024): https://aclanthology.org/2024.findings-acl.892/
- Niu et al., "RAGTruth" (ACL 2024): https://aclanthology.org/2024.acl-long.585/
- Sankararaman et al., "Provenance" (EMNLP Industry 2024): https://aclanthology.org/2024.emnlp-industry.97/
- Fedus et al., "Switch Transformers" (JMLR 2022): https://jmlr.org/papers/v23/21-0998.html
- Gu & Dao, "Mamba" (arXiv 2312.00752): https://arxiv.org/abs/2312.00752
- Quach et al., "Conformal Language Modeling" (ICLR 2024): https://proceedings.iclr.cc/paper_files/paper/2024/hash/31421b112e5f7faf4fc577b74e45dab2-Abstract-Conference.html
- Rubin-Toles et al., "Conformal Language Model Reasoning with Coherent Factuality" (ICLR 2025): https://proceedings.iclr.cc/paper_files/paper/2025/hash/679fcceef65c3d855aa885bd024542c1-Abstract-Conference.html
- Gulati et al., "Do We Need New Agent Benchmarks?" (arXiv 2508.08948): https://arxiv.org/abs/2508.08948
