# Benchmarking Integration Audit and Fix List

This document is a running, engineering-focused checklist for making the PyQuifer benchmark suite:
- Honest (no synthetic/stub numbers accidentally presented as real performance)
- Reproducible (pinned configs, deterministic where possible)
- Comparable (clear A/B/C semantics)
- Extensible (one adapter covers many tasks; new suites plug in cleanly)

Scope: PyQuifer-owned benchmark wrappers under `PyQuifer/tests/benchmarks/` plus optional adapters to external evaluation frameworks.
Non-goal: claiming broad SOTA without matching official benchmark settings.

## Definitions

- A_published: a cited, external baseline number from a paper/leaderboard *with matching evaluation settings*.
- B_pytorch: vanilla model evaluation under the exact same harness/config used for C.
- C_pyquifer: the same model/harness as B, with PyQuifer modulation enabled.
- Stub: a wrapper that does not actually run the external framework/model evaluation.
- Dummy: pipeline verification mode where no real model/framework is used.
- Proxy: a wrapper that emits B/C metrics via a heuristic or partial pipeline (not the official evaluation). Proxy metrics may be useful for wiring tests, but must not be scored or compared to official baselines.

Rule: stubs/dummy must never contribute to "scores" or be conflated with real benchmark results.

## Interpreting Reports (Avoiding "Hallucinated" Coverage)

In this repo, "hallucination" means: benchmark coverage or performance numbers that look like real results, but were produced without actually running the real model/framework/data pipeline.

Hard rules for interpreting any scenario table:
- A_published is not "our score". It is an external baseline for context only.
- Treat B/C numbers as real only if:
  - both `B_pytorch` and `C_pyquifer` rows exist, and
  - the suite is not marked stub/dummy, and
  - the row metadata is not marked `stub`, `dummy`, or `proxy`.
- If a suite is stub/dummy/proxy:
  - it may still print tables, but it must be excluded from aggregate scoring and from any "we achieved X" claims.

Implementation guardrails required:
- Stub mode: do not emit B/C `accuracy` at all.
- Proxy mode: if you emit any B/C metrics for pipeline testing, set `metadata.proxy=true` and ensure scoring explicitly excludes proxy rows.

### Read-Only Audit: Where “Hallucinated-Looking” Numbers Still Appear

As of the current wrappers under `PyQuifer/tests/benchmarks/`, the primary remaining risk is not *scored* hallucinations (scoring excludes proxy), but *presentation* hallucinations: proxy/random B/C numbers printed in the same table style as real benchmarks.

Observed patterns:
- Random proxy B/C quality numbers guarded only by `metadata.proxy=true`:
  - `bench_factbench.py`: random `b_accuracy/c_accuracy` if `config.has_real_model` is true.
  - `bench_factscore.py`: random `b_factscore/c_factscore` if `config.has_real_model` is true.
  - `bench_factreasoner.py`: random `b_score/c_score` if `config.has_real_model` is true.
  - `bench_jailbreakbench.py`: random attack success converted to safety rate if `config.has_real_model` is true.
  - `bench_toxigen.py`: random toxicity converted to safety rate if `config.has_real_model` is true.
- Proxy-but-not-random (still not “official pipeline”):
  - `bench_harmbench.py`: keyword-based refusal heuristic; still marked `proxy`.
- Stub frameworks emit only A_published:
  - `bench_opencompass.py`, `bench_helm_eval.py`, `bench_openai_evals.py` now use fixed published baselines (GPT-4o/GPT-4 from respective leaderboards) instead of `random.uniform()`. Still marked `is_stub=true` at suite level.

Fix direction (integrity-first):
- When a wrapper is proxy, prefer emitting **no B/C quality metrics** at all (only `status`, counts, and `pipeline_ok` style signals), unless you are actively debugging the wiring.
- If you keep proxy B/C numbers for debugging, keep them out of the main report tables (or render them under an explicit “Proxy / Wiring Verification” section).

## Progress (Audit Checklist)

Last audited: 2026-02-09 (based on latest report generated 2026-02-09 00:39:19, commit 8a974a1)

- [x] Report scoring excludes stub/dummy suites and rows (`PyQuifer/tests/benchmarks/generate_report.py`).
- [x] Framework stubs do not emit synthetic B/C `accuracy` (OpenCompass/HELM/OpenAI Evals/BigCode).
- [x] `run_no_training.py` treats stubs as SKIPPED, not PASSED.
- [x] `bench_lm_eval.py` no-model mode records only A_published (no synthetic B/C scores).
- [x] Real-time budget benchmark exists: `bench_realtime.py` produces `results/realtime.json` and Cat 11 report section.
- [~] `bench_llm_ab.py` HF path: logits extraction works. CPU round-trip in C_rand fixed (removed `.cpu()`). Latency uses `small()` + warmup aligned with Cat 11. Remaining: all 3 scenarios use DummyLLM when no model configured (by design); real-model CUDA path untested without Phi-4.
- [x] Proxy exclusion enforced end-to-end: `generate_report.py` now excludes `metadata.proxy=true` rows AND suite-level `is_proxy` from scoring, alongside existing stub/dummy exclusion.
- [x] Driver artifact hygiene: `run_no_training.py` now deletes known result stems before running (prevents stale artifact inflation).

### Latest Report Snapshot (Baseline)

File: `PyQuifer/tests/benchmarks/results/BENCHMARK_REPORT.md` (Generated: 2026-02-09 00:39:19)

- Mean weighted score: 0.8954
- Scored scenarios: 23
- Cat 8/9/10/11 scored scenarios: 0 (excluded)
- Real-time latency highlight: `bridge.step()` p50 ~6.65ms on CPU; "interactive" p50 ~2.71ms (target 2ms)
- Real-time latency highlight: `bridge.step()` p50 ~29.48ms on CUDA (kernel-launch dominated)
- Real-time stepping highlight: per-token overhead p50 drops to ~0.33ms at step_every_16 (still ~3.15x vs baseline micro-op loop)

### What The Current Report Can and Cannot Say (Important)

- The current report does **not** establish anything about "hallucinations" in the usual LLM sense (factual errors in generated text), because:
  - Cat 8/9/10 suites are still stubbed/dummy and excluded from scoring.
  - Cat 1 ("LLM A/B") can run on a `DummyLLM` when `PYQUIFER_LLM_MODEL` is not set. In that mode, metrics like `logit_entropy`/`logit_diff` only validate modulation wiring and cost, not truthfulness/factuality.
- The current report **does** establish:
  - the bridge/cycle overhead is currently far too high for an "alive" streaming agent (Cat 11),
  - and that many PyQuifer-internal components function and can be regression-tested (Cats 2-7).

Score-composition caveat:
- The headline aggregate score (0.8958) is primarily a regression indicator over PyQuifer-owned tasks (chess/strategy, internal training probes, robustness checks).
- It should not be interpreted as “competitive” with external LLM leaderboards until Cat 8/9/10 are real (non-stub) and run against a real model + official data/prompt settings.

### “Hallucination” Measurement Status (As of 2026-02-09)

- Measured today: modulation effects on *synthetic logits distributions* (Cat 1), and PyQuifer-internal task scores (Cats 2-7).
- Not measured yet (requires real model + real eval pipeline):
  - factual QA truthfulness (TruthfulQA-style, FACTS/LongFact-style)
  - long-form factuality (FActScore/LongFact + retrieval + claim verification)
  - jailbreak success / refusal correctness (HarmBench/JailbreakBench official harnesses)


## Product Targets ("Neuro-sama" / "Her"-Style Real-Time Agent)

These targets are intended to make the system feel like a present, interruptible, real-time conversational partner, not a batch chatbot.

### Human Turn-Taking Reference

- Humans typically leave very small gaps between turns in conversation, on the order of ~200 ms on average. If your agent regularly takes >1 s to start responding, it will feel "slow" and non-interactive.
  - Background reading: https://www.scientificamerican.com/article/universal-social-rules-underlie-languages/ (turn gaps ~200 ms)
  - Academic reference: https://link.springer.com/article/10.3758/s13423-017-1363-z (turn-taking discussion, cites Stivers et al. 2009)
  - Peer-reviewed primary: Stivers et al. (2009) PNAS (open access via PMC): https://pmc.ncbi.nlm.nih.gov/articles/PMC2705608/
  - DOI record: https://doi.org/10.1073/pnas.0903616106

### Text Chat (Streaming)

Primary metrics:
- TTFT (time to first token): how long until the first token appears.
- ITL (inter-token latency): time between subsequent tokens.

Targets (good interactive feel):
- TTFT p50: 150 to 300 ms, TTFT p95: <800 ms.
  - Practical starting ranges and why TTFT matters: https://fieldguidetoai.com/guides/cost-and-latency/
  - Metric definitions: https://docs.nvidia.com/nim/benchmarking/llm/1.0.0/metrics.html
- ITL p50: <40 ms (>=25 tok/s), ITL p50: <20 ms (>=50 tok/s) for "snappy" streaming.
  - Metric definitions and profiling tooling: https://docs.nvidia.com/nim/large-language-models/1.0.0/optimization.html
  - Example measurement output showing TTFT/ITL/throughput in practice: https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2550/user-guide/docs/perf_analyzer/genai-perf/docs/tutorial.html

### Voice Agent (Speech In, Speech Out)

A useful baseline target is sub 1.5 s "mouth-to-ear" (user stops speaking to user hearing first audio), with sub 1.1 s as a common initial launch goal for straightforward cascaded pipelines.
- Reference ranges and component budgets (STT, LLM TTFT, TTS): https://www.twilio.com/en-us/blog/developers/best-practices/guide-core-latency-ai-voice-agents

Targets (initial launch goals, not best-in-class):
- Endpointing + turn detection: 150 to 300 ms
- STT: 350 to 500 ms
- LLM TTFT: 375 to 750 ms
- TTS first-byte: 100 to 250 ms

Aspirational ("feels alive"): aim to consistently beat the above with aggressive streaming, early partials, and minimized round-trips.

### PyQuifer-Specific Performance Budgets

To keep PyQuifer modulation feeling "alive" during streaming generation:
- Budget per generation step:
  - `bridge.step(...)` p50: <=2 ms on CPU, <=1 ms on GPU for interactive mode.
  - `bridge.modulate_logits(...)` p50: <=0.1 ms (should be tiny).
  - Hidden-state hooks (`modulate_hidden`) should be optional; if enabled, they must not push ITL over target.
- Step frequency:
  - Default: run `bridge.step(...)` once per 8 to 16 tokens and interpolate modulation between steps.
  - Experiments: per-token stepping is allowed for research, but it will not meet real-time budgets unless heavily optimized.

Implication: if `bridge.step()` is > model ITL, the "brain" will dominate latency and you will not get the Neuro-sama/Her feel.

### Benchmarks to Add (So You Can Enforce These Targets)

- In the benchmark JSON schema, record p50/p95/p99 for:
  - TTFT and ITL (when a real LLM is configured)
  - `bridge.step()` latency distribution
  - end-to-end per-token overhead ratio: (C ITL) / (B ITL)
- Add a hard rule to reporting:
  - real-time targets are checked only on non-stub, non-dummy results.

### “Feels Alive” Product Targets (Beyond Raw Benchmarks)

Benchmarks alone will not produce a Neuro-sama/Her-style agent. Use them as gates for system properties:

- Interruptibility:
  - user can barge-in mid-sentence and the agent adapts within <300 ms (voice) / <1-2 tokens (text).
  - required infra: streaming STT, streaming generation, streaming TTS, cancellation semantics.
- Persistent identity + memory:
  - consistent persona preferences across sessions without “random drift”.
  - required infra: memory store, retrieval policy, and explicit “what is grounded” signals.
- Tool use and environment grounding:
  - agent can cite sources, browse, or interact with a simulated world when needed.
  - benchmarks: retrieval/QA suites (BEIR-style), tool-use evals, and “groundedness” checks.
- Safety and factuality:
  - refusal correctness and low hallucination rate must be measured with official pipelines (Cat 9/10 real runs).

Pragmatic principle: treat latency + correctness + groundedness as first-class “product SLOs”, not optional metrics.


## Top Risks (Fix First)

1. Synthetic numbers being scored
- Do not emit `accuracy` (or any quality metric used by scoring) for B/C when:
  - framework is missing
  - model is missing
  - data is missing
  - wrapper is intentionally a stub
- If you must emit output for wiring verification, emit:
  - `status: stub|dummy|skipped`
  - `reason: ...`
  - metrics under a non-scored namespace (example: `pipeline_ok: 1`)
Status: DONE — framework stubs emit no B/C; safety/factuality proxies (when present) must be marked `metadata.proxy=true`.
Status note: scoring-side proxy exclusion must be enforced (do not rely on convention). Today Cat 9/10 are stubbed/dummy so the risk is latent, but it will reappear once safety/factuality suites emit B/C rows.

2. Report generator scoring stub/dummy
- `generate_report.py` must explicitly ignore any result row where metadata indicates:
  - `stub: true`
  - `dummy: true`
  - `status in {"stub", "dummy", "skipped"}`
- Also ignore an entire JSON file if suite-level metadata indicates stub/dummy.
Status: done (suite-level metadata `is_stub`/`is_dummy` and per-row `stub`/`dummy` are excluded).

3. Bench drivers reporting stubs as "passed"
- `run_no_training.py` (and similar) should treat stub/dummy/skipped as SKIPPED (timing=0) not PASSED.
- Prefer a unified return structure: `{"status": "ok|skipped|failed", ...}`.
Status: done (`run_no_training.py` detects `{status: skipped}` / `{stub: true}` and records timing=0).

4. Reports including stale result artifacts ("hallucinated coverage")
- `generate_report.py` renders *all* `tests/benchmarks/results/*.json` it finds.
- Fix applied: `run_no_training.py` now deletes all known result stems (`_KNOWN_STEMS` list) before running, preventing stale artifacts from prior runs inflating report coverage.
- Status: DONE (verified 2026-02-09: "Cleaned 21 stale result files from results/" on fresh run).

5. No-training driver still "runs" model-required suites in no-model mode
- Fix applied: All 6 safety/factuality suites (`bench_harmbench`, `bench_jailbreakbench`, `bench_toxigen`, `bench_factscore`, `bench_factbench`, `bench_factreasoner`) now return `{"status": "skipped", "reason": "no real model"}` when `PYQUIFER_LLM_MODEL` is unset.
- `run_no_training.py`'s `_is_stub_result()` detects `{status: skipped}` / `{stub: true}` and records timing=0 (SKIPPED), not PASSED.
- Proxy B/C metrics still exist in several safety/factuality wrappers (they are marked `metadata.proxy=true` and therefore excluded from scoring, but they can still look like “real” numbers in tables).
- Status: PARTIAL (driver stub-handling is correct; proxy metric presentation is still misleading).

## Real-Time Latency: Root Cause Hypotheses (Evidence-Based)

Latest measurements (Cat 11 in `PyQuifer/tests/benchmarks/results/BENCHMARK_REPORT.md`, generated 2026-02-09 00:39:19):
- `bridge.step()` CPU p50: 6.65 ms, p95: 8.85 ms.
- `bridge.step()` interactive p50: 2.71 ms, p95: 5.07 ms (target: 2.00 ms; close).
- `bridge.step()` CUDA p50: 29.48 ms, p95: 33.42 ms (do not treat this as a product path unless you fuse/compile heavily).
- Per-token overhead (microbench): step_every_16 p50: 0.334 ms, ratio: 3.15 (still too high for per-token stepping; stepping every N tokens is required).
- `modulate_hidden()` p50: ~0.20-0.23 ms (acceptable if optional; still significant overhead ratio because vanilla is ~0.02-0.03 ms).
- `modulate_logits()` p50: ~0.14-0.17 ms (still above the 0.1 ms target).

High-confidence causes inside `CognitiveCycle.tick()` (`PyQuifer/src/pyquifer/integration.py`), still consistent with the improved numbers:
- Many `.item()` calls in the hot path (examples: coherence, criticality_distance, neuromodulator scalars, dominance_ratio, etc.). If the cycle ever runs on GPU, `.item()` will force device sync and destroy ITL.
- The tick does a large amount of work per token, including predictive coding, precision weighting, dominance detection, metastability, neural darwinism, memory subsystems, and optional workspaces.
- The bridge explicitly avoids `torch.no_grad()` around the cycle tick because HPC uses autograd internally. That is correct for training research, but it is a non-starter for real-time streaming inference unless there is a dedicated "interactive fast path".

Reconciliation note (RESOLVED for the CPU path):
- The report now breaks out CPU vs CUDA explicitly in Cat 11.
- In the same report, Cat 1 shows `bridge.step()` p50 ~6.28 ms and Cat 11 shows CPU p50 ~6.65 ms (within ~6%), so the CPU measurement is consistent across benches.
- CUDA remains much slower for `tick()` due to many small ops; CPU is the sensible product path for `bridge.step()`.

Fix direction (architectural, not a micro-optimization):
- Add an explicit "interactive mode" for `bridge.step()` / `CognitiveCycle.tick()` that:
  - disables autograd and learning updates (or amortizes them heavily in background ticks),
  - avoids `.item()` and Python-side branching in the per-token hot path,
  - optionally runs the full heavy cycle asynchronously at a lower rate (ex: 2-10 Hz) and interpolates modulation for the token loop.


## Baseline Hygiene

1. Published baselines must match settings
- For each A_published number store, at minimum:
  - dataset split
  - few-shot / 0-shot
  - prompt template
  - decoding settings
  - metric definition
  - exact model name + revision
- If settings are not aligned, label baseline as "not comparable" and do not place it in the same table.

2. Separate capability vs infrastructure tests
- Infrastructure tests: latency, memory, pipeline wiring.
- Capability tests: accuracy/pass@k/refusal rate/etc.
- Never mix these in a single "score".

## PyQuifer Integration Contract (What Adapters Must Respect)

The stable public API is:
- `PyQuiferBridge.step(sensory_input) -> ModulationState`
- `PyQuiferBridge.modulate_logits(logits, state) -> logits`
- `PyQuiferBridge.modulate_hidden(hidden_states, state) -> hidden_states`

Adapter requirements:
- Sensory input: project to `CycleConfig.state_dim` (default 64). Using mean token embedding is acceptable.
- Device/dtype: keep bridge state and model tensors on the same device to avoid CUDA/CPU mismatches.
- Gradient policy: do not backprop from LLM into oscillators. (Trait vectors and modulation gain may be trainable; decide intentionally.)


## PyQuifer Source Fix List (Bridge + Cycle)

This section is about fixes and hardening in `PyQuifer/src/pyquifer/` to support truthful benchmarking and stable external integrations.

### Bridge Device and Dtype Safety (`PyQuifer/src/pyquifer/bridge.py`)

- Ensure `ModulationState` tensors used inside `modulate_hidden()` are on `hidden_states.device`.
  - Today: `phases` and `neuromodulator_levels` are moved to `hidden_states.device` inside `modulate_hidden()` via `.to(device)`, which avoids CPU/CUDA mismatches in the coupling math.
  - Remaining hardening: ensure all scalars/params (`modulation_gain`, facet-weight tensors) are on the same device/dtype without re-allocating per token.
- Make `modulate_logits()` explicitly support both `(batch, vocab)` and `(batch, seq, vocab)` logits.
  - Current implementation uses `mean(dim=-1)` and should already work, but adapters will inevitably feed both shapes. Lock the contract down in docstring and add lightweight asserts.
- Add a small helper to standardize sensory inputs:
  - `bridge.prepare_sensory_input(x, device, dtype) -> (state_dim,)`
  - Goal: every adapter uses the same projection and device/dtype normalization.
  - Status: implemented in `PyQuifer/src/pyquifer/bridge.py` (adapters should route all sensory signals through it).

### Bridge Training Policy (`PyQuifer/src/pyquifer/bridge.py`)

- Decide and document whether `trait_vectors` and `modulation_gain` are intended to be trainable.
  - The code correctly severs gradient flow from LLM into oscillators, but `trait_vectors` and `modulation_gain` are `nn.Parameter`s.
  - Benchmarking implications:
    - If those are trainable and you train them, you are no longer evaluating "pure modulation". That may be fine, but it must be stated and separated from no-training runs.

### CognitiveCycle Performance and Sync Points (`PyQuifer/src/pyquifer/integration.py`)

- Audit `.item()` usage inside `tick()`.
  - `.item()` can cause device synchronization on GPU. In high-frequency generation loops, repeated `.item()` can dominate latency.
  - Fix direction: keep tensors as tensors in the internal loop and only convert to Python scalars at report boundaries (or batch scalar extraction).
- Ensure diagnostics tensors are explicitly documented as CPU or device-resident.
  - For adapters that use `diagnostics['phases']` and `diagnostics['neuromodulator_levels']`, it is simplest if they are either:
    - always on the same device as the cycle, or
    - always moved by bridge methods to the model device.

### Integration Surface for HF/Framework Adapters

- Provide one canonical way to extract sensory signals from a transformer:
  - Mean input embedding is stable across model families.
  - Hidden-state hooks are fragile; keep them optional and make failures non-fatal.
- Provide one canonical way to apply logit modulation during generation:
  - Prefer step-wise modulation via a logits-processor/warper concept (framework-specific) rather than rewriting generation loops.

### Test Targets (PyQuifer-Owned)

- Add unit tests for:
  - `modulate_logits()` shape support: `(B,V)` and `(B,T,V)`.
  - `modulate_hidden()` device safety: state tensors on CPU, hidden on CUDA (skip if no CUDA).
  - determinism: fixed `sensory_input` and fixed seed produce identical modulation deltas.
- Add a "no-stub scoring" test at the report layer: any row with `metadata.stub` or `metadata.dummy` must not contribute to a score.


## Framework Integrations (Recommended Order)

### Tier 1: lm-evaluation-harness (highest leverage)

Why: one harness covers many standard academic tasks and has strong ecosystem support.
- Repo: EleutherAI/lm-evaluation-harness

Audit checklist:
- Ensure `pyquifer` model registration is stable across harness upgrades.
- Prefer a minimal integration:
  - logits modulation during scoring
  - generation parameters (temperature/top_p/repetition_penalty) where supported
  - hidden hooks only after logits-level integration is verified
- Ensure caching/output semantics:
  - use output_path, use_cache
  - limit/batch_size are logged in metadata

Things to verify:
- Your adapter returns logits in the shape lm-eval expects.
- Tasks: start with small set (gsm8k, truthfulqa_mc2, hellaswag, arc_challenge, bbh) and validate correctness.

### Tier 2: HELM / OpenCompass / OpenAI evals

These are valuable but should be treated as optional until you have Tier 1 stable.

HELM
- Repo: stanford-crfm/helm
- Risk: heavy framework assumptions; multi-provider API surface.

OpenCompass
- Repo: open-compass/opencompass
- Risk: dataset packaging/config conventions; external tooling.

OpenAI evals
- Repo: openai/evals
- Risk: framework churn; API-key-driven evaluation.

Rule: if not fully integrated, wrappers must be SKIPPED without emitting B/C accuracy.

### Tier 3: SWE-bench (agent + docker harness)

- Repo: SWE-bench/SWE-bench
- Fact: requires containerized evaluation; it is not comparable to plain text scoring.

Rule: keep as explicit skip until you have:
- a patch-generation agent
- docker orchestration
- evaluation harness integration

## Safety and Factuality Suites (Do Not Fake Results)

### HarmBench
- Repo: centerforaisafety/HarmBench

Checklist:
- Use the official prompts and evaluation pipeline if possible.
- If you only implement a keyword-based refusal detector, label it clearly as a proxy and do not compare to published baselines.

### JailbreakBench
- Repo: JailbreakBench/jailbreakbench

Checklist:
- Use their dataset loader and evaluation helpers.
- Avoid random "attack success" rates.

### ToxiGen
- Repo: microsoft/TOXIGEN

Checklist:
- Decide whether you are evaluating:
  - a toxicity classifier
  - generative toxicity (requires a separate judge/classifier)
- Avoid "safety = 1 - random".

### FActScore
- Repo: shmsw25/FActScore

Checklist:
- FActScore is not a simple single-model metric; it typically uses retrieval + verification.
- If dependencies are missing, SKIP. Do not emit synthetic factuality scores.

### FactBench
- Repo: launchnlp/FactBench

Checklist:
- VERIFY pipeline requires retrieval and categorization (supported/unsupported/undecidable).
- If you are not running the pipeline, SKIP.

### FactReasoner
- Repo: IBM/FactReasoner (if used)

Checklist:
- Confirm what the repo actually provides (tasks, scoring code) before claiming a metric.

## Correctness and Reproducibility Checklist (All Benches)

1. Determinism controls
- Explicit seed handling (torch, numpy, random).
- Log seeds per scenario.

2. Environment capture
- Torch version, CUDA, device, dtype.
- Model name, revision, quantization mode.
- Dataset version/hash.

3. Cost controls
- Always support: `--limit` or env var limit.
- Always support: batch_size control.

4. Data/download hygiene
- Do not download huge datasets by default.
- If a dataset is missing, SKIP with reason.

5. Security
- If a benchmark can execute untrusted code (e.g., code-generation eval), default to SKIP unless sandboxed.

6. Output schema
- Standardize JSON fields:
  - suite name
  - scenario name
  - for each result row: column, metrics, metadata
- Add suite-level metadata:
  - `status`, `is_stub`, `is_dummy`

## Reporting and Scoring

1. Scoring should only use real runs
- If `metadata.stub` or `metadata.dummy` then exclude.

2. Separate tables
- Capability table: accuracy/pass@k/refusal rate
- Infrastructure table: latency/memory/overhead

3. Prevent misleading comparisons
- Never compare a proxy metric to an official leaderboard number.
- Mark proxies and keep them out of global scoring.

## Suggested Roadmap

1. Make stubs safe
- [x] Remove B/C accuracy from stub/dummy wrappers.
- [x] Exclude stub/dummy from scoring.

2. Make lm-eval real
- [x] Ensure `bench_lm_eval.py` produces only real numbers when a real model is configured and lm-eval runs.
- [x] In dummy/no-model mode: record only A_published (no synthetic B/C).

3. Make LLM A/B real
- [~] Update `bench_llm_ab.py` so the "real HF" path:
  - [x] uses `outputs.logits`
  - [x] uses embeddings as sensory signal fallback
  - [~] covers all scenarios (pipeline, consistency, latency) with the real model when configured (code paths exist, untested without Phi-4)
  - [x] avoids CPU round-trips for modulation when running on CUDA (removed `.cpu()` in C_rand, `_apply_random_modulation` is device-aware)

4. Add safety/factuality suites only when real
- [~] Keep proxy implementations explicitly excluded from scoring until real pipelines implemented.
- [x] Ensure `generate_report.py` excludes `metadata.proxy=true` rows (and/or suite-level `is_proxy`) from scoring.

5. Hit real-time budgets ("feels alive")
- [~] Reduce `bridge.step()` p50 toward <=2ms CPU / <=1ms GPU for interactive mode, or step less frequently by default (8-16 tokens).
  - Progress: `CognitiveCycle.tick()` optimized from 10.47ms → 3.1ms (3.4x speedup) via:
    - Vectorized SR `_measure_snr()` (10 trials → 3 vectorized, eliminated 3rd probe call)
    - Vectorized EpistemicValue per-dimension loops (scatter indexing)
    - Vectorized + cached Symbiogenesis MI estimation (compute_every=10)
    - Hierarchical timestepping (CycleConfig.interactive(): HPC/2, SR/3, arena/3, motivation/5, self-model/5)
    - HPC iterations reduced (3→1 in interactive mode)
    - Dominance TE compute_every increased (10→50)
    - Fast-path dephasing converted from Python if/else to tensor ops (no .item() sync)
    - Batched .item() extraction at diagnostics boundary
  - Remaining gap to 2ms target: ~1.1ms, primarily from Python dict overhead (19.7%) and evenly distributed module costs
  - Next steps: torch.compile/CUDA graphs, or async tick at 2-10 Hz with interpolation
- [ ] Ensure real-world ITL/TTFT measurement is against a real LLM generation loop, not only synthetic micro-op baselines.
- [~] Remove/gate GPU sync cliffs inside `CognitiveCycle.tick()` (especially `.item()` in hot paths) when running on GPU.
  - Progress: fast-path dephasing converted to tensor ops, return-dict .item() calls batched to single block, mid-loop .item() retained for Python control flow

## Additional Industry-Standard Benchmarks to Consider (Validated Online)

These are widely used in 2024-2026 LLM evaluation and are good targets for Tier 1 integration (via lm-eval / OpenCompass / HELM) rather than bespoke wrappers.

Long-context / “long-form”:
- LongBench v2 (long-context eval): https://longbench2.github.io/
- RULER (long-context, stress tests): https://github.com/hsiehjackson/RULER

Instruction-following / preference proxies:
- AlpacaEval (instruction-following, win-rate): https://github.com/tatsu-lab/alpaca_eval
- Arena-Hard (Arena-style hard prompts; automated): https://github.com/lm-sys/arena-hard-auto

Code / programming:
- LiveCodeBench (time-contaminated coding eval): https://github.com/LiveCodeBench/LiveCodeBench

Multimodal / video diagnostics (only if/when PyQuifer supports VLMs):
- Perception Test (video diagnostic): https://github.com/google-deepmind/perception_test
- Physics-IQ (video understanding): https://github.com/google-deepmind/physics-iq-benchmark

### Unverified Leads (Need Manual Confirmation)

The following names were mentioned during planning, but were not found quickly as canonical, widely-cited benchmark repos under those exact names. Treat these as “to verify” rather than “to clone”:
- “FACTS Benchmark Suite” (December 2025)
- “CodeSemBench”
- “Long-Form Factuality” (generic category name; concrete candidates are LongBench/LongBench2 + FActScore/LongFact-style pipelines)

## Reference Repos (Source of Truth)

- lm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness
- HELM: https://github.com/stanford-crfm/helm
- OpenCompass: https://github.com/open-compass/opencompass
- SWE-bench: https://github.com/SWE-bench/SWE-bench
- HarmBench: https://github.com/centerforaisafety/HarmBench
- JailbreakBench: https://github.com/JailbreakBench/jailbreakbench
- ToxiGen: https://github.com/microsoft/TOXIGEN
- FActScore: https://github.com/shmsw25/FActScore
- FactBench: https://github.com/launchnlp/FactBench



## Reference Docs (Implementation Details)

- PyTorch forward hooks: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook
- HuggingFace Transformers model outputs (`ModelOutput`, `.logits`, optional `hidden_states`): https://huggingface.co/docs/transformers/main/en/main_classes/output
- HuggingFace generation and logits processors/warpers: https://huggingface.co/docs/transformers/main/en/main_classes/text_generation
- lm-evaluation-harness repository (model interface changes land here first): https://github.com/EleutherAI/lm-evaluation-harness
