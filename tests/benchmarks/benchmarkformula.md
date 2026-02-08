# PyQuifer Benchmark Formula (Working Guide)

This document defines a **benchmarking methodology** for PyQuifer, and a broader
"universal" AI benchmarking checklist so results remain comparable to
state-of-the-art (SOTA) evaluation practice.

Goal: produce results that are (1) meaningful to users, (2) comparable to baselines,
and (3) robust against common benchmark failure modes (selection bias, hardware bias,
"warm-cache" illusions, cherry-picked metrics).

Scope constraint: this is a **benchmarking guide**. It does not claim SOTA numbers by itself.
When you incorporate external benchmark results (online), treat them as references until
they are reproduced under the same settings.

---

## 0) Project Understanding (What PyQuifer Is)

PyQuifer is a PyTorch library for **dynamical/oscillatory computation** and a multi-module
"cognitive cycle" (`pyquifer.integration.CognitiveCycle`) plus an external bridge
(`pyquifer.bridge.PyQuiferBridge`) intended to modulate LLM sampling/hidden states.

Benchmarking PyQuifer needs to cover:

1. **Microbenchmarks**: single-module performance and scaling
2. **System benchmarks**: end-to-end `CognitiveCycle.tick()` performance and stability
3. **Application benchmarks**: realistic integration usage (bridge into a transformer loop)

---

## 0.1) Universal AI Benchmarking: What "Good" Looks Like

Most benchmark controversies come from measuring the wrong thing (or measuring the
right thing in a biased way). A modern AI benchmark suite generally tries to cover:

1. Capability: accuracy / strength / task success
2. Generalization: transfer, distribution shift, adversarial cases
3. Calibration: confidence matches correctness
4. Robustness: noise, perturbations, out-of-distribution
5. Efficiency: latency, compute, memory, energy
6. Safety/harms: toxicity, bias, jailbreak resistance (if user-facing)
7. Reproducibility: stable methodology and reporting

This guide treats chess/game benchmarks as a fun, rigorous application domain and
also provides a generic evaluation skeleton applicable to other tasks.

## 0.2) How PyQuifer Fits "General AI" Evaluation

PyQuifer is not (by itself) a foundation model. To evaluate it "against general AI stuff",
you typically benchmark it as a **component** that changes the behavior of another system.

Common PyQuifer roles:

- **Controller**: `CognitiveCycle.tick()` produces modulation signals over time.
- **Adapter/bridge**: `PyQuiferBridge` modifies sampling parameters and/or hidden states.
- **Diagnostics**: criticality / metastability / IIT-like metrics are computed to characterize
  internal dynamics.

Therefore, the right question is usually:

"When PyQuifer is integrated into an existing baseline system, does it improve capability,
robustness, calibration, or efficiency under a fixed budget?"

This document defines A/B-style benchmarking patterns to answer that cleanly.

## 1) Baselines and Comparators

To avoid baseline bias, every benchmark scenario must declare a baseline of the same
functional scope.

### 1.1 Required baselines (always include)

- **Plain PyTorch reference**: the simplest correct implementation using tensor ops.
- **Optimized PyTorch reference**: same math, but using best-practice optimizations:
  - vectorization, avoiding Python loops
  - `torch.compile` (where applicable)
  - AMP (`torch.autocast`) where numerically valid

### 1.2 Recommended external comparators (only when apples-to-apples exists)

Use external systems only when the benchmark semantics match:

- ODE / continuous-time: `torchdiffeq`, `torchdyn`
- Spiking: `norse`, `snntorch`
- Neuromorphic: `lava` (if matched)

If the external tool does not match semantics, use it only as qualitative context.

---

## 1.3) "State of the art" benchmark suites to align with (general AI)

This document intentionally does not copy external numbers. Instead, it names the
benchmark ecosystems whose methodology is widely accepted, so PyQuifer results can
be mapped onto them.

Language/reasoning (LLMs):

- BIG-bench / BIG-bench Hard (BBH)
- MMLU / MMLU-Pro style multi-domain tests
- ARC(-like) generalization benchmarks
- Truthfulness/hallucination: TruthfulQA-style evaluations
- Instruction-following: MT-Bench, Arena-style Elo (pairwise preference)
- Tool-use: agentic benchmarks (e.g., SWE-bench for coding agents)

Performance methodology:

- MLPerf (training and inference rules)
- TorchBench (PyTorch-level performance regression methodology)

Game/RL style:

- OpenSpiel / standard Elo/TrueSkill evaluation patterns
- SPRT match methodology (used in chess engine testing)

## 2) Benchmark Suite Design (What to Measure)

Benchmarks should be reported as a **dashboard**. A single scalar score is optional.

### 2.1 Performance metrics (hardware-dependent)

Always report distributions, not just means:

- **Throughput**: steps/sec or samples/sec
- **Latency**: p50 / p95 / p99 per step
- **Compute** (recommended):
  - FLOPs/step (estimated)
  - achieved TFLOPs and % of theoretical peak (roofline-style)
  - bytes moved / step (approx) and memory-bandwidth utilization
- **Memory (GPU)**:
  - peak allocated (`torch.cuda.max_memory_allocated()`)
  - peak reserved (`torch.cuda.max_memory_reserved()`)
- **Compilation overhead**: first-run vs steady-state (if using `torch.compile`)

Optional (when instrumentation exists):

- **Energy**: joules per step / sample (external power measurement)
- **Utilization**: GPU SM utilization, memory bandwidth utilization (external tooling)

### 2.2 Correctness / functional metrics (hardware-agnostic)

Benchmarks must not reward incorrect shortcuts.

- **Numerical correctness**: compare outputs to a reference within tolerance
- **Determinism envelope**: fixed-seed runs match within tolerance
- **Training stability** (if training included): grad norms, NaNs/Infs, loss monotonicity

Additional general AI metrics (when the benchmark is task-level rather than module-level):

- **Accuracy / success rate** on a fixed dataset
- **Pass@k** for generative tasks where multiple attempts are allowed
- **Pairwise win-rate / preference rate** vs baselines (convertible to Elo)
- **Calibration**:
  - Expected Calibration Error (ECE)
  - Brier score / negative log-likelihood where probabilities exist
- **Robustness under perturbation**:
  - noise, paraphrases, adversarial distractors, counterfactual edits
- **Fairness / subgroup** reporting (if humans are in the loop or data has subgroups)

### 2.3 Dynamics-specific metrics (PyQuifer-specific)

Choose metrics that map to PyQuifer's intended dynamics:

- **Synchronization**: Kuramoto order parameter `R(t)`
  - summary: mean `R`, peak `R`, time-to-threshold (e.g. ticks until `R>=0.8`)
- **Metastability**: metastability index, chimera index (if enabled)
- **Criticality**: branching ratio estimate, distance to target criticality
- **Information flow**: transfer entropy / dominance ratio (when computed)
- **Integration / complexity**: IIT/Phi approximations as diagnostics (compute sparsely)

Important: many of these are expensive. Define a **measurement schedule**
(e.g., compute every N ticks) so results reflect realistic usage.

---

## 2.5) "PyQuifer vs General AI" (A/B Integration Benchmarks)

If you want to evaluate PyQuifer as a general-purpose improvement to an LLM/agent,
define an A/B harness where the *only* difference is whether PyQuifer is active.

### 2.5.1 A/B conditions

Minimum:

- **A (baseline)**: the model/agent with fixed decoding + fixed prompts/tools.
- **B (PyQuifer)**: same model/agent, but use PyQuifer to modulate generation.

Recommended additional conditions:

- **B1 (light)**: only temperature/top_p/repetition modulation via `modulate_logits()`.
- **B2 (deep)**: hidden-state injection via `modulate_hidden()`.
- **B3 (control)**: random modulation with the same amplitude budget (tests placebo effects).

### 2.5.2 What counts as an improvement

Define a primary metric and a budget.

Examples:

- "Improve success rate at the same latency"
- "Improve calibration (ECE) at the same accuracy"
- "Improve robustness under paraphrases without increasing token usage"

Avoid vague claims like "smarter" without a measurable definition.

### 2.5.3 Task families (general AI)

Pick tasks that cover multiple failure modes:

- **Reasoning**: math, logic, long-context consistency
- **Knowledge**: multi-domain QA
- **Tool use**: function calling / retrieval / coding
- **Planning/control**: multi-step decision making
- **Robustness**: paraphrase sets, adversarial distractors, noisy inputs
- **Calibration**: confidence and probability accuracy

### 2.5.4 Report extra costs

When you add PyQuifer, always report overhead:

- added latency per token or per step
- extra memory
- extra GPU/CPU utilization

This prevents "improvements" that are just spending more compute.

---

## 2.4) Chess / Game Benchmarking (Domain Example)

Chess is a great benchmarking domain because it has:

- clear ground-truth evaluation for many subproblems (tablebases)
- strong open baselines (Stockfish, Leela Chess Zero)
- a mature evaluation methodology (SPRT, Elo, fixed time controls)

If you benchmark PyQuifer on chess-like tasks, separate the benchmark into layers:

### A) Position-level evaluation (static)

- **Move quality**: compare chosen move vs engine PV at a fixed depth/time
  - metrics: top-1 match rate, centipawn loss distribution, blunder rate
- **Outcome classification** (win/draw/loss) where tablebases apply
  - use Syzygy-like tablebases for endgames when possible
- **Specialized sets** (fun + diagnostic):
  - fortress detection
  - zugzwang / stalemate traps
  - tactical puzzles (forks, pins, mates)

### B) Engine-level strength evaluation (dynamic)

- **Elo / win-rate** vs baselines under fixed conditions
  - fixed time control or fixed nodes
  - fixed opening book
  - report draw rate separately
- **SPRT** (Sequential Probability Ratio Test) for efficient strength validation

### C) Efficiency

- **Time to move** distribution (p50/p95)
- **Nodes/sec** or "positions/sec" where relevant
- **Energy per move** (if measuring power)

### D) Bias and leakage in chess benchmarks

- Avoid cherry-picked positions (selection bias)
- Separate "seen" vs "unseen" sets (data contamination)
- Include adversarial near-fortresses to test false positives

## 3) Biases and Benchmark Failure Modes

Use this as a pre-publication checklist.

### 3.1 Selection bias

- Don't benchmark only "easy" regimes (e.g., trivially synchronized parameters).
- Use a parameter grid and report aggregate results.

### 3.2 Hardware bias

- Always report exact hardware + driver + torch version.
- Avoid cross-hardware claims unless normalized or reproduced.

### 3.3 Warmup / caching illusions

- Report **cold start** and **steady state** separately.
- Include warmup iterations and state them explicitly.

### 3.4 Metric gaming

- Throughput-only benchmarks can hide correctness regressions.
- Always pair performance with correctness checks.

### 3.6 Data contamination and evaluation leakage (general AI)

For any benchmark derived from public corpora (or common puzzle sets), treat leakage as a
first-class risk:

- Record dataset provenance and hash
- Prefer newly-generated or procedurally-generated test cases when feasible
- For LLM-style evaluations, avoid including benchmark items in prompts or few-shot examples

### 3.7 Prompt sensitivity / harness bias (LLMs)

LLM results can vary dramatically based on formatting and instructions.

- Use a fixed evaluation harness
- Report variance across prompt templates if using prompting
- Keep system prompts, tool schemas, and decoding params pinned

### 3.8 Budget mismatch (compute laundering)

The easiest way to accidentally cheat is to give the PyQuifer condition more compute.
Treat "budget" as a first-class constraint:

- cap total wall time
- cap tokens generated
- cap FLOPs (if you can estimate)
- cap memory

If PyQuifer uses more compute, report that explicitly and treat it as a tradeoff.

### 3.5 Reproducibility gaps

- Run multiple seeds and report mean +/- CI (bootstrap CI preferred).
- Save configs, commit hash, and exact command lines.

---

## 4) The Benchmark Formula (How to Score)

Prefer dashboards. If you must publish a single score, use an explicit multi-objective formula.

### 4.1 Normalized sub-scores

For each scenario `s`, compute normalized values relative to a baseline `b`:

- **Speedup**: `S_speed(s) = throughput_pyquifer / throughput_baseline`
- **Latency improvement**: `S_lat(s) = p95_baseline / p95_pyquifer`
- **Memory efficiency**: `S_mem(s) = peak_mem_baseline / peak_mem_pyquifer`

Clamp each to a max (e.g., 10x) to avoid one outlier dominating.

### 4.2 Robustness / correctness gates

Define binary gates (fail => scenario score = 0):

- correctness within tolerance
- no NaNs/infs in outputs (and grads if training)
- determinism envelope passes (within tolerance)

### 4.3 Weighted geometric mean (recommended)

Geometric mean discourages "optimize one metric to infinity".

```
Score(s) = gate(s) * exp(
  w_speed * ln(S_speed(s)) +
  w_lat   * ln(S_lat(s))   +
  w_mem   * ln(S_mem(s))
)
```

Suggested default weights:

- `w_speed = 0.4`
- `w_lat   = 0.4`
- `w_mem   = 0.2`

Aggregate across scenarios:

```
Overall = geometric_mean_s( Score(s) )
```

If you include task accuracy, incorporate it as a gate (minimum acceptable) and/or as
a multiplicative factor.

### 4.4 Task-level accuracy integration (optional)

If the scenario has an accuracy metric `A(s)` in [0, 1], one safe pattern is:

- Gate: require `A(s) >= A_min`
- Then include it: `Score(s) *= (A(s) / A_baseline(s))^w_acc`

This prevents meaningless speedups from dominating if the system is wrong.

### 4.5 Budget-aware scoring (recommended for A/B agent tests)

For integrated A/B agent benchmarks, consider explicitly adding a budget penalty.

Example:

- Let `A(s)` be task success in [0, 1]
- Let `C(s)` be added cost ratio, e.g. `latency_pyquifer / latency_baseline`

Then one simple family is:

```
Score(s) = gate(s) * A(s) / C(s)^lambda
```

Where `lambda` controls how harshly you penalize extra cost.

---

## 5) Benchmark Matrix (What Scenarios to Run)

Define a matrix where each row is a scenario and each column is a metric.
At minimum, include CPU and CUDA (if available).

### 5.1 Microbench scenarios

- Oscillator step (`LearnableKuramotoBank`)
- Multi-bank step (`FrequencyBank`)
- HPC step (`HierarchicalPredictiveCoding`)
- Criticality update (`CriticalityController`)

Parameters to sweep:

- number of oscillators (e.g., 16, 32, 64, 128, 256)
- topology (global vs sparse/learnable)
- dtype (`fp32`, `bf16` where valid)
- batch size (where applicable)

### 5.2 System scenarios

- `CognitiveCycle.tick()` steady-state:
  - ticks/sec at fixed `CycleConfig`
  - p95 latency per tick
  - peak memory
- Optional subsystems on/off via config flags (STP, mean-field, etc.)

### 5.3 Application scenarios

- `PyQuiferBridge.step()` in a loop (token-stream style)
- `modulate_logits()` on representative `(batch, vocab)` shapes
- `modulate_hidden()` on representative transformer hidden shapes

### 5.3.1 LLM/agent A/B scenarios (bridged)

If you have an actual agent loop, define standard scenarios:

- short-form QA (latency sensitive)
- long-form reasoning (robustness to drift)
- tool calls (correctness under schema constraints)
- adversarial prompts (safety/policy adherence if applicable)

For each, report success, calibration, and overhead.

---

## 5.4 Universal benchmark matrix (general AI)

If you want PyQuifer to be evaluated as a general AI component (not only a library),
add a task-level matrix. Example axes:

- Task type: classification, generation, planning, control
- Supervision: zero-shot, few-shot, finetuned
- Shift: in-distribution, out-of-distribution, adversarial
- Constraints: latency budget, memory budget, energy budget
- Output requirements: calibrated probabilities, explanations, tool traces

This makes it clear what PyQuifer improves (and what it does not).

## 6) Reporting Template

### Environment

- Repo commit:
- OS:
- CPU:
- GPU:
- CUDA / driver:
- PyTorch:
- Python:

### Scenario definition

- Scenario name:
- Module(s):
- Inputs shapes:
- Dtype:
- Seeds:
- Warmup:
- Measured iterations:

### Results

| Metric | PyQuifer | Baseline | Ratio |
|---|---:|---:|---:|
| throughput | | | |
| latency p50 | | | |
| latency p95 | | | |
| latency p99 | | | |
| peak memory | | | |

Correctness:

- max abs error vs reference:
- max rel error vs reference:
- determinism envelope:

---

## 7) Using Online Benchmarks Responsibly

If you compare against published benchmarks (TorchBench, MLPerf, library-specific repos),
treat them as **external references**:

- only quote numbers if you can match: hardware, dtype, batch, and scenario
- otherwise quote them as "context", not as a direct performance claim

Recommended workflow:

1. Use published benchmarks to pick representative workloads.
2. Reproduce locally with the same settings.
3. Publish command lines + configs so others can replicate.

---

## 8) Next Steps

1. Convert this guide into executable benchmark scripts under `tests/benchmarks/`.
2. Add JSON/CSV output and a plotting notebook.
3. Add CI smoke benchmarks (tiny sizes) to detect performance regressions.

***

## Appendix A: Suggested "State of the Art" Benchmark Sources to Review

When you do web research, look for:

- **MLPerf**: training/inference methodology norms
- **TorchBench**: PyTorch performance regression suites
- **DeepBench**: GEMM/convolution microbench patterns
- **NeuroBench**: neuromorphic/spiking benchmark suites (relevant to PyQuifer motifs)

And for general AI evaluation methodology:

- **HELM** (evaluation categories + reporting discipline)
- **EleutherAI LM Evaluation Harness** (reproducible LLM eval harness patterns)
- **Chatbot Arena / Elo**-style preference evaluation methodology

This appendix intentionally does not copy numbers; it is a guide for what to align with.

***

## Appendix B: Common "Gotchas" Checklist

- [ ] Separate cold-start vs steady-state
- [ ] Pin dtype and precision mode
- [ ] Pin seeds and run multiple seeds
- [ ] Disable/enable `torch.compile` explicitly
- [ ] Verify correctness vs reference
- [ ] Avoid measuring only a "happy path"

***

## Document status

This is a living document. When you add new benchmark scripts, update:

- scenario list
- baseline definitions
- scoring weights (if using a scalar score)
