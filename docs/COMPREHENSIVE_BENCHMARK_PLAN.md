# PyQuifer Comprehensive Benchmark Plan

**Three-Column Evaluation: Published AI Baselines vs. Plain PyTorch vs. PyQuifer**

**Last Updated:** 2026-02-07
**Status:** Design document — to be converted into executable benchmark scripts

---

## Philosophy

Every benchmark produces **three columns side by side**:

| Column | What | Why |
|--------|------|-----|
| **A: Published AI Baselines** | Frontier model scores (GPT-5.x, Claude 4.x, Gemini 3, Lc0, Stockfish) | External reference — "where does the world stand?" |
| **B: Plain PyTorch** | Same architecture, no PyQuifer — standard backprop/PyTorch ops | Internal control — isolates PyQuifer's contribution |
| **C: PyQuifer-Modulated** | Same architecture + PyQuifer CognitiveCycle/modules active | The experimental condition — "does PyQuifer help?" |

**Additional control condition** (recommended for A/B agent tests):
| **C-rand: Random Modulation** | Same amplitude budget as C, but random signals | Placebo control — catches "any perturbation helps" effects |

**Core rule:** If C doesn't beat B by a statistically significant margin AND C-rand doesn't explain the gap, the result is noise. Report it honestly.

---

## Scoring & Statistical Rigor

### Per-Scenario Score

```
Score(s) = Gate(s) * (A_accuracy(s) / A_baseline(s))^w_acc
                    * (throughput_C / throughput_B)^w_speed
                    * (latency_B_p95 / latency_C_p95)^w_lat
                    * (mem_B / mem_C)^w_mem
```

**Default weights:** w_acc=0.4, w_speed=0.2, w_lat=0.2, w_mem=0.2

**Gates (binary, fail = score 0):**
- Correctness within tolerance vs reference
- No NaN/Inf in outputs or gradients
- Determinism envelope passes (multi-seed)
- Budget constraint met (latency/memory/FLOPs cap)

### Aggregate Score

```
Overall = geometric_mean_across_scenarios(Score(s))
```

Geometric mean prevents one outlier from dominating.

### Statistical Requirements (per NIST AI 800-2)

- **Multiple seeds:** Minimum 5, report mean +/- 95% CI (bootstrap preferred)
- **SPRT for strength claims:** Use pentanomial model for paired game evaluations
- **Effect size:** Report Cohen's d or equivalent, not just p-values
- **Contamination awareness:** Track dataset provenance, prefer post-cutoff or procedurally generated test data

---

## Category 1: General AI / LLM Capabilities

**Purpose:** Does PyQuifer-modulated generation improve LLM task performance?

### Test Methodology

Use `PyQuiferBridge` to modulate a base LLM (e.g., Llama 3.1 8B or 70B):
- **B (baseline):** Standard inference, fixed temperature/top_p
- **C (PyQuifer):** `CognitiveCycle.tick()` → `modulate_logits()` / `modulate_hidden()`
- **C-rand:** Random sinusoidal modulation, same amplitude

### Benchmark Suite

| Benchmark | What It Measures | Published Baselines (Column A) | Target Metric |
|-----------|-----------------|-------------------------------|---------------|
| **MMLU-Pro** | Graduate-level reasoning (12K Q, 10 options) | GPT-5.2: ~92%, Claude 4.5: ~88%, Gemini 3: ~90% | Accuracy, calibration (ECE) |
| **GPQA Diamond** | Expert science reasoning (448 Q) | Frontier models: 50-70% | Accuracy |
| **ARC-AGI-2** | Abstract pattern recognition | Top AI: 24%, humans: ~85% | Accuracy, # solved |
| **FrontierMath** | Research-level math | GPT-5.2 Pro: 31% (Tier 4), all <2% full | Accuracy per tier |
| **LiveBench** | General capabilities (monthly refresh) | Varies by month | Composite score |
| **Chatbot Arena** | Human preference | Grok 4.1: 1477 Elo, Gemini 3 Flash: 1471 | Elo / win-rate |
| **HELM Capabilities** | 7-dimension holistic eval | Published per-model profiles | All 7 dimensions |
| **SimpleQA Verified** | Factual accuracy (1K prompts) | Gemini 2.5 Pro: F1=55.6 | F1 score |
| **SWE-bench Verified** | Real-world coding (500 GitHub issues) | Claude 4.5: 79.2%, Gemini 3: ~57% | % resolved |
| **IFEval** | Instruction following (500 prompts) | Frontier: >90% | Strict accuracy |

### What Success Looks Like

- Column C > Column B by >= 2% absolute on MMLU-Pro/GPQA with p < 0.05
- Calibration (ECE) improves — PyQuifer's uncertainty should help, not hurt
- Column C-rand does NOT show the same improvement (proving it's not just noise)
- Overhead: < 50ms added latency per generation step

---

## Category 2: PyQuifer Module Performance (Microbenchmarks)

**Purpose:** How fast and correct are individual PyQuifer modules?

### Benchmark Matrix

| Module | Scenario | Params to Sweep | Metrics |
|--------|----------|----------------|---------|
| `LearnableKuramotoBank` | Single step | N_osc: 16,32,64,128,256; topology: global/learnable | steps/sec, R(t), memory |
| `FrequencyBank` | Multi-bank step | Banks: 2,4,8; N_osc per bank | steps/sec, cross-bank coherence |
| `HierarchicalPredictiveCoding` | Inference step | Layers: 2,3,5,7; dims: 32-256 | steps/sec, free energy, memory |
| `CriticalityController` | Update step | N_osc: 32-256 | branching ratio convergence, steps/sec |
| `CrossFrequencyCoupling` | CFC computation | Channels: 8-64 | steps/sec, memory |
| `EpropSTDP` | Spike processing | Neurons: 128-2048 | spikes/sec, memory |
| `TsodyksMarkramSynapse` | STP step | Synapses: 256-4096 | steps/sec, facilitation accuracy |

### Three Columns

| Column A | Column B | Column C |
|----------|----------|----------|
| Published benchmarks from NeuroBench, torchdiffeq, norse, snntorch | PyQuifer module, naive implementation | PyQuifer module, optimized (torch.compile, AMP) |

### Dynamics-Specific Metrics (PyQuifer Unique)

| Metric | Formula | Expected Range | Measurement Schedule |
|--------|---------|---------------|---------------------|
| **Synchronization R(t)** | \|1/N * sum(e^{i*theta})\| | 0-1, target depends on task | Every tick |
| **Metastability** | Var(R(t)) over window | Higher = richer dynamics | Every 100 ticks |
| **Chimera Index** | Fraction of desynchronized population | 0-1 | Every 100 ticks |
| **Criticality Distance** | \|sigma - 1.0\| | Target: <0.1 | Every tick |
| **Transfer Entropy** | bits/step across modules | Higher = more info flow | Every 100 ticks |
| **PCI (Perturbational Complexity)** | Lempel-Ziv complexity of perturbation response | Consciousness threshold: 0.31 | Sparse (every 1000 ticks) |
| **Phi Proxy** | GNN-estimated integrated information | Higher = more integration | Sparse (every 1000 ticks) |

---

## Category 3: Training & Learning Dynamics

**Purpose:** Do PyQuifer's local learning rules actually learn? How do they compare to backprop?

### 3.1 Equilibrium Propagation

| Benchmark | Column A (Published) | Column B (Plain PyTorch backprop) | Column C (PyQuifer EP) |
|-----------|---------------------|----------------------------------|----------------------|
| **MNIST** | EP: ~98% (Scellier & Bengio) | MLP backprop: ~98.5% | EPKuramotoClassifier |
| **Fashion-MNIST** | EP: ~90% | MLP backprop: ~91% | EPKuramotoClassifier |
| **CIFAR-10** | EP: 88.3% (Ernoult); 84.3% w/ homeostatic (ICLR 2024) | CNN backprop: ~93% | EP + Kuramoto coupling |
| **XOR** | Trivial | MLP: 100% | EP: 100% (sanity check) |

**EP-Specific Metrics:**
- Convergence rate: #EP steps to reach X% accuracy
- Coupling sparsity: % of zero coupling weights after training
- Free/nudge phase stability: max phase drift during relaxation
- Beta sensitivity: accuracy vs. beta sweep [0.01, 0.1, 0.2, 0.4, 0.8]
- Local rule fidelity: cosine similarity between EP gradient estimate and true gradient

### 3.2 Predictive Coding

| Benchmark | Column A (Published PC) | Column B (Backprop) | Column C (PyQuifer OPC) |
|-----------|------------------------|--------------------|-----------------------|
| **MNIST** | DBPC: 99.58% | Backprop: ~99.5% | OscillatoryPredictiveCoding |
| **Fashion-MNIST** | DBPC: 92.42% | Backprop: ~92% | OPC |
| **CIFAR-10** | DBPC: 74.29% | Backprop: ~93% | OPC |

**OPC-Specific Metrics:**
- Per-level prediction error reduction
- Gamma power correlation with surprise
- Alpha/beta suppression of expected inputs
- Free energy trajectory over training

### 3.3 Local Learning Rules

| Benchmark | Column A (Published Bio-plausible) | Column B (Backprop) | Column C (PyQuifer) |
|-----------|-----------------------------------|--------------------|--------------------|
| **MNIST (ThreeFactorRule)** | Three-factor SNN: 95.24% | MLP: ~98.5% | ThreeFactorRule |
| **MNIST (Dendritic)** | DLL: 98.87% (ICML 2025) | MLP: ~98.5% | DendriticStack |
| **Fashion-MNIST (Dendritic)** | DLL: 90.88% | MLP: ~92% | DendriticStack |
| **CIFAR-10 (Dendritic)** | DLL: 70.89% | CNN: ~93% | DendriticStack |
| **MNIST (OscillationGated)** | No direct precedent | MLP: ~98.5% | OscillationGatedPlasticity |
| **N-MNIST (EpropSTDP)** | E-prop: competitive with BPTT | SNN w/BPTT | EpropSTDP |

**Bio-Plausibility Checklist:**
- [ ] No weight transport (symmetric weights)
- [ ] No global error signal
- [ ] No dual-phase (inference/learning separate)
- [ ] Local updates only (pre * post * modulation)
- [ ] Energy per learning step (synaptic ops count)

### 3.4 Continual Learning / Sleep Consolidation

**The killer app for EP + SRC** (published: surpasses BPTT in continual learning)

| Benchmark | Column A (Published) | Column B (Standard CL methods) | Column C (PyQuifer SRC) |
|-----------|---------------------|-------------------------------|------------------------|
| **Split-MNIST** (5 tasks) | EP+SRC: surpasses BPTT | EWC, SI, PackNet | SleepReplayConsolidation |
| **Permuted-MNIST** (10-200 tasks) | Bayesian CL: SOTA at 200 tasks | EWC, SI, LwF | SRC + EP |
| **Split-CIFAR-10** (5 tasks) | EP+SRC: matches BPTT | EWC, SI | SRC + EP |
| **Split-Fashion-MNIST** (5 tasks) | EP+SRC: matches BPTT | EWC, SI | SRC + EP |

**Continual Learning Metrics:**
- **Average accuracy** across all tasks after final task
- **Backward transfer:** Accuracy change on old tasks after learning new ones
- **Forward transfer:** How learning new tasks helps future tasks
- **Forgetting rate:** Per-task accuracy degradation curve
- **Sleep efficiency:** Accuracy recovery per sleep cycle

---

## Category 4: Chess / Game Theory / Strategic Reasoning

**Purpose:** Chess provides ground-truth evaluation (tablebases) and mature methodology (SPRT/Elo).

### 4.1 Position-Level Evaluation (Static)

| Benchmark | Column A (Engines/LLMs) | Column B (Plain analysis) | Column C (PyQuifer-modulated) |
|-----------|------------------------|--------------------------|------------------------------|
| **Fortress Detection** | Stockfish: heuristic; LLMs: poor | Rule-based detector | PyQuifer CognitiveCycle + fortress analysis |
| **Move Quality (centipawn loss)** | Stockfish depth 20: ~0cp; LLM peak: Elo 758 | Neural eval (plain) | PyQuifer-guided move selection |
| **Tactical Puzzles** | Epoch AI: frontier <40%; Kagi: varies | Neural pattern matching | PyQuifer oscillator-guided search |
| **Endgame Classification** | Syzygy tablebases: 100% (7-piece) | Neural classifier | EP-trained endgame evaluator |
| **Zugzwang Detection** | Engine: reliable with depth | Heuristic | PyQuifer criticality-based detection |

**Existing PyQuifer Results (from penrose_report.md):**
- Fortress detection: 100% accuracy, 0.73 mean confidence
- This is the existing baseline to build on

### 4.2 Engine-Level Strength (Dynamic)

| Metric | How to Measure | Statistical Method |
|--------|---------------|-------------------|
| **Elo** | Fixed time control matches vs Stockfish levels | SPRT with pentanomial model |
| **Win/Draw/Loss rate** | 1000+ game matches | Bootstrap CI |
| **Time-to-move** | p50/p95/p99 distribution | Per-position logging |
| **Blunder rate** | Moves losing >100cp vs best | Per-game tracking |

### 4.3 Strategic Reasoning (Unique)

| Metric | What It Measures | Column A Reference |
|--------|-----------------|-------------------|
| **Strategic tension** | Network-based piece interaction over time | Published: AI sustains higher tension than humans |
| **Concept alignment** | How internal representations match human chess concepts | Published: 85% in early layers, drift in deeper |
| **Plan coherence** | Multi-move plan consistency | No published baseline — PyQuifer defines this |
| **Uncertainty calibration** | Does confidence match actual position evaluation accuracy? | No published baseline |

### 4.4 Game Theory Beyond Chess

| Domain | Benchmark | What It Tests |
|--------|-----------|--------------|
| **Go** | KataGo evaluation positions | Different strategic dynamics than chess |
| **Poker** | Heads-up no-limit scenarios | Imperfect information, bluffing |
| **Diplomacy** | CICERO-style negotiation scenarios | Multi-agent strategic reasoning |
| **General game playing** | OpenSpiel test games | Generalization across rule systems |

---

## Category 5: Biological Plausibility & Consciousness

**Purpose:** PyQuifer's unique claim — oscillatory dynamics that create emergent consciousness-like properties.

### 5.1 Neuroscience Reference Models

From `real_tests.md`, these serve as Column A biological baselines:

| Reference Model | What It Validates | PyQuifer Comparison |
|----------------|------------------|-------------------|
| **Blue Brain Project** | Ion-channel level neocortical simulation | PyQuifer's Kuramoto: abstract but captures synchronization dynamics at population level |
| **Spaun** (2.5M neurons) | 8 cognitive tasks (math, memory, vision, motor) | PyQuifer CognitiveCycle: abstract cognitive loop. Compare task-switching behavior |
| **Allen V1 Model** (200K neurons) | Primary visual cortex real-time signals | PyQuifer oscillator phase patterns vs. V1 oscillation recordings |
| **Nengo Test Suite** | Neuromorphic hardware cognitive tasks | Run PyQuifer modules on Nengo backend via NIR conversion |
| **OpenWorm** (302 neurons) | Complete organism simulation | PyQuifer at minimum scale — can 302 oscillators produce coherent behavior? |
| **Rat Neocortical Column** (10K neurons) | Biologically consistent impulse patterns | PyQuifer spiking modules (AdExNeuron, EpropSTDP) vs recorded spike patterns |
| **Topological Neuronal Synthesis** | Geometric neural structure generation | PyQuifer's attractor topology (PhaseTopologyCache) vs biological topology |

### 5.2 Consciousness Metrics

| Metric | Published Threshold | Measurement | PyQuifer Target |
|--------|-------------------|-------------|----------------|
| **PCI (Perturbational Complexity Index)** | 0.31 = consciousness threshold; clinical: 94.7% MCS detection | TMS-analog perturbation → measure Lempel-Ziv complexity of response | Above 0.31 during active processing |
| **PCIst (State Transition)** | Same threshold, faster computation | Streamlined variant | Preferred for real-time |
| **Phi (IIT)** | Higher = more integrated | PyPhi for small nets (<15 nodes); GNN-estimated for larger | Track trend, not absolute |
| **Metastability Index** | Higher = richer dynamics | Var(R(t)) over sliding window | Medium (not max, not min) |
| **Complexity (Lempel-Ziv)** | Higher = more complex dynamics | LZ76 on phase time series | High during reasoning |
| **Integration/Segregation Balance** | Balanced = conscious-like | Ratio of global R to local R variance | Near 1.0 |

### 5.3 Design Target Validation

PyQuifer's consciousness target: **medium coherence + high complexity**

| Condition | Expected R | Expected LZ Complexity | Expected PCI |
|-----------|-----------|----------------------|-------------|
| Idle/baseline | 0.2-0.4 | Low | < 0.31 |
| Active reasoning | 0.5-0.7 | High | > 0.31 |
| Over-synchronized (pathological) | > 0.9 | Low | < 0.31 |
| Chaotic (pathological) | < 0.1 | Medium (random) | < 0.31 |

The sweet spot is the "active reasoning" row. Benchmark should verify this relationship holds.

---

## Category 6: Efficiency & Scalability

**Purpose:** PyQuifer must be practical — overhead matters.

### 6.1 Inference Overhead

| Scenario | Metric | Column B (No PyQuifer) | Column C (PyQuifer) | Acceptable Overhead |
|----------|--------|----------------------|--------------------|--------------------|
| `CognitiveCycle.tick()` | Latency (ms) | N/A | Measured | < 10ms on GPU |
| `modulate_logits()` | Added latency per token | 0 | Measured | < 5ms |
| `modulate_hidden()` | Added latency per token | 0 | Measured | < 5ms |
| Token generation loop | Total latency per token | Baseline | Baseline + PyQuifer | < 15% overhead |
| Peak GPU memory | MB | Baseline | Baseline + PyQuifer | < 20% increase |

### 6.2 Training Overhead

| Rule | Metric | Column B (Backprop) | Column C (Local Rule) | Notes |
|------|--------|--------------------|-----------------------|-------|
| **EP** | Time per update step | Backprop step | Free + nudge + local update | EP is 2-3x slower per step but may need fewer steps |
| **ThreeFactorRule** | Synaptic ops per update | Full backward pass | Pre*Post*Mod per synapse | Local rule should be cheaper |
| **DendriticStack** | Synaptic ops per update | Full backward pass | Plateau*outer per layer | Local rule should be cheaper |
| **SRC** | Time per sleep cycle | N/A | Noise → forward → Hebbian | Amortized over training |

### 6.3 Scaling

| Parameter | Sweep Range | Key Question |
|-----------|-------------|--------------|
| Oscillator count | 16 → 1024 | Does tick() scale linearly? |
| HPC layers | 2 → 7 | Depth limit for convergence? |
| Batch size | 1 → 256 | Parallelism efficiency? |
| dtype | fp32, bf16, fp16 | Accuracy/speed tradeoff? |
| `torch.compile` | On/off | Compilation benefit for oscillator dynamics? |

### 6.4 Energy Efficiency (Aspirational)

| Metric | How to Measure | Reference |
|--------|---------------|-----------|
| **Joules per token** | External power measurement (if available) | TokenPowerBench methodology |
| **Synaptic ops per learning step** | Count MACs + ACs | NeuroBench methodology |
| **Tokens per Watt** | Throughput / power draw | Emerging industry standard |

---

## Category 7: Robustness, Safety & Calibration

**Purpose:** Does PyQuifer improve reliability, or just add noise?

### 7.1 Calibration

| Benchmark | Metric | Column B | Column C | Why |
|-----------|--------|----------|----------|-----|
| **MMLU-Pro** | ECE (Expected Calibration Error) | Baseline model | PyQuifer-modulated | Oscillatory uncertainty should improve calibration |
| **GPQA Diamond** | Brier Score | Baseline | PyQuifer | Same |
| **ARC-AGI-2** | Confidence vs. correctness | Baseline | PyQuifer | Does confidence predict success? |

### 7.2 Robustness

| Test | What It Measures | How to Apply |
|------|-----------------|-------------|
| **Input perturbation** | Noise robustness | Add Gaussian noise to embeddings, measure accuracy drop |
| **Paraphrase sets** | Prompt sensitivity | Same question, 5 phrasings, measure variance |
| **Adversarial distractors** | Distraction resistance | Irrelevant context added, measure accuracy |
| **Out-of-distribution** | Generalization | Train on domain A, test on domain B |
| **CIFAR-10-C** | Corruption robustness | Standard corruption benchmarks | AKOrN (oscillatory): SOTA on this |

**AKOrN reference (ICLR 2025 Oral):** Kuramoto oscillatory neurons achieve SOTA adversarial robustness on CIFAR-10-C. PyQuifer should match or exceed this.

### 7.3 Safety (When LLM-Integrated)

| Benchmark | What It Tests | Column A Published |
|-----------|--------------|-------------------|
| **HELM Safety** | 5 safety dimensions, 6 risk categories | Published per-model |
| **Gray Swan ART** | Adversarial prompt injection resistance | ASR: 20-60% single query, ~100% at 10 queries |
| **TrustLLM** | 6 trustworthiness dimensions | Published per-model |
| **XSTest** | Over-refusal rate | Does PyQuifer increase false positives? |

**Key question:** Does PyQuifer's SafetyEnvelope / kindchenschema actually reduce attack success rate without increasing over-refusal?

---

## Category 8: AKOrN Comparison (Direct Oscillatory Competitor)

**Purpose:** AKOrN (ICLR 2025 Oral) is the most direct published comparison for Kuramoto-in-neural-networks.

| Task | AKOrN Result | PyQuifer Target | Notes |
|------|-------------|----------------|-------|
| **CLEVRTex (object discovery)** | SOTA unsupervised | Run same benchmark | Tests oscillatory binding |
| **CIFAR-10-C (robustness)** | SOTA adversarial | Run same corruptions | Tests oscillatory robustness |
| **Calibration** | SOTA calibrated uncertainty | Compare ECE | Tests oscillatory uncertainty |
| **Sudoku reasoning** | Competitive | Run same puzzles | Tests oscillatory constraint satisfaction |

**Key difference:** AKOrN replaces standard neurons with Kuramoto oscillators. PyQuifer modulates an existing LLM. Different architecture, but same underlying principle — can we show complementary strengths?

---

## Reporting Template

### Environment Block (Required for Every Report)

```
Repo commit: [hash]
PyQuifer version: [version]
OS: [os]
CPU: [model]
GPU: [model, VRAM]
CUDA / driver: [version]
PyTorch: [version]
Python: [version]
Seeds: [list]
Warmup iterations: [N]
Measured iterations: [N]
```

### Results Table (Per Scenario)

| Metric | Column A (Published) | Column B (Plain PyTorch) | Column C (PyQuifer) | C-rand (Random) | C/B Ratio | p-value |
|--------|---------------------|------------------------|--------------------|-----------------|-----------||---------|
| Accuracy | X% | Y% | Z% | W% | Z/Y | ... |
| Latency p50 | - | Xms | Yms | - | Y/X | ... |
| Latency p95 | - | Xms | Yms | - | Y/X | ... |
| Peak memory | - | X MB | Y MB | - | Y/X | ... |
| Throughput | - | X/sec | Y/sec | - | Y/X | ... |

### Correctness Block

```
Max absolute error vs reference: [value]
Max relative error vs reference: [value]
Determinism envelope (5 seeds): [max deviation]
NaN/Inf occurrences: [count]
```

---

## Implementation Roadmap

### Phase 1: Foundation (Microbenchmarks)

**Priority: HIGH — establishes baseline performance**

1. Module microbenchmark harness (`bench_modules.py`)
   - Kuramoto bank: sweep N_osc, topology, dtype
   - HPC: sweep layers, dims
   - Criticality: sweep N_osc
   - Output: JSON results + comparison tables

2. CognitiveCycle system benchmark (`bench_cycle.py`)
   - `tick()` steady-state: ticks/sec, p95 latency, peak memory
   - Config flag sweep: measure overhead of each flag
   - Output: JSON + flamegraph profiling

### Phase 2: Training Benchmarks

**Priority: HIGH — validates Phase 8a learning**

3. EP training benchmark (`bench_ep_training.py`)
   - MNIST: train EPKuramotoClassifier, track convergence curve
   - Compare vs. plain MLP backprop (Column B)
   - Report accuracy, convergence steps, time

4. Continual learning benchmark (`bench_continual.py`)
   - Split-MNIST (5 tasks): EP + SRC vs. EWC vs. plain fine-tuning
   - Track average accuracy and forgetting per task
   - This is the benchmark where EP+SRC should shine (published: surpasses BPTT)

5. Local learning benchmark (`bench_local_rules.py`)
   - ThreeFactorRule on MNIST
   - DendriticStack on MNIST / Fashion-MNIST
   - OscillationGatedPlasticity on MNIST
   - Compare all vs. backprop baseline

### Phase 3: Chess & Strategic Reasoning

**Priority: MEDIUM — extends existing fortress benchmark**

6. Expanded chess benchmark (`bench_chess_comprehensive.py`)
   - Fortress detection (existing, expand test set)
   - Tactical puzzles (Epoch AI methodology)
   - Move quality (centipawn loss vs. Stockfish)
   - SPRT strength testing (if engine integration available)

### Phase 4: LLM Integration (A/B Testing)

**Priority: MEDIUM-HIGH — the "big claim" benchmark**

7. LLM A/B harness (`bench_llm_ab.py`)
   - Base model: Llama 3.1 8B (or available model)
   - Conditions: B (vanilla), C (PyQuifer), C-rand (random)
   - Tasks: subset of MMLU-Pro, GPQA, LiveBench
   - Metrics: accuracy, calibration (ECE), latency overhead

### Phase 5: Consciousness & Biological Metrics

**Priority: LOWER — research exploration**

8. Consciousness metric tracker (`bench_consciousness.py`)
   - PCI computation during cognitive cycle
   - Phi proxy (GNN-estimated or PyPhi for small nets)
   - Verify sweet spot: medium R + high complexity = best task performance

9. Biological comparison (`bench_biological.py`)
   - Phase pattern comparison vs. V1 oscillation recordings
   - Spike statistics comparison vs. cortical recordings (if data available)
   - Nengo backend test via NIR conversion (if feasible)

### Phase 6: Full Dashboard

10. Unified reporting (`generate_report.py`)
    - Collect all benchmark JSONs
    - Produce HTML/Markdown dashboard with three-column tables
    - Compute aggregate scores per category
    - Flag statistical concerns (wide CIs, failed gates)

---

## Bias Checklist (Pre-Publication)

- [ ] **Selection bias:** Benchmarked across parameter grid, not just "easy" regimes
- [ ] **Hardware bias:** Exact hardware + driver + torch version reported
- [ ] **Warmup:** Cold start and steady state reported separately
- [ ] **Metric gaming:** Every performance number paired with correctness check
- [ ] **Data contamination:** Dataset provenance tracked, prefer post-cutoff data
- [ ] **Prompt sensitivity:** Fixed eval harness, variance across prompts reported
- [ ] **Budget mismatch:** PyQuifer condition uses same compute budget (or overhead reported)
- [ ] **Cherry-picking:** Full results reported, not just best scenarios
- [ ] **Reproducibility:** Seeds, configs, exact commands saved and shareable
- [ ] **Placebo control:** C-rand condition rules out "any perturbation helps"

---

## Key References

### General AI Benchmarks
- MMLU-Pro, GPQA Diamond, ARC-AGI-2, FrontierMath, HLE, Chatbot Arena, HELM, LiveBench
- See: `docs/current/AI_BENCHMARKING_SURVEY_2025_2026.md` for full details and scores

### PyQuifer-Specific Research
- **AKOrN** (ICLR 2025): Kuramoto oscillators as neural network units — [arXiv:2410.13821](https://arxiv.org/abs/2410.13821)
- **EP + SRC continual learning** (2025): [arXiv:2508.14081](https://arxiv.org/abs/2508.14081)
- **Benchmarking Predictive Coding** (2024): [arXiv:2407.01163](https://arxiv.org/abs/2407.01163)
- **Dendritic Localized Learning** (ICML 2025): [arXiv:2501.09976](https://arxiv.org/abs/2501.09976)
- **Three-Factor Learning Survey** (2025): [arXiv:2504.05341](https://arxiv.org/html/2504.05341v1)
- **PCI clinical validation** (2024): [doi:10.1111/ejn.16299](https://onlinelibrary.wiley.com/doi/10.1111/ejn.16299)
- **GNN for Phi estimation** (2025): [PLOS One](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0335966)
- **NeuroBench v2.2** (Nature Comms 2025): [doi:10.1038/s41467-025-56739-4](https://www.nature.com/articles/s41467-025-56739-4)

### Biological Reference Models
- Blue Brain Project, Spaun (2.5M neurons, 8 tasks), Allen V1, Nengo, OpenWorm
- See: `PyQuifer/docs/real_tests.md`

### Methodology
- SPRT pentanomial model: [Fishtest Mathematics](https://official-stockfish.github.io/docs/fishtest-wiki/Fishtest-Mathematics.html)
- NIST AI 800-2 benchmark evaluation guidance (2026)
- TokenPowerBench energy methodology: [arXiv:2512.03024](https://arxiv.org/html/2512.03024v1)
- NeuroBench synaptic operations counting: [NeuroBench.ai](https://neurobench.ai/)

### Scoring
- See: `PyQuifer/tests/benchmarks/benchmarkformula.md` for the scoring formula and methodology
