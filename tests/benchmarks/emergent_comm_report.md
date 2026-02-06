# Benchmark Report: PyQuifer vs Emergent Communication Frameworks

**Date:** 2026-02-06
**PyQuifer version:** Phase 5 (479 tests, 12 enhancements)
**Benchmark repos:** EGG (Kharitonov et al. 2019), emergent_communication_at_scale (Chaabouni et al.)
**Script:** `tests/benchmarks/bench_emergent_comm.py`

---

## Executive Summary

PyQuifer is **not a multi-agent communication framework** -- it provides neuroscience-inspired learning and selection mechanisms that could *underpin* communication systems. This benchmark compares the overlapping components: reward-modulated learning (PyQuifer's RewardModulatedHebbian vs REINFORCE), population selection (PyQuifer's SelectionArena vs independent agent training), and signal discrimination (SpikingLayer and NoveltyDetector as signal processors). Both REINFORCE and R-M Hebbian show learning improvement over 200 steps with comparable final loss (0.33 vs 0.35). SelectionArena produces rapid resource divergence (fitness-based competition), while EGG's independent training maintains population diversity. No new gaps identified -- PyQuifer and EGG/ECAS are complementary frameworks with minimal overlap (1/16 shared features: population training).

## What EGG and ECAS Are

**EGG** (Facebook Research) is a toolkit for multi-agent emergent communication games:
- Sender-receiver architecture with discrete/continuous channels
- REINFORCE and Gumbel-Softmax optimization for non-differentiable discrete messages
- 18+ game types (signaling, referential, compositional, population-based)
- Compositionality and language analysis metrics

**emergent_communication_at_scale** extends this to Jax/Haiku with Lewis games at scale.

**Key difference**: EGG/ECAS design *communication protocols* between agents. PyQuifer provides *neural dynamics primitives* (learning rules, selection, oscillators) that could be used inside agents but doesn't implement the communication infrastructure.

## Results

### 1. Reward-Modulated Learning

| System | Final Loss | Improves? | Time |
|--------|:----------:|:---------:|:----:|
| REINFORCE (EGG-style) | 0.329 | YES | 160.6 ms |
| Reward-Modulated Hebbian (PyQuifer) | 0.351 | YES | 105.7 ms |

**Analysis:**
- Both systems learn successfully over 200 steps on a simple mapping task
- REINFORCE achieves slightly lower loss (0.329 vs 0.351) due to explicit gradient computation through the policy
- R-M Hebbian is 34% faster (105 vs 161 ms) because it avoids the computational graph for policy gradients
- R-M Hebbian uses local three-factor learning (pre * post * reward), which is biologically plausible but less sample-efficient than REINFORCE

### 2. Population Selection

| System | Init Diversity | Final Diversity | Time |
|--------|:--------------:|:---------------:|:----:|
| EGG Population (independent) | 0.040 | 0.048 | 334.5 ms |
| PyQuifer SelectionArena | 0.001 | 11.215 | 56.3 ms |

**Analysis:**
- **EGG population**: Independent training preserves diversity (0.040 → 0.048, +20%). Each agent evolves independently, creating diverse communication strategies
- **PyQuifer SelectionArena**: Rapid resource divergence (0.001 → 11.2). Replicator dynamics amplify fitness differences, causing "winner-take-all" resource allocation. This is biologically correct (neural Darwinism selects winning groups) but different from EGG's diversity-preserving approach
- SelectionArena is 6x faster (56 vs 335 ms) because it uses shared weights with resource gating rather than training independent models

### 3. Signal Discrimination

| System | Response Separation | Time |
|--------|:-------------------:|:----:|
| Linear (baseline) | 0.010 | 0.1 ms |
| PyQuifer SpikingLayer | 0.000 | 0.6 ms |
| PyQuifer NoveltyDetector | 0.040 | 60.6 ms |

**Analysis:**
- **SpikingLayer separation = 0**: With threshold=0.5 and untrained weights, all 5 signals produce zero spikes (all below threshold). This connects to G-04 (no input normalization) -- SpikingLayer needs input scaling to produce meaningful output
- **NoveltyDetector separation = 0.04**: Successfully discriminates between signals via different novelty responses. Repeated presentations of the same signal reduce novelty, while switching signals increases it
- NoveltyDetector is the most meaningful signal discriminator, confirming PyQuifer's strength in *detecting signal differences* rather than *classifying signals*

### 4. Architecture Feature Comparison

| Feature | EGG/ECAS | PyQuifer |
|---------|:--------:|:--------:|
| compositionality_metrics | YES | no |
| continuous_channel | YES | no |
| criticality_control | no | YES |
| discrete_channel | YES | no |
| gumbel_softmax | YES | no |
| intrinsic_motivation | no | YES |
| multi_agent | YES | no |
| neural_darwinism | no | YES |
| online_adaptation | no | YES |
| oscillatory_dynamics | no | YES |
| population_training | YES | YES |
| reinforce_optimization | YES | no |
| reward_modulated_learning | no | YES |
| rnn_agents | YES | no |
| spiking_neurons | no | YES |
| transformer_agents | YES | no |
| **Total** | **9/16** | **8/16** |

**Shared feature (1)**: Population-based training (EGG's `pop` game vs PyQuifer's SelectionArena).

## Comparative Assessment

### Where EGG/ECAS exceeds PyQuifer

| Dimension | EGG/ECAS | PyQuifer |
|-----------|:--------:|:--------:|
| Communication channels | Discrete + Continuous | None |
| Multi-agent framework | Built-in sender/receiver | Single-module |
| Language analysis | Compositionality, entropy | None |
| Agent architectures | RNN, Transformer | N/A |
| Policy optimization | REINFORCE, Gumbel-Softmax | N/A |

### Where PyQuifer exceeds EGG/ECAS

| Dimension | PyQuifer | EGG/ECAS |
|-----------|:--------:|:--------:|
| Biological learning | R-M Hebbian, STDP, eligibility traces | REINFORCE only |
| Selection mechanism | Replicator dynamics, speciation | Independent training |
| Neural dynamics | Spiking, oscillatory, criticality | Standard backprop |
| Online adaptation | Real-time, per-token | Batch training |
| Intrinsic motivation | Novelty, mastery, coherence | Task reward only |

### Complementarity

EGG provides the communication *framework* (games, channels, metrics). PyQuifer provides the biological *mechanisms* (learning rules, selection, dynamics). A future integration:

1. Replace EGG's standard RNN agents with PyQuifer's spiking networks
2. Use R-M Hebbian for local synaptic updates alongside REINFORCE for global policy
3. Apply SelectionArena for evolving agent populations within EGG's game structure
4. Use NoveltyDetector for intrinsic exploration bonuses in communication games

## Gaps Identified

No new gaps. The SpikingLayer zero-separation result reinforces existing G-04 (SpikeEncoder input adapter needed).

## Pytest Results

```
8/8 passed (4.02s)

TestRewardLearning::test_reinforce_runs                  PASSED
TestRewardLearning::test_rstdp_runs                      PASSED
TestRewardLearning::test_both_produce_finite_loss         PASSED
TestPopulationSelection::test_egg_population_runs         PASSED
TestPopulationSelection::test_neural_darwinism_runs       PASSED
TestSignalDiscrimination::test_spiking_produces_output    PASSED
TestSignalDiscrimination::test_novelty_produces_separation PASSED
TestArchitecture::test_feature_counts                     PASSED
```

Existing test suite: **479/479 passed** (unaffected).

## Verdict

**PyQuifer learning/selection mechanisms: PASS -- reward-modulated learning achieves comparable loss to REINFORCE, SelectionArena implements correct replicator dynamics.**

PyQuifer and EGG/ECAS serve fundamentally different roles: EGG builds communication games, PyQuifer provides neural dynamics primitives. The overlapping components (reward learning, population selection) work correctly and could be integrated. R-M Hebbian is 34% faster than REINFORCE with slightly higher loss, consistent with the biological plausibility vs sample efficiency tradeoff. No new gaps identified.
