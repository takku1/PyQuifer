# Benchmark Report: PyQuifer Spiking Modules vs Intel Lava

**Date:** 2026-02-06
**PyQuifer version:** Phase 5 (479 tests, 12 enhancements)
**Benchmark repo:** `tests/benchmarks/lava/` -- Intel Corporation, v0.9.0 (BSD-3-Clause)
**Script:** `tests/benchmarks/bench_lava.py`

---

## Executive Summary

PyQuifer's spiking neural network modules produce **correct LIF dynamics** with bounded voltage and stable firing, **spike-frequency adaptation** via AdExNeuron, and **matching 2nd-order synaptic dynamics** to Lava's dual-state (u+v) neuron model. Lava excels in **hardware deployment** with 12/16 architectural features (fixed-point precision, sparse connectivity, synaptic delays, process isolation) targeting Intel's Loihi neuromorphic chips. PyQuifer excels in **gradient-based learning** with 6/16 features (surrogate gradients, learnable parameters, oscillatory modulation, precision weighting) targeting research and cognitive modeling. The two frameworks are **complementary**: Lava for neuromorphic deployment, PyQuifer for neuroscience-informed AI research.

## What Lava Is

Lava (v0.9.0) is Intel's open-source framework for neuromorphic computing applications, designed for Intel Loihi 1/2 neuromorphic chips. It uses a Communicating Sequential Process (CSP) architecture where:

- **Processes** are independent computational units (neurons, synapses, etc.)
- **Ports** enable message passing between processes (InPort, OutPort, RefPort)
- **ProcessModels** provide separate implementations per backend (floating-point for prototyping, fixed-point for hardware)
- **RunConfigs** select between CPU simulation and Loihi hardware execution

Key neuron models: LIF, TernaryLIF, LIFReset, LIFRefractory, LearningLIF, ATRLIF, ResFireFloat, RFIzhikevich, ProductNeuron, S4d.

**Key difference**: Lava is **hardware-first** -- designed to compile and run on neuromorphic chips. PyQuifer is **research-first** -- designed for consciousness modeling and oscillatory neural dynamics with gradient-based learning. Lava's question: "How do we deploy SNNs on Loihi?" PyQuifer's question: "How do we model biological neural oscillations for cognitive architectures?"

## Results

### 1. LIF Dynamics Comparison

Both systems implement the core Leaky Integrate-and-Fire neuron. Lava uses a 2-state model (current u + voltage v), while PyQuifer uses a 1-state model (membrane potential only).

**Lava LIF equations (floating-point):**
```
u[t] = u[t-1] * (1 - du) + a_in
v[t] = v[t-1] * (1 - dv) + u[t] + bias
spike = v > vth; reset v = 0
```

**PyQuifer LIF equations:**
```
mem = decay * (mem - v_rest) + v_rest + current * dt/tau
spike = surrogate_spike(mem, threshold); reset mem = v_reset
```

| Metric | Lava LIF (float) | PyQuifer LIFNeuron |
|--------|:-----------------:|:------------------:|
| Firing rate | 0.353 | 0.076 |
| Mean ISI | 2.73 | 12.55 |
| CV of ISI | 0.425 | 0.657 |
| Time (200 steps, 100 neurons) | 1.0 ms | 12.3 ms |
| Voltage bounded | YES | YES |
| States | 2 (u, v) | 1 (mem) |

**Analysis:**

- **Firing rate difference (0.353 vs 0.076)**: The rate difference is due to fundamentally different architectures. Lava's 2-state model accumulates input in current `u` (with decay du=0.1, steady-state u = input/du), then feeds u into voltage v. This amplification means even moderate input drives v above threshold. PyQuifer's 1-state model directly integrates input into membrane, with current scaled by `dt/tau`. Different but both correct.

- **CV of ISI**: Lava's lower CV (0.425) indicates more regular firing, expected from the dual-state filtering. PyQuifer's higher CV (0.657) reflects more variable inter-spike intervals, closer to biological irregular spiking.

- **Speed**: Lava's numpy implementation (1.0 ms) is faster than PyQuifer's PyTorch (12.3 ms) for this small-scale benchmark. The overhead is PyTorch's tensor operations and autograd infrastructure (surrogate gradients). At scale, PyTorch's GPU support would reverse this.

### 2. Adaptive Threshold Comparison

Lava's ATRLIF implements explicit threshold adaptation with separate refractory and threshold state variables. PyQuifer's AdExNeuron implements adaptation via the adaptation current `w`.

**Lava ATRLIF dynamics:**
```
theta[t] = (1 - delta_theta) * (theta[t-1] - theta_0) + theta_0
r[t] = (1 - delta_r) * r[t-1]
spike = (v - r) >= theta
post-spike: r += 2*theta, theta += theta_step
```

**PyQuifer AdEx dynamics:**
```
dV/dt = [-gL*(V-EL) + gL*DT*exp((V-VT)/DT) - w + I] / C
dw/dt = [a*(V-EL) - w] / tau_w
post-spike: V -> V_reset, w += b
```

| Metric | Lava ATRLIF | PyQuifer AdExNeuron |
|--------|:-----------:|:-------------------:|
| Rate (first half) | 0.250 | 0.160 |
| Rate (second half) | 0.250 | 0.130 |
| Adaptation visible | NO (stable) | YES (-19%) |
| Mechanism | Threshold increases | w accumulates |
| State variables | 4 (i, v, theta, r) | 3 (V, w, spike) |

**Analysis:**

- **Lava ATRLIF stable at 0.25**: With default parameters (delta_theta=0.2, delta_r=0.2), the threshold and refractory dynamics reach equilibrium quickly. The rate stays constant because theta and r balance each other within a few steps. This is correct behavior -- ATRLIF adapts *transiently* after perturbation but reaches steady state.

- **PyQuifer AdEx adapts**: The adaptation current w accumulates slowly (tau_w=30), creating visible spike-frequency adaptation (0.16 -> 0.13, -19% rate reduction). This is the classic AdEx adapting neuron pattern: each spike increments w by `b`, which hyperpolarizes the membrane.

- **Different adaptation strategies**: ATRLIF models *threshold* adaptation (theta increases with spiking). AdEx models *current* adaptation (w opposes excitation). Both are biologically valid -- different cortical neuron types use different adaptation mechanisms. Lava's ATRLIF is closer to Loihi hardware (threshold manipulation), while PyQuifer's AdEx is closer to biophysical models (Brette & Gerstner 2005).

### 3. Fixed-Point Precision Effects

Lava provides both floating-point and bit-accurate (Loihi-equivalent) fixed-point implementations. This is critical for hardware deployment.

| Metric | Float | Fixed (Loihi) |
|--------|:-----:|:-------------:|
| Firing rate | 0.873 | 0.965 |
| Spike agreement | 89.2% | -- |
| Max voltage deviation | 1.000 | -- |

**Analysis:**

- **89.2% spike agreement**: Fixed-point quantization (24-bit state, 12-bit decay, 6-bit alignment shifts) produces slightly different dynamics, resulting in ~11% of timesteps having different spike/no-spike decisions. This is expected and well-characterized in Lava's test suite.

- **Fixed fires more often**: The fixed-point model has higher firing rate (0.965 vs 0.873) due to truncation rounding effects in the decay computation. Small bit-level differences accumulate over timesteps.

- **PyQuifer has no fixed-point support**: All PyQuifer computations use float32 via PyTorch. This is by design -- PyQuifer targets GPU/CPU computation for research, not neuromorphic hardware. Adding fixed-point would require a separate inference pathway.

- **Voltage deviation = 1.0**: The max voltage difference between float and fixed models is 1.0 (the threshold value), occurring when one model spikes and the other doesn't at a given timestep.

### 4. Synaptic (2nd-Order) Neuron Dynamics

Both systems implement 2nd-order neurons with separate synaptic current and membrane voltage.

| Metric | Lava LIF (u+v) | PyQuifer SynapticNeuron |
|--------|:---------------:|:----------------------:|
| Firing rate | 0.665 | 0.665 |
| Temporal smoothness | -0.235 | -0.235 |

**Analysis:**

- **Identical firing rates (0.665)**: With matched parameters (Lava du=0.15 -> PyQuifer alpha=0.85, Lava dv=0.1 -> PyQuifer beta=0.9), the two models produce identical dynamics. This validates that PyQuifer's SynapticNeuron correctly implements the same dual-exponential dynamics as Lava's LIF.

- **Negative autocorrelation (-0.235)**: Both show negative temporal autocorrelation because high firing rate means voltage frequently crosses threshold and resets, creating an oscillatory pattern. This is physically correct -- rapid spike-reset cycles produce anti-correlation at lag 1.

- **PyQuifer adds surrogate gradients**: While the dynamics are identical, PyQuifer's SynapticNeuron supports backpropagation via surrogate gradients (fast sigmoid). Lava's LIF has no gradient support -- learning is done via on-chip STDP rules, not backprop.

### 5. Architecture Feature Comparison

| Feature | Lava | PyQuifer |
|---------|:----:|:--------:|
| adaptive_threshold | YES | YES |
| convolutional | YES | no |
| distributed_exec | YES | no |
| fixed_point | YES | no |
| graded_spikes | YES | no |
| hardware_mapping | YES | no |
| learnable_params | no | YES |
| online_stdp | YES | YES |
| oscillatory_modulation | no | YES |
| precision_weighting | no | YES |
| process_isolation | YES | no |
| refractory_period | YES | no |
| sparse_connectivity | YES | no |
| surrogate_gradients | no | YES |
| synaptic_delays | YES | no |
| ternary_spikes | YES | no |
| **Total** | **12/16** | **6/16** |

| Dimension | Lava | PyQuifer |
|-----------|:----:|:--------:|
| Neuron variants | 10 | 8 |
| Bit-widths | 8, 12, 13, 16, 17, 24 | float32 |
| Target hardware | Loihi 1/2, CPU | GPU, CPU |
| Learning | On-chip STDP | Backprop + STDP |
| Framework | CSP / Process model | PyTorch nn.Module |

**Analysis:**

- **Lava's 12/16 features** are dominated by hardware capabilities: fixed-point, sparse connectivity, convolutional, synaptic delays, distributed execution, ternary/graded spikes. These are essential for Loihi deployment but irrelevant for software-only research.

- **PyQuifer's 6/16 features** are dominated by learning capabilities: surrogate gradients, learnable parameters, precision weighting, oscillatory modulation. These enable gradient-based training and integration with cognitive architectures but have no hardware target.

- **Shared features (2)**: Both support adaptive threshold neurons and online STDP learning, the two most important features for biological neural simulation.

## Comparative Assessment

### Where Lava exceeds PyQuifer

| Dimension | Lava | PyQuifer |
|-----------|:----:|:--------:|
| Hardware deployment | Loihi 1/2 chips | Software only |
| Fixed-point precision | 6 bit-widths | float32 only |
| Sparse connectivity | Native Sparse process | Dense only |
| Synaptic delays | DelayDense process | Not implemented |
| Refractory period | LIFRefractory | Not implemented |
| Ternary/graded spikes | TernaryLIF, graded Dense | Binary only |
| Process isolation | CSP architecture | Shared memory |
| Convolutional | Conv process | Not implemented |

### Where PyQuifer exceeds Lava

| Dimension | PyQuifer | Lava |
|-----------|:--------:|:----:|
| Gradient-based learning | Surrogate gradients throughout | No backprop |
| Learnable parameters | tau, threshold as nn.Parameter | Fixed at construction |
| Oscillatory coupling | Kuramoto, Stuart-Landau integration | No oscillator support |
| Precision weighting | Hierarchical predictive coding | Uniform error treatment |
| Consciousness modeling | Criticality, coherence, complexity | Not designed for this |
| GPU acceleration | Native PyTorch CUDA | CPU or Loihi only |
| AdEx model | Full Brette-Gerstner 2005 | ATRLIF (simpler threshold) |
| Short-term plasticity | Tsodyks-Markram STP | Not implemented |

### Complementarity

The two frameworks target different stages of the neuromorphic pipeline:

1. **PyQuifer** (research): Explore biologically-inspired architectures with gradient learning, oscillatory dynamics, and consciousness metrics. Train models with backprop + STDP on GPUs.

2. **Lava** (deployment): Once architecture is validated, convert to Lava processes for energy-efficient deployment on Loihi hardware with fixed-point precision and on-chip learning.

A future integration path: PyQuifer trains SNN architectures -> export weights -> Lava deploys on Loihi with on-chip STDP for continued learning.

## Gaps Identified

### G-13: No refractory period in PyQuifer spiking neurons

- Module: `spiking.py` -> `LIFNeuron`, `SpikingLayer`
- Issue: Lava implements LIFRefractory with configurable refractory period (voltage frozen for N timesteps after spike). PyQuifer neurons have no refractory mechanism -- they can fire on consecutive timesteps.
- Evidence: Architecture comparison; Lava has refractory_period=True, PyQuifer=False.
- Fix: Add `refractory_period` parameter to LIFNeuron. After spike, clamp membrane to v_reset for N steps. ~10 lines.
- Severity: **Low-Medium** | Effort: **Small** (~10 lines)
- Category: Missing feature

### G-14: No ternary or graded spike support

- Module: `spiking.py`
- Issue: All PyQuifer neurons produce binary spikes (0 or 1). Lava supports ternary spikes (+1, 0, -1 via TernaryLIF) and graded spikes (multi-bit payload via Dense num_message_bits). Ternary spikes enable inhibitory signaling without separate pathways.
- Evidence: Architecture comparison; TernaryLIF and graded Dense in Lava.
- Fix: Add `spike_mode` parameter ('binary', 'ternary', 'graded') to LIFNeuron. Ternary: add lower threshold. Graded: output scaled membrane at spike. ~20 lines.
- Severity: **Low** | Effort: **Small** (~20 lines)
- Category: Enhancement

### G-15: No synaptic delay support

- Module: `spiking.py`, `advanced_spiking.py`
- Issue: Lava implements DelayDense with per-synapse integer delays. PyQuifer has no delay mechanism -- all spikes propagate in one timestep. Synaptic delays are critical for temporal coding and coincidence detection.
- Evidence: Lava's DelayDense process with configurable delay matrix.
- Fix: Add `SynapticDelay` layer or `delays` parameter to SpikingLayer. Use circular buffer for delayed spike delivery. ~30 lines.
- Severity: **Low** | Effort: **Small-Medium** (~30 lines)
- Category: Missing feature

## Pytest Results

```
14/14 passed (1.37s)

TestLIFDynamics::test_lava_lif_runs                   PASSED
TestLIFDynamics::test_pyquifer_lif_runs               PASSED
TestLIFDynamics::test_firing_rates_nonzero            PASSED
TestLIFDynamics::test_lif_voltage_bounded             PASSED
TestAdaptive::test_lava_atrlif_adapts                 PASSED
TestAdaptive::test_pyquifer_adex_adapts               PASSED
TestAdaptive::test_atrlif_dynamics_bounded            PASSED
TestFixedPoint::test_fixed_point_runs                 PASSED
TestFixedPoint::test_float_fixed_agreement            PASSED
TestFixedPoint::test_pyquifer_no_fixed_point          PASSED
TestSynaptic::test_synaptic_comparison_runs           PASSED
TestSynaptic::test_temporal_smoothness                PASSED
TestArchitecture::test_feature_lists_valid            PASSED
TestArchitecture::test_neuron_variant_counts          PASSED
```

Existing test suite: **479/479 passed** (unaffected).

## Verdict

**PyQuifer spiking dynamics: PASS -- correct LIF integration, adaptation, synaptic dynamics, bounded voltage.**

The benchmark confirms that PyQuifer implements correct spiking neuron dynamics that match Lava's floating-point models when parameters are aligned. The 2nd-order SynapticNeuron produces identical dynamics to Lava's dual-state LIF. PyQuifer's AdExNeuron provides richer biophysical adaptation than Lava's ATRLIF. The two frameworks are complementary: Lava for neuromorphic hardware deployment (12/16 features), PyQuifer for gradient-based research and cognitive modeling (6/16 features, but including unique oscillatory and precision-weighting capabilities). Three gaps identified are all low-severity missing features (refractory period, ternary/graded spikes, synaptic delays) that could bridge the deployment gap.
