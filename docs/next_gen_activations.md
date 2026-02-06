# Next-Generation Activations in PyQuifer: A Research Program
_Revision incorporating external expert feedback._

## Introduction

This document outlines a research program to develop and integrate Tier 6 (Adaptive) and Tier 7 (Self-Regulating) systems within the `PyQuifer` framework. Moving beyond "architectural poetry," this program defines a series of concrete, falsifiable experiments designed to bridge the gap between abstract theory and a working, dynamic AI system.

**Guiding Philosophy (Research vs. Engineering):**
This roadmap is unapologetically a **research program**. The primary goal is to ask and answer fundamental questions about the possibility of creating self-regulating AI dynamics. Each phase is designed to produce robust, publishable findings. Engineering a final "product" is a secondary goal that will follow from the successful completion of the research.

---

## Tier 6: Hypernetwork-Generated Activations & Adaptive Nonlinearities

### 1. Conceptual Framework
Tier 6 proposes a shift towards **adaptive nonlinearities**, where a hypernetwork `H` dynamically generates the parameters `θ_A` for an activation function `A(x; θ_A)` based on a context vector `c`. This is inspired by the learnable splines in **Fourier Kolmogorov-Arnold Networks (FKAN, NeurIPS 2023)**.

### 2. Theoretical Grounding
*   **Neuromodulation:** The hypernetwork `H` acts as a functional analog of a neuromodulatory system, translating a global system state `c` into changes in local neuronal processing.
*   **Active Inference (FEP):** The hypernetwork is part of the agent's generative model, refining the model's structure to better predict sensory evidence, with the context `c` potentially representing the model's own precision estimates.

### 3. Bleeding-Edge Research & Future Directions (Post-2023)
The field is advancing rapidly. Our research will leverage and extend several key concepts:
*   **Automated Discovery:** Techniques like surrogate-based optimization (NeurIPS 2023) and LLM-driven evolutionary search (**`EvolveAct`**) can be used to build a library of high-performing activations for the hypernetwork to select from.
*   **Advanced Parametric Functions:** Commercially available, powerful parametric functions like **LEAF (2023)** will serve as crucial benchmarks for our more complex hypernetwork models.
*   **Implicit Hypernetworks:** Recent work reformulating attention as a hypernetwork suggests that the hypernetwork itself should be a non-trivial computational module, a principle we will adopt.

---

## Tier 7: Homeostatic Regulation of System Dynamics

### 1. Conceptual Framework: A Biological Computationalist Approach
Grounded in **Biological Computationalism**, this research views computation as an emergent property of a system's intrinsic, self-regulating dynamics. The core of Tier 7 is a **homeostatic regulator** (`M`) that implements principles of homeostatic plasticity (Dohare et al., 2023) to enforce stability and complexity in the core computational modules.

### 2. The "Observer Problem" & The IDB Metric
A deep literature search confirms that a scalable, differentiable proxy for Integrated Information (`Φ`) is an unsolved problem. To proceed scientifically, we will rename our goal:
*   **From "Φ-proxy" to "IDB (Integration-Differentiation Balance)":** We are not measuring consciousness. We are engineering a metric that we hypothesize correlates with the interesting and complex dynamics necessary for higher cognition.
*   **Pragmatic First Step:** The initial IDB metric will be a heuristic based on **spectral coherence** of the system's oscillators—differentiable, computable, and capturing the trade-off between disorganized chaos and rigid synchrony. The invention of a more principled, novel IDB metric remains a high-risk, high-reward goal for later stages.

### 3. Architectural Proposal: Homeostasis as the Meta-Controller
The Tier 7 feedback loop is a system of **homeostatic regulation**. A **Monitor (`G`)** computes the IDB metric and other state variables (e.g., mean firing rates). A **Homeostatic Regulator (`M`)**, operating on a slower timescale, receives this information and applies corrective updates to the core dynamics' parameters, `θ_F`. The hypothesis is that consciousness-like properties will *emerge* from a system that successfully maintains its own dynamic stability in a complex regime.

---

## Experimental Roadmap & Research Program

### Phase 0: Baseline Characterization
*   **Objective:** To establish a rigorous, quantitative baseline for the system's behavior *without* meta-control.
*   **Method:** Run the full `PyQuifer` system with static, hand-tuned parameters (`θ_F`) on a suite of benchmark tasks. Record all relevant metrics: task performance, IDB score, mean activity, parameter drift, etc.
*   **Success Criteria:** A comprehensive "control condition" dataset against which all future experiments will be measured.

### Phase 1: Tier 6 Implementation & Validation
*   **Experiment 1.1: `SplineHyperActivation` Benchmarking:** Implement and test the `SplineHyperActivation` module against baselines (GELU, LEAF) as previously outlined.
*   **Experiment 1.2 (Sanity Check): Proving the IDB Metric's Value:**
    *   **Objective:** To prove that the IDB metric is not just an arbitrary number, but actually predicts something useful about the system's state.
    *   **Method:** Using the baseline model from Phase 0, manually set the core parameters `θ_F` to values that the IDB metric predicts are "good" (high IDB) and "bad" (low IDB). Run the system on a learning task.
    *   **Success Criteria:** A clear correlation is observed—e.g., the high-IDB configurations lead to faster learning, more stable gradients, or better generalization than the low-IDB configurations.

### Phase 2: Open-Loop & Simple Closed-Loop Control
*   **Experiment 2.1: Landscape Mapping:**
    *   **Objective:** To produce a "phase diagram" of the system's dynamics, as previously described. This is a core scientific contribution.
    *   **Method:** Systematically sweep `θ_F` parameters (e.g., `coupling_strength`, `noise_level`) and plot the resulting IDB score, creating a 2D map of the system's "homeostatic landscape."
*   **Experiment 2.2: PI Control of Homeostasis:**
    *   **Objective:** To build the first, simplest closed-loop system that regulates its own internal state.
    *   **Method:** Implement a simple PI controller as the **Homeostatic Regulator (`M`)**. The controller's goal is to maintain the IDB score at a target setpoint by adjusting one or two parameters from `θ_F`.
    *   **Success Criteria:** The system demonstrates robust self-regulation, returning to its IDB setpoint after being perturbed. This proves the dynamics are controllable.

### Phase 3: Advanced Control & Analysis
*   **Experiment 3.1: Task-Conditioned Homeostasis:**
    *   **Objective:** To build a controller that learns to balance internal stability with external task demands.
    *   **Method:** Frame the problem for an appropriate algorithm.
        *   **Controller:** For a low-dimensional action space (2-3 parameters), use a derivative-free optimization method like **CMA-ES**, which is more sample-efficient than complex RL. PPO will be reserved for higher-dimensional control problems.
        *   **Reward:** The reward function will be `R = -L_task - β * |IDB_error|`, rewarding task performance while penalizing deviation from the homeostatic setpoint.
*   **Experiment 3.2: The "Catastrophe" Experiment:**
    *   **Objective:** To validate that the system possesses the rich, nonlinear dynamics claimed in its theoretical foundation.
    *   **Method:** Using the landscape map from 2.1, identify a parameter region corresponding to a bifurcation point (a phase transition). Design an experiment that intentionally pushes the controller to steer the system across this boundary.
    *   **Success Criteria:** The system's global state exhibits a sudden, discontinuous jump, which is documented and analyzed. This provides powerful evidence of complex, emergent behavior.

### Cross-Phase Mandates
*   **Failure Mode Analysis & Safety:** All controller experiments must include pre-defined safety bounds on parameter adjustments to prevent pathological states (e.g., seizure-like synchronization, oscillatory instability, total quiescence). System monitors must be in place to detect and flag these failure modes.
*   **Explicit Timescale Separation:** The ratio between the fast core dynamics and the slow controller updates is a critical hyperparameter. All experiments must explicitly define and study this ratio, starting with a conservative target (e.g., 100:1) and testing its impact on stability and performance.