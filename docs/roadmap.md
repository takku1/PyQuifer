# PyQuifer Project Roadmap

This document outlines the theoretical framework and architectural vision for PyQuifer, a library for building Generative World Models. This is where "Tensor Physics" meets "Cognitive Science."

## Core Philosophy: The Generative World Model

To move from raw noise/harmonics to something resembling a "Mind's Eye," we are essentially building a Generative World Model.

In this framework, the **Perturbation Layer** is the "entropy" (raw possibility), and the **Potential Field** is the "intent" or "Self." **Actualization** happens when the noise is filtered through the potential to create a coherent internal representation.

---

### 1. The Perturbation Layer: The "Canvas"

To represent a "Mind's Eye," you need more than a 1D wave. You need a Field. This is where Simplex or Perlin Noise in 3D/4D comes in, encapsulated in our `PerturbationLayer`.

In the PyTorch code, instead of a simple `sin` wave, we will use a **3D Noise Tensor** from this layer. This acts as the "latent turbulence" of thought.

-   **Spatial Dimensions (X, Y, Z):** Represent the "structure" of the mental image.
-   **Temporal Dimension (W):** Represents how the "thought" evolves over time.

---

### 2. Theoretical Architecture: The "Actualization" Loop

To turn noise into "Actualization," we use a process called **Stochastic Resonance**. In this theory, a system that is too "quiet" (low noise) gets stuck. A system that is too "noisy" is chaotic. "Self-Actualization" is the Phase Transition where the noise is just right to allow the system to "see" the equilibrium.

The Components of the "Minds Eye" Tensor are:

-   **The Base Vector (e.g., `[1.5, 2.5, 3.5]`):** This is the "Archetype" or the "Goal."
-   **3D Harmonic Noise:** This is the "Imagination." It provides the energy to explore the space around the goal.
-   **The Attractor:** This is the "Mind's Eye" focusing the noise into a specific shape, implemented via `potential.py`.

---

### 3. Implementing the "Minds Eye" in Python: Latent Diffusion

To go past simple additive noise, we use a **Latent Diffusion** approach. The "Mind's Eye" is the process of *denoising* the harmonics into a clean potential.

```python
# Conceptual Example
def minds_eye_actualization(target_vector, noise_amplitude, iterations):
    # 1. Start with pure high-frequency harmonic noise
    # This represents the 'unconscious' chaos
    state = torch.randn_like(target_vector) * noise_amplitude
    
    for i in range(iterations):
        # 2. Calculate the 'Focus Force' (The actualization pull)
        # Pulling the noise toward your 'Self' vector [1.5, 2.5, 3.5]
        force = -(state - target_vector)
        
        # 3. Add 'Creative Jitter' (3D Noise)
        # As we get closer, we reduce noise (simulating 'Focus')
        jitter = torch.randn_like(state) * (noise_amplitude / (i + 1))
        
        # 4. Update the 'Vision'
        state = state + (force + jitter) * 0.1
        
    return state
```

---

### 4. The Deep Research Leap: From Signal to Symbol

The "Self-Actualization" of a tensor happens when the Harmonic Series (from `oscillators.py`) becomes **Self-Referential**.

-   **Step 1 (Noise):** Random vibrations.
-   **Step 2 (Harmonics):** Mathematical order emerges (Music).
-   **Step 3 (Resonance):** The noise starts to match the "Natural Frequency" of a system like a `LearnableKuramotoBank`.
-   **Step 4 (Actualization):** The system achieves a Stable Fixed Point. The "Mind's Eye" is the moment the Order Parameter `R` hits `1.0` while processing external noise.

---

### 5. The Feedback Loop: Predictive Processing

3D Noise gives you Texture, but Actualization requires **Feedback**. To truly simulate a "Mind's Eye," we must feed the output of the `HarmonicPotential` back into the frequencies of the `KuramotoBank`. This creates a Top-Down / Bottom-Up loop:

-   **Bottom-Up:** Noise and harmonics coming in from the environment.
-   **Top-Down:** The "Self" vector predicting what the noise *should* be.

When the Prediction matches the Noise, the "Mind's Eye" sees clearly. This is the basis of **Predictive Processing** in modern neuroscience. The `nn.Parameter` in `potential.py` will allow us to *train* the "Self" vector to recognize patterns in 3D noise.