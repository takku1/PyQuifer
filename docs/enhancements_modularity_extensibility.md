# Modularity and Extensibility Enhancements for PyQuifer

This document proposes improvements to PyQuifer's architecture to enhance its modularity, making it easier to extend, modify, and integrate new components without significant refactoring.

## 1. Clarify and Modularize `Snake` Activation Function Usage

**Current State:**
The `Snake` activation function is defined in `src/pyquifer/oscillators.py` alongside the `LearnableKuramotoBank`. Its presence there is somewhat unexpected, as it is a general-purpose activation function not inherently tied to oscillators. Its specific usage within the overall PyQuifer architecture (i.e., which layers or models employ it) is not immediately clear from the current code structure.

**Proposed Enhancements:**

*   **Dedicated `activations.py` Module:** Create a new module, e.g., `src/pyquifer/activations.py`, to house `Snake` and any other custom or specialized activation functions. This improves logical grouping and makes it easier to find and manage activation functions.
*   **Explicit Integration:** If `Snake` is intended for use within `GenerativeWorldModel`, `MindEyeActualization`, or other `nn.Module`s (e.g., in internal `nn.Linear` layers), make this explicit. Add comments or modify the respective modules to clearly show where `Snake` is applied. For example:
    ```python
    # In GenerativeWorldModel or MindEyeActualization:
    self.some_linear_layer = nn.Linear(input_dim, output_dim)
    self.activation = Snake()
    # ...
    output = self.activation(self.some_linear_layer(x))
    ```
*   **Documentation:** Update the project documentation to explain the purpose and intended use cases for `Snake` within the PyQuifer framework.

**Impact:**
A dedicated module for activation functions improves code organization and discoverability. Explicit integration makes the architecture easier to understand and allows for more flexible experimentation with different activation functions in various parts of the model.

## 2. Configurable Potential Functions in `MultiAttractorPotential`

**Current State:**
The `MultiAttractorPotential` (in `src/pyquifer/potentials.py`) hardcodes the potential function as an inverse-squared distance relationship (`-strengths / dist_sq`). While this is a common choice, different potential functions can lead to vastly different emergent dynamics and might be desirable for various applications or theoretical explorations.

**Proposed Enhancements:**

*   **Pluggable Potential Kernels:** Refactor `MultiAttractorPotential` to accept a configurable "potential kernel" or "interaction function" during initialization. This kernel would be a callable (e.g., a small `nn.Module` or a function) that takes `distances` (or `squared_distances`) and `strengths` as input and returns the potential contribution.
    ```python
    # Example:
    def inverse_squared_kernel(dist_sq, strength):
        return -strength / (dist_sq + 1e-8)

    def gaussian_kernel(dist_sq, strength, sigma):
        return -strength * torch.exp(-dist_sq / (2 * sigma**2))

    class MultiAttractorPotential(...):
        def __init__(self, ..., potential_kernel: Callable = inverse_squared_kernel, kernel_params: dict = None):
            super().__init__()
            self.potential_kernel = potential_kernel
            self.kernel_params = kernel_params if kernel_params is not None else {}
            # ...

        def forward(self, ...):
            # ...
            potential_per_attractor = self.potential_kernel(dist_sq, strengths, **self.kernel_params)
            # ...
    ```
*   **Pre-defined Kernels and Customization:** Provide a few common potential kernels (e.g., inverse-squared, Gaussian, Lennard-Jones like) as part of the library, while allowing users to easily implement and plug in their own custom functions.

**Impact:**
This enhancement greatly improves the flexibility and extensibility of the `MultiAttractorPotential` module, enabling researchers to experiment with different interaction dynamics within the potential field without modifying the core class.

## 3. Implement a Flexible Logging and Visualization Framework

**Current State:**
The current project relies on `print()` statements for basic feedback during initialization, data ingestion, and simulation loops. While useful for quick debugging, this approach lacks the sophistication needed for comprehensive monitoring, analysis, and visualization of complex model dynamics over long training runs.

**Proposed Enhancements:**

*   **Structured Logging:**
    *   **Python `logging` Module:** Integrate Python's standard `logging` module throughout the codebase. Configure different logging levels (DEBUG, INFO, WARNING, ERROR) to control verbosity.
    *   **Configurable Loggers:** Allow users to easily configure log file outputs, console outputs, and formatters via a configuration file (e.g., `logging.conf` or a dictionary configuration).
*   **Experiment Tracking and Metrics Visualization:**
    *   **TensorBoard/Weights & Biases (WandB):** Integrate with a popular experiment tracking tool like TensorBoard (built-in with PyTorch) or Weights & Biases. This would allow for:
        *   **Scalar Logging:** Tracking loss, Kuramoto R, actualization strength, learning rates, etc., over epochs/steps.
        *   **Histogram Logging:** Visualizing distributions of learnable parameters (e.g., `natural_frequencies`, `attractor_positions`) over time.
        *   **Image Logging:** Visualizing slices of the `generated_noise_field`, potential landscapes (2D/3D), or phase plots of oscillators.
        *   **Model Graph Visualization:** Automatically logging the computational graph.
    *   **Custom Callbacks:** Provide a mechanism (e.g., a base `LoggerCallback` class) that users can extend to implement custom logging behavior or to interface with other experiment tracking tools.
*   **Basic In-notebook Visualizations:** Enhance the `if __name__ == '__main__':` blocks and add utility functions for basic plotting (e.g., `matplotlib`, `seaborn`) that can be used directly in Jupyter notebooks for quick inspection.

**Impact:**
A robust logging and visualization framework is crucial for understanding the behavior of complex generative models like PyQuifer. It will significantly improve the development workflow, facilitate hyperparameter tuning, and aid in presenting research results. This makes the library more research-friendly and user-friendly for complex model analysis.
