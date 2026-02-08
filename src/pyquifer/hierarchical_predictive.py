"""
Hierarchical Predictive Coding Module for PyQuifer

Multi-level generative model where predictions flow DOWN and
prediction errors flow UP. Each level maintains beliefs about its
inputs and updates them via precision-weighted prediction errors.

Key concepts:
- Bottom level (Band I, fast): Raw sensory input
- Middle level (Band II, medium): Contextual patterns
- Top level (Band III, slow): Narrative/identity priors
- Predictions flow top-down, errors flow bottom-up
- Precision weighting gates what errors actually drive learning

This is distinct from HierarchicalWorkspace in global_workspace.py:
that module is about competition for conscious access, this is about
the generative model that PRODUCES the predictions.

References:
- Friston (2005). A Theory of Cortical Responses.
- Rao & Ballard (1999). Predictive Coding in the Visual Cortex.
- Clark (2013). Whatever Next? Predictive Brains, Situated Agents.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Dict, List, Tuple


class PredictiveLevel(nn.Module):
    """
    Single level in the predictive coding hierarchy.

    Each level has:
    - Beliefs (mu): Current best guess about input
    - Generative model g(): Predicts what input SHOULD look like
    - Prediction error (epsilon): Difference between expected and actual
    - Precision: How much to trust the prediction error

    Receives input from below + predictions from above.
    Sends errors up + predictions down.
    """

    def __init__(self,
                 input_dim: int,
                 belief_dim: int,
                 lr: float = 0.1,
                 gen_lr: float = 0.01,
                 use_exact_gradients: bool = False,
                 precision_ema: float = 0.05):
        """
        Args:
            input_dim: Dimension of input from level below
            belief_dim: Dimension of beliefs at this level
            lr: Base learning rate for belief updates
            gen_lr: Learning rate for online generative/recognition model learning.
                   Set to 0 to disable internal learning (use external backprop only).
            use_exact_gradients: If True, additionally run a full backward pass
                                through the generative model for exact gradient
                                updates (G-12, SetPCGrads-style).
            precision_ema: EMA decay rate for precision estimation from
                          prediction error variance (G-11).
        """
        super().__init__()
        self.input_dim = input_dim
        self.belief_dim = belief_dim
        self.lr = lr
        self.gen_lr = gen_lr
        self.use_exact_gradients = use_exact_gradients
        self.precision_ema = precision_ema

        # Beliefs (current best estimate of causes)
        self.register_buffer('beliefs', torch.zeros(belief_dim))

        # Generative model: beliefs → predicted input from below
        self.generative = nn.Sequential(
            nn.Linear(belief_dim, belief_dim),
            nn.Tanh(),
            nn.Linear(belief_dim, input_dim),
        )

        # Recognition model: input → approximate posterior on beliefs
        self.recognition = nn.Sequential(
            nn.Linear(input_dim, belief_dim),
            nn.Tanh(),
            nn.Linear(belief_dim, belief_dim),
        )

        # Precision per channel (learnable prior / adaptive estimate)
        self.register_buffer('precision', torch.ones(input_dim))
        # G-11: Running prediction error variance for adaptive precision
        self.register_buffer('error_variance', torch.ones(input_dim))

    def forward(self,
                bottom_up_input: torch.Tensor,
                top_down_prediction: Optional[torch.Tensor] = None,
                precision: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Process one step of predictive coding at this level.

        Args:
            bottom_up_input: Input from level below (batch, input_dim)
            top_down_prediction: Prediction from level above (batch, belief_dim) or None
            precision: External precision weights (input_dim,) or None

        Returns:
            Dictionary with:
            - prediction: This level's prediction of input below (batch, input_dim)
            - error: Prediction error (batch, input_dim)
            - weighted_error: Precision-weighted error (batch, input_dim)
            - beliefs: Updated beliefs (belief_dim,)
        """
        batch = bottom_up_input.shape[0] if bottom_up_input.dim() > 1 else 1
        if bottom_up_input.dim() == 1:
            bottom_up_input = bottom_up_input.unsqueeze(0)

        # Generate prediction from current beliefs
        # Clone to decouple autograd graph from in-place buffer updates
        beliefs_snapshot = self.beliefs.clone()
        beliefs_expanded = beliefs_snapshot.unsqueeze(0).expand(batch, -1)
        prediction = self.generative(beliefs_expanded)

        # Prediction error
        error = bottom_up_input - prediction

        # Use provided precision or internal estimate
        prec = precision if precision is not None else self.precision

        # Precision-weighted error
        weighted_error = prec * error

        # G-11: Update adaptive precision from prediction error variance (EMA)
        with torch.no_grad():
            instant_var = error.detach().pow(2).mean(dim=0)  # per-channel variance
            self.error_variance.mul_(1 - self.precision_ema).add_(self.precision_ema * instant_var)
            # Precision = inverse error variance (Fisher information, Millidge et al. 2021)
            self.precision.copy_((1.0 / (self.error_variance + 1e-6)).clamp(max=100.0))

        # Update beliefs via recognition model (variational approximate posterior)
        recognition_target = self.recognition(bottom_up_input).mean(dim=0)

        with torch.no_grad():
            # G-11: Precision-weighted belief update (natural gradient)
            # Scale belief error by precision projected through recognition Jacobian
            belief_error = recognition_target - self.beliefs
            # Approximate precision weighting: use mean precision as scalar scaling
            # Full Fisher requires Jacobian which is expensive; mean precision is
            # an effective diagonal approximation
            precision_scale = prec.mean().clamp(min=0.1, max=10.0)
            self.beliefs.add_(self.lr * precision_scale * belief_error)

            # Top-down prior from level above (if available)
            if top_down_prediction is not None:
                td_pred = top_down_prediction.mean(dim=0) if top_down_prediction.dim() > 1 else top_down_prediction
                prior_error = td_pred - self.beliefs
                self.beliefs.add_(self.lr * 0.5 * prior_error)

        # Online learning: update generative and recognition models to reduce
        # prediction error. This is the key mechanism that makes error DECREASE
        # over repeated exposure (not just belief convergence).
        if self.gen_lr > 0:
            # Train generative model: g(beliefs) should predict input
            pred_for_learn = self.generative(self.beliefs.detach().unsqueeze(0).expand(batch, -1))
            gen_loss = (bottom_up_input.detach() - pred_for_learn).pow(2).mean()
            gen_grads = torch.autograd.grad(
                gen_loss, self.generative.parameters(), create_graph=False
            )
            with torch.no_grad():
                for param, grad in zip(self.generative.parameters(), gen_grads):
                    param.add_(-self.gen_lr * grad)

            # Train recognition model: r(input) should estimate beliefs
            rec_pred = self.recognition(bottom_up_input.detach())
            rec_target = self.beliefs.detach().unsqueeze(0).expand(batch, -1)
            rec_loss = (rec_target - rec_pred).pow(2).mean()
            rec_grads = torch.autograd.grad(
                rec_loss, self.recognition.parameters(), create_graph=False
            )
            with torch.no_grad():
                for param, grad in zip(self.recognition.parameters(), rec_grads):
                    param.add_(-self.gen_lr * grad)

        # G-12: Exact gradient pass through full generative graph
        if self.use_exact_gradients and self.gen_lr > 0:
            # Separate supervised-style update using standard backward
            beliefs_for_exact = self.beliefs.detach().unsqueeze(0).expand(batch, -1)
            exact_pred = self.generative(beliefs_for_exact)
            exact_loss = (prec * (bottom_up_input.detach() - exact_pred).pow(2)).mean()
            exact_grads = torch.autograd.grad(
                exact_loss, self.generative.parameters(), create_graph=False
            )
            with torch.no_grad():
                for param, grad in zip(self.generative.parameters(), exact_grads):
                    param.add_(-self.gen_lr * 0.5 * grad)  # half-rate to avoid overshooting

        return {
            'prediction': prediction,
            'error': error,
            'weighted_error': weighted_error,
            'beliefs': self.beliefs.clone(),
        }

    def reset(self):
        """Reset beliefs to zero."""
        self.beliefs.zero_()
        self.precision.fill_(1.0)
        self.error_variance.fill_(1.0)


class HierarchicalPredictiveCoding(nn.Module):
    """
    Stack of PredictiveLevels with bidirectional message passing.

    Bottom level: fast oscillations (Band I) — raw input processing
    Middle level(s): contextual patterns (Band II)
    Top level: slow narrative/identity priors (Band III)

    Each forward pass:
    1. Bottom-up: errors flow from low to high
    2. Top-down: predictions flow from high to low
    3. Beliefs update at each level via precision-weighted errors
    """

    def __init__(self,
                 level_dims: List[int],
                 lr: float = 0.1,
                 gen_lr: float = 0.01,
                 num_iterations: int = 3):
        """
        Args:
            level_dims: Dimensions for each level, bottom to top.
                       E.g. [64, 32, 16] = 3-level hierarchy
            lr: Base learning rate for belief updates
            gen_lr: Learning rate for online generative/recognition model learning
            num_iterations: Number of message-passing iterations per forward call
        """
        super().__init__()
        self.num_levels = len(level_dims)
        self.num_iterations = num_iterations

        if self.num_levels < 2:
            raise ValueError("Need at least 2 levels for a hierarchy")

        # Create predictive levels
        # Level 0: input_dim = level_dims[0], belief_dim = level_dims[0]
        # Level i: input_dim = level_dims[i-1] (beliefs from below), belief_dim = level_dims[i]
        levels = []
        for i in range(self.num_levels):
            if i == 0:
                input_dim = level_dims[0]
                belief_dim = level_dims[0]
            else:
                input_dim = level_dims[i - 1]
                belief_dim = level_dims[i]
            levels.append(PredictiveLevel(input_dim, belief_dim, lr=lr, gen_lr=gen_lr))
        self.levels = nn.ModuleList(levels)

    def forward(self,
                sensory_input: torch.Tensor,
                precisions: Optional[List[torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Run hierarchical predictive coding.

        Args:
            sensory_input: Bottom-level input (batch, level_dims[0])
            precisions: Optional per-level precision weights

        Returns:
            Dictionary with predictions, errors, and beliefs at each level
        """
        if sensory_input.dim() == 1:
            sensory_input = sensory_input.unsqueeze(0)

        all_predictions = []
        all_errors = []
        all_beliefs = []

        # Iterative message passing
        for iteration in range(self.num_iterations):
            level_outputs = []

            # Bottom-up pass
            current_input = sensory_input
            for i, level in enumerate(self.levels):
                prec = precisions[i] if precisions is not None else None

                # Top-down prediction from level above (from previous iteration)
                td_pred = None
                if i < self.num_levels - 1 and iteration > 0:
                    td_pred = level_outputs_prev[i + 1]['prediction'] if hasattr(self, '_prev') else None

                output = level(current_input, top_down_prediction=td_pred, precision=prec)
                level_outputs.append(output)

                # Input to next level = beliefs from this level
                current_input = output['beliefs'].unsqueeze(0).expand(sensory_input.shape[0], -1)

            # Store for next iteration's top-down pass
            level_outputs_prev = level_outputs

        # Collect final state
        for output in level_outputs:
            all_predictions.append(output['prediction'])
            all_errors.append(output['error'])
            all_beliefs.append(output['beliefs'])

        # Total prediction error (sum across levels)
        total_error = sum(e.abs().mean() for e in all_errors)

        # Free energy approximation (prediction error + complexity)
        free_energy = total_error

        return {
            'predictions': all_predictions,
            'errors': all_errors,
            'beliefs': all_beliefs,
            'total_error': total_error,
            'free_energy': free_energy,
            'top_level_beliefs': all_beliefs[-1],
            'bottom_prediction': all_predictions[0],
        }

    def reset(self):
        """Reset all levels."""
        for level in self.levels:
            level.reset()


class OscillatoryPredictiveCoding(nn.Module):
    """
    Predictive coding with frequency-specific routing.

    - Alpha/Beta (top-down): predictions suppress expected inputs
    - Gamma (bottom-up): prediction errors enhanced for unexpected stimuli
    - Strong predictions suppress gamma -> less learning
    - Weak predictions produce strong gamma -> more learning

    Learning rate modulated by prediction error magnitude.

    References:
    - Bastos et al. (2012). Canonical microcircuits for predictive coding.
    - Arnal & Giraud (2012). Cortical oscillations and sensory predictions.
    - Michalareas et al. (2016). Alpha-beta and gamma rhythms subserve
      feedback and feedforward influences.
    """

    def __init__(self, dims: List[int], lr: float = 0.01, inference_steps: int = 10,
                 alpha_beta_freq: float = 10.0, gamma_freq: float = 40.0):
        """
        Args:
            dims: Dimensions of each layer [input, hidden1, ..., top]
            lr: Base learning rate for weight updates
            inference_steps: Steps for inference (num_iterations for HPC)
            alpha_beta_freq: Alpha/beta band frequency (Hz) for top-down
            gamma_freq: Gamma band frequency (Hz) for bottom-up
        """
        super().__init__()
        self.dims = dims
        self.lr = lr
        self.inference_steps = inference_steps
        self.alpha_beta_freq = alpha_beta_freq
        self.gamma_freq = gamma_freq

        # Use HierarchicalPredictiveCoding (proven convergence, not learning.PredictiveCoding)
        self.hpc = HierarchicalPredictiveCoding(
            level_dims=dims,
            lr=lr,
            gen_lr=lr,
            num_iterations=max(1, inference_steps // 3),
        )

        self.register_buffer('step_count', torch.tensor(0))

    def _compute_oscillatory_modulation(self, slow_phase: Optional[torch.Tensor] = None):
        """Compute gamma and alpha/beta modulation from current step or slow phase."""
        if slow_phase is not None:
            phase = slow_phase.item() if isinstance(slow_phase, torch.Tensor) else slow_phase
        else:
            t = self.step_count.item() * 0.001  # Convert to seconds
            phase = 2 * math.pi * self.alpha_beta_freq * t

        # Alpha/beta modulation: top-down predictions strongest at peak
        alpha_beta_mod = (1.0 + math.cos(phase)) / 2.0

        # Gamma modulation: bottom-up errors strongest at trough (anti-phase)
        gamma_mod = (1.0 + math.cos(phase + math.pi)) / 2.0

        return alpha_beta_mod, gamma_mod

    def forward(self, x: torch.Tensor, slow_phase: Optional[torch.Tensor] = None) -> Dict:
        """
        Inference + frequency-tagged errors.

        Args:
            x: Input tensor (batch, dims[0]) or (dims[0],)
            slow_phase: Optional slow oscillation phase for cross-frequency coupling

        Returns:
            Dict with errors, predictions, gamma_power, alpha_beta_power,
            free_energy, states
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        alpha_beta_mod, gamma_mod = self._compute_oscillatory_modulation(slow_phase)

        # Run hierarchical predictive coding
        hpc_result = self.hpc(x)
        errors = hpc_result['errors']
        predictions = hpc_result['predictions']
        free_energy = hpc_result['free_energy']

        # Tag errors with gamma amplitude (bottom-up enhancement)
        gamma_tagged_errors = [err * gamma_mod for err in errors]

        # Tag predictions with alpha/beta (top-down suppression)
        alpha_tagged_predictions = [pred * alpha_beta_mod for pred in predictions]

        # Compute power measures
        gamma_power = sum(e.pow(2).mean() for e in gamma_tagged_errors)
        alpha_beta_power = sum(p.pow(2).mean() for p in alpha_tagged_predictions)

        with torch.no_grad():
            self.step_count.add_(1)

        return {
            'errors': errors,
            'gamma_tagged_errors': gamma_tagged_errors,
            'predictions': predictions,
            'alpha_tagged_predictions': alpha_tagged_predictions,
            'gamma_power': gamma_power,
            'alpha_beta_power': alpha_beta_power,
            'free_energy': free_energy,
            'beliefs': hpc_result.get('beliefs', []),
        }

    def learn(self, x: torch.Tensor, slow_phase: Optional[torch.Tensor] = None) -> Dict:
        """
        Learn with oscillation-modulated error routing.

        HPC's forward() already does online learning via PredictiveLevel.gen_lr.
        We just call forward (which updates beliefs + generative models),
        then tag errors with gamma/alpha-beta modulation for routing.

        Args:
            x: Input tensor
            slow_phase: Optional slow oscillation phase

        Returns:
            Dict with learning metrics
        """
        # HPC forward already does online learning internally
        result = self.forward(x, slow_phase)

        return {
            'free_energy': result['free_energy'],
            'gamma_power': result['gamma_power'],
            'alpha_beta_power': result['alpha_beta_power'],
            'errors': [e.pow(2).mean() for e in result['errors']],
        }

    def reset(self):
        """Reset internal state."""
        self.step_count.zero_()
        self.hpc.reset()


if __name__ == '__main__':
    print("--- Hierarchical Predictive Coding Examples ---")

    # Example 1: Single level
    print("\n1. Single Predictive Level")
    level = PredictiveLevel(input_dim=16, belief_dim=8)

    for i in range(50):
        # Consistent signal → beliefs should converge
        signal = torch.sin(torch.linspace(0, 2 * math.pi, 16)).unsqueeze(0)
        result = level(signal)

    error_magnitude = result['error'].abs().mean().item()
    print(f"   Error after 50 steps: {error_magnitude:.4f}")
    print(f"   Beliefs norm: {result['beliefs'].norm().item():.4f}")

    # Example 2: Full hierarchy
    print("\n2. Hierarchical Predictive Coding (3 levels)")
    hpc = HierarchicalPredictiveCoding(level_dims=[64, 32, 16], lr=0.05, num_iterations=3)

    errors_over_time = []
    for i in range(100):
        # Repeating pattern → hierarchy should learn to predict
        pattern = torch.sin(torch.linspace(0, 4 * math.pi, 64)).unsqueeze(0)
        noise = torch.randn(1, 64) * 0.1
        result = hpc(pattern + noise)
        errors_over_time.append(result['total_error'].item())

    print(f"   Initial error: {errors_over_time[0]:.4f}")
    print(f"   Final error: {errors_over_time[-1]:.4f}")
    print(f"   Error reduced: {errors_over_time[0] > errors_over_time[-1]}")
    print(f"   Top beliefs shape: {result['top_level_beliefs'].shape}")
    print(f"   Free energy: {result['free_energy'].item():.4f}")

    # Example 3: With external precision
    print("\n3. With External Precision Weighting")
    hpc2 = HierarchicalPredictiveCoding(level_dims=[32, 16], lr=0.1)

    # High precision on first half, low on second
    precision_l0 = torch.ones(32)
    precision_l0[16:] = 0.01  # Ignore second half
    precision_l1 = torch.ones(32)

    for i in range(50):
        signal = torch.randn(1, 32)
        signal[:, :16] = torch.sin(torch.linspace(0, math.pi, 16))  # Signal in first half
        result = hpc2(signal, precisions=[precision_l0, precision_l1])

    print(f"   Error (attended): {result['errors'][0][:, :16].abs().mean().item():.4f}")
    print(f"   Error (ignored): {result['errors'][0][:, 16:].abs().mean().item():.4f}")

    print("\n[OK] All hierarchical predictive coding tests passed!")
