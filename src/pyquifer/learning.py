"""
Learning Rules Beyond Backpropagation for PyQuifer

Implements biologically plausible learning rules that complement or
replace standard gradient descent:
- Eligibility traces: Mark synapses for future reward modulation
- Reward-modulated learning: Hebbian + global reward signal
- Contrastive Hebbian: Learn from difference between states
- Three-factor learning: Pre * Post * Reward

These enable learning from sparse/delayed rewards and support
the intrinsic motivation system.

Based on work by Hebb, Gerstner, Sutton, Friston.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Dict, Callable, List


class EligibilityTrace(nn.Module):
    """
    Eligibility traces for credit assignment over time.

    Marks recently active synapses as "eligible" for modification
    when a reward signal arrives later. Enables learning from
    delayed rewards without full backprop through time.

    The trace decays exponentially, creating a temporal credit window.
    """

    def __init__(self,
                 shape: Tuple[int, ...],
                 decay_rate: float = 0.95,
                 accumulation_rate: float = 0.1):
        """
        Args:
            shape: Shape of the parameter to track (e.g., weight matrix shape)
            decay_rate: How fast traces decay (0-1, higher = longer memory)
            accumulation_rate: How fast new activity accumulates
        """
        super().__init__()
        self.decay_rate = decay_rate
        self.accumulation_rate = accumulation_rate

        # The eligibility trace
        self.register_buffer('trace', torch.zeros(shape))

        # Separate traces for LTP and LTD (Dale's law: excitatory/inhibitory)
        # Access via .trace_pos and .trace_neg after forward()
        self.register_buffer('trace_pos', torch.zeros(shape))
        self.register_buffer('trace_neg', torch.zeros(shape))

    def forward(self,
                activity: torch.Tensor,
                pre_activity: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Update eligibility trace based on current activity.

        Args:
            activity: Current activity/gradient signal (same shape as trace)
            pre_activity: Optional presynaptic activity for Hebbian-style traces

        Returns:
            Current trace value
        """
        # Decay existing trace and accumulate new activity
        with torch.no_grad():
            self.trace.mul_(self.decay_rate)

            if pre_activity is not None:
                # Hebbian-style: trace depends on pre-post correlation
                if activity.dim() == 1 and pre_activity.dim() == 1:
                    hebbian = torch.outer(activity, pre_activity)
                else:
                    hebbian = activity * pre_activity
                self.trace.add_(self.accumulation_rate * hebbian)
            else:
                self.trace.add_(self.accumulation_rate * activity)

            # Update signed traces
            self.trace_pos.copy_(torch.relu(self.trace))
            self.trace_neg.copy_(torch.relu(-self.trace))

        return self.trace.clone()

    def apply_reward(self, reward: torch.Tensor, learning_rate: float = 0.01) -> torch.Tensor:
        """
        Compute weight update based on trace and reward.

        Args:
            reward: Reward signal (scalar or broadcastable)
            learning_rate: Learning rate for the update

        Returns:
            Weight delta to apply
        """
        # Reward modulates the trace to produce update
        delta = learning_rate * reward * self.trace
        return delta

    def reset(self):
        """Reset traces to zero."""
        self.trace.zero_()
        self.trace_pos.zero_()
        self.trace_neg.zero_()


class RewardModulatedHebbian(nn.Module):
    """
    Three-factor learning rule: Pre * Post * Reward.

    Combines local Hebbian correlation with global reward modulation.
    Synapses that were active before reward get strengthened.
    This bridges unsupervised Hebbian learning with goal-directed behavior.

    Can use PyQuifer's intrinsic motivation signals as reward.
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 trace_decay: float = 0.9,
                 learning_rate: float = 0.01,
                 weight_decay: float = 0.0001):
        """
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            trace_decay: Eligibility trace decay rate
            learning_rate: Base learning rate
            weight_decay: L2 regularization
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Learnable weights
        self.weights = nn.Parameter(
            torch.randn(output_dim, input_dim) * 0.1
        )
        self.bias = nn.Parameter(torch.zeros(output_dim))

        # Eligibility trace for weights
        self.trace = EligibilityTrace(
            (output_dim, input_dim),
            decay_rate=trace_decay
        )

        # Activity buffers
        self.register_buffer('last_input', torch.zeros(input_dim))
        self.register_buffer('last_output', torch.zeros(output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with trace update.

        Args:
            x: Input tensor (batch, input_dim) or (input_dim,)

        Returns:
            Output tensor
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Linear transform
        output = torch.matmul(x, self.weights.T) + self.bias

        # Update traces with Hebbian correlation
        # Average over batch for trace update
        with torch.no_grad():
            self.last_input = x.mean(dim=0)
            self.last_output = output.mean(dim=0)
            self.trace(self.last_output, self.last_input)

        return output.squeeze(0) if output.shape[0] == 1 else output

    def reward_update(self, reward: float) -> None:
        """
        Apply reward-modulated update to weights.

        Args:
            reward: Reward signal (positive = strengthen, negative = weaken)
        """
        reward_tensor = torch.tensor(reward, device=self.weights.device)

        with torch.no_grad():
            # Get update from trace
            delta = self.trace.apply_reward(reward_tensor, self.learning_rate)

            # Apply update with weight decay
            self.weights.mul_(1 - self.weight_decay).add_(delta)

    def reset_traces(self):
        """Reset eligibility traces."""
        self.trace.reset()


class ContrastiveHebbian(nn.Module):
    """
    Contrastive Hebbian Learning (CHL).

    Learns from the difference between two network states:
    - "Free" phase: Network settles freely
    - "Clamped" phase: Output is clamped to target

    Update = (clamped correlation) - (free correlation)

    This is more biologically plausible than backprop and
    naturally handles recurrent networks.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 learning_rate: float = 0.01,
                 settle_steps: int = 10):
        """
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            learning_rate: Learning rate
            settle_steps: Steps to let network settle
        """
        super().__init__()
        self.settle_steps = settle_steps
        self.learning_rate = learning_rate

        # Weights (symmetric for energy-based settling)
        self.W_ih = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.1)
        self.W_ho = nn.Parameter(torch.randn(output_dim, hidden_dim) * 0.1)

        # For recurrent settling
        self.W_hh = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.05)

    def settle(self,
               x: torch.Tensor,
               target: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Let network settle to equilibrium.

        Args:
            x: Input
            target: Optional target for clamped phase

        Returns:
            hidden: Hidden state at equilibrium
            output: Output state at equilibrium
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Initialize hidden
        hidden = torch.tanh(torch.matmul(x, self.W_ih.T))

        # Settle
        for _ in range(self.settle_steps):
            # Recurrent update
            hidden_input = torch.matmul(x, self.W_ih.T)
            hidden_recurrent = torch.matmul(hidden, self.W_hh.T)

            if target is not None:
                # Clamped: also get signal from output
                output_feedback = torch.matmul(target, self.W_ho)
                hidden = torch.tanh(hidden_input + hidden_recurrent + output_feedback * 0.5)
            else:
                hidden = torch.tanh(hidden_input + hidden_recurrent)

        # Compute output
        if target is not None:
            output = target  # Clamped
        else:
            output = torch.tanh(torch.matmul(hidden, self.W_ho.T))

        return hidden, output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (free phase)."""
        _, output = self.settle(x, target=None)
        return output.squeeze(0) if output.shape[0] == 1 else output

    def learn(self, x: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Perform contrastive learning step.

        Args:
            x: Input
            target: Target output

        Returns:
            Dictionary with learning statistics
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if target.dim() == 1:
            target = target.unsqueeze(0)

        # Free phase
        h_free, o_free = self.settle(x, target=None)

        # Clamped phase
        h_clamp, o_clamp = self.settle(x, target=target)

        # Compute correlations
        with torch.no_grad():
            batch_size = x.shape[0]

            # Input-hidden correlation difference
            corr_ih_free = torch.matmul(h_free.T, x)
            corr_ih_clamp = torch.matmul(h_clamp.T, x)
            delta_ih = (corr_ih_clamp - corr_ih_free) / batch_size

            # Hidden-hidden correlation difference
            corr_hh_free = torch.matmul(h_free.T, h_free)
            corr_hh_clamp = torch.matmul(h_clamp.T, h_clamp)
            delta_hh = (corr_hh_clamp - corr_hh_free) / batch_size

            # Hidden-output correlation difference
            corr_ho_free = torch.matmul(o_free.T, h_free)
            corr_ho_clamp = torch.matmul(o_clamp.T, h_clamp)
            delta_ho = (corr_ho_clamp - corr_ho_free) / batch_size

            # Apply updates
            self.W_ih.add_(self.learning_rate * delta_ih)
            self.W_hh.add_(self.learning_rate * delta_hh)
            self.W_ho.add_(self.learning_rate * delta_ho)

        return {
            'delta_ih_norm': delta_ih.norm(),
            'delta_hh_norm': delta_hh.norm(),
            'delta_ho_norm': delta_ho.norm(),
            'free_output': o_free,
            'target_error': (o_free - target).pow(2).mean()
        }


class PredictiveCoding(nn.Module):
    """
    Predictive Coding / Free Energy Minimization.

    Each layer predicts the layer below. Learning minimizes
    prediction error at each level. This is Friston's Free
    Energy Principle in neural network form.

    Natural fit for PyQuifer's generative world model.
    """

    def __init__(self,
                 dims: List[int],
                 learning_rate: float = 0.01,
                 inference_steps: int = 10,
                 precision: float = 1.0):
        """
        Args:
            dims: Dimensions of each layer [input, hidden1, ..., top]
            learning_rate: Learning rate for weight updates
            inference_steps: Steps for inference (settling)
            precision: Inverse variance of prediction errors
        """
        super().__init__()
        self.dims = dims
        self.num_layers = len(dims)
        self.learning_rate = learning_rate
        self.inference_steps = inference_steps
        self.precision = precision

        # Top-down prediction weights
        self.predictors = nn.ModuleList([
            nn.Linear(dims[i + 1], dims[i], bias=True)
            for i in range(len(dims) - 1)
        ])

        # Bottom-up error weights (for inference)
        self.error_weights = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1], bias=False)
            for i in range(len(dims) - 1)
        ])

    def predict(self, states: List[torch.Tensor]) -> List[torch.Tensor]:
        """Generate predictions for each layer."""
        predictions = []
        for i in range(self.num_layers - 1):
            pred = self.predictors[i](states[i + 1])
            predictions.append(pred)
        return predictions

    def compute_errors(self,
                       states: List[torch.Tensor],
                       predictions: List[torch.Tensor]) -> List[torch.Tensor]:
        """Compute prediction errors."""
        errors = []
        for i in range(len(predictions)):
            error = states[i] - predictions[i]
            errors.append(error * self.precision)
        return errors

    def inference_step(self,
                       states: List[torch.Tensor],
                       errors: List[torch.Tensor],
                       step_size: float = 0.1) -> List[torch.Tensor]:
        """Update internal states to minimize free energy."""
        new_states = [states[0]]  # Input is fixed

        for i in range(1, self.num_layers):
            # Gradient of free energy w.r.t. state
            # = precision * error_from_below - precision * error_to_above
            grad = torch.zeros_like(states[i])

            # Error from layer below (through error_weights)
            if i < self.num_layers - 1:
                grad = grad - self.error_weights[i - 1](errors[i - 1])

            # Error to layer above (through predictor)
            if i < len(errors):
                # Predictor[i-1] predicts layer i-1 from layer i
                # Error propagates back through predictor weights
                grad = grad + torch.matmul(errors[i - 1], self.predictors[i - 1].weight)

            new_state = states[i] - step_size * grad
            new_state = torch.clamp(new_state, -10.0, 10.0)  # Prevent explosion
            new_states.append(new_state)

        return new_states

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Run inference to minimize prediction error.

        Args:
            x: Input tensor

        Returns:
            states: Inferred states at each layer
            free_energy: Total prediction error (free energy)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Initialize states
        states = [x]
        for i in range(1, self.num_layers):
            states.append(torch.randn(x.shape[0], self.dims[i], device=x.device) * 0.1)

        # Inference loop
        for _ in range(self.inference_steps):
            predictions = self.predict(states)
            errors = self.compute_errors(states, predictions)
            states = self.inference_step(states, errors)

        # Compute final free energy
        predictions = self.predict(states)
        errors = self.compute_errors(states, predictions)
        free_energy = sum(e.pow(2).sum() for e in errors)

        return states, free_energy

    def learn(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Learn by minimizing prediction error.

        Args:
            x: Input observation

        Returns:
            Dictionary with learning statistics
        """
        states, free_energy = self.forward(x)
        predictions = self.predict(states)
        errors = self.compute_errors(states, predictions)

        # Update predictors and error weights to reduce errors
        with torch.no_grad():
            batch_size = x.shape[0]
            for i, (pred, err, state_above) in enumerate(zip(self.predictors, errors, states[1:])):
                # Predictor update: reduce prediction error
                delta_w = -self.learning_rate * torch.matmul(err.T, state_above) / batch_size
                delta_b = -self.learning_rate * err.mean(dim=0)

                pred.weight.add_(delta_w)
                pred.bias.add_(delta_b)

                # Error weight update: soft alignment toward predictor transpose
                # Biological predictive coding uses approximate reciprocal weights
                target_err_w = pred.weight.detach().T
                self.error_weights[i].weight.mul_(1 - self.learning_rate).add_(
                    self.learning_rate * target_err_w
                )

        return {
            'free_energy': free_energy,
            'top_state': states[-1],
            'errors': [e.pow(2).mean() for e in errors]
        }


class DifferentiablePlasticity(nn.Module):
    """
    Differentiable Hebbian plasticity (Miconi et al. 2018).

    Combines fixed weights with a fast Hebbian trace:
        y = tanh(x @ (W + alpha * H))

    Where:
    - W: Slow (backprop-trained) weights
    - H: Fast Hebbian trace, updated each step: H += eta * outer(y, x)
    - alpha: Learnable scalar controlling plasticity influence
    - eta: Learnable learning rate for Hebbian trace
    - H is clamped to [-1, 1] to prevent runaway

    This enables rapid adaptation within a single episode while
    maintaining the stability of learned baseline weights.

    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        eta_init: Initial Hebbian learning rate
        alpha_init: Initial plasticity influence
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 eta_init: float = 0.01,
                 alpha_init: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Slow weights (trained by backprop)
        self.W = nn.Parameter(torch.randn(output_dim, input_dim) * 0.1)

        # Learnable plasticity parameters
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.eta = nn.Parameter(torch.tensor(eta_init))

        # Fast Hebbian trace (not a parameter — updated online)
        self.register_buffer('H', torch.zeros(output_dim, input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with Hebbian-augmented weights.

        Args:
            x: Input tensor (..., input_dim)

        Returns:
            Output tensor (..., output_dim)
        """
        # Effective weight = fixed + plasticity * Hebbian
        W_eff = self.W + self.alpha * self.H
        y = torch.tanh(x @ W_eff.T)

        # Update Hebbian trace
        with torch.no_grad():
            if x.dim() == 1:
                outer = torch.outer(y, x)
            else:
                # Batch: average outer product
                outer = torch.einsum('bi,bj->ij', y, x) / x.shape[0]

            self.H.add_(self.eta.item() * outer)
            self.H.clamp_(-1.0, 1.0)

        return y

    def reset(self):
        """Reset Hebbian trace (e.g., between episodes)."""
        self.H.zero_()


class LearnableEligibilityTrace(nn.Module):
    """
    Eligibility trace with optionally learnable decay parameter.

    Extends EligibilityTrace: when learnable=True, the decay_rate
    becomes an nn.Parameter that can be optimized by backprop.

    Args:
        shape: Shape of the parameter to track
        decay_rate: Initial decay rate (0-1)
        accumulation_rate: How fast new activity accumulates
        learnable: If True, decay_rate is an nn.Parameter
    """

    def __init__(self,
                 shape: Tuple[int, ...],
                 decay_rate: float = 0.95,
                 accumulation_rate: float = 0.1,
                 learnable: bool = False):
        super().__init__()
        self.accumulation_rate = accumulation_rate
        self.learnable = learnable

        if learnable:
            # Store in logit space for unconstrained optimization
            # sigmoid(logit) = decay_rate
            logit = math.log(decay_rate / (1 - decay_rate + 1e-8))
            self.decay_logit = nn.Parameter(torch.tensor(logit))
        else:
            self.decay_rate_val = decay_rate

        self.register_buffer('trace', torch.zeros(shape))

    @property
    def decay_rate(self) -> torch.Tensor:
        if self.learnable:
            return torch.sigmoid(self.decay_logit)
        else:
            return torch.tensor(self.decay_rate_val, device=self.trace.device)

    def forward(self,
                activity: torch.Tensor,
                pre_activity: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Update eligibility trace based on current activity.

        Args:
            activity: Current activity signal (same shape as trace)
            pre_activity: Optional presynaptic activity

        Returns:
            Current trace value
        """
        decay = self.decay_rate

        with torch.no_grad():
            self.trace.mul_(decay.item() if isinstance(decay, torch.Tensor) else decay)

            if pre_activity is not None:
                if activity.dim() == 1 and pre_activity.dim() == 1:
                    hebbian = torch.outer(activity, pre_activity)
                else:
                    hebbian = activity * pre_activity
                self.trace.add_(self.accumulation_rate * hebbian)
            else:
                self.trace.add_(self.accumulation_rate * activity)

        return self.trace.clone()

    def apply_reward(self, reward: torch.Tensor, learning_rate: float = 0.01) -> torch.Tensor:
        """Compute weight update from trace and reward."""
        return learning_rate * reward * self.trace

    def reset(self):
        """Reset trace to zero."""
        self.trace.zero_()


if __name__ == '__main__':
    print("--- Learning Rules Beyond Backprop Examples ---")

    # Example 1: Eligibility Traces
    print("\n1. Eligibility Traces")
    trace = EligibilityTrace((5, 3), decay_rate=0.9)

    # Accumulate activity
    for i in range(10):
        post = torch.randn(5)
        pre = torch.randn(3)
        trace(post, pre)
        if i % 3 == 0:
            print(f"   Step {i}: trace norm = {trace.trace.norm().item():.4f}")

    # Apply reward
    delta = trace.apply_reward(torch.tensor(1.0), learning_rate=0.1)
    print(f"   Reward update norm: {delta.norm().item():.4f}")

    # Example 2: Reward-Modulated Hebbian
    print("\n2. Reward-Modulated Hebbian Learning")
    rml = RewardModulatedHebbian(input_dim=4, output_dim=2)

    # Run some forward passes
    for i in range(20):
        x = torch.randn(4)
        y = rml(x)

        # Occasional reward
        if i % 5 == 4:
            rml.reward_update(reward=1.0)
            print(f"   Step {i}: Rewarded! Weight norm = {rml.weights.norm().item():.4f}")

    # Example 3: Contrastive Hebbian
    print("\n3. Contrastive Hebbian Learning")
    chl = ContrastiveHebbian(input_dim=4, hidden_dim=8, output_dim=2, settle_steps=5)

    for epoch in range(10):
        x = torch.randn(4)
        target = torch.randn(2)

        result = chl.learn(x, target)

        if epoch % 3 == 0:
            print(f"   Epoch {epoch}: error = {result['target_error'].item():.4f}")

    # Example 4: Predictive Coding
    print("\n4. Predictive Coding")
    pc = PredictiveCoding(dims=[8, 16, 4], inference_steps=10)

    for epoch in range(20):
        x = torch.randn(3, 8)  # Batch of 3

        result = pc.learn(x)

        if epoch % 5 == 0:
            print(f"   Epoch {epoch}: free_energy = {result['free_energy'].item():.4f}")

    # Final inference
    states, fe = pc.forward(torch.randn(8))
    print(f"   Final top state shape: {states[-1].shape}")
