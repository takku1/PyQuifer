"""
Prospective Configuration — infer-then-modify learning paradigm.

A fundamentally different learning approach from backpropagation:
instead of propagating errors backward, first infer what the target
activity should be (prospective phase), then modify weights to
consolidate that activity (modification phase).

Key classes:
- ProspectiveInference: Infer target activity before weight update
- ProspectiveHebbian: Modify weights to consolidate inferred activity
- InferThenModify: Full learning loop

References:
- Song et al. (2024). Inferring neural activity before plasticity as a
  foundation for learning beyond backpropagation. Nature Neuroscience.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProspectiveInference(nn.Module):
    """Infer target activity BEFORE weight update.

    During the prospective phase, activity is relaxed toward
    the target by iteratively updating neuron activities while
    keeping weights fixed. This creates a "prospective" activity
    pattern that represents what the network SHOULD do.

    Args:
        dim: Layer dimension.
        num_iterations: Number of inference iterations.
        inference_lr: Step size for activity inference.
        beta: Target influence strength.
    """

    def __init__(
        self,
        dim: int,
        num_iterations: int = 20,
        inference_lr: float = 0.1,
        beta: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_iterations = num_iterations
        self.inference_lr = inference_lr
        self.beta = beta

        # Energy function parameters
        self.W = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(dim))

    def energy(
        self,
        activity: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute energy of current activity.

        E = -0.5 * a^T W a - b^T a + beta * ||a - target||^2

        Args:
            activity: (..., dim) current activity
            target: Optional (..., dim) target activity

        Returns:
            (...,) energy values
        """
        # Hopfield-like energy
        Wa = F.linear(activity, self.W)
        energy = -0.5 * (activity * Wa).sum(dim=-1) - (self.bias * activity).sum(dim=-1)

        # Target nudge
        if target is not None:
            energy = energy + self.beta * ((activity - target) ** 2).sum(dim=-1)

        return energy

    def infer(
        self,
        initial_activity: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run prospective inference to find target activity.

        Iteratively updates activity to minimize energy while
        being nudged toward the target.

        Args:
            initial_activity: (..., dim) starting activity
            target: Optional (..., dim) desired activity

        Returns:
            Dict with 'activity' (inferred), 'energy_trajectory',
            'convergence'
        """
        activity = initial_activity.clone().requires_grad_(True)
        energies = []

        for _ in range(self.num_iterations):
            e = self.energy(activity, target)
            total_energy = e.sum()
            energies.append(total_energy.item())

            # Compute gradient of energy w.r.t. activity
            grad = torch.autograd.grad(total_energy, activity, create_graph=False)[0]

            # Update activity (gradient descent on energy)
            with torch.no_grad():
                activity = activity - self.inference_lr * grad
                activity = torch.tanh(activity)  # Bound activations
                activity.requires_grad_(True)

        final_energy = self.energy(activity.detach(), target)

        convergence = torch.tensor(0.0)
        if len(energies) >= 2:
            convergence = torch.tensor(abs(energies[-1] - energies[-2]))

        return {
            'activity': activity.detach(),
            'energy_trajectory': torch.tensor(energies),
            'convergence': convergence,
            'final_energy': final_energy,
        }


class ProspectiveHebbian(nn.Module):
    """Modify weights to consolidate inferred activity.

    After prospective inference determines the target activity,
    this module updates weights using a Hebbian-like rule:
    dW = lr * (a_prospective - a_current) * a_input^T

    This is a LOCAL learning rule — no backpropagation needed.

    Args:
        input_dim: Input dimension.
        output_dim: Output dimension.
        lr: Learning rate for weight modification.
    """

    def __init__(self, input_dim: int, output_dim: int, lr: float = 0.01):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr

        self.weight = nn.Parameter(torch.randn(output_dim, input_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass.

        Args:
            x: (..., input_dim)

        Returns:
            (..., output_dim)
        """
        return F.linear(x, self.weight, self.bias)

    def modify(
        self,
        input_activity: torch.Tensor,
        current_output: torch.Tensor,
        prospective_output: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Hebbian weight modification.

        Args:
            input_activity: (..., input_dim) input to this layer
            current_output: (..., output_dim) current output
            prospective_output: (..., output_dim) desired output

        Returns:
            Dict with 'weight_update_norm', 'activity_error'
        """
        # Error signal: difference between prospective and current
        error = prospective_output - current_output

        # Hebbian update: dW = lr * error * input^T
        if input_activity.dim() > 1:
            dW = self.lr * torch.einsum('...o,...i->oi', error, input_activity)
            dW = dW / max(1, input_activity.shape[0])  # Average over batch
        else:
            dW = self.lr * torch.outer(error, input_activity)

        with torch.no_grad():
            self.weight.add_(dW)
            self.bias.add_(self.lr * error.mean(dim=0) if error.dim() > 1 else self.lr * error)

        return {
            'weight_update_norm': dW.norm(),
            'activity_error': error.norm(),
        }


class InferThenModify(nn.Module):
    """Full prospective configuration learning loop.

    Two-phase learning:
    1. Inference phase: Find prospective activity (what neurons SHOULD do)
    2. Modification phase: Update weights to consolidate prospective activity

    This is fundamentally different from backpropagation:
    - No global error signal
    - No backward pass through the network
    - Local learning rules only
    - Biologically plausible

    Args:
        dims: List of layer dimensions [input, hidden..., output].
        num_inference_steps: Steps for prospective inference.
        inference_lr: Inference step size.
        modification_lr: Weight modification step size.
        beta: Target influence strength.
    """

    def __init__(
        self,
        dims: List[int],
        num_inference_steps: int = 20,
        inference_lr: float = 0.1,
        modification_lr: float = 0.01,
        beta: float = 1.0,
    ):
        super().__init__()
        self.dims = dims
        self.num_layers = len(dims) - 1

        # Prospective inference modules (one per hidden layer)
        self.inference_modules = nn.ModuleList([
            ProspectiveInference(d, num_inference_steps, inference_lr, beta)
            for d in dims[1:]
        ])

        # Weight layers
        self.layers = nn.ModuleList([
            ProspectiveHebbian(dims[i], dims[i+1], modification_lr)
            for i in range(self.num_layers)
        ])

        # Activation function
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass (no learning).

        Args:
            x: (..., dims[0]) input

        Returns:
            (..., dims[-1]) output
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.num_layers - 1:
                x = self.activation(x)
        return x

    def learn(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Run infer-then-modify learning.

        Args:
            x: (..., dims[0]) input
            target: (..., dims[-1]) desired output

        Returns:
            Dict with 'output', 'loss', per-layer 'inference' and 'modification' stats
        """
        # Phase 1: Forward pass to get current activities
        activities = [x]
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < self.num_layers - 1:
                h = self.activation(h)
            activities.append(h)

        current_output = activities[-1]
        loss = F.mse_loss(current_output, target)

        # Phase 2: Prospective inference (backward from target)
        prospective = [None] * (self.num_layers + 1)
        prospective[-1] = target

        for i in range(self.num_layers - 1, -1, -1):
            if prospective[i + 1] is not None:
                inference_result = self.inference_modules[i].infer(
                    activities[i + 1], prospective[i + 1]
                )
                prospective[i + 1] = inference_result['activity']

        # Phase 3: Weight modification
        total_update_norm = 0.0
        for i in range(self.num_layers):
            if prospective[i + 1] is not None:
                mod_result = self.layers[i].modify(
                    activities[i].detach(),
                    activities[i + 1].detach(),
                    prospective[i + 1].detach(),
                )
                total_update_norm += mod_result['weight_update_norm'].item()

        # New output after modification
        new_output = self.forward(x)
        new_loss = F.mse_loss(new_output, target)

        return {
            'output': new_output,
            'loss': new_loss,
            'old_loss': loss,
            'improvement': (loss - new_loss).item(),
            'total_update_norm': total_update_norm,
        }
