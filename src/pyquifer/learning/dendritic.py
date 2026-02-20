"""
Dendritic Credit Assignment for PyQuifer

Two-compartment neurons: basal (feedforward) + apical (top-down teaching).
Local error signals via plateau potentials — no backprop contamination.

Basal dendrite: standard feedforward input
Apical dendrite: top-down prediction from higher layer
Plateau potential: local error signal without contaminating feedforward path

plateau = sigma(V_apical - theta_apical)
dw_ff/dt = eta * plateau * (x_i - <x_i>)

References:
- Guerguiev et al. (2017). Towards deep learning with segregated dendrites. eLife.
- Sacramento et al. (2018). Dendritic cortical microcircuits.
- Richards & Bhatt (2018). The dendritic error hypothesis.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn


class DendriticNeuron(nn.Module):
    """
    Two-compartment neuron: basal (feedforward) + apical (top-down teaching).

    Basal dendrite: standard feedforward input
    Apical dendrite: top-down prediction from higher layer
    Plateau potential: local error signal without contaminating feedforward

    plateau = sigmoid(V_apical - threshold)
    dw_ff/dt = eta * plateau * (x_basal - mean(x_basal))
    """

    def __init__(self, input_dim: int, output_dim: int,
                 apical_dim: Optional[int] = None,
                 plateau_threshold: float = 0.0,
                 lr: float = 0.01):
        """
        Args:
            input_dim: Basal (feedforward) input dimension
            output_dim: Number of output neurons
            apical_dim: Apical (top-down) input dimension. Defaults to output_dim.
            plateau_threshold: Threshold for plateau potential generation
            lr: Local learning rate for weight updates
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.apical_dim = apical_dim or output_dim
        self.plateau_threshold = plateau_threshold
        self.lr = lr

        # Basal (feedforward) weights
        self.W_basal = nn.Parameter(torch.randn(output_dim, input_dim) * 0.1)
        self.bias = nn.Parameter(torch.zeros(output_dim))

        # Apical (top-down) weights
        self.W_apical = nn.Parameter(torch.randn(output_dim, self.apical_dim) * 0.1)

        # Buffers for local learning
        self.register_buffer('last_basal_input', torch.zeros(input_dim))
        self.register_buffer('last_output', torch.zeros(output_dim))
        self.register_buffer('last_plateau', torch.zeros(output_dim))

    def forward(self, x_basal: torch.Tensor,
                x_apical: Optional[torch.Tensor] = None) -> Dict:
        """
        Forward pass. Returns output, plateau_potential, basal_activation.

        Args:
            x_basal: Feedforward input (input_dim,) or (batch, input_dim)
            x_apical: Optional top-down teaching signal (apical_dim,) or (batch, apical_dim)

        Returns:
            Dict with output, plateau_potential, basal_activation
        """
        was_1d = x_basal.dim() == 1
        if was_1d:
            x_basal = x_basal.unsqueeze(0)

        # Basal (feedforward) activation
        basal_activation = torch.tanh(x_basal @ self.W_basal.T + self.bias)

        # Apical (top-down) plateau potential
        if x_apical is not None:
            if x_apical.dim() == 1:
                x_apical = x_apical.unsqueeze(0)
            v_apical = x_apical @ self.W_apical.T
            plateau = torch.sigmoid(v_apical - self.plateau_threshold)
        else:
            plateau = torch.zeros_like(basal_activation)

        # Output is purely feedforward (apical does NOT contaminate output)
        output = basal_activation

        # Store for local learning
        with torch.no_grad():
            self.last_basal_input.copy_(x_basal.mean(dim=0))
            self.last_output.copy_(output.mean(dim=0))
            self.last_plateau.copy_(plateau.mean(dim=0))

        if was_1d:
            output = output.squeeze(0)
            plateau = plateau.squeeze(0)
            basal_activation = basal_activation.squeeze(0)

        return {
            'output': output,
            'plateau_potential': plateau,
            'basal_activation': basal_activation,
        }

    def local_update(self) -> torch.Tensor:
        """
        Apply dendritic credit assignment (local, no backprop). Returns delta_w.

        Uses last stored plateau potential and basal input for the update:
        dw = lr * plateau * outer(output, x_basal - mean(x_basal))
        """
        with torch.no_grad():
            # Center the input (subtract mean)
            centered_input = self.last_basal_input - self.last_basal_input.mean()

            # Weight update: plateau-gated Hebbian
            delta_w = self.lr * torch.outer(
                self.last_plateau * self.last_output,
                centered_input,
            )

            with torch.no_grad():
                self.W_basal.add_(delta_w)

        return delta_w

    def reset(self):
        """Reset stored activations."""
        self.last_basal_input.zero_()
        self.last_output.zero_()
        self.last_plateau.zero_()


class DendriticStack(nn.Module):
    """
    Stack of DendriticNeurons forming a hierarchical network.
    Bottom-up feedforward + top-down teaching signals.
    Connects to HierarchicalPredictiveCoding for top-down predictions.
    """

    def __init__(self, dims: List[int], lr: float = 0.01,
                 plateau_threshold: float = 0.0):
        """
        Args:
            dims: Dimensions of each layer [input, hidden1, ..., output]
            lr: Local learning rate
            plateau_threshold: Threshold for plateau potentials
        """
        super().__init__()
        self.dims = dims
        self.num_layers = len(dims) - 1

        if self.num_layers < 1:
            raise ValueError("Need at least 2 dimensions for a stack")

        # Build layers
        layers = []
        for i in range(self.num_layers):
            # Apical dim = dimension of the NEXT layer's output (top-down signal)
            # For the top layer, no apical input
            apical_dim = dims[i + 2] if i + 1 < self.num_layers else None
            layers.append(DendriticNeuron(
                input_dim=dims[i],
                output_dim=dims[i + 1],
                apical_dim=apical_dim,
                plateau_threshold=plateau_threshold,
                lr=lr,
            ))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor,
                top_down_predictions: Optional[List[torch.Tensor]] = None) -> Dict:
        """
        Forward + optional top-down teaching. Returns all layer outputs.

        Args:
            x: Input tensor (dims[0],) or (batch, dims[0])
            top_down_predictions: Optional list of top-down signals per layer.
                                  Length should be num_layers. Use None entries to skip.

        Returns:
            Dict with outputs (list), plateau_potentials (list), final_output
        """
        outputs = []
        plateaus = []
        current = x

        for i, layer in enumerate(self.layers):
            # Get top-down signal for this layer
            td = None
            if top_down_predictions is not None and i < len(top_down_predictions):
                td = top_down_predictions[i]

            result = layer(current, x_apical=td)
            outputs.append(result['output'])
            plateaus.append(result['plateau_potential'])
            current = result['output']

        return {
            'outputs': outputs,
            'plateau_potentials': plateaus,
            'final_output': outputs[-1],
        }

    def learn(self) -> Dict:
        """
        Apply local dendritic updates to all layers.

        Returns:
            Dict with delta_norms per layer
        """
        delta_norms = []
        for layer in self.layers:
            delta = layer.local_update()
            delta_norms.append(delta.norm().item())

        return {
            'delta_norms': delta_norms,
            'mean_delta_norm': sum(delta_norms) / len(delta_norms),
        }

    def reset(self):
        """Reset all layers."""
        for layer in self.layers:
            layer.reset()
