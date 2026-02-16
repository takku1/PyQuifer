"""
Dendritic Localized Learning (DLL) — SOTA bio-plausible learning.

Extends dendritic.py with the full DLL algorithm: multi-compartment
pyramidal neurons with local error signals in dendrites. No global
error propagation — each layer learns from locally available information.

Key classes:
- PyramidalNeuron: Multi-compartment with basal/apical/soma
- DendriticLocalizedLearning: Full DLL algorithm
- DendriticErrorSignal: Local error representation in dendrites

References:
- Guerguiev et al. (2017). Towards deep learning with segregated dendrites. eLife.
- Sacramento et al. (2018). Dendritic cortical microcircuits.
- Payeur et al. (2021). Burst-dependent synaptic plasticity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple, List


class DendriticErrorSignal(nn.Module):
    """Local error representation in apical dendrites.

    Instead of backpropagating errors globally, the error signal
    is represented locally in apical dendrites as the difference
    between top-down prediction and actual somatic activity.

    Args:
        dim: Neuron dimension.
        feedback_dim: Dimension of top-down feedback.
    """

    def __init__(self, dim: int, feedback_dim: int = 0):
        super().__init__()
        self.dim = dim
        feedback_dim = feedback_dim if feedback_dim > 0 else dim

        # Feedback projection (top-down)
        self.feedback_proj = nn.Linear(feedback_dim, dim, bias=False)

        # Error gate: modulates how much error influences learning
        self.error_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )

    def compute_error(
        self,
        somatic_activity: torch.Tensor,
        top_down_signal: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute local dendritic error.

        Args:
            somatic_activity: (..., dim) current neuron output
            top_down_signal: (..., feedback_dim) feedback from higher layer

        Returns:
            Dict with 'error', 'gated_error', 'error_magnitude'
        """
        prediction = self.feedback_proj(top_down_signal)
        error = prediction - somatic_activity

        # Gate the error based on context
        gate_input = torch.cat([somatic_activity, prediction], dim=-1)
        gate = self.error_gate(gate_input)
        gated_error = error * gate

        return {
            'error': error,
            'gated_error': gated_error,
            'error_magnitude': error.norm(dim=-1),
        }


class PyramidalNeuron(nn.Module):
    """Multi-compartment pyramidal neuron.

    Three-compartment model:
    - Basal dendrites: receive feedforward input from lower layer
    - Soma: integrate basal + apical, produce output
    - Apical dendrites: receive feedback from higher layer + lateral

    Learning uses burst-dependent plasticity: when apical input
    drives the soma past a burst threshold, basal synapses are
    potentiated (strengthened).

    Args:
        basal_dim: Feedforward input dimension.
        apical_dim: Feedback input dimension.
        soma_dim: Somatic output dimension.
        burst_threshold: Threshold for burst-dependent plasticity.
    """

    def __init__(
        self,
        basal_dim: int,
        apical_dim: int,
        soma_dim: int,
        burst_threshold: float = 0.5,
    ):
        super().__init__()
        self.basal_dim = basal_dim
        self.apical_dim = apical_dim
        self.soma_dim = soma_dim
        self.burst_threshold = burst_threshold

        # Basal dendrite: feedforward
        self.basal = nn.Linear(basal_dim, soma_dim)

        # Apical dendrite: feedback
        self.apical = nn.Linear(apical_dim, soma_dim)

        # Somatic integration
        self.soma_gate = nn.Sequential(
            nn.Linear(soma_dim * 2, soma_dim),
            nn.Sigmoid(),
        )

        # Error signal module
        self.error_signal = DendriticErrorSignal(soma_dim, apical_dim)

        # Burst tracking
        self.register_buffer('_burst_rate', torch.tensor(0.0))

    def forward(
        self,
        basal_input: torch.Tensor,
        apical_input: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute pyramidal neuron output.

        Args:
            basal_input: (..., basal_dim) feedforward input
            apical_input: Optional (..., apical_dim) feedback input

        Returns:
            Dict with 'soma_output', 'basal_drive', 'apical_drive',
            'burst_probability'
        """
        basal_drive = self.basal(basal_input)

        if apical_input is not None:
            apical_drive = self.apical(apical_input)
        else:
            apical_drive = torch.zeros_like(basal_drive)

        # Somatic integration with gating
        gate_input = torch.cat([basal_drive, apical_drive], dim=-1)
        gate = self.soma_gate(gate_input)
        soma_output = torch.tanh(basal_drive + gate * apical_drive)

        # Burst detection
        burst_prob = torch.sigmoid(
            (apical_drive.abs().mean(dim=-1) - self.burst_threshold) * 5
        )

        with torch.no_grad():
            self._burst_rate.lerp_(burst_prob.mean(), 0.01)

        return {
            'soma_output': soma_output,
            'basal_drive': basal_drive,
            'apical_drive': apical_drive,
            'burst_probability': burst_prob,
        }


class DendriticLocalizedLearning(nn.Module):
    """Full Dendritic Localized Learning algorithm.

    Multi-layer network of pyramidal neurons that learns using
    only locally available information:
    1. Feedforward: basal dendrites propagate input upward
    2. Feedback: apical dendrites carry top-down predictions
    3. Learning: burst-dependent plasticity using local errors

    No global backpropagation — each layer learns independently.

    Args:
        dims: List of layer dimensions [input, hidden..., output].
        apical_dim: Dimension for feedback connections.
        lr: Learning rate for local updates.
        burst_threshold: Burst detection threshold.
    """

    def __init__(
        self,
        dims: List[int],
        apical_dim: int = 0,
        lr: float = 0.01,
        burst_threshold: float = 0.5,
    ):
        super().__init__()
        self.dims = dims
        self.num_layers = len(dims) - 1
        self.lr = lr

        apical_dim = apical_dim if apical_dim > 0 else dims[-1]

        # Pyramidal neuron layers
        self.neurons = nn.ModuleList()
        for i in range(self.num_layers):
            ad = dims[i + 1] if i < self.num_layers - 1 else apical_dim
            self.neurons.append(
                PyramidalNeuron(dims[i], ad, dims[i + 1], burst_threshold)
            )

        # Feedback connections (top-down)
        self.feedback = nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.feedback.append(
                nn.Linear(dims[i + 2], dims[i + 1], bias=False)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass (feedforward only).

        Args:
            x: (..., dims[0])

        Returns:
            (..., dims[-1])
        """
        for neuron in self.neurons:
            result = neuron(x)
            x = result['soma_output']
        return x

    def learn(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Run DLL learning step.

        1. Forward pass: collect all layer activities
        2. Feedback pass: send top-down predictions
        3. Local learning: update weights using burst-dependent plasticity

        Args:
            x: (..., dims[0]) input
            target: (..., dims[-1]) desired output

        Returns:
            Dict with 'output', 'loss', 'layer_errors', 'burst_rates'
        """
        # Phase 1: Forward pass
        activities = [x]
        layer_results = []

        h = x
        for neuron in self.neurons:
            result = neuron(h)
            h = result['soma_output']
            activities.append(h)
            layer_results.append(result)

        output = activities[-1]
        loss = F.mse_loss(output, target)

        # Phase 2: Feedback pass (top-down)
        feedback_signals = [None] * self.num_layers
        feedback_signals[-1] = target  # Direct target for output layer

        for i in range(self.num_layers - 2, -1, -1):
            feedback_signals[i] = self.feedback[i](activities[i + 2].detach())

        # Phase 3: Local learning
        layer_errors = []
        burst_rates = []

        for i in range(self.num_layers):
            neuron = self.neurons[i]
            basal_input = activities[i].detach()

            if feedback_signals[i] is not None:
                # Compute dendritic error
                error_result = neuron.error_signal.compute_error(
                    activities[i + 1].detach(),
                    feedback_signals[i].detach(),
                )

                # Burst-dependent plasticity
                burst_prob = layer_results[i]['burst_probability']
                error = error_result['gated_error']

                # Update basal weights: potentiate when bursting
                with torch.no_grad():
                    if basal_input.dim() > 1:
                        dW = self.lr * torch.einsum(
                            '...o,...i->oi',
                            error * burst_prob.unsqueeze(-1),
                            basal_input,
                        ) / max(1, basal_input.shape[0])
                    else:
                        dW = self.lr * torch.outer(
                            error * burst_prob.unsqueeze(-1),
                            basal_input,
                        )
                    neuron.basal.weight.add_(dW)

                layer_errors.append(error_result['error_magnitude'].mean().item())
            else:
                layer_errors.append(0.0)

            burst_rates.append(neuron._burst_rate.item())

        # Compute new output after learning
        new_output = self.forward(x)
        new_loss = F.mse_loss(new_output, target)

        return {
            'output': new_output,
            'loss': new_loss,
            'old_loss': loss,
            'improvement': (loss - new_loss).item(),
            'layer_errors': layer_errors,
            'burst_rates': burst_rates,
        }
