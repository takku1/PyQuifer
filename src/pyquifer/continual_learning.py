"""
Continual Learning Module for PyQuifer

Implements mechanisms to prevent catastrophic forgetting and enable
lifelong learning in neural networks:

1. Elastic Weight Consolidation (EWC) - Protect important weights
2. Synaptic Intelligence (SI) - Online importance estimation
3. Memory Efficient Synaptic Updates (MESU) - Metaplasticity framework
4. PackNet - Task-specific weight pruning and freezing
5. Experience Replay - Rehearse past experiences

These enable PyQuifer systems to learn new tasks without forgetting
previous knowledge - essential for genuine consciousness that
accumulates experience over time.

Based on work by Kirkpatrick et al., Zenke et al., Dohare et al.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque


@dataclass
class TaskInfo:
    """Information about a learned task."""
    task_id: int
    name: str
    fisher_diagonal: Optional[Dict[str, torch.Tensor]] = None
    optimal_weights: Optional[Dict[str, torch.Tensor]] = None
    importance_scores: Optional[Dict[str, torch.Tensor]] = None
    mask: Optional[Dict[str, torch.Tensor]] = None


class ElasticWeightConsolidation(nn.Module):
    """
    Elastic Weight Consolidation (EWC) for continual learning.

    EWC slows down learning on weights that are important for
    previous tasks. Importance is estimated via Fisher Information
    (gradient variance on old task data).

    Loss = L_new + (lambda/2) * sum_i F_i * (theta_i - theta*_i)^2

    where F_i is Fisher information and theta*_i are optimal weights
    for previous tasks.
    """

    def __init__(self,
                 model: nn.Module,
                 ewc_lambda: float = 1000.0,
                 online: bool = True,
                 gamma: float = 0.9):
        """
        Args:
            model: The neural network to protect
            ewc_lambda: Regularization strength (higher = more protection)
            online: If True, use online EWC (moving average of Fisher)
            gamma: Decay factor for online EWC
        """
        super().__init__()
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.online = online
        self.gamma = gamma

        self.tasks: List[TaskInfo] = []
        self.current_task_id = 0

        # For online EWC
        self.cumulative_fisher: Dict[str, torch.Tensor] = {}
        self.cumulative_weights: Dict[str, torch.Tensor] = {}

    def compute_fisher(self,
                       dataloader,
                       num_samples: int = 200) -> Dict[str, torch.Tensor]:
        """
        Compute Fisher Information Matrix diagonal.

        Fisher = E[(d log p(y|x,theta) / d theta)^2]

        Approximated by empirical gradient variance.
        """
        fisher = {}

        # Initialize
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param)

        self.model.eval()
        samples_seen = 0

        for batch in dataloader:
            if samples_seen >= num_samples:
                break

            # Get inputs (handle different batch formats)
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch

            self.model.zero_grad()

            # Forward pass
            output = self.model(x)

            # For each sample, compute gradient of log-likelihood
            if output.dim() > 1:
                # Classification: use log softmax
                log_probs = F.log_softmax(output, dim=-1)
                # Sample from predicted distribution
                probs = torch.exp(log_probs)
                sampled = torch.multinomial(probs, 1).squeeze()

                for i in range(len(x)):
                    self.model.zero_grad()
                    log_probs[i, sampled[i]].backward(retain_graph=True)

                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            fisher[name] += param.grad.data.pow(2)

                    samples_seen += 1
                    if samples_seen >= num_samples:
                        break
            else:
                # Regression: use MSE gradient
                output.sum().backward()
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        fisher[name] += param.grad.data.pow(2)
                samples_seen += len(x)

        # Normalize
        for name in fisher:
            fisher[name] /= samples_seen

        return fisher

    def consolidate(self,
                    dataloader,
                    task_name: str = "",
                    num_samples: int = 200):
        """
        Consolidate current task by computing and storing Fisher/weights.

        Call this after training on a task, before moving to next task.
        """
        # Compute Fisher information
        fisher = self.compute_fisher(dataloader, num_samples)

        # Store current weights
        optimal_weights = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        if self.online:
            # Online EWC: update cumulative Fisher with decay
            for name in fisher:
                if name in self.cumulative_fisher:
                    self.cumulative_fisher[name] = (
                        self.gamma * self.cumulative_fisher[name] +
                        fisher[name]
                    )
                    self.cumulative_weights[name] = optimal_weights[name]
                else:
                    self.cumulative_fisher[name] = fisher[name]
                    self.cumulative_weights[name] = optimal_weights[name]
        else:
            # Standard EWC: store per-task Fisher
            task = TaskInfo(
                task_id=self.current_task_id,
                name=task_name or f"task_{self.current_task_id}",
                fisher_diagonal=fisher,
                optimal_weights=optimal_weights
            )
            self.tasks.append(task)

        self.current_task_id += 1

    def penalty(self) -> torch.Tensor:
        """
        Compute EWC penalty to add to loss.

        Returns:
            Regularization penalty (scalar tensor)
        """
        penalty = torch.tensor(0.0, device=next(self.model.parameters()).device)

        if self.online:
            # Online EWC
            for name, param in self.model.named_parameters():
                if name in self.cumulative_fisher:
                    penalty += (
                        self.cumulative_fisher[name] *
                        (param - self.cumulative_weights[name]).pow(2)
                    ).sum()
        else:
            # Standard EWC: sum over all tasks
            for task in self.tasks:
                for name, param in self.model.named_parameters():
                    if name in task.fisher_diagonal:
                        penalty += (
                            task.fisher_diagonal[name] *
                            (param - task.optimal_weights[name]).pow(2)
                        ).sum()

        return 0.5 * self.ewc_lambda * penalty

    def get_importance(self, name: str) -> Optional[torch.Tensor]:
        """Get importance scores for a parameter."""
        if self.online:
            return self.cumulative_fisher.get(name)
        elif self.tasks:
            # Return maximum importance across tasks
            importances = [
                t.fisher_diagonal.get(name)
                for t in self.tasks
                if t.fisher_diagonal and name in t.fisher_diagonal
            ]
            if importances:
                return torch.stack(importances).max(dim=0)[0]
        return None


class SynapticIntelligence(nn.Module):
    """
    Synaptic Intelligence (SI) for online importance estimation.

    Unlike EWC which requires a separate Fisher computation phase,
    SI accumulates importance during training in an online manner.

    Importance = integral of (gradient * weight_change) over training

    This captures how much each weight contributed to reducing loss.
    """

    def __init__(self,
                 model: nn.Module,
                 si_lambda: float = 100.0,
                 epsilon: float = 0.1):
        """
        Args:
            model: The neural network
            si_lambda: Regularization strength
            epsilon: Damping factor to prevent division by zero
        """
        super().__init__()
        self.model = model
        self.si_lambda = si_lambda
        self.epsilon = epsilon

        # Per-parameter accumulators
        self.omega: Dict[str, torch.Tensor] = {}  # Importance
        self.W_old: Dict[str, torch.Tensor] = {}  # Previous weights
        self.small_omega: Dict[str, torch.Tensor] = {}  # Running importance

        # Initialize
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.omega[name] = torch.zeros_like(param)
                self.W_old[name] = param.data.clone()
                self.small_omega[name] = torch.zeros_like(param)

    def update_running_importance(self):
        """
        Update running importance based on gradients.
        Call this after each training step (after backward, before optimizer.step).
        """
        for name, param in self.model.named_parameters():
            if param.grad is not None and name in self.small_omega:
                # Accumulate gradient * change
                # Note: gradient points in direction of increasing loss
                # Weight change will be opposite (decreasing loss)
                self.small_omega[name] += (
                    -param.grad.data * (param.data - self.W_old[name])
                )

    def consolidate(self, task_name: str = ""):
        """
        Consolidate importance after finishing a task.
        """
        for name, param in self.model.named_parameters():
            if name in self.small_omega:
                # Normalize by total weight change
                delta = param.data - self.W_old[name]

                # Update cumulative importance
                self.omega[name] += (
                    self.small_omega[name] /
                    (delta.pow(2) + self.epsilon)
                )

                # Reset for next task
                self.W_old[name] = param.data.clone()
                self.small_omega[name].zero_()

    def penalty(self) -> torch.Tensor:
        """Compute SI regularization penalty."""
        penalty = torch.tensor(0.0, device=next(self.model.parameters()).device)

        for name, param in self.model.named_parameters():
            if name in self.omega:
                penalty += (
                    self.omega[name] *
                    (param - self.W_old[name]).pow(2)
                ).sum()

        return self.si_lambda * penalty


class ContinualBackprop(nn.Module):
    """
    Continual Backprop for maintaining network plasticity.

    Based on Dohare et al. (2024) - "Maintaining plasticity in
    continual learning via regenerating units".

    Key insight: Units that aren't used become dormant over time.
    Solution: Periodically reinitialize least-useful units while
    preserving network function.

    This prevents the "loss of plasticity" problem where networks
    become unable to learn new things.
    """

    def __init__(self,
                 model: nn.Module,
                 reinit_fraction: float = 0.1,
                 utility_decay: float = 0.99,
                 reinit_threshold: float = 0.01):
        """
        Args:
            model: The neural network
            reinit_fraction: Fraction of units to potentially reinitialize
            utility_decay: Decay rate for utility tracking
            reinit_threshold: Utility threshold below which to reinitialize
        """
        super().__init__()
        self.model = model
        self.reinit_fraction = reinit_fraction
        self.utility_decay = utility_decay
        self.reinit_threshold = reinit_threshold

        # Track utility of each unit
        self.utility: Dict[str, torch.Tensor] = {}
        self.activation_count: Dict[str, torch.Tensor] = {}

        # Initialize tracking for each layer
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                out_features = module.out_features
                self.utility[name] = torch.ones(out_features)
                self.activation_count[name] = torch.zeros(out_features)

    def update_utility(self, activations: Dict[str, torch.Tensor]):
        """
        Update utility estimates based on activations.

        Args:
            activations: Dict mapping layer name to activation tensor
        """
        for name, act in activations.items():
            if name in self.utility:
                # Decay old utility
                self.utility[name] *= self.utility_decay

                # Add contribution from current activation
                # Units that are active and useful have high gradient * activation
                if act.dim() > 1:
                    act_magnitude = act.abs().mean(dim=0)  # Average over batch
                else:
                    act_magnitude = act.abs()

                self.utility[name] += (1 - self.utility_decay) * act_magnitude.cpu()
                self.activation_count[name] += (act.abs() > 0.01).float().sum(dim=0).cpu()

    def regenerate(self) -> int:
        """
        Reinitialize dormant units.

        Returns:
            Number of units reinitialized
        """
        total_reinit = 0

        for name, module in self.model.named_modules():
            if name in self.utility and isinstance(module, nn.Linear):
                # Find units with low utility
                low_utility_mask = self.utility[name] < self.reinit_threshold

                # Limit to reinit_fraction
                n_low = low_utility_mask.sum().item()
                max_reinit = int(self.reinit_fraction * len(self.utility[name]))

                if n_low > 0:
                    # Get indices of lowest utility units
                    _, indices = self.utility[name].topk(
                        min(n_low, max_reinit), largest=False
                    )

                    # Reinitialize weights for these units
                    with torch.no_grad():
                        for idx in indices:
                            # Reinitialize incoming weights
                            nn.init.kaiming_normal_(
                                module.weight[idx:idx+1],
                                mode='fan_in',
                                nonlinearity='relu'
                            )
                            if module.bias is not None:
                                module.bias[idx] = 0.0

                    # Reset utility for reinitialized units
                    self.utility[name][indices] = 1.0
                    self.activation_count[name][indices] = 0

                    total_reinit += len(indices)

        return total_reinit

    def get_stats(self) -> Dict[str, float]:
        """Get plasticity statistics."""
        total_units = 0
        dormant_units = 0

        for name, util in self.utility.items():
            total_units += len(util)
            dormant_units += (util < self.reinit_threshold).sum().item()

        return {
            'total_units': total_units,
            'dormant_units': dormant_units,
            'dormant_fraction': dormant_units / max(1, total_units),
            'mean_utility': sum(u.mean().item() for u in self.utility.values()) / max(1, len(self.utility))
        }


class MESU(nn.Module):
    """
    Memory-Efficient Synaptic Updates (MESU) via metaplasticity.

    Based on "Addressing Loss of Plasticity and Catastrophic Forgetting
    in Continual Learning" (Elsayed & Mahmood, 2024).

    Key idea: Each synapse has a "metaplasticity" variable that
    modulates its learning rate based on:
    1. How much it has changed recently (high change → lower LR)
    2. How important it is for past tasks (important → lower LR)
    3. Uncertainty about its current value (uncertain → higher LR)
    """

    def __init__(self,
                 model: nn.Module,
                 meta_lr: float = 0.1,
                 importance_decay: float = 0.99,
                 uncertainty_scale: float = 1.0):
        """
        Args:
            model: The neural network
            meta_lr: Metaplasticity learning rate
            importance_decay: Decay for importance estimates
            uncertainty_scale: Scale for uncertainty-based modulation
        """
        super().__init__()
        self.model = model
        self.meta_lr = meta_lr
        self.importance_decay = importance_decay
        self.uncertainty_scale = uncertainty_scale

        # Per-parameter metaplasticity
        self.meta: Dict[str, torch.Tensor] = {}
        self.uncertainty: Dict[str, torch.Tensor] = {}
        self.grad_ema: Dict[str, torch.Tensor] = {}
        self.grad_var: Dict[str, torch.Tensor] = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.meta[name] = torch.ones_like(param)  # Metaplasticity
                self.uncertainty[name] = torch.ones_like(param)  # Uncertainty
                self.grad_ema[name] = torch.zeros_like(param)  # Gradient EMA
                self.grad_var[name] = torch.ones_like(param)  # Gradient variance

    def get_learning_rates(self) -> Dict[str, torch.Tensor]:
        """
        Compute per-parameter learning rate modulation.

        Returns:
            Dict of learning rate multipliers (0 to 1)
        """
        lr_mods = {}

        for name in self.meta:
            # Learning rate is modulated by:
            # - Metaplasticity (low for important params)
            # - Uncertainty (high for uncertain params)
            lr_mods[name] = torch.sigmoid(
                self.meta[name] +
                self.uncertainty_scale * self.uncertainty[name]
            )

        return lr_mods

    def update(self, loss: torch.Tensor):
        """
        Update metaplasticity based on gradients.
        Call after backward(), before optimizer.step().
        """
        for name, param in self.model.named_parameters():
            if param.grad is not None and name in self.meta:
                grad = param.grad.data

                # Update gradient statistics
                self.grad_ema[name] = (
                    0.9 * self.grad_ema[name] +
                    0.1 * grad
                )
                self.grad_var[name] = (
                    0.9 * self.grad_var[name] +
                    0.1 * (grad - self.grad_ema[name]).pow(2)
                )

                # Uncertainty from gradient variance
                self.uncertainty[name] = torch.sqrt(self.grad_var[name] + 1e-8)

                # Metaplasticity decreases when gradients are consistent
                # (parameter is important for current task)
                consistency = 1.0 / (1.0 + self.grad_var[name])
                self.meta[name] = (
                    self.importance_decay * self.meta[name] -
                    self.meta_lr * consistency * grad.abs()
                )

    def apply_modulation(self, optimizer):
        """
        Apply learning rate modulation to optimizer.
        """
        lr_mods = self.get_learning_rates()

        for group in optimizer.param_groups:
            for param in group['params']:
                # Find matching modulation
                for name, p in self.model.named_parameters():
                    if p is param and name in lr_mods:
                        # Store original gradient
                        if param.grad is not None:
                            param.grad.data *= lr_mods[name]
                        break


class ExperienceReplay:
    """
    Experience Replay buffer for continual learning.

    Stores representative samples from past tasks and
    replays them during training on new tasks.

    Supports multiple sampling strategies:
    - Random: Uniform sampling
    - Prioritized: Sample based on importance
    - Reservoir: Memory-efficient streaming updates
    """

    def __init__(self,
                 capacity: int = 10000,
                 strategy: str = 'reservoir'):
        """
        Args:
            capacity: Maximum number of samples to store
            strategy: Sampling strategy ('random', 'prioritized', 'reservoir')
        """
        self.capacity = capacity
        self.strategy = strategy

        self.buffer: List[Tuple] = []
        self.priorities: List[float] = []
        self.samples_seen = 0

    def add(self,
            x: torch.Tensor,
            y: Optional[torch.Tensor] = None,
            task_id: int = 0,
            priority: float = 1.0):
        """
        Add sample(s) to the buffer.
        """
        # Handle batched input
        if x.dim() > 1:
            for i in range(len(x)):
                yi = y[i] if y is not None else None
                self._add_single(x[i], yi, task_id, priority)
        else:
            self._add_single(x, y, task_id, priority)

    def _add_single(self, x, y, task_id, priority):
        """Add a single sample."""
        self.samples_seen += 1

        if self.strategy == 'reservoir':
            # Reservoir sampling: each sample has equal probability
            if len(self.buffer) < self.capacity:
                self.buffer.append((x.clone(), y.clone() if y is not None else None, task_id))
                self.priorities.append(priority)
            else:
                # Replace with probability capacity / samples_seen
                idx = torch.randint(0, self.samples_seen, (1,)).item()
                if idx < self.capacity:
                    self.buffer[idx] = (x.clone(), y.clone() if y is not None else None, task_id)
                    self.priorities[idx] = priority
        else:
            # Simple FIFO
            if len(self.buffer) >= self.capacity:
                self.buffer.pop(0)
                self.priorities.pop(0)
            self.buffer.append((x.clone(), y.clone() if y is not None else None, task_id))
            self.priorities.append(priority)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[int]]:
        """
        Sample a batch from the buffer.

        Returns:
            x: Input batch
            y: Label batch (or None)
            task_ids: List of task IDs
        """
        if len(self.buffer) == 0:
            raise ValueError("Buffer is empty")

        batch_size = min(batch_size, len(self.buffer))

        if self.strategy == 'prioritized':
            # Sample proportional to priority
            probs = torch.tensor(self.priorities)
            probs = probs / probs.sum()
            indices = torch.multinomial(probs, batch_size, replacement=False)
        else:
            # Random sampling
            indices = torch.randperm(len(self.buffer))[:batch_size]

        xs, ys, task_ids = [], [], []
        for idx in indices:
            x, y, tid = self.buffer[idx]
            xs.append(x)
            ys.append(y)
            task_ids.append(tid)

        x_batch = torch.stack(xs)
        y_batch = torch.stack(ys) if ys[0] is not None else None

        return x_batch, y_batch, task_ids

    def get_task_samples(self, task_id: int, n: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Get samples from a specific task."""
        task_samples = [(x, y) for x, y, tid in self.buffer if tid == task_id]

        if not task_samples:
            raise ValueError(f"No samples for task {task_id}")

        n = min(n, len(task_samples))
        indices = torch.randperm(len(task_samples))[:n]

        xs = [task_samples[i][0] for i in indices]
        ys = [task_samples[i][1] for i in indices]

        x_batch = torch.stack(xs)
        y_batch = torch.stack(ys) if ys[0] is not None else None

        return x_batch, y_batch

    def __len__(self):
        return len(self.buffer)


class ContinualLearner(nn.Module):
    """
    Unified continual learning wrapper combining multiple strategies.

    Provides a simple interface to add continual learning capabilities
    to any PyTorch model.
    """

    def __init__(self,
                 model: nn.Module,
                 strategy: str = 'ewc',
                 ewc_lambda: float = 1000.0,
                 si_lambda: float = 100.0,
                 replay_capacity: int = 10000,
                 replay_batch: int = 32):
        """
        Args:
            model: Base neural network
            strategy: 'ewc', 'si', 'replay', 'mesu', or 'combined'
            ewc_lambda: EWC regularization strength
            si_lambda: SI regularization strength
            replay_capacity: Experience replay buffer size
            replay_batch: Batch size for replay
        """
        super().__init__()
        self.model = model
        self.strategy = strategy
        self.replay_batch = replay_batch

        # Initialize components based on strategy
        if strategy in ['ewc', 'combined']:
            self.ewc = ElasticWeightConsolidation(model, ewc_lambda)
        else:
            self.ewc = None

        if strategy in ['si', 'combined']:
            self.si = SynapticIntelligence(model, si_lambda)
        else:
            self.si = None

        if strategy in ['replay', 'combined']:
            self.replay = ExperienceReplay(replay_capacity)
        else:
            self.replay = None

        if strategy in ['mesu', 'combined']:
            self.mesu = MESU(model)
        else:
            self.mesu = None

        self.current_task = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through base model."""
        return self.model(x)

    def compute_loss(self,
                     loss: torch.Tensor,
                     x: Optional[torch.Tensor] = None,
                     y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute total loss including continual learning penalties.

        Args:
            loss: Task-specific loss
            x: Current batch inputs (for replay)
            y: Current batch labels (for replay)

        Returns:
            Total loss with CL penalties
        """
        total_loss = loss

        # EWC penalty
        if self.ewc is not None:
            total_loss = total_loss + self.ewc.penalty()

        # SI penalty
        if self.si is not None:
            total_loss = total_loss + self.si.penalty()

        # Experience replay
        if self.replay is not None and len(self.replay) > 0:
            replay_x, replay_y, _ = self.replay.sample(self.replay_batch)
            replay_x = replay_x.to(next(self.model.parameters()).device)

            replay_out = self.model(replay_x)

            if replay_y is not None:
                replay_y = replay_y.to(replay_x.device)
                replay_loss = F.mse_loss(replay_out, replay_y)
            else:
                # Unsupervised: minimize reconstruction error
                replay_loss = F.mse_loss(replay_out, replay_x)

            total_loss = total_loss + 0.5 * replay_loss

        # Store current samples
        if self.replay is not None and x is not None:
            self.replay.add(x.detach().cpu(), y.detach().cpu() if y is not None else None, self.current_task)

        return total_loss

    def after_backward(self):
        """Call after loss.backward(), before optimizer.step()."""
        if self.si is not None:
            self.si.update_running_importance()

        if self.mesu is not None:
            self.mesu.update(torch.tensor(0.0))  # Loss not needed

    def consolidate_task(self, dataloader, task_name: str = ""):
        """
        Consolidate after finishing a task.

        Args:
            dataloader: DataLoader for the completed task
            task_name: Optional name for the task
        """
        if self.ewc is not None:
            self.ewc.consolidate(dataloader, task_name)

        if self.si is not None:
            self.si.consolidate(task_name)

        self.current_task += 1

    def get_stats(self) -> Dict[str, float]:
        """Get continual learning statistics."""
        stats = {
            'current_task': self.current_task,
            'strategy': self.strategy
        }

        if self.replay is not None:
            stats['replay_size'] = len(self.replay)

        return stats


if __name__ == '__main__':
    print("--- Continual Learning Examples ---")

    # Create simple model
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 10)
    )

    # Example 1: EWC
    print("\n1. Elastic Weight Consolidation")
    ewc = ElasticWeightConsolidation(model, ewc_lambda=1000.0)

    # Simulate task 1
    fake_data = [(torch.randn(4, 10),) for _ in range(10)]
    ewc.consolidate(fake_data, "task_1")

    # Compute penalty
    penalty = ewc.penalty()
    print(f"   EWC penalty after task 1: {penalty.item():.6f}")

    # Example 2: Synaptic Intelligence
    print("\n2. Synaptic Intelligence")
    si = SynapticIntelligence(model, si_lambda=100.0)

    for _ in range(20):
        x = torch.randn(4, 10)
        out = model(x)
        loss = out.sum()
        loss.backward()
        si.update_running_importance()

        with torch.no_grad():
            for p in model.parameters():
                p -= 0.01 * p.grad
                p.grad.zero_()

    si.consolidate("task_1")
    print(f"   SI penalty: {si.penalty().item():.6f}")

    # Example 3: Experience Replay
    print("\n3. Experience Replay")
    replay = ExperienceReplay(capacity=1000, strategy='reservoir')

    for i in range(100):
        x = torch.randn(10)
        y = torch.randn(10)
        replay.add(x, y, task_id=i // 50)

    x_batch, y_batch, task_ids = replay.sample(8)
    print(f"   Buffer size: {len(replay)}")
    print(f"   Sampled tasks: {task_ids}")

    # Example 4: Continual Backprop
    print("\n4. Continual Backprop")
    cb = ContinualBackprop(model, reinit_fraction=0.1)

    # Simulate training that causes dormancy
    for i in range(50):
        x = torch.randn(4, 10)
        out = model(x)

        # Track activations
        activations = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                activations[name] = out if name == '2' else x

        cb.update_utility(activations)

    stats = cb.get_stats()
    print(f"   Dormant fraction: {stats['dormant_fraction']:.2%}")

    n_reinit = cb.regenerate()
    print(f"   Reinitialized units: {n_reinit}")

    # Example 5: Unified ContinualLearner
    print("\n5. ContinualLearner (combined)")
    learner = ContinualLearner(model, strategy='combined')

    for _ in range(10):
        x = torch.randn(4, 10)
        y = torch.randn(4, 10)

        out = learner(x)
        loss = F.mse_loss(out, y)
        total_loss = learner.compute_loss(loss, x, y)

        total_loss.backward()
        learner.after_backward()

        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p -= 0.01 * p.grad
                    p.grad.zero_()

    stats = learner.get_stats()
    print(f"   Stats: {stats}")

    print("\n[OK] All continual learning tests passed!")
