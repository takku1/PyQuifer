"""Continual Learning / Sleep Consolidation Benchmark — Three-Column Evaluation.

Dual-mode:
  pytest:  python -m pytest tests/benchmarks/bench_continual.py -v --timeout=120
  CLI:     python tests/benchmarks/bench_continual.py

The "killer benchmark" — EP+SRC published to surpass BPTT on continual learning.

Column A: Published EP+SRC results (surpasses BPTT on Split-MNIST/Fashion-MNIST)
Column B: Plain MLP fine-tuning (catastrophic forgetting baseline) + EWC
Column C: PyQuifer EP + SleepReplayConsolidation

Scenarios:
  1. Split-MNIST (5 tasks, 2 digits each)
  2. Split-Fashion-MNIST (5 tasks)
  3. Permuted-MNIST (10 tasks, fixed permutations)
  4. Split-CIFAR-10 (5 tasks, 2 classes each)

Metrics:
  - Average accuracy across all tasks after final task
  - Per-task forgetting curve
  - Backward transfer
  - Forward transfer
  - Sleep cycle recovery
"""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from harness import (
    BenchmarkResult, BenchmarkSuite, MemoryTracker, MetricCollector,
    bootstrap_ci, get_device, permuted_mnist, set_seed, split_cifar10,
    split_fashion_mnist, split_mnist, timer,
)

# ---------------------------------------------------------------------------
# Baseline Models
# ---------------------------------------------------------------------------

class PlainMLP(nn.Module):
    """Simple MLP for sequential task training."""
    def __init__(self, input_dim: int = 784, hidden: int = 256, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.fc3 = nn.Linear(hidden // 2, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1) if x.dim() > 2 else x
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))

    def get_linear_layers(self) -> List[nn.Linear]:
        return [self.fc1, self.fc2, self.fc3]


# ---------------------------------------------------------------------------
# EWC (Elastic Weight Consolidation) — simple diagonal Fisher
# ---------------------------------------------------------------------------

class EWC:
    """Simple EWC implementation with diagonal Fisher approximation."""

    def __init__(self, model: nn.Module, lambda_ewc: float = 100.0):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.fisher: Dict[str, torch.Tensor] = {}
        self.star_params: Dict[str, torch.Tensor] = {}

    def compute_fisher(self, dataloader, device, num_samples: int = 200):
        self.model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()
                  if p.requires_grad}
        count = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            x = x.view(x.size(0), -1)
            self.model.zero_grad()
            out = self.model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data.pow(2) * x.size(0)
            count += x.size(0)
            if count >= num_samples:
                break

        for n in fisher:
            fisher[n] /= max(count, 1)
            if n in self.fisher:
                self.fisher[n] = self.fisher[n] + fisher[n]
            else:
                self.fisher[n] = fisher[n]

        self.star_params = {n: p.data.clone()
                           for n, p in self.model.named_parameters()
                           if p.requires_grad}

    def penalty(self) -> torch.Tensor:
        if not self.fisher:
            return torch.tensor(0.0)
        total = torch.tensor(0.0, device=next(self.model.parameters()).device)
        for n, p in self.model.named_parameters():
            if n in self.fisher:
                total = total + (self.fisher[n] * (p - self.star_params[n]).pow(2)).sum()
        return self.lambda_ewc * total


# ---------------------------------------------------------------------------
# EP + Sleep Consolidation Classifier
# ---------------------------------------------------------------------------

class EPSleepClassifier(nn.Module):
    """EPKuramotoClassifier + SleepReplayConsolidation for continual learning."""

    def __init__(self, input_dim: int = 784, num_classes: int = 10,
                 num_osc: int = 64, ep_steps: int = 20,
                 sleep_lr: float = 0.001, sleep_steps: int = 50):
        super().__init__()
        from pyquifer.equilibrium_propagation import EPKuramotoClassifier
        from pyquifer.memory_consolidation import SleepReplayConsolidation

        self.ep_model = EPKuramotoClassifier(
            input_dim=input_dim, num_classes=num_classes,
            num_oscillators=num_osc, ep_lr=0.01, ep_beta=0.1,
            free_steps=ep_steps, nudge_steps=ep_steps,
        )

        self.src = SleepReplayConsolidation(
            layer_dims=[input_dim, num_osc], sleep_lr=sleep_lr,
            num_replay_steps=sleep_steps,
        )

    def forward(self, x):
        return self.ep_model(x)

    def train_step(self, x, y, optimizer):
        result = self.ep_model.ep_train_step(x, y)
        cls_loss = result['classification_loss']
        optimizer.zero_grad()
        cls_loss.backward()
        optimizer.step()
        return result

    def sleep_cycle(self) -> Dict:
        encoder_layer = self.ep_model.encoder[0]
        self.src.set_weights_from_network([encoder_layer])
        result = self.src.sleep_cycle()
        consolidated = self.src.get_weights()
        with torch.no_grad():
            encoder_layer.weight.copy_(consolidated[0])
        return result


# ---------------------------------------------------------------------------
# Training Helpers
# ---------------------------------------------------------------------------

def train_on_task(model: nn.Module, train_loader, device,
                  optimizer, num_epochs: int = 5,
                  ewc: Optional[EWC] = None,
                  max_samples: int = 0) -> List[float]:
    model.train()
    losses = []
    for _ in range(num_epochs):
        total_loss, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x = x.view(x.size(0), -1)
            out = model(x)
            loss = F.cross_entropy(out, y)
            if ewc is not None:
                loss = loss + ewc.penalty()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            n += y.size(0)
            if max_samples > 0 and n >= max_samples:
                break
        losses.append(total_loss / max(n, 1))
    return losses


def train_ep_on_task(model: EPSleepClassifier, train_loader, device,
                     optimizer, num_epochs: int = 5,
                     max_samples: int = 0,
                     ep_samples_per_batch: int = 4) -> List[float]:
    model.train()
    losses = []
    for _ in range(num_epochs):
        total_loss, n = 0.0, 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            x_flat = x_batch.view(x_batch.size(0), -1)

            logits = model(x_flat)
            loss = F.cross_entropy(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y_batch.size(0)

            num_ep = min(ep_samples_per_batch, x_flat.size(0))
            for i in range(num_ep):
                model.train_step(x_flat[i], y_batch[i], optimizer)

            n += y_batch.size(0)
            if max_samples > 0 and n >= max_samples:
                break
        losses.append(total_loss / max(n, 1))
    return losses


def evaluate_all_tasks(model, task_test_loaders, device) -> List[float]:
    model.eval()
    accs = []
    with torch.no_grad():
        for loader in task_test_loaders:
            correct, total = 0, 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                x = x.view(x.size(0), -1)
                out = model(x)
                if isinstance(out, dict):
                    out = out.get("final_output", out.get("output"))
                pred = out.argmax(dim=-1)
                correct += (pred == y).sum().item()
                total += y.size(0)
            accs.append(correct / max(total, 1))
    return accs


# ---------------------------------------------------------------------------
# Forward Transfer Helper
# ---------------------------------------------------------------------------

def compute_forward_transfer(model, test_loaders, device,
                             num_classes: int = 10) -> List[float]:
    """Evaluate accuracy on each task BEFORE training on it (minus random baseline)."""
    random_baseline = 1.0 / num_classes
    forward_transfers = []
    for loader in test_loaders:
        accs = evaluate_all_tasks(model, [loader], device)
        ft = accs[0] - random_baseline
        forward_transfers.append(round(ft, 4))
    return forward_transfers


# ---------------------------------------------------------------------------
# Benchmark Scenarios
# ---------------------------------------------------------------------------

def bench_continual(dataset: str = "mnist", num_tasks: int = 5,
                    epochs_per_task: int = 5, seed: int = 42,
                    max_train: int = 0, max_test: int = 0,
                    num_osc: int = 64, ep_steps: int = 20,
                    sleep_steps: int = 50, ewc_lambda: float = 100.0
                    ) -> MetricCollector:
    """Run full continual learning benchmark on Split-MNIST or Split-Fashion-MNIST."""
    device = get_device()
    set_seed(seed)

    split_fn = split_mnist if dataset == "mnist" else split_fashion_mnist
    tasks = split_fn(num_tasks=num_tasks, batch_size=128,
                     max_samples_per_task=max_train)
    test_loaders = [t[1] for t in tasks]
    train_loaders = [t[0] for t in tasks]

    name = f"Split-{dataset.upper()} ({num_tasks} tasks)"
    mc = MetricCollector(name)

    # Column A: Published
    if dataset == "mnist":
        mc.record("A_published", "avg_accuracy", 0.95,
                  {"source": "EP+SRC surpasses BPTT (Laborieux et al.)"})
    else:
        mc.record("A_published", "avg_accuracy", 0.88,
                  {"source": "EP+SRC Fashion-MNIST estimate"})

    # --- Column B1: Naive fine-tuning ---
    set_seed(seed)
    mlp_naive = PlainMLP().to(device)
    opt_naive = torch.optim.Adam(mlp_naive.parameters(), lr=1e-3)
    naive_accs_after_each: List[List[float]] = []
    naive_peak_accs: List[float] = [0.0] * num_tasks

    for t_idx in range(num_tasks):
        # Forward transfer: evaluate BEFORE training on this task
        ft = evaluate_all_tasks(mlp_naive, [test_loaders[t_idx]], device)
        if t_idx > 0:
            mc.record("B_pytorch", f"forward_transfer_task_{t_idx}",
                      round(ft[0] - 0.1, 4))  # minus random baseline for 10 classes

        train_on_task(mlp_naive, train_loaders[t_idx], device, opt_naive,
                      num_epochs=epochs_per_task, max_samples=max_train)
        accs = evaluate_all_tasks(mlp_naive, test_loaders, device)
        naive_accs_after_each.append(accs)
        for i, a in enumerate(accs):
            naive_peak_accs[i] = max(naive_peak_accs[i], a)

    final_naive = naive_accs_after_each[-1]
    avg_naive = sum(final_naive) / len(final_naive)
    mc.record("B_pytorch", "avg_accuracy", round(avg_naive, 4),
              {"method": "naive_fine_tuning"})
    for i, a in enumerate(final_naive):
        mc.record("B_pytorch", f"task_{i}_accuracy", round(a, 4))

    # Backward transfer
    bwt_naive = 0.0
    for i in range(num_tasks - 1):
        bwt_naive += final_naive[i] - naive_accs_after_each[i][i]
    bwt_naive /= max(num_tasks - 1, 1)
    mc.record("B_pytorch", "backward_transfer", round(bwt_naive, 4))

    # Per-task forgetting
    for i in range(num_tasks):
        forgetting = naive_peak_accs[i] - final_naive[i]
        mc.record("B_pytorch", f"forgetting_task_{i}", round(forgetting, 4))

    # --- Column B2: EWC ---
    set_seed(seed)
    mlp_ewc = PlainMLP().to(device)
    opt_ewc = torch.optim.Adam(mlp_ewc.parameters(), lr=1e-3)
    ewc = EWC(mlp_ewc, lambda_ewc=ewc_lambda)
    ewc_accs_after_each: List[List[float]] = []

    for t_idx in range(num_tasks):
        train_on_task(mlp_ewc, train_loaders[t_idx], device, opt_ewc,
                      num_epochs=epochs_per_task, ewc=ewc,
                      max_samples=max_train)
        ewc.compute_fisher(train_loaders[t_idx], device)
        accs = evaluate_all_tasks(mlp_ewc, test_loaders, device)
        ewc_accs_after_each.append(accs)

    final_ewc = ewc_accs_after_each[-1]
    avg_ewc = sum(final_ewc) / len(final_ewc)
    mc.record("B_pytorch", "avg_accuracy_ewc", round(avg_ewc, 4),
              {"method": "EWC", "lambda": ewc_lambda})

    bwt_ewc = 0.0
    for i in range(num_tasks - 1):
        bwt_ewc += final_ewc[i] - ewc_accs_after_each[i][i]
    bwt_ewc /= max(num_tasks - 1, 1)
    mc.record("B_pytorch", "backward_transfer_ewc", round(bwt_ewc, 4))

    # --- Column C: EP + SleepReplayConsolidation ---
    set_seed(seed)
    ep_sleep = EPSleepClassifier(
        num_osc=num_osc, ep_steps=ep_steps, sleep_steps=sleep_steps,
    ).to(device)
    opt_ep = torch.optim.Adam(
        list(ep_sleep.ep_model.encoder.parameters()) +
        list(ep_sleep.ep_model.readout.parameters()),
        lr=1e-3,
    )

    ep_accs_after_each: List[List[float]] = []
    sleep_recovery: List[Tuple[float, float]] = []
    ep_peak_accs: List[float] = [0.0] * num_tasks

    for t_idx in range(num_tasks):
        # Forward transfer
        if t_idx > 0:
            ft = evaluate_all_tasks(ep_sleep, [test_loaders[t_idx]], device)
            mc.record("C_pyquifer", f"forward_transfer_task_{t_idx}",
                      round(ft[0] - 0.1, 4))

        train_ep_on_task(ep_sleep, train_loaders[t_idx], device, opt_ep,
                         num_epochs=epochs_per_task,
                         max_samples=max_train if max_train > 0 else 0,
                         ep_samples_per_batch=4)

        accs_before = evaluate_all_tasks(ep_sleep, test_loaders, device)
        avg_before = sum(accs_before) / len(accs_before)

        ep_sleep.sleep_cycle()

        accs_after = evaluate_all_tasks(ep_sleep, test_loaders, device)
        avg_after = sum(accs_after) / len(accs_after)
        sleep_recovery.append((avg_before, avg_after))
        ep_accs_after_each.append(accs_after)
        for i, a in enumerate(accs_after):
            ep_peak_accs[i] = max(ep_peak_accs[i], a)

    final_ep = ep_accs_after_each[-1]
    avg_ep = sum(final_ep) / len(final_ep)
    mc.record("C_pyquifer", "avg_accuracy", round(avg_ep, 4))
    for i, a in enumerate(final_ep):
        mc.record("C_pyquifer", f"task_{i}_accuracy", round(a, 4))

    bwt_ep = 0.0
    for i in range(num_tasks - 1):
        bwt_ep += final_ep[i] - ep_accs_after_each[i][i]
    bwt_ep /= max(num_tasks - 1, 1)
    mc.record("C_pyquifer", "backward_transfer", round(bwt_ep, 4))

    # Per-task forgetting
    for i in range(num_tasks):
        forgetting = ep_peak_accs[i] - final_ep[i]
        mc.record("C_pyquifer", f"forgetting_task_{i}", round(forgetting, 4))

    # Sleep recovery metrics
    for t_idx, (before, after) in enumerate(sleep_recovery):
        mc.record("C_pyquifer", f"sleep_recovery_task_{t_idx}",
                  round(after - before, 4))

    return mc


# ---------------------------------------------------------------------------
# Scenario: Permuted-MNIST
# ---------------------------------------------------------------------------

def bench_permuted_mnist(num_tasks: int = 10, epochs_per_task: int = 5,
                         seed: int = 42, max_train: int = 0,
                         max_test: int = 0, num_osc: int = 64,
                         ep_steps: int = 20, sleep_steps: int = 50
                         ) -> MetricCollector:
    """Permuted-MNIST continual learning benchmark."""
    device = get_device()
    set_seed(seed)

    tasks = permuted_mnist(num_tasks=num_tasks, batch_size=128,
                           max_samples=max_train)
    test_loaders = [t[1] for t in tasks]
    train_loaders = [t[0] for t in tasks]

    mc = MetricCollector(f"Permuted-MNIST ({num_tasks} tasks)")

    # Column A: Published
    mc.record("A_published", "avg_accuracy", 0.93,
              {"source": "Bayesian CL SOTA ~93%"})

    # --- Column B: Naive fine-tuning ---
    set_seed(seed)
    mlp = PlainMLP().to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    mlp_peak_accs = [0.0] * num_tasks

    for t_idx in range(num_tasks):
        train_on_task(mlp, train_loaders[t_idx], device, opt,
                      num_epochs=epochs_per_task, max_samples=max_train)
        accs = evaluate_all_tasks(mlp, test_loaders, device)
        for i, a in enumerate(accs):
            mlp_peak_accs[i] = max(mlp_peak_accs[i], a)

    final_accs = evaluate_all_tasks(mlp, test_loaders, device)
    avg_acc = sum(final_accs) / len(final_accs)
    mc.record("B_pytorch", "avg_accuracy", round(avg_acc, 4))

    for i in range(num_tasks):
        forgetting = mlp_peak_accs[i] - final_accs[i]
        mc.record("B_pytorch", f"forgetting_task_{i}", round(forgetting, 4))

    # --- Column C: EP + SRC ---
    set_seed(seed)
    ep_sleep = EPSleepClassifier(
        num_osc=num_osc, ep_steps=ep_steps, sleep_steps=sleep_steps,
    ).to(device)
    opt_ep = torch.optim.Adam(
        list(ep_sleep.ep_model.encoder.parameters()) +
        list(ep_sleep.ep_model.readout.parameters()),
        lr=1e-3,
    )
    ep_peak_accs = [0.0] * num_tasks

    for t_idx in range(num_tasks):
        # Forward transfer
        if t_idx > 0:
            ft = evaluate_all_tasks(ep_sleep, [test_loaders[t_idx]], device)
            mc.record("C_pyquifer", f"forward_transfer_task_{t_idx}",
                      round(ft[0] - 0.1, 4))

        train_ep_on_task(ep_sleep, train_loaders[t_idx], device, opt_ep,
                         num_epochs=epochs_per_task,
                         max_samples=max_train if max_train > 0 else 0,
                         ep_samples_per_batch=4)
        ep_sleep.sleep_cycle()
        accs = evaluate_all_tasks(ep_sleep, test_loaders, device)
        for i, a in enumerate(accs):
            ep_peak_accs[i] = max(ep_peak_accs[i], a)

    final_ep = evaluate_all_tasks(ep_sleep, test_loaders, device)
    avg_ep = sum(final_ep) / len(final_ep)
    mc.record("C_pyquifer", "avg_accuracy", round(avg_ep, 4))

    for i in range(num_tasks):
        forgetting = ep_peak_accs[i] - final_ep[i]
        mc.record("C_pyquifer", f"forgetting_task_{i}", round(forgetting, 4))

    return mc


# ---------------------------------------------------------------------------
# Scenario: Split-CIFAR-10
# ---------------------------------------------------------------------------

def bench_continual_cifar10(num_tasks: int = 5, epochs_per_task: int = 5,
                            seed: int = 42, max_train: int = 0,
                            max_test: int = 0, num_osc: int = 64,
                            ep_steps: int = 20, sleep_steps: int = 50
                            ) -> MetricCollector:
    """Split-CIFAR-10 continual learning benchmark."""
    device = get_device()
    set_seed(seed)

    tasks = split_cifar10(num_tasks=num_tasks, batch_size=128,
                          max_samples_per_task=max_train)
    test_loaders = [t[1] for t in tasks]
    train_loaders = [t[0] for t in tasks]

    mc = MetricCollector(f"Split-CIFAR-10 ({num_tasks} tasks)")

    # Column A: Published
    mc.record("A_published", "avg_accuracy", 0.90,
              {"source": "EP+SRC matches BPTT ~90%"})

    # --- Column B: MLP with input_dim=3072 ---
    set_seed(seed)
    mlp = PlainMLP(input_dim=3072).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    mlp_peak_accs = [0.0] * num_tasks

    for t_idx in range(num_tasks):
        train_on_task(mlp, train_loaders[t_idx], device, opt,
                      num_epochs=epochs_per_task, max_samples=max_train)
        accs = evaluate_all_tasks(mlp, test_loaders, device)
        for i, a in enumerate(accs):
            mlp_peak_accs[i] = max(mlp_peak_accs[i], a)

    final_accs = evaluate_all_tasks(mlp, test_loaders, device)
    avg_acc = sum(final_accs) / len(final_accs)
    mc.record("B_pytorch", "avg_accuracy", round(avg_acc, 4))

    for i in range(num_tasks):
        forgetting = mlp_peak_accs[i] - final_accs[i]
        mc.record("B_pytorch", f"forgetting_task_{i}", round(forgetting, 4))

    # --- Column C: EP + SRC with input_dim=3072 ---
    set_seed(seed)
    ep_sleep = EPSleepClassifier(
        input_dim=3072, num_osc=num_osc, ep_steps=ep_steps,
        sleep_steps=sleep_steps,
    ).to(device)
    opt_ep = torch.optim.Adam(
        list(ep_sleep.ep_model.encoder.parameters()) +
        list(ep_sleep.ep_model.readout.parameters()),
        lr=1e-3,
    )
    ep_peak_accs = [0.0] * num_tasks

    for t_idx in range(num_tasks):
        train_ep_on_task(ep_sleep, train_loaders[t_idx], device, opt_ep,
                         num_epochs=epochs_per_task,
                         max_samples=max_train if max_train > 0 else 0,
                         ep_samples_per_batch=4)
        ep_sleep.sleep_cycle()
        accs = evaluate_all_tasks(ep_sleep, test_loaders, device)
        for i, a in enumerate(accs):
            ep_peak_accs[i] = max(ep_peak_accs[i], a)

    final_ep = evaluate_all_tasks(ep_sleep, test_loaders, device)
    avg_ep = sum(final_ep) / len(final_ep)
    mc.record("C_pyquifer", "avg_accuracy", round(avg_ep, 4))

    for i in range(num_tasks):
        forgetting = ep_peak_accs[i] - final_ep[i]
        mc.record("C_pyquifer", f"forgetting_task_{i}", round(forgetting, 4))

    return mc


# ---------------------------------------------------------------------------
# Full Suite (CLI mode)
# ---------------------------------------------------------------------------

def run_full_suite(num_seeds: int = 3, epochs_per_task: int = 5):
    print("=" * 60, flush=True)
    print("Continual Learning Benchmark — Full Suite", flush=True)
    print("=" * 60, flush=True)

    suite = BenchmarkSuite("Continual Learning Benchmarks")

    # Split-MNIST and Split-Fashion-MNIST
    for dataset in ["mnist", "fashion_mnist"]:
        print(f"\n--- Split-{dataset.upper()} ---", flush=True)
        avg_naive, avg_ewc, avg_ep = [], [], []
        for seed in range(num_seeds):
            mc = bench_continual(
                dataset=dataset, epochs_per_task=epochs_per_task, seed=seed,
                num_osc=64, ep_steps=10, sleep_steps=20,
            )
            for r in mc.results:
                if r.column == "B_pytorch":
                    if "avg_accuracy" in r.metrics:
                        avg_naive.append(r.metrics["avg_accuracy"])
                    if "avg_accuracy_ewc" in r.metrics:
                        avg_ewc.append(r.metrics["avg_accuracy_ewc"])
                elif r.column == "C_pyquifer" and "avg_accuracy" in r.metrics:
                    avg_ep.append(r.metrics["avg_accuracy"])
            print(f"  Seed {seed}: naive={avg_naive[-1]:.4f} "
                  f"EWC={avg_ewc[-1]:.4f} EP+SRC={avg_ep[-1]:.4f}", flush=True)

        agg = MetricCollector(f"Split-{dataset.upper()} (aggregated)")
        if avg_naive:
            lo, hi = bootstrap_ci(avg_naive)
            agg.record("B_pytorch", "avg_accuracy_naive",
                       round(sum(avg_naive)/len(avg_naive), 4),
                       {"ci_95": [round(lo, 4), round(hi, 4)]})
        if avg_ewc:
            lo, hi = bootstrap_ci(avg_ewc)
            agg.record("B_pytorch", "avg_accuracy_ewc",
                       round(sum(avg_ewc)/len(avg_ewc), 4),
                       {"ci_95": [round(lo, 4), round(hi, 4)]})
        if avg_ep:
            lo, hi = bootstrap_ci(avg_ep)
            agg.record("C_pyquifer", "avg_accuracy",
                       round(sum(avg_ep)/len(avg_ep), 4),
                       {"ci_95": [round(lo, 4), round(hi, 4)]})
        suite.add(agg)

    # Permuted-MNIST
    print("\n--- Permuted-MNIST ---", flush=True)
    mc_perm = bench_permuted_mnist(num_tasks=5, epochs_per_task=epochs_per_task,
                                   ep_steps=10, sleep_steps=20)
    suite.add(mc_perm)

    # Split-CIFAR-10 (capped dataset — EP single-sample updates are expensive)
    print("\n--- Split-CIFAR-10 ---", flush=True)
    mc_cifar = bench_continual_cifar10(epochs_per_task=epochs_per_task,
                                       ep_steps=10, sleep_steps=20,
                                       max_train=5000, max_test=1000)
    suite.add(mc_cifar)

    json_path = str(Path(__file__).parent / "results" / "continual.json")
    suite.to_json(json_path)
    print(f"\nResults saved to {json_path}")
    print("\n" + suite.to_markdown())


# ---------------------------------------------------------------------------
# Pytest Classes (smoke tests)
# ---------------------------------------------------------------------------

class TestContinualMNIST:
    def test_split_mnist_runs(self):
        mc = bench_continual(
            dataset="mnist", num_tasks=2, epochs_per_task=1,
            seed=0, max_train=64, max_test=32,
            num_osc=16, ep_steps=5, sleep_steps=10,
        )
        columns = {r.column for r in mc.results}
        assert "B_pytorch" in columns
        assert "C_pyquifer" in columns

    def test_split_mnist_has_forgetting(self):
        mc = bench_continual(
            dataset="mnist", num_tasks=2, epochs_per_task=1,
            seed=0, max_train=64, max_test=32,
            num_osc=16, ep_steps=5, sleep_steps=10,
        )
        metric_keys = set()
        for r in mc.results:
            metric_keys.update(r.metrics.keys())
        assert any("forgetting" in k for k in metric_keys)

    def test_ewc_penalty(self):
        device = get_device()
        model = PlainMLP().to(device)
        ewc = EWC(model, lambda_ewc=10.0)
        assert ewc.penalty().item() == 0.0

    def test_ep_sleep_classifier_forward(self):
        device = get_device()
        model = EPSleepClassifier(num_osc=16, ep_steps=5, sleep_steps=10).to(device)
        x = torch.randn(2, 784, device=device)
        out = model(x)
        assert out.shape == (2, 10)

    def test_sleep_cycle_runs(self):
        device = get_device()
        model = EPSleepClassifier(num_osc=16, ep_steps=5, sleep_steps=10).to(device)
        result = model.sleep_cycle()
        assert "mean_delta_norm" in result


class TestContinualFashionMNIST:
    def test_split_fashion_mnist_runs(self):
        mc = bench_continual(
            dataset="fashion_mnist", num_tasks=2, epochs_per_task=1,
            seed=0, max_train=64, max_test=32,
            num_osc=16, ep_steps=5, sleep_steps=10,
        )
        columns = {r.column for r in mc.results}
        assert "B_pytorch" in columns


class TestPermutedMNIST:
    def test_permuted_mnist_runs(self):
        mc = bench_permuted_mnist(
            num_tasks=2, epochs_per_task=1,
            seed=0, max_train=64, max_test=32,
            num_osc=16, ep_steps=5, sleep_steps=10,
        )
        columns = {r.column for r in mc.results}
        assert "B_pytorch" in columns
        assert "C_pyquifer" in columns

    def test_permuted_mnist_has_forgetting(self):
        mc = bench_permuted_mnist(
            num_tasks=2, epochs_per_task=1,
            seed=0, max_train=64, max_test=32,
            num_osc=16, ep_steps=5, sleep_steps=10,
        )
        metric_keys = set()
        for r in mc.results:
            metric_keys.update(r.metrics.keys())
        assert any("forgetting" in k for k in metric_keys)


class TestContinualCIFAR10:
    def test_split_cifar10_runs(self):
        mc = bench_continual_cifar10(
            num_tasks=2, epochs_per_task=1,
            seed=0, max_train=64, max_test=32,
            num_osc=16, ep_steps=5, sleep_steps=10,
        )
        columns = {r.column for r in mc.results}
        assert "B_pytorch" in columns
        assert "C_pyquifer" in columns


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_full_suite()
