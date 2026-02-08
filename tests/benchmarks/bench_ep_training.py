"""Equilibrium Propagation Training Benchmark — Three-Column Evaluation.

Dual-mode:
  pytest:  python -m pytest tests/benchmarks/bench_ep_training.py -v --timeout=120
  CLI:     python tests/benchmarks/bench_ep_training.py

Column A: Published EP accuracy (Scellier & Bengio, Laborieux et al.)
Column B: Plain MLP/CNN backprop baseline
Column C: EPKuramotoClassifier from PyQuifer

Scenarios:
  1. MNIST classification
  2. Fashion-MNIST classification
  3. CIFAR-10 classification
  4. XOR sanity check
  5. Convergence speed (steps to 90%, 95%, 97%)
  6. Beta sensitivity sweep (extended with 0.8)
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from harness import (
    BenchmarkResult, BenchmarkSuite, MemoryTracker, MetricCollector,
    bootstrap_ci, compute_accuracy, get_device, load_cifar10,
    load_fashion_mnist, load_mnist, set_seed, timer,
)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class PlainMLP(nn.Module):
    """Baseline MLP trained with standard backprop."""
    def __init__(self, input_dim: int = 784, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1) if x.dim() > 2 else x)


class PlainCNN(nn.Module):
    """Baseline CNN for CIFAR-10 trained with standard backprop."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            # Flattened input — reshape to image
            x = x.view(x.size(0), 3, 32, 32)
        elif x.dim() == 3:
            x = x.unsqueeze(0)
        feat = self.features(x)
        return self.classifier(feat.view(feat.size(0), -1))


# ---------------------------------------------------------------------------
# Training Loops
# ---------------------------------------------------------------------------

def train_mlp_epoch(model: nn.Module, loader, optimizer, device) -> float:
    """Train one epoch, return mean loss."""
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x = x.view(x.size(0), -1)
        out = model(x)
        loss = F.cross_entropy(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        n += y.size(0)
    return total_loss / n


def train_cnn_epoch(model: nn.Module, loader, optimizer, device) -> float:
    """Train one CNN epoch (keeps image dimensions), return mean loss."""
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = F.cross_entropy(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        n += y.size(0)
    return total_loss / n


def compute_cnn_accuracy(model: nn.Module, dataloader, device) -> float:
    """Accuracy for CNN (keeps spatial dims)."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


def train_ep_epoch(model, loader, optimizer, device,
                   max_samples: int = 0,
                   ep_samples_per_batch: int = 4) -> float:
    """Train one EP epoch — hybrid: batch backprop + periodic EP coupling updates."""
    model.train()
    total_loss = 0.0
    n = 0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        x_flat = x_batch.view(x_batch.size(0), -1)

        # 1. Batch forward/backward for encoder+readout
        logits = model(x_flat)
        loss = F.cross_entropy(logits, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y_batch.size(0)

        # 2. EP coupling updates on a few samples per batch
        num_ep = min(ep_samples_per_batch, x_flat.size(0))
        for i in range(num_ep):
            model.ep_train_step(x_flat[i], y_batch[i])

        n += y_batch.size(0)
        if max_samples > 0 and n >= max_samples:
            break
    return total_loss / max(n, 1)


# ---------------------------------------------------------------------------
# EP-specific Metrics
# ---------------------------------------------------------------------------

def compute_coupling_sparsity(model) -> float:
    """Fraction of near-zero coupling weights in EP model."""
    if not hasattr(model, 'oscillators'):
        return 0.0
    coupling = model.oscillators.coupling.detach()
    near_zero = (coupling.abs() < 1e-3).float().mean().item()
    return round(near_zero, 4)


# ---------------------------------------------------------------------------
# Scenario 1 & 2: MNIST / Fashion-MNIST Classification
# ---------------------------------------------------------------------------

def bench_classification(dataset: str = "mnist", num_epochs: int = 10,
                         seed: int = 42, max_train: int = 0,
                         max_test: int = 0, num_osc: int = 64,
                         ep_steps: int = 20) -> MetricCollector:
    """Train MLP and EP classifier, compare accuracy."""
    device = get_device()
    set_seed(seed)

    loader_fn = load_mnist if dataset == "mnist" else load_fashion_mnist
    train_loader = loader_fn(train=True, batch_size=128, max_samples=max_train)
    test_loader = loader_fn(train=False, batch_size=256, max_samples=max_test)

    mc = MetricCollector(f"{dataset.upper()} Classification")

    # Column A: Published results
    if dataset == "mnist":
        mc.record("A_published", "accuracy", 0.9777,
                  {"source": "Laborieux et al. 2021"})
    else:
        mc.record("A_published", "accuracy", 0.90,
                  {"source": "EP Fashion-MNIST estimate"})

    # Column B: Plain MLP backprop
    mlp = PlainMLP().to(device)
    opt_mlp = torch.optim.Adam(mlp.parameters(), lr=1e-3)

    with timer() as t_mlp, MemoryTracker() as mem_mlp:
        for _ in range(num_epochs):
            train_mlp_epoch(mlp, train_loader, opt_mlp, device)

    acc_mlp = compute_accuracy(mlp, test_loader, device)
    mc.record("B_pytorch", "accuracy", round(acc_mlp, 4))
    mc.record("B_pytorch", "time_ms", round(t_mlp["elapsed_ms"], 1))
    mc.record("B_pytorch", "peak_memory_mb", mem_mlp.peak_mb)

    # Column C: EPKuramotoClassifier
    from pyquifer.equilibrium_propagation import EPKuramotoClassifier
    ep_model = EPKuramotoClassifier(
        input_dim=784, num_classes=10, num_oscillators=num_osc,
        ep_lr=0.01, ep_beta=0.1,
        free_steps=ep_steps, nudge_steps=ep_steps,
    ).to(device)
    opt_ep = torch.optim.Adam(
        list(ep_model.encoder.parameters()) + list(ep_model.readout.parameters()),
        lr=1e-3,
    )

    with timer() as t_ep, MemoryTracker() as mem_ep:
        for _ in range(num_epochs):
            train_ep_epoch(ep_model, train_loader, opt_ep, device,
                           max_samples=max_train if max_train > 0 else 0,
                           ep_samples_per_batch=4)

    acc_ep = compute_accuracy(ep_model, test_loader, device)
    sparsity = compute_coupling_sparsity(ep_model)
    mc.record("C_pyquifer", "accuracy", round(acc_ep, 4))
    mc.record("C_pyquifer", "time_ms", round(t_ep["elapsed_ms"], 1))
    mc.record("C_pyquifer", "peak_memory_mb", mem_ep.peak_mb)
    mc.record("C_pyquifer", "coupling_sparsity", sparsity)

    return mc


# ---------------------------------------------------------------------------
# Scenario 3: CIFAR-10 Classification
# ---------------------------------------------------------------------------

def bench_cifar10(num_epochs: int = 10, seed: int = 42,
                  max_train: int = 0, max_test: int = 0,
                  num_osc: int = 64, ep_steps: int = 20) -> MetricCollector:
    """CIFAR-10: CNN baseline vs EP classifier."""
    device = get_device()
    set_seed(seed)

    train_loader = load_cifar10(train=True, batch_size=128, max_samples=max_train)
    test_loader = load_cifar10(train=False, batch_size=256, max_samples=max_test)

    mc = MetricCollector("CIFAR-10 Classification")

    # Column A: Published EP results
    mc.record("A_published", "accuracy", 0.883,
              {"source": "Ernoult et al. EP CIFAR-10"})
    mc.record("A_published", "accuracy_homeostatic", 0.843,
              {"source": "ICLR 2024 homeostatic EP"})

    # Column B: PlainCNN backprop
    cnn = PlainCNN().to(device)
    opt_cnn = torch.optim.Adam(cnn.parameters(), lr=1e-3)

    with timer() as t_b, MemoryTracker() as mem_b:
        for _ in range(num_epochs):
            train_cnn_epoch(cnn, train_loader, opt_cnn, device)

    acc_cnn = compute_cnn_accuracy(cnn, test_loader, device)
    mc.record("B_pytorch", "accuracy", round(acc_cnn, 4))
    mc.record("B_pytorch", "time_ms", round(t_b["elapsed_ms"], 1))
    mc.record("B_pytorch", "peak_memory_mb", mem_b.peak_mb)

    # Column C: EPKuramotoClassifier with input_dim=3072
    from pyquifer.equilibrium_propagation import EPKuramotoClassifier
    ep_model = EPKuramotoClassifier(
        input_dim=3072, num_classes=10, num_oscillators=num_osc,
        ep_lr=0.01, ep_beta=0.1,
        free_steps=ep_steps, nudge_steps=ep_steps,
    ).to(device)
    opt_ep = torch.optim.Adam(
        list(ep_model.encoder.parameters()) + list(ep_model.readout.parameters()),
        lr=1e-3,
    )

    with timer() as t_c, MemoryTracker() as mem_c:
        for _ in range(num_epochs):
            train_ep_epoch(ep_model, train_loader, opt_ep, device,
                           max_samples=max_train if max_train > 0 else 0,
                           ep_samples_per_batch=4)

    acc_ep = compute_accuracy(ep_model, test_loader, device)
    sparsity = compute_coupling_sparsity(ep_model)
    mc.record("C_pyquifer", "accuracy", round(acc_ep, 4))
    mc.record("C_pyquifer", "time_ms", round(t_c["elapsed_ms"], 1))
    mc.record("C_pyquifer", "peak_memory_mb", mem_c.peak_mb)
    mc.record("C_pyquifer", "coupling_sparsity", sparsity)

    return mc


# ---------------------------------------------------------------------------
# Scenario 4: XOR Sanity Check
# ---------------------------------------------------------------------------

def bench_xor(num_epochs: int = 200, seed: int = 42) -> MetricCollector:
    """XOR sanity check — all methods must reach 100%."""
    device = get_device()
    set_seed(seed)

    mc = MetricCollector("XOR Sanity Check")

    # XOR dataset
    xor_x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32, device=device)
    xor_y = torch.tensor([0, 1, 1, 0], dtype=torch.long, device=device)

    # Column A: trivial
    mc.record("A_published", "accuracy", 1.0, {"source": "trivial (XOR)"})

    # Column B: MLP
    set_seed(seed)
    mlp = PlainMLP(input_dim=2, num_classes=2).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=0.01)
    for _ in range(num_epochs):
        out = mlp(xor_x)
        loss = F.cross_entropy(out, xor_y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = mlp(xor_x).argmax(dim=-1)
        acc_b = (pred == xor_y).float().mean().item()
    mc.record("B_pytorch", "accuracy", round(acc_b, 4))

    # Column C: EP
    from pyquifer.equilibrium_propagation import EPKuramotoClassifier
    set_seed(seed)
    ep_model = EPKuramotoClassifier(
        input_dim=2, num_classes=2, num_oscillators=8,
        ep_lr=0.01, ep_beta=0.1, free_steps=10, nudge_steps=10,
    ).to(device)
    opt_ep = torch.optim.Adam(
        list(ep_model.encoder.parameters()) + list(ep_model.readout.parameters()),
        lr=0.01,
    )
    for _ in range(num_epochs):
        logits = ep_model(xor_x)
        loss = F.cross_entropy(logits, xor_y)
        opt_ep.zero_grad()
        loss.backward()
        opt_ep.step()
        # EP coupling updates
        for i in range(4):
            ep_model.ep_train_step(xor_x[i], xor_y[i])

    with torch.no_grad():
        pred = ep_model(xor_x).argmax(dim=-1)
        acc_c = (pred == xor_y).float().mean().item()
    mc.record("C_pyquifer", "accuracy", round(acc_c, 4))

    return mc


# ---------------------------------------------------------------------------
# Scenario 5: Convergence Speed
# ---------------------------------------------------------------------------

def bench_convergence(seed: int = 42, max_train: int = 0,
                      max_test: int = 0, num_epochs: int = 10,
                      ep_steps: int = 20) -> MetricCollector:
    """Measure steps to reach 90%, 95% accuracy on MNIST."""
    device = get_device()
    set_seed(seed)

    train_loader = load_mnist(train=True, batch_size=128, max_samples=max_train)
    test_loader = load_mnist(train=False, batch_size=256, max_samples=max_test)

    mc = MetricCollector("Convergence Speed (MNIST)")
    thresholds = [0.90, 0.95]

    # Column B: MLP
    mlp = PlainMLP().to(device)
    opt_mlp = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    for epoch in range(1, num_epochs + 1):
        train_mlp_epoch(mlp, train_loader, opt_mlp, device)
        acc = compute_accuracy(mlp, test_loader, device)
        for th in thresholds:
            key = f"epochs_to_{int(th*100)}pct"
            existing = None
            for r in mc.results:
                if r.column == "B_pytorch" and key in r.metrics:
                    existing = r.metrics[key]
            if existing is None and acc >= th:
                mc.record("B_pytorch", key, epoch)
    mc.record("B_pytorch", "final_accuracy", round(acc, 4))

    # Column C: EP
    from pyquifer.equilibrium_propagation import EPKuramotoClassifier
    ep_model = EPKuramotoClassifier(
        input_dim=784, num_classes=10, num_oscillators=64,
        free_steps=ep_steps, nudge_steps=ep_steps,
    ).to(device)
    opt_ep = torch.optim.Adam(
        list(ep_model.encoder.parameters()) + list(ep_model.readout.parameters()),
        lr=1e-3,
    )

    for epoch in range(1, num_epochs + 1):
        train_ep_epoch(ep_model, train_loader, opt_ep, device,
                       max_samples=max_train if max_train > 0 else 0,
                       ep_samples_per_batch=4)
        acc = compute_accuracy(ep_model, test_loader, device)
        for th in thresholds:
            key = f"epochs_to_{int(th*100)}pct"
            existing = None
            for r in mc.results:
                if r.column == "C_pyquifer" and key in r.metrics:
                    existing = r.metrics[key]
            if existing is None and acc >= th:
                mc.record("C_pyquifer", key, epoch)
    mc.record("C_pyquifer", "final_accuracy", round(acc, 4))

    return mc


# ---------------------------------------------------------------------------
# Scenario 6: Beta Sensitivity Sweep (extended with 0.8)
# ---------------------------------------------------------------------------

def bench_beta_sweep(betas: Optional[List[float]] = None,
                     seed: int = 42, max_train: int = 0,
                     max_test: int = 0, num_epochs: int = 5,
                     ep_steps: int = 20) -> MetricCollector:
    """Sweep EP beta parameter, report accuracy at each."""
    if betas is None:
        betas = [0.01, 0.05, 0.1, 0.2, 0.4, 0.8]

    device = get_device()
    mc = MetricCollector("Beta Sensitivity Sweep (MNIST)")

    train_loader = load_mnist(train=True, batch_size=128, max_samples=max_train)
    test_loader = load_mnist(train=False, batch_size=256, max_samples=max_test)

    from pyquifer.equilibrium_propagation import EPKuramotoClassifier

    for beta in betas:
        set_seed(seed)
        ep_model = EPKuramotoClassifier(
            input_dim=784, num_classes=10, num_oscillators=64,
            ep_lr=0.01, ep_beta=beta,
            free_steps=ep_steps, nudge_steps=ep_steps,
        ).to(device)
        opt = torch.optim.Adam(
            list(ep_model.encoder.parameters()) + list(ep_model.readout.parameters()),
            lr=1e-3,
        )

        for _ in range(num_epochs):
            train_ep_epoch(ep_model, train_loader, opt, device,
                           max_samples=max_train if max_train > 0 else 0,
                           ep_samples_per_batch=4)

        acc = compute_accuracy(ep_model, test_loader, device)
        sparsity = compute_coupling_sparsity(ep_model)
        mc.record("C_pyquifer", f"accuracy_beta={beta}", round(acc, 4))
        mc.record("C_pyquifer", f"sparsity_beta={beta}", sparsity)

    return mc


# ---------------------------------------------------------------------------
# Full Suite (CLI mode)
# ---------------------------------------------------------------------------

def run_full_suite(num_seeds: int = 3, num_epochs: int = 5):
    """Full benchmark with multiple seeds — produces JSON results."""
    print("=" * 60, flush=True)
    print("EP Training Benchmark — Full Suite", flush=True)
    print("=" * 60, flush=True)

    suite = BenchmarkSuite("EP Training Benchmarks")
    results_dir = Path(__file__).parent / "results"

    # MNIST and Fashion-MNIST classification with multiple seeds
    for dataset in ["mnist", "fashion_mnist"]:
        print(f"\n--- {dataset.upper()} Classification ---", flush=True)
        accs_b, accs_c = [], []
        for seed in range(num_seeds):
            mc = bench_classification(
                dataset=dataset, num_epochs=num_epochs, seed=seed,
                num_osc=64, ep_steps=15,
                max_train=10000, max_test=2000,
            )
            for r in mc.results:
                if r.column == "B_pytorch" and "accuracy" in r.metrics:
                    accs_b.append(r.metrics["accuracy"])
                elif r.column == "C_pyquifer" and "accuracy" in r.metrics:
                    accs_c.append(r.metrics["accuracy"])
            print(f"  Seed {seed}: B={accs_b[-1] if accs_b else '?':.4f} "
                  f"C={accs_c[-1] if accs_c else '?':.4f}", flush=True)

        agg = MetricCollector(f"{dataset.upper()} Classification (aggregated)")
        if dataset == "mnist":
            agg.record("A_published", "accuracy", 0.9777)
        else:
            agg.record("A_published", "accuracy", 0.90)
        if accs_b:
            lo, hi = bootstrap_ci(accs_b)
            agg.record("B_pytorch", "accuracy", round(sum(accs_b)/len(accs_b), 4),
                       {"ci_95": [round(lo, 4), round(hi, 4)]})
        if accs_c:
            lo, hi = bootstrap_ci(accs_c)
            agg.record("C_pyquifer", "accuracy", round(sum(accs_c)/len(accs_c), 4),
                       {"ci_95": [round(lo, 4), round(hi, 4)]})
        suite.add(agg)

    # CIFAR-10 (capped dataset size — EP single-sample updates are expensive)
    print("\n--- CIFAR-10 Classification ---", flush=True)
    mc_cifar = bench_cifar10(num_epochs=num_epochs, num_osc=64, ep_steps=15,
                             max_train=10000, max_test=2000)
    suite.add(mc_cifar)

    # XOR
    print("\n--- XOR Sanity Check ---", flush=True)
    mc_xor = bench_xor()
    suite.add(mc_xor)

    # Convergence
    print("\n--- Convergence Speed ---", flush=True)
    mc_conv = bench_convergence(num_epochs=num_epochs, ep_steps=15,
                                max_train=10000, max_test=2000)
    suite.add(mc_conv)

    # Beta sweep
    print("\n--- Beta Sensitivity ---", flush=True)
    mc_beta = bench_beta_sweep(num_epochs=3, ep_steps=15,
                               max_train=10000, max_test=2000)
    suite.add(mc_beta)

    # Save results
    json_path = str(results_dir / "ep_training.json")
    suite.to_json(json_path)
    print(f"\nResults saved to {json_path}")
    print("\n" + suite.to_markdown())


# ---------------------------------------------------------------------------
# Pytest Classes (smoke test mode)
# ---------------------------------------------------------------------------

class TestEPClassification:
    """Smoke tests — small subsets, few epochs, just verify no crashes."""

    def test_mnist_mlp_trains(self):
        device = get_device()
        set_seed(0)
        loader = load_mnist(train=True, batch_size=64, max_samples=256)
        model = PlainMLP().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss = train_mlp_epoch(model, loader, opt, device)
        assert loss < 5.0, f"MLP loss too high: {loss}"

    def test_mnist_ep_trains(self):
        device = get_device()
        set_seed(0)
        from pyquifer.equilibrium_propagation import EPKuramotoClassifier
        model = EPKuramotoClassifier(
            input_dim=784, num_classes=10, num_oscillators=16,
            free_steps=5, nudge_steps=5,
        ).to(device)
        opt = torch.optim.Adam(
            list(model.encoder.parameters()) + list(model.readout.parameters()),
            lr=1e-3,
        )
        loader = load_mnist(train=True, batch_size=32, max_samples=64)
        loss = train_ep_epoch(model, loader, opt, device, max_samples=32)
        assert loss < 10.0, f"EP loss too high: {loss}"

    def test_classification_scenario_runs(self):
        mc = bench_classification(
            dataset="mnist", num_epochs=1, seed=0,
            max_train=128, max_test=64,
            num_osc=16, ep_steps=5,
        )
        assert len(mc.results) >= 3
        table = mc.to_markdown_table()
        assert "accuracy" in table


class TestEPCIFAR10:
    def test_cifar10_scenario_runs(self):
        mc = bench_cifar10(
            num_epochs=1, seed=0,
            max_train=128, max_test=64,
            num_osc=16, ep_steps=5,
        )
        assert len(mc.results) >= 3
        columns = {r.column for r in mc.results}
        assert "A_published" in columns
        assert "B_pytorch" in columns
        assert "C_pyquifer" in columns

    def test_plain_cnn_forward(self):
        device = get_device()
        cnn = PlainCNN().to(device)
        x = torch.randn(2, 3, 32, 32, device=device)
        out = cnn(x)
        assert out.shape == (2, 10)


class TestEPXOR:
    def test_xor_scenario_runs(self):
        mc = bench_xor(num_epochs=50, seed=0)
        assert len(mc.results) >= 3
        columns = {r.column for r in mc.results}
        assert "A_published" in columns
        assert "B_pytorch" in columns
        assert "C_pyquifer" in columns


class TestEPConvergence:
    def test_convergence_runs(self):
        mc = bench_convergence(seed=0, max_train=128, max_test=64,
                               num_epochs=2, ep_steps=5)
        assert len(mc.results) >= 1


class TestEPBetaSweep:
    def test_beta_sweep_runs(self):
        mc = bench_beta_sweep(betas=[0.05, 0.2], seed=0,
                              max_train=128, max_test=64,
                              num_epochs=1, ep_steps=5)
        assert len(mc.results) >= 1

    def test_beta_sweep_includes_sparsity(self):
        mc = bench_beta_sweep(betas=[0.1], seed=0,
                              max_train=64, max_test=32,
                              num_epochs=1, ep_steps=5)
        metric_keys = set()
        for r in mc.results:
            metric_keys.update(r.metrics.keys())
        assert any("sparsity" in k for k in metric_keys)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_full_suite()
