"""Shared benchmark framework for PyQuifer three-column evaluation.

Provides timing, memory tracking, metric collection, dataset loading,
statistical utilities, and report generation. No PyQuifer imports —
pure infrastructure reused by all benchmark scripts.

Dual-mode support:
  pytest: import utilities directly
  CLI:    `from harness import *`
"""
from __future__ import annotations

import json
import math
import os
import platform
import random
import subprocess
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# ---------------------------------------------------------------------------
# Configurable Seeds
# ---------------------------------------------------------------------------

DEFAULT_SEEDS = 5


def get_num_seeds() -> int:
    """Return number of seeds: PYQUIFER_BENCH_SEEDS env var or DEFAULT_SEEDS."""
    return int(os.environ.get("PYQUIFER_BENCH_SEEDS", DEFAULT_SEEDS))

# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

@contextmanager
def timer():
    """Context manager that yields a dict with ``elapsed_ms`` on exit."""
    result: Dict[str, float] = {"elapsed_ms": 0.0}
    start = time.perf_counter()
    yield result
    result["elapsed_ms"] = (time.perf_counter() - start) * 1000.0


class MemoryTracker:
    """Track peak GPU (if available) or CPU memory delta."""

    def __init__(self):
        self.has_cuda = torch.cuda.is_available()

    def __enter__(self):
        if self.has_cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            self._start = torch.cuda.memory_allocated()
        else:
            self._start = 0
        return self

    def __exit__(self, *exc):
        if self.has_cuda:
            torch.cuda.synchronize()
            self.peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            self.delta_mb = (torch.cuda.memory_allocated() - self._start) / (1024 ** 2)
        else:
            self.peak_mb = 0.0
            self.delta_mb = 0.0

    @property
    def summary(self) -> Dict[str, float]:
        return {"peak_mb": round(self.peak_mb, 2),
                "delta_mb": round(self.delta_mb, 2)}


# ---------------------------------------------------------------------------
# Metric Collection
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """One row of results for a single column in a scenario."""
    name: str
    column: str          # "A_published", "B_pytorch", "C_pyquifer", "C_rand"
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricCollector:
    """Collects BenchmarkResults across columns for a scenario."""

    def __init__(self, scenario_name: str):
        self.scenario_name = scenario_name
        self.results: List[BenchmarkResult] = []

    def record(self, column: str, metric: str, value: float,
               metadata: Optional[Dict[str, Any]] = None):
        # Find or create result for this column
        for r in self.results:
            if r.column == column:
                r.metrics[metric] = value
                if metadata:
                    r.metadata.update(metadata)
                return
        self.results.append(BenchmarkResult(
            name=self.scenario_name, column=column,
            metrics={metric: value},
            metadata=metadata or {}
        ))

    def add_result(self, result: BenchmarkResult):
        self.results.append(result)

    def to_dict(self) -> Dict:
        return {
            "scenario": self.scenario_name,
            "results": [asdict(r) for r in self.results],
        }

    def to_json(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    def to_markdown_table(self) -> str:
        """Three-column comparison table."""
        if not self.results:
            return ""
        all_metrics = sorted({m for r in self.results for m in r.metrics})
        columns = sorted({r.column for r in self.results})

        header = f"### {self.scenario_name}\n\n"
        header += "| Metric |"
        for c in columns:
            label = c.replace("A_published", "Published").replace(
                "B_pytorch", "PyTorch Backprop").replace(
                "C_pyquifer", "PyQuifer").replace(
                "C_rand", "Random Modulation")
            header += f" {label} |"
        header += "\n|" + "---|" * (len(columns) + 1) + "\n"

        rows = []
        for m in all_metrics:
            row = f"| {m} |"
            for c in columns:
                val = None
                for r in self.results:
                    if r.column == c and m in r.metrics:
                        val = r.metrics[m]
                if val is not None:
                    row += f" {_fmt(val)} |"
                else:
                    row += " — |"
            rows.append(row)
        return header + "\n".join(rows)

    def compute_ratio(self, metric: str) -> Dict[str, Any]:
        """Compute C/B ratio for a metric."""
        b_val, c_val = None, None
        for r in self.results:
            if r.column == "B_pytorch" and metric in r.metrics:
                b_val = r.metrics[metric]
            elif r.column == "C_pyquifer" and metric in r.metrics:
                c_val = r.metrics[metric]
        if b_val is None or c_val is None or b_val == 0:
            return {"ratio": None}
        ratio = c_val / b_val
        return {"ratio": round(ratio, 4), "C": c_val, "B": b_val}


def _fmt(v) -> str:
    if isinstance(v, str):
        return v
    if abs(v) >= 100:
        return f"{v:.1f}"
    if abs(v) >= 1:
        return f"{v:.2f}"
    return f"{v:.4f}"


class BenchmarkSuite:
    """Aggregates multiple MetricCollectors into a single report."""

    def __init__(self, suite_name: str):
        self.suite_name = suite_name
        self.collectors: List[MetricCollector] = []

    def add(self, collector: MetricCollector):
        self.collectors.append(collector)

    def to_json(self, path: str):
        data = {
            "suite": self.suite_name,
            "scenarios": [c.to_dict() for c in self.collectors],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def to_markdown(self) -> str:
        lines = [f"# {self.suite_name}\n"]
        for c in self.collectors:
            lines.append(c.to_markdown_table())
            lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dataset Utilities
# ---------------------------------------------------------------------------

def _get_data_root() -> str:
    return str(Path(__file__).resolve().parent / "data")


def load_mnist(train: bool = True, batch_size: int = 128,
               max_samples: int = 0) -> DataLoader:
    from torchvision import datasets, transforms
    ds = datasets.MNIST(_get_data_root(), train=train, download=True,
                        transform=transforms.ToTensor())
    if max_samples > 0:
        ds = Subset(ds, list(range(min(max_samples, len(ds)))))
    return DataLoader(ds, batch_size=batch_size, shuffle=train)


def load_fashion_mnist(train: bool = True, batch_size: int = 128,
                       max_samples: int = 0) -> DataLoader:
    from torchvision import datasets, transforms
    ds = datasets.FashionMNIST(_get_data_root(), train=train, download=True,
                               transform=transforms.ToTensor())
    if max_samples > 0:
        ds = Subset(ds, list(range(min(max_samples, len(ds)))))
    return DataLoader(ds, batch_size=batch_size, shuffle=train)


def load_cifar10(train: bool = True, batch_size: int = 128,
                 max_samples: int = 0) -> DataLoader:
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    ds = datasets.CIFAR10(_get_data_root(), train=train, download=True,
                          transform=transform)
    if max_samples > 0:
        ds = Subset(ds, list(range(min(max_samples, len(ds)))))
    return DataLoader(ds, batch_size=batch_size, shuffle=train)


def split_mnist(num_tasks: int = 5, batch_size: int = 128,
                max_samples_per_task: int = 0) -> List[Tuple[DataLoader, DataLoader]]:
    """Split MNIST into sequential tasks (2 digits per task)."""
    from torchvision import datasets, transforms
    train_ds = datasets.MNIST(_get_data_root(), train=True, download=True,
                              transform=transforms.ToTensor())
    test_ds = datasets.MNIST(_get_data_root(), train=False, download=True,
                             transform=transforms.ToTensor())
    tasks = []
    digits_per_task = 10 // num_tasks
    for t in range(num_tasks):
        task_digits = list(range(t * digits_per_task, (t + 1) * digits_per_task))
        train_idx = [i for i, (_, y) in enumerate(train_ds)
                     if y in task_digits]
        test_idx = [i for i, (_, y) in enumerate(test_ds)
                    if y in task_digits]
        if max_samples_per_task > 0:
            train_idx = train_idx[:max_samples_per_task]
            test_idx = test_idx[:max_samples_per_task]
        train_loader = DataLoader(Subset(train_ds, train_idx),
                                  batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(Subset(test_ds, test_idx),
                                 batch_size=batch_size, shuffle=False)
        tasks.append((train_loader, test_loader))
    return tasks


def split_fashion_mnist(num_tasks: int = 5, batch_size: int = 128,
                        max_samples_per_task: int = 0) -> List[Tuple[DataLoader, DataLoader]]:
    """Split Fashion-MNIST into sequential tasks (2 classes per task)."""
    from torchvision import datasets, transforms
    train_ds = datasets.FashionMNIST(_get_data_root(), train=True, download=True,
                                     transform=transforms.ToTensor())
    test_ds = datasets.FashionMNIST(_get_data_root(), train=False, download=True,
                                    transform=transforms.ToTensor())
    tasks = []
    classes_per_task = 10 // num_tasks
    for t in range(num_tasks):
        task_classes = list(range(t * classes_per_task, (t + 1) * classes_per_task))
        train_idx = [i for i, (_, y) in enumerate(train_ds)
                     if y in task_classes]
        test_idx = [i for i, (_, y) in enumerate(test_ds)
                    if y in task_classes]
        if max_samples_per_task > 0:
            train_idx = train_idx[:max_samples_per_task]
            test_idx = test_idx[:max_samples_per_task]
        train_loader = DataLoader(Subset(train_ds, train_idx),
                                  batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(Subset(test_ds, test_idx),
                                 batch_size=batch_size, shuffle=False)
        tasks.append((train_loader, test_loader))
    return tasks


def split_cifar10(num_tasks: int = 5, batch_size: int = 128,
                  max_samples_per_task: int = 0) -> List[Tuple[DataLoader, DataLoader]]:
    """Split CIFAR-10 into sequential tasks (2 classes per task)."""
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_ds = datasets.CIFAR10(_get_data_root(), train=True, download=True,
                                 transform=transform)
    test_ds = datasets.CIFAR10(_get_data_root(), train=False, download=True,
                                transform=transform)
    tasks = []
    classes_per_task = 10 // num_tasks
    for t in range(num_tasks):
        task_classes = list(range(t * classes_per_task, (t + 1) * classes_per_task))
        train_idx = [i for i, (_, y) in enumerate(train_ds)
                     if y in task_classes]
        test_idx = [i for i, (_, y) in enumerate(test_ds)
                    if y in task_classes]
        if max_samples_per_task > 0:
            train_idx = train_idx[:max_samples_per_task]
            test_idx = test_idx[:max_samples_per_task]
        train_loader = DataLoader(Subset(train_ds, train_idx),
                                  batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(Subset(test_ds, test_idx),
                                 batch_size=batch_size, shuffle=False)
        tasks.append((train_loader, test_loader))
    return tasks


def permuted_mnist(num_tasks: int = 10, batch_size: int = 128,
                   max_samples: int = 0) -> List[Tuple[DataLoader, DataLoader]]:
    """Generate Permuted-MNIST tasks with fixed permutation seeds.

    Each task uses the full 10 classes with a unique pixel permutation.
    Task 0 = original MNIST (identity permutation).
    """
    from torchvision import datasets, transforms

    train_ds = datasets.MNIST(_get_data_root(), train=True, download=True,
                               transform=transforms.ToTensor())
    test_ds = datasets.MNIST(_get_data_root(), train=False, download=True,
                              transform=transforms.ToTensor())
    tasks = []
    for t in range(num_tasks):
        rng = np.random.RandomState(seed=t * 1000)
        perm = rng.permutation(784) if t > 0 else np.arange(784)
        perm_tensor = torch.from_numpy(perm).long()

        class PermutedSubset(torch.utils.data.Dataset):
            def __init__(self, base_ds, perm_t, max_n):
                self.base_ds = base_ds
                self.perm_t = perm_t
                self.length = min(max_n, len(base_ds)) if max_n > 0 else len(base_ds)

            def __len__(self):
                return self.length

            def __getitem__(self, idx):
                x, y = self.base_ds[idx]
                x_flat = x.view(-1)[self.perm_t].view(1, 28, 28)
                return x_flat, y

        train_loader = DataLoader(
            PermutedSubset(train_ds, perm_tensor, max_samples),
            batch_size=batch_size, shuffle=True,
        )
        test_loader = DataLoader(
            PermutedSubset(test_ds, perm_tensor, max_samples),
            batch_size=batch_size, shuffle=False,
        )
        tasks.append((train_loader, test_loader))
    return tasks


# ---------------------------------------------------------------------------
# Statistical Utilities
# ---------------------------------------------------------------------------

def bootstrap_ci(values: List[float], confidence: float = 0.95,
                 n_bootstrap: int = 1000) -> Tuple[float, float]:
    """Compute bootstrap confidence interval."""
    if len(values) < 2:
        v = values[0] if values else 0.0
        return (v, v)
    arr = np.array(values)
    means = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(np.mean(sample))
    means.sort()
    lo = (1 - confidence) / 2
    hi = 1 - lo
    return (float(np.percentile(means, lo * 100)),
            float(np.percentile(means, hi * 100)))


def compute_accuracy(model: torch.nn.Module, dataloader: DataLoader,
                     device: torch.device) -> float:
    """Standard classification accuracy."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            x = x.view(x.size(0), -1)
            out = model(x)
            if isinstance(out, dict):
                out = out.get("final_output", out.get("output", out.get("logits")))
            pred = out.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Cohen's d
# ---------------------------------------------------------------------------

def cohens_d(group1: List[float], group2: List[float]) -> float:
    """Compute Cohen's d effect size (pooled SD) between two groups."""
    a1, a2 = np.array(group1, dtype=np.float64), np.array(group2, dtype=np.float64)
    n1, n2 = len(a1), len(a2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = np.var(a1, ddof=1), np.var(a2, ddof=1)
    pooled_sd = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return 0.0
    return float((np.mean(a1) - np.mean(a2)) / pooled_sd)


# ---------------------------------------------------------------------------
# Gate Checks
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    """Binary pass/fail gate check result."""
    name: str
    passed: bool
    detail: str = ""


def check_gates(
    metrics: Dict[str, float],
    nan_inf_keys: Optional[List[str]] = None,
    correctness_key: Optional[str] = None,
    correctness_tol: float = 0.01,
    correctness_ref: float = 0.0,
    determinism_keys: Optional[List[str]] = None,
    determinism_envelope: float = 0.05,
    determinism_ref: Optional[Dict[str, float]] = None,
    budget_key: Optional[str] = None,
    budget_max: float = float("inf"),
) -> List[GateResult]:
    """Check binary pass/fail gates on benchmark metrics.

    Args:
        metrics: Dict of metric_name -> value.
        nan_inf_keys: Keys to check for NaN/Inf.
        correctness_key: Metric key to check within tolerance of correctness_ref.
        correctness_tol: Absolute tolerance for correctness gate.
        correctness_ref: Reference value for correctness comparison.
        determinism_keys: Keys to check are within envelope of determinism_ref.
        determinism_envelope: Max relative deviation allowed.
        determinism_ref: Reference values (from first seed run).
        budget_key: Metric key that must be <= budget_max.
        budget_max: Maximum allowed value for budget_key.

    Returns:
        List of GateResult, one per check.
    """
    results: List[GateResult] = []

    # NaN/Inf gate
    if nan_inf_keys:
        for k in nan_inf_keys:
            v = metrics.get(k, 0.0)
            ok = math.isfinite(v)
            results.append(GateResult(
                name=f"nan_inf_{k}", passed=ok,
                detail=f"{k}={v}" if not ok else "",
            ))

    # Correctness gate
    if correctness_key and correctness_key in metrics:
        v = metrics[correctness_key]
        ok = abs(v - correctness_ref) <= correctness_tol
        results.append(GateResult(
            name=f"correctness_{correctness_key}", passed=ok,
            detail=f"{correctness_key}={v:.4f} vs ref={correctness_ref:.4f} tol={correctness_tol}",
        ))

    # Determinism gate
    if determinism_keys and determinism_ref:
        for k in determinism_keys:
            v = metrics.get(k)
            ref = determinism_ref.get(k)
            if v is not None and ref is not None and ref != 0:
                dev = abs(v - ref) / abs(ref)
                ok = dev <= determinism_envelope
                results.append(GateResult(
                    name=f"determinism_{k}", passed=ok,
                    detail=f"deviation={dev:.4f}" if not ok else "",
                ))

    # Budget gate
    if budget_key and budget_key in metrics:
        v = metrics[budget_key]
        ok = v <= budget_max
        results.append(GateResult(
            name=f"budget_{budget_key}", passed=ok,
            detail=f"{budget_key}={v:.2f} > max={budget_max:.2f}" if not ok else "",
        ))

    return results


# ---------------------------------------------------------------------------
# Weighted Scoring
# ---------------------------------------------------------------------------

def compute_scenario_score(
    accuracy: float,
    speed_ratio: float = 1.0,
    latency_ratio: float = 1.0,
    memory_ratio: float = 1.0,
    gates: Optional[List[GateResult]] = None,
) -> float:
    """Compute weighted scenario score per the benchmark plan formula.

    Score = Gate * (acc)^0.4 * (speed)^0.2 * (lat)^0.2 * (mem)^0.2

    Args:
        accuracy: Classification accuracy [0, 1].
        speed_ratio: C_steps_per_sec / B_steps_per_sec (higher is better).
        latency_ratio: B_latency / C_latency (higher is better).
        memory_ratio: B_peak_mb / C_peak_mb (higher is better).
        gates: List of GateResults; if any fail, score is 0.

    Returns:
        Weighted score (0.0 if any gate fails).
    """
    if gates:
        if not all(g.passed for g in gates):
            return 0.0

    # Clamp ratios to avoid negative/zero issues in power
    def _clamp(v: float) -> float:
        return max(v, 1e-6)

    return (
        _clamp(accuracy) ** 0.4
        * _clamp(speed_ratio) ** 0.2
        * _clamp(latency_ratio) ** 0.2
        * _clamp(memory_ratio) ** 0.2
    )


# ---------------------------------------------------------------------------
# Environment Block
# ---------------------------------------------------------------------------

def get_environment_block() -> str:
    """Return a markdown block with environment info for reports."""
    lines = ["## Environment\n"]

    # Git commit
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[2]),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        lines.append(f"- **Git commit:** {git_hash}")
    except Exception:
        lines.append("- **Git commit:** unknown")

    # PyTorch
    lines.append(f"- **PyTorch:** {torch.__version__}")

    # CUDA / GPU
    if torch.cuda.is_available():
        lines.append(f"- **GPU:** {torch.cuda.get_device_name(0)}")
        lines.append(f"- **CUDA:** {torch.version.cuda}")
    else:
        lines.append("- **GPU:** None (CPU only)")

    # CPU / OS
    lines.append(f"- **CPU:** {platform.processor() or platform.machine()}")
    lines.append(f"- **OS:** {platform.system()} {platform.release()}")
    lines.append(f"- **Python:** {platform.python_version()}")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def generate_report(results_dir: str, output_path: str):
    """Collect all JSON results and produce a unified markdown report."""
    results_path = Path(results_dir)
    output = Path(output_path)
    json_files = sorted(results_path.glob("*.json"))

    lines = ["# PyQuifer Comprehensive Benchmark Report\n",
             f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
             get_environment_block()]

    scores = []
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        suite_name = data.get("suite", jf.stem)
        lines.append(f"\n## {suite_name}\n")
        for scenario in data.get("scenarios", []):
            mc = MetricCollector(scenario["scenario"])
            for r in scenario.get("results", []):
                mc.add_result(BenchmarkResult(**r))
            lines.append(mc.to_markdown_table())
            lines.append("")
            # Compute C/B accuracy ratio if available
            ratio = mc.compute_ratio("accuracy")
            if ratio.get("ratio") is not None:
                scores.append(ratio["ratio"])

    if scores:
        geo_mean = math.exp(sum(math.log(s) for s in scores) / len(scores))
        lines.append(f"\n## Aggregate Score\n")
        lines.append(f"Geometric mean of C/B accuracy ratios: **{geo_mean:.4f}**\n")
        lines.append(f"Number of scenarios: {len(scores)}\n")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return str(output)
