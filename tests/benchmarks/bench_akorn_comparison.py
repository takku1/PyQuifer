"""
AKOrN Benchmark: PyQuifer vs Official (ICLR 2025)

Fair comparison on a synthetic binding task — oscillatory phase grouping.
Both implementations receive tokens belonging to K latent groups and must
learn to synchronize in-group phases while desynchronizing across groups.

Metrics:
  1. Binding Accuracy: % of token-pairs correctly grouped by phase clustering
  2. Phase Separation: mean inter-group phase distance / mean intra-group distance
  3. Convergence Speed: steps to reach 90% binding accuracy
  4. Throughput: tokens/sec at inference
  5. Energy landscape: monotonic decrease during dynamics (stability)

The task is NEUTRAL — not designed to favor either approach.
"""

import sys
import time
import math
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- PyQuifer AKOrN ---
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pyquifer.cognition.binding.visual import AKOrNLayer, AKOrNBlock

# --- Official AKOrN ---
OFFICIAL_ROOT = Path(__file__).resolve().parents[3] / "nom_nom" / "akorn"
sys.path.insert(0, str(OFFICIAL_ROOT))
from source.layers.klayer import KLayer
from source.layers.kutils import normalize as k_normalize, reshape, reshape_back
from source.layers.common_layers import ReadOutConv


# ============================================================
# Synthetic binding dataset
# ============================================================

def make_binding_task(
    batch_size: int = 32,
    num_tokens: int = 16,
    feature_dim: int = 64,
    num_groups: int = 4,
    noise_std: float = 0.3,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create a synthetic binding task.

    Each token belongs to one of `num_groups` clusters. In-group tokens share
    a latent prototype with additive noise. The task: learn to bind (synchronize)
    tokens that share a group and separate those that don't.

    Returns:
        features: (B, N, D) token features
        labels:   (B, N) integer group labels in [0, num_groups)
    """
    # Random prototypes per group — well-separated in feature space
    prototypes = torch.randn(num_groups, feature_dim, device=device)
    prototypes = F.normalize(prototypes, dim=-1) * 2.0  # unit-sphere scaled

    # Assign tokens to groups uniformly
    labels = torch.randint(0, num_groups, (batch_size, num_tokens), device=device)

    # Build features = prototype[label] + noise
    features = prototypes[labels] + noise_std * torch.randn(
        batch_size, num_tokens, feature_dim, device=device
    )

    return features, labels


# ============================================================
# Wrappers to make both models comparable
# ============================================================

class PyQuiferAKOrN(nn.Module):
    """Our AKOrN: vector-valued Kuramoto on hyperspheres with attention coupling."""

    def __init__(self, dim: int, n: int = 4, num_heads: int = 4,
                 num_steps: int = 10, gamma: float = 1.0):
        super().__init__()
        self.layer = AKOrNLayer(
            dim=dim,
            n=n,
            num_heads=num_heads,
            gamma=gamma,
            num_steps=num_steps,
            coupling_mode="attention",
        )
        self.name = "PyQuifer (vector Kuramoto)"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, N, D) features after oscillatory binding."""
        return self.layer(x)


class OfficialAKOrN(nn.Module):
    """Official AKOrN: vector-valued Kuramoto on hyperspheres.

    Adapted to work with (B, N, D) token inputs instead of (B, C, H, W) images.
    We use J='attn' mode (attention connectivity) for fair comparison.
    """

    def __init__(self, dim: int, n: int = 4, num_steps: int = 10,
                 gamma: float = 1.0, heads: int = 8):
        super().__init__()
        self.dim = dim
        self.n = n
        self.T = num_steps
        self.gamma = gamma

        # Official KLayer expects (B, C, ...) spatial format.
        # For token inputs we'll reshape to (B, C, N, 1) pseudo-image.
        self.klayer = KLayer(
            n=n, ch=dim, J="attn", c_norm="gn",
            use_omega=False, heads=heads, apply_proj=True,
        )
        # Readout: phase-invariant norm mapping
        self.readout = ReadOutConv(dim, dim, n)

        # Conditioning projection (features -> bias c)
        self.cond_proj = nn.Linear(dim, dim)

        self.name = "Official (vector Kuramoto)"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, D) -> (B, N, D) after oscillatory binding."""
        B, N, D = x.shape

        # Reshape tokens to pseudo-image: (B, D, N, 1)
        x_img = x.permute(0, 2, 1).unsqueeze(-1)  # (B, D, N, 1)

        # Conditioning bias from features
        c = self.cond_proj(x)  # (B, N, D)
        c_img = c.permute(0, 2, 1).unsqueeze(-1)  # (B, D, N, 1)

        # Random initial state (key design choice of official AKOrN)
        z = torch.randn_like(x_img)
        z = k_normalize(z, self.n)

        # Run Kuramoto dynamics
        xs, es = self.klayer(z, c_img, T=self.T, gamma=self.gamma)
        z_final = xs[-1]  # (B, D, N, 1)

        # Phase-invariant readout
        out = self.readout(z_final)  # (B, D, N, 1)
        out = out.squeeze(-1).permute(0, 2, 1)  # (B, N, D)

        return out

    def get_phases(self, x: torch.Tensor) -> torch.Tensor:
        """Returns phase states (B, ch//n, n, N) for analysis."""
        B, N, D = x.shape
        x_img = x.permute(0, 2, 1).unsqueeze(-1)
        c = self.cond_proj(x).permute(0, 2, 1).unsqueeze(-1)
        z = torch.randn_like(x_img)
        z = k_normalize(z, self.n)
        xs, _ = self.klayer(z, c, T=self.T, gamma=self.gamma)
        z_final = xs[-1].squeeze(-1)  # (B, D, N)
        z_reshaped = z_final.unflatten(1, (-1, self.n))  # (B, D//n, n, N)
        return z_reshaped


# ============================================================
# Metrics
# ============================================================

def phase_clustering_accuracy(
    features: torch.Tensor, labels: torch.Tensor, model: nn.Module
) -> float:
    """Measure how well phase similarity predicts group membership.

    For each pair of tokens, check if same-group pairs have higher feature
    similarity than cross-group pairs after binding. Reports AUC-like accuracy.
    """
    with torch.no_grad():
        out = model(features)  # (B, N, D)

    B, N, D = out.shape

    # Cosine similarity between all token pairs
    out_norm = F.normalize(out, dim=-1)
    sim = torch.bmm(out_norm, out_norm.transpose(1, 2))  # (B, N, N)

    # Ground truth: same group = 1, different = 0
    same_group = (labels.unsqueeze(1) == labels.unsqueeze(2)).float()

    # Mask diagonal (self-similarity always 1)
    mask = ~torch.eye(N, dtype=torch.bool, device=features.device).unsqueeze(0)
    mask = mask.expand(B, -1, -1)
    sim = sim[mask].reshape(B, -1)
    same_group = same_group[mask].reshape(B, -1)

    # For each batch: accuracy = fraction of pairs where
    # sim > median correctly predicts same-group
    accs = []
    for b in range(B):
        s = sim[b]
        g = same_group[b]
        threshold = s.median()
        pred = (s > threshold).float()
        acc = (pred == g).float().mean().item()
        accs.append(acc)

    return float(np.mean(accs))


def phase_separation_ratio(
    features: torch.Tensor, labels: torch.Tensor, model: nn.Module
) -> float:
    """Ratio of inter-group distance to intra-group distance in output space.

    Higher = better separation. A random model scores ~1.0.
    """
    with torch.no_grad():
        out = model(features)

    B, N, D = out.shape
    out_norm = F.normalize(out, dim=-1)

    ratios = []
    for b in range(B):
        intra_dists = []
        inter_dists = []
        for i in range(N):
            for j in range(i + 1, N):
                d = 1.0 - (out_norm[b, i] * out_norm[b, j]).sum().item()
                if labels[b, i] == labels[b, j]:
                    intra_dists.append(d)
                else:
                    inter_dists.append(d)

        if intra_dists and inter_dists:
            intra = np.mean(intra_dists)
            inter = np.mean(inter_dists)
            if intra > 1e-8:
                ratios.append(inter / intra)

    return float(np.mean(ratios)) if ratios else 0.0


def measure_throughput(
    model: nn.Module, dim: int, num_tokens: int = 16,
    batch_size: int = 32, num_iters: int = 50, device: str = "cpu",
) -> float:
    """Tokens per second at inference."""
    model.eval()
    x = torch.randn(batch_size, num_tokens, dim, device=device)

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            model(x)
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(num_iters):
        with torch.no_grad():
            model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    total_tokens = batch_size * num_tokens * num_iters
    return total_tokens / elapsed


def energy_monotonicity(
    features: torch.Tensor, model: nn.Module
) -> Dict[str, float]:
    """Check if energy decreases monotonically during dynamics.

    A well-behaved oscillator system should have non-increasing energy.
    Returns fraction of steps where energy decreased and total energy drop.
    """
    # Only works for official model which tracks energy
    if isinstance(model, OfficialAKOrN):
        B, N, D = features.shape
        x_img = features.permute(0, 2, 1).unsqueeze(-1)
        c = model.cond_proj(features).permute(0, 2, 1).unsqueeze(-1)
        z = torch.randn_like(x_img)
        z = k_normalize(z, model.n)
        with torch.no_grad():
            _, es = model.klayer(z, c, T=model.T, gamma=model.gamma)
        es = torch.stack(es)  # (T+1, B)
        diffs = es[1:] - es[:-1]  # (T, B)
        frac_decreasing = (diffs <= 0).float().mean().item()
        total_drop = (es[0] - es[-1]).mean().item()
        return {"frac_decreasing": frac_decreasing, "total_drop": total_drop}

    # For PyQuifer — track order parameter R instead (should increase)
    elif isinstance(model, PyQuiferAKOrN):
        B, N, D = features.shape
        layer = model.layer
        # Replicate forward to track R over steps
        c = layer.cond_proj(features)
        c = c.transpose(1, 2)
        c = layer.c_norm(c)
        c = c.transpose(1, 2)

        x_state = torch.randn(B, N, D, device=features.device)
        x_g = x_state.view(B, N, layer.num_groups, layer.n)
        x_g = F.normalize(x_g, dim=-1)
        x_state = x_g.view(B, N, D)

        Rs = []
        with torch.no_grad():
            Rs.append(layer.order_parameter(x_state).mean().item())
            for _ in range(layer.num_steps):
                x_state, _ = layer._kuramoto_step(x_state, c, features)
                Rs.append(layer.order_parameter(x_state).mean().item())
        Rs = np.array(Rs)
        diffs = Rs[1:] - Rs[:-1]
        frac_increasing = (diffs >= 0).mean()
        return {"frac_increasing": float(frac_increasing), "R_final": float(Rs[-1])}

    return {}


# ============================================================
# Training loop for binding task
# ============================================================

def train_binding(
    model: nn.Module,
    num_epochs: int = 50,
    batch_size: int = 32,
    num_tokens: int = 16,
    feature_dim: int = 64,
    num_groups: int = 4,
    lr: float = 1e-3,
    device: str = "cpu",
) -> List[Dict]:
    """Train model on contrastive binding loss.

    Loss: push same-group features together, different-group features apart.
    Uses NT-Xent style contrastive loss on the output representations.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    for epoch in range(num_epochs):
        model.train()
        features, labels = make_binding_task(
            batch_size, num_tokens, feature_dim, num_groups, device=device
        )

        out = model(features)  # (B, N, D)
        out_norm = F.normalize(out, dim=-1)

        # Contrastive loss: same-group pairs should have high similarity
        sim = torch.bmm(out_norm, out_norm.transpose(1, 2))  # (B, N, N)
        same_group = (labels.unsqueeze(1) == labels.unsqueeze(2)).float()

        # Mask diagonal
        mask = ~torch.eye(num_tokens, dtype=torch.bool, device=device).unsqueeze(0)
        mask = mask.expand(batch_size, -1, -1)
        sim_masked = sim[mask].reshape(batch_size, -1)
        target = same_group[mask].reshape(batch_size, -1)

        # Binary cross-entropy on similarity vs group membership
        loss = F.binary_cross_entropy_with_logits(
            sim_masked * 5.0,  # temperature scaling
            target,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Evaluate every 5 epochs
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            model.eval()
            eval_features, eval_labels = make_binding_task(
                64, num_tokens, feature_dim, num_groups, device=device
            )
            acc = phase_clustering_accuracy(eval_features, eval_labels, model)
            sep = phase_separation_ratio(eval_features, eval_labels, model)
            record = {
                "epoch": epoch,
                "loss": loss.item(),
                "binding_accuracy": acc,
                "separation_ratio": sep,
            }
            history.append(record)
            print(f"  [{model.name}] epoch {epoch:3d}  "
                  f"loss={loss.item():.4f}  acc={acc:.3f}  sep={sep:.2f}")

    return history


# ============================================================
# Main benchmark
# ============================================================

@dataclass
class BenchmarkResult:
    model_name: str = ""
    param_count: int = 0
    final_binding_accuracy: float = 0.0
    final_separation_ratio: float = 0.0
    best_binding_accuracy: float = 0.0
    epochs_to_90pct: int = -1  # -1 = never reached
    throughput_tokens_per_sec: float = 0.0
    energy_stability: Dict = field(default_factory=dict)
    training_history: List[Dict] = field(default_factory=list)


def run_benchmark(
    feature_dim: int = 64,
    num_tokens: int = 16,
    num_groups: int = 4,
    num_epochs: int = 80,
    num_steps: int = 10,
    device: str = "cpu",
) -> List[BenchmarkResult]:
    """Run full benchmark comparing both AKOrN implementations."""

    print("=" * 70)
    print("AKOrN BENCHMARK: PyQuifer vs Official (ICLR 2025)")
    print("=" * 70)
    print(f"Config: dim={feature_dim}, tokens={num_tokens}, groups={num_groups}, "
          f"steps={num_steps}, epochs={num_epochs}, device={device}")
    print()

    results = []

    # --- Model 1: PyQuifer (ours) ---
    print("-" * 50)
    print("Training PyQuifer AKOrN (vector Kuramoto on hyperspheres)...")
    print("-" * 50)
    pyq_model = PyQuiferAKOrN(
        dim=feature_dim, n=4, num_heads=4,
        num_steps=num_steps, gamma=1.0,
    )
    pyq_params = sum(p.numel() for p in pyq_model.parameters())
    print(f"  Parameters: {pyq_params:,}")

    pyq_history = train_binding(
        pyq_model, num_epochs=num_epochs, feature_dim=feature_dim,
        num_tokens=num_tokens, num_groups=num_groups, device=device,
    )

    # Evaluate
    pyq_model.eval()
    eval_feat, eval_lab = make_binding_task(128, num_tokens, feature_dim, num_groups, device=device)
    pyq_acc = phase_clustering_accuracy(eval_feat, eval_lab, pyq_model)
    pyq_sep = phase_separation_ratio(eval_feat, eval_lab, pyq_model)
    pyq_throughput = measure_throughput(pyq_model, feature_dim, num_tokens, device=device)
    pyq_energy = energy_monotonicity(eval_feat[:8], pyq_model)

    best_acc = max(h["binding_accuracy"] for h in pyq_history)
    epochs_90 = next((h["epoch"] for h in pyq_history if h["binding_accuracy"] >= 0.9), -1)

    results.append(BenchmarkResult(
        model_name="PyQuifer (vector Kuramoto)",
        param_count=pyq_params,
        final_binding_accuracy=pyq_acc,
        final_separation_ratio=pyq_sep,
        best_binding_accuracy=best_acc,
        epochs_to_90pct=epochs_90,
        throughput_tokens_per_sec=pyq_throughput,
        energy_stability=pyq_energy,
        training_history=pyq_history,
    ))
    print()

    # --- Model 2: Official AKOrN ---
    print("-" * 50)
    print("Training Official AKOrN (vector Kuramoto on hyperspheres)...")
    print("-" * 50)
    try:
        off_model = OfficialAKOrN(
            dim=feature_dim, n=4, num_steps=num_steps, gamma=1.0, heads=4,
        )
        off_params = sum(p.numel() for p in off_model.parameters())
        print(f"  Parameters: {off_params:,}")

        off_history = train_binding(
            off_model, num_epochs=num_epochs, feature_dim=feature_dim,
            num_tokens=num_tokens, num_groups=num_groups, device=device,
        )

        off_model.eval()
        off_acc = phase_clustering_accuracy(eval_feat, eval_lab, off_model)
        off_sep = phase_separation_ratio(eval_feat, eval_lab, off_model)
        off_throughput = measure_throughput(off_model, feature_dim, num_tokens, device=device)
        off_energy = energy_monotonicity(eval_feat[:8], off_model)

        best_acc_off = max(h["binding_accuracy"] for h in off_history)
        epochs_90_off = next(
            (h["epoch"] for h in off_history if h["binding_accuracy"] >= 0.9), -1
        )

        results.append(BenchmarkResult(
            model_name="Official (vector Kuramoto)",
            param_count=off_params,
            final_binding_accuracy=off_acc,
            final_separation_ratio=off_sep,
            best_binding_accuracy=best_acc_off,
            epochs_to_90pct=epochs_90_off,
            throughput_tokens_per_sec=off_throughput,
            energy_stability=off_energy,
            training_history=off_history,
        ))
    except Exception as e:
        print(f"  ERROR: Official model failed — {e}")
        import traceback
        traceback.print_exc()
        results.append(BenchmarkResult(
            model_name="Official (vector Kuramoto)",
            final_binding_accuracy=-1,
        ))

    # --- Model 3: Baseline (standard attention, no oscillators) ---
    print()
    print("-" * 50)
    print("Training Baseline (standard multi-head attention, no oscillators)...")
    print("-" * 50)

    class AttentionBaseline(nn.Module):
        """Standard multi-head attention — no oscillatory dynamics at all."""
        def __init__(self, dim, num_heads=4):
            super().__init__()
            self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
            self.norm = nn.LayerNorm(dim)
            self.name = "Baseline (standard attention)"

        def forward(self, x):
            out, _ = self.attn(x, x, x)
            return self.norm(x + out)

    baseline = AttentionBaseline(feature_dim)
    base_params = sum(p.numel() for p in baseline.parameters())
    print(f"  Parameters: {base_params:,}")

    base_history = train_binding(
        baseline, num_epochs=num_epochs, feature_dim=feature_dim,
        num_tokens=num_tokens, num_groups=num_groups, device=device,
    )

    baseline.eval()
    base_acc = phase_clustering_accuracy(eval_feat, eval_lab, baseline)
    base_sep = phase_separation_ratio(eval_feat, eval_lab, baseline)
    base_throughput = measure_throughput(baseline, feature_dim, num_tokens, device=device)

    best_acc_base = max(h["binding_accuracy"] for h in base_history)
    epochs_90_base = next(
        (h["epoch"] for h in base_history if h["binding_accuracy"] >= 0.9), -1
    )

    results.append(BenchmarkResult(
        model_name="Baseline (standard attention)",
        param_count=base_params,
        final_binding_accuracy=base_acc,
        final_separation_ratio=base_sep,
        best_binding_accuracy=best_acc_base,
        epochs_to_90pct=epochs_90_base,
        throughput_tokens_per_sec=base_throughput,
        training_history=base_history,
    ))

    # --- Report ---
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Model':<35} {'Params':>8} {'Accuracy':>10} {'Sep Ratio':>10} "
          f"{'Best Acc':>10} {'>90%':>6} {'Tok/s':>12}")
    print("-" * 95)
    for r in results:
        e90 = str(r.epochs_to_90pct) if r.epochs_to_90pct >= 0 else "never"
        print(f"{r.model_name:<35} {r.param_count:>8,} {r.final_binding_accuracy:>10.3f} "
              f"{r.final_separation_ratio:>10.2f} {r.best_binding_accuracy:>10.3f} "
              f"{e90:>6} {r.throughput_tokens_per_sec:>12,.0f}")

    # Energy stability
    print()
    print("Energy/Coherence Stability:")
    for r in results:
        if r.energy_stability:
            print(f"  {r.model_name}: {r.energy_stability}")

    # Analysis
    print()
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    pyq_r = results[0]
    if len(results) > 1 and results[1].final_binding_accuracy >= 0:
        off_r = results[1]
        if pyq_r.final_binding_accuracy > off_r.final_binding_accuracy:
            print(f"  -> PyQuifer WINS on binding accuracy "
                  f"({pyq_r.final_binding_accuracy:.3f} vs {off_r.final_binding_accuracy:.3f})")
        elif off_r.final_binding_accuracy > pyq_r.final_binding_accuracy:
            print(f"  -> Official WINS on binding accuracy "
                  f"({off_r.final_binding_accuracy:.3f} vs {pyq_r.final_binding_accuracy:.3f})")
        else:
            print(f"  -> TIE on binding accuracy ({pyq_r.final_binding_accuracy:.3f})")

        speed_ratio = pyq_r.throughput_tokens_per_sec / max(off_r.throughput_tokens_per_sec, 1)
        print(f"  -> PyQuifer is {speed_ratio:.1f}x {'faster' if speed_ratio > 1 else 'slower'} "
              f"({pyq_r.throughput_tokens_per_sec:,.0f} vs {off_r.throughput_tokens_per_sec:,.0f} tok/s)")

    if len(results) > 2:
        base_r = results[2]
        osc_advantage = pyq_r.final_binding_accuracy - base_r.final_binding_accuracy
        print(f"  -> Oscillatory advantage over baseline: "
              f"{osc_advantage:+.3f} accuracy points")

    # Save JSON
    out_dir = Path(__file__).parent
    out_path = out_dir / "akorn_comparison_results.json"
    json_results = []
    for r in results:
        d = asdict(r)
        json_results.append(d)
    with open(out_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AKOrN Benchmark")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    run_benchmark(
        feature_dim=args.dim,
        num_tokens=args.tokens,
        num_groups=args.groups,
        num_epochs=args.epochs,
        num_steps=args.steps,
        device=args.device,
    )
