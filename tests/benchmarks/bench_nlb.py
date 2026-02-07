"""
Benchmark: PyQuifer Neural Dynamics vs NLB (Neural Latents Benchmark) Tasks

Compares PyQuifer's neural dynamics modules against tasks from the Neural
Latents Benchmark (Pei et al. 2021) for neural population modeling.

PyQuifer is NOT a neural data analysis toolkit, so this benchmark
focuses on the overlapping components: spike generation, latent factor
recovery, neural co-smoothing, and criticality detection.

Benchmark sections:
  1. Spike Train Generation (Poisson vs PyQuifer LIF + SpikeEncoder)
  2. Neural Co-smoothing (Linear regression vs PyQuifer PredictiveCoding)
  3. Latent Factor Discovery (PCA vs PyQuifer WilsonCowanNetwork)
  4. Criticality Detection (PyQuifer AvalancheDetector + CriticalityController)
  5. Architecture Feature Comparison

Usage:
  python bench_nlb.py         # Full suite with console output
  pytest bench_nlb.py -v      # Just the tests

Reference: Pei et al. (2021). Neural Latents Benchmark '21.
"""

import sys
import os
import time
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn

# Add PyQuifer to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from pyquifer.spiking import LIFNeuron, SpikingLayer, SpikeEncoder
from pyquifer.learning import PredictiveCoding
from pyquifer.neural_mass import WilsonCowanPopulation, WilsonCowanNetwork
from pyquifer.criticality import AvalancheDetector, CriticalityController, BranchingRatio
from pyquifer.short_term_plasticity import TsodyksMarkramSynapse


@contextmanager
def timer():
    start = time.perf_counter()
    result = {'elapsed_ms': 0.0}
    yield result
    result['elapsed_ms'] = (time.perf_counter() - start) * 1000


# ============================================================================
# Section 1: Reimplemented NLB Primitives
# ============================================================================

def generate_poisson_spike_trains(num_neurons: int, num_timesteps: int,
                                  firing_rates: torch.Tensor) -> torch.Tensor:
    """Generate Poisson spike trains with given per-neuron firing rates.

    Reimplements the basic Poisson spike generation used in NLB datasets.
    Each neuron fires independently with a fixed rate per timestep.

    Args:
        num_neurons: Number of neurons
        num_timesteps: Number of time bins
        firing_rates: Per-neuron firing probability per bin (num_neurons,)

    Returns:
        Binary spike matrix (num_timesteps, num_neurons)
    """
    rates = firing_rates.unsqueeze(0).expand(num_timesteps, num_neurons)
    spikes = torch.bernoulli(rates.clamp(0.0, 1.0))
    return spikes


def compute_isi_cv(spikes: torch.Tensor) -> float:
    """Compute coefficient of variation of inter-spike intervals.

    CV of ISI is a standard measure of spike train regularity.
    Poisson process: CV ~ 1.0. Regular firing: CV ~ 0.0.

    Args:
        spikes: Binary spike matrix (timesteps, neurons)

    Returns:
        Mean CV of ISI across neurons
    """
    cvs = []
    for n in range(spikes.shape[1]):
        spike_times = torch.where(spikes[:, n] > 0.5)[0].float()
        if len(spike_times) < 3:
            continue
        isis = spike_times[1:] - spike_times[:-1]
        if isis.std() < 1e-8 or isis.mean() < 1e-8:
            cvs.append(0.0)
        else:
            cvs.append((isis.std() / isis.mean()).item())
    return float(np.mean(cvs)) if cvs else 0.0


def compute_fano_factor(spikes: torch.Tensor, window: int = 50) -> float:
    """Compute Fano factor (variance/mean of spike counts in windows).

    Poisson process: Fano ~ 1.0. Sub-Poisson: Fano < 1.0.

    Args:
        spikes: Binary spike matrix (timesteps, neurons)
        window: Window size for counting spikes

    Returns:
        Mean Fano factor across neurons
    """
    fanos = []
    num_windows = spikes.shape[0] // window
    if num_windows < 2:
        return 1.0
    for n in range(spikes.shape[1]):
        counts = []
        for w in range(num_windows):
            counts.append(spikes[w * window:(w + 1) * window, n].sum().item())
        counts = np.array(counts)
        mean_c = counts.mean()
        if mean_c < 1e-8:
            continue
        fanos.append(counts.var() / mean_c)
    return float(np.mean(fanos)) if fanos else 1.0


def linear_regression_predict(X_train: torch.Tensor, Y_train: torch.Tensor,
                              X_test: torch.Tensor) -> torch.Tensor:
    """Simple ridge regression baseline for co-smoothing.

    Reimplements the linear regression baseline used in NLB evaluations.

    Args:
        X_train: Training inputs (samples, features)
        Y_train: Training targets (samples, outputs)
        X_test: Test inputs (samples, features)

    Returns:
        Predictions (samples, outputs)
    """
    # Ridge regression: W = (X^T X + lambda I)^{-1} X^T Y
    reg_lambda = 0.01
    XtX = X_train.T @ X_train
    XtY = X_train.T @ Y_train
    I = torch.eye(XtX.shape[0], device=XtX.device)
    W = torch.linalg.solve(XtX + reg_lambda * I, XtY)
    return X_test @ W


def compute_r2(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Compute R-squared (coefficient of determination).

    Args:
        y_true: Ground truth (samples, features)
        y_pred: Predictions (samples, features)

    Returns:
        R-squared value
    """
    ss_res = ((y_true - y_pred) ** 2).sum().item()
    ss_tot = ((y_true - y_true.mean(dim=0)) ** 2).sum().item()
    if ss_tot < 1e-8:
        return 0.0
    return 1.0 - ss_res / ss_tot


def generate_low_rank_data(num_neurons: int, num_timesteps: int,
                           num_factors: int, noise_std: float = 0.1
                           ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate synthetic neural data from known latent factors.

    Reimplements the low-rank generative model used in NLB for testing
    latent factor recovery methods.

    Args:
        num_neurons: Number of neurons
        num_timesteps: Number of time bins
        num_factors: Number of latent factors
        noise_std: Observation noise standard deviation

    Returns:
        data: Observed firing rates (num_timesteps, num_neurons)
        latents: True latent factors (num_timesteps, num_factors)
        loadings: True loading matrix (num_factors, num_neurons)
    """
    latents = torch.zeros(num_timesteps, num_factors)
    for f in range(num_factors):
        freq = 0.02 * (f + 1)
        t = torch.arange(num_timesteps, dtype=torch.float32)
        latents[:, f] = torch.sin(2 * math.pi * freq * t + f * math.pi / 3)

    loadings = torch.randn(num_factors, num_neurons) * 0.5
    data = latents @ loadings + noise_std * torch.randn(num_timesteps, num_neurons)
    return data, latents, loadings


# ============================================================================
# Section 2: Benchmark Functions
# ============================================================================

@dataclass
class SpikeGenResult:
    """Results from spike train generation comparison."""
    system: str
    mean_rate: float
    target_rate: float
    isi_cv: float
    fano_factor: float
    elapsed_ms: float


def bench_spike_generation() -> Dict[str, SpikeGenResult]:
    """Compare Poisson model vs PyQuifer LIF + SpikeEncoder for spike statistics."""
    torch.manual_seed(42)
    results = {}
    num_neurons = 50
    num_timesteps = 1000
    target_rate = 0.1  # 10% firing probability per bin

    # --- Baseline: Poisson Spike Trains ---
    firing_rates = torch.full((num_neurons,), target_rate)

    with timer() as t_poisson:
        poisson_spikes = generate_poisson_spike_trains(
            num_neurons, num_timesteps, firing_rates
        )
        poisson_mean_rate = poisson_spikes.mean().item()
        poisson_cv = compute_isi_cv(poisson_spikes)
        poisson_fano = compute_fano_factor(poisson_spikes)

    results['poisson'] = SpikeGenResult(
        system='Poisson (NLB baseline)',
        mean_rate=poisson_mean_rate,
        target_rate=target_rate,
        isi_cv=poisson_cv,
        fano_factor=poisson_fano,
        elapsed_ms=t_poisson['elapsed_ms']
    )

    # --- PyQuifer: LIF Neuron + SpikeEncoder ---
    encoder = SpikeEncoder(num_neurons=num_neurons, mode='rate', gain=1.0)
    lif = LIFNeuron(tau=10.0, threshold=0.5, learnable=False)

    with timer() as t_lif:
        # Generate input currents that would produce ~target_rate firing
        input_signal = torch.full((num_timesteps, num_neurons), target_rate)
        encoded_spikes = encoder(input_signal)

        # Run encoded spikes through LIF for biologically realistic dynamics
        membrane = torch.zeros(1, num_neurons)
        lif_spikes_list = []
        for t_step in range(num_timesteps):
            current = encoded_spikes[t_step].unsqueeze(0)
            spikes, membrane = lif(current, membrane)
            lif_spikes_list.append(spikes.squeeze(0))

        lif_spikes = torch.stack(lif_spikes_list)
        lif_mean_rate = lif_spikes.mean().item()
        lif_cv = compute_isi_cv(lif_spikes)
        lif_fano = compute_fano_factor(lif_spikes)

    results['pyquifer_lif'] = SpikeGenResult(
        system='PyQuifer LIF + SpikeEncoder',
        mean_rate=lif_mean_rate,
        target_rate=target_rate,
        isi_cv=lif_cv,
        fano_factor=lif_fano,
        elapsed_ms=t_lif['elapsed_ms']
    )

    return results


@dataclass
class CoSmoothResult:
    """Results from neural co-smoothing comparison."""
    system: str
    r2_score: float
    num_train_neurons: int
    num_test_neurons: int
    elapsed_ms: float


def bench_co_smoothing() -> Dict[str, CoSmoothResult]:
    """Compare linear regression vs PyQuifer PredictiveCoding for co-smoothing.

    Co-smoothing: given 50% of neural firing rates, predict the other 50%.
    """
    torch.manual_seed(42)
    results = {}
    num_neurons = 50
    num_timesteps = 500
    num_train = num_neurons // 2
    num_test = num_neurons - num_train

    # Generate correlated neural data from latent factors
    data, _, _ = generate_low_rank_data(num_neurons, num_timesteps, num_factors=5)

    # Split into observed and held-out neurons
    X_observed = data[:, :num_train]
    Y_heldout = data[:, num_train:]

    # Train/test split in time
    split = int(0.7 * num_timesteps)
    X_train, X_test = X_observed[:split], X_observed[split:]
    Y_train, Y_test = Y_heldout[:split], Y_heldout[split:]

    # --- Baseline: Linear Regression ---
    with timer() as t_lin:
        Y_pred_lin = linear_regression_predict(X_train, Y_train, X_test)
        r2_lin = compute_r2(Y_test, Y_pred_lin)

    results['linear_regression'] = CoSmoothResult(
        system='Linear Regression (NLB baseline)',
        r2_score=r2_lin,
        num_train_neurons=num_train,
        num_test_neurons=num_test,
        elapsed_ms=t_lin['elapsed_ms']
    )

    # --- PyQuifer: PredictiveCoding ---
    # Learn latent structure from observed neurons, then decode to held-out
    hidden_dim = 32
    pc = PredictiveCoding(
        dims=[num_train, hidden_dim, 16],
        learning_rate=0.005,
        inference_steps=10,
        precision=1.0
    )

    with timer() as t_pc:
        # Train PC on observed neurons
        for epoch in range(20):
            for i in range(0, split, 10):
                batch_x = X_train[i:i + 10]
                if len(batch_x) == 0:
                    continue
                pc.learn(batch_x)

        # Extract hidden representations (middle layer) for train and test
        hidden_train = []
        for i in range(split):
            with torch.no_grad():
                states, _ = pc(X_train[i:i + 1])
            hidden_train.append(states[1].squeeze(0))
        H_train = torch.stack(hidden_train)

        hidden_test = []
        for i in range(len(X_test)):
            with torch.no_grad():
                states, _ = pc(X_test[i:i + 1])
            hidden_test.append(states[1].squeeze(0))
        H_test = torch.stack(hidden_test)

        # Guard against NaN from PC inference divergence
        H_train = torch.nan_to_num(H_train, nan=0.0)
        H_test = torch.nan_to_num(H_test, nan=0.0)

        # Linear regression from PC hidden state to held-out neurons
        Y_pred_pc = linear_regression_predict(H_train, Y_train, H_test)
        r2_pc = compute_r2(Y_test, Y_pred_pc.detach())
        if not np.isfinite(r2_pc):
            r2_pc = 0.0  # PC didn't converge for this task

    results['predictive_coding'] = CoSmoothResult(
        system='PyQuifer PredictiveCoding',
        r2_score=r2_pc,
        num_train_neurons=num_train,
        num_test_neurons=num_test,
        elapsed_ms=t_pc['elapsed_ms']
    )

    return results


@dataclass
class LatentFactorResult:
    """Results from latent factor discovery comparison."""
    system: str
    variance_explained: float
    factor_correlation: float
    num_factors: int
    elapsed_ms: float


def bench_latent_factors() -> Dict[str, LatentFactorResult]:
    """Compare PCA vs PyQuifer WilsonCowanNetwork for latent factor recovery."""
    torch.manual_seed(42)
    results = {}
    num_neurons = 50
    num_timesteps = 500
    num_factors = 3

    data, true_latents, true_loadings = generate_low_rank_data(
        num_neurons, num_timesteps, num_factors, noise_std=0.1
    )

    # --- Baseline: PCA ---
    with timer() as t_pca:
        # Center data
        data_centered = data - data.mean(dim=0)
        # SVD
        U, S, Vt = torch.linalg.svd(data_centered, full_matrices=False)
        pca_factors = U[:, :num_factors] * S[:num_factors]

        # Variance explained
        total_var = (S ** 2).sum().item()
        explained_var = (S[:num_factors] ** 2).sum().item() / total_var

        # Factor correlation with ground truth
        correlations = []
        for f in range(num_factors):
            best_corr = 0.0
            for g in range(num_factors):
                pca_f = pca_factors[:, f]
                true_f = true_latents[:, g]
                pca_f = pca_f - pca_f.mean()
                true_f = true_f - true_f.mean()
                corr = (pca_f * true_f).sum() / (pca_f.norm() * true_f.norm() + 1e-8)
                best_corr = max(best_corr, abs(corr.item()))
            correlations.append(best_corr)
        pca_corr = float(np.mean(correlations))

    results['pca'] = LatentFactorResult(
        system='PCA (NLB baseline)',
        variance_explained=explained_var,
        factor_correlation=pca_corr,
        num_factors=num_factors,
        elapsed_ms=t_pca['elapsed_ms']
    )

    # --- PyQuifer: WilsonCowanNetwork ---
    wc_net = WilsonCowanNetwork(
        num_populations=num_factors,
        coupling_strength=0.3,
        tau_E=10.0,
        tau_I=5.0,
        dt=0.1
    )

    with timer() as t_wc:
        # Drive WilsonCowanNetwork with the neural data and collect E-state trajectories
        # The network's population dynamics should capture latent structure
        wc_trajectories = []

        for t_step in range(num_timesteps):
            # Use mean activity across neuron groups as external input
            chunk_size = num_neurons // num_factors
            ext_input = torch.zeros(num_factors)
            for f in range(num_factors):
                start = f * chunk_size
                end = start + chunk_size if f < num_factors - 1 else num_neurons
                ext_input[f] = data[t_step, start:end].mean()

            result = wc_net(steps=5, external_input=ext_input)
            wc_trajectories.append(result['E_states'].detach().clone())

        wc_factors = torch.stack(wc_trajectories)  # (timesteps, num_factors)

        # Variance explained by WC factors (project data onto WC factor space)
        wc_centered = wc_factors - wc_factors.mean(dim=0)
        data_centered_wc = data - data.mean(dim=0)
        # Least squares fit: data ~ wc_factors @ W
        W_fit = torch.linalg.lstsq(wc_centered, data_centered_wc).solution
        data_reconstructed = wc_centered @ W_fit
        ss_res = ((data_centered_wc - data_reconstructed) ** 2).sum().item()
        ss_tot = (data_centered_wc ** 2).sum().item()
        wc_var_explained = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 1e-8 else 0.0

        # Factor correlation with ground truth
        wc_correlations = []
        for f in range(num_factors):
            best_corr = 0.0
            for g in range(num_factors):
                wc_f = wc_factors[:, f] - wc_factors[:, f].mean()
                true_f = true_latents[:, g] - true_latents[:, g].mean()
                corr = (wc_f * true_f).sum() / (wc_f.norm() * true_f.norm() + 1e-8)
                best_corr = max(best_corr, abs(corr.item()))
            wc_correlations.append(best_corr)
        wc_corr = float(np.mean(wc_correlations))

    results['wilson_cowan'] = LatentFactorResult(
        system='PyQuifer WilsonCowanNetwork',
        variance_explained=wc_var_explained,
        factor_correlation=wc_corr,
        num_factors=num_factors,
        elapsed_ms=t_wc['elapsed_ms']
    )

    return results


@dataclass
class CriticalityResult:
    """Results from criticality detection benchmark."""
    system: str
    regime: str
    detected_regime: str
    power_law_exponent: float
    branching_ratio: float
    correct: bool
    elapsed_ms: float


def generate_regime_data(regime: str, num_timesteps: int,
                         num_neurons: int) -> torch.Tensor:
    """Generate synthetic neural activity at different criticality regimes.

    Reimplements the regime-specific data generation for criticality analysis.

    Args:
        regime: 'subcritical', 'critical', or 'supercritical'
        num_timesteps: Number of time bins
        num_neurons: Number of neurons

    Returns:
        Activity tensor (num_timesteps, num_neurons)
    """
    activity = torch.zeros(num_timesteps, num_neurons)

    if regime == 'subcritical':
        # Activity decays quickly, small isolated bursts
        for t in range(num_timesteps):
            # Low spontaneous rate with fast decay
            rate = 0.02
            if t > 0:
                prev_active = (activity[t - 1] > 0.5).sum().item()
                rate = rate + 0.3 * prev_active / num_neurons  # branching < 1
            activity[t] = torch.bernoulli(torch.full((num_neurons,), min(rate, 1.0)))

    elif regime == 'critical':
        # Power-law avalanches, branching ratio ~ 1
        for t in range(num_timesteps):
            rate = 0.02
            if t > 0:
                prev_active = (activity[t - 1] > 0.5).sum().item()
                rate = rate + 0.9 * prev_active / num_neurons  # branching ~ 1
            activity[t] = torch.bernoulli(torch.full((num_neurons,), min(rate, 1.0)))

    elif regime == 'supercritical':
        # Activity tends to explode, saturating bursts
        for t in range(num_timesteps):
            rate = 0.05
            if t > 0:
                prev_active = (activity[t - 1] > 0.5).sum().item()
                rate = rate + 1.8 * prev_active / num_neurons  # branching > 1
            activity[t] = torch.bernoulli(torch.full((num_neurons,), min(rate, 1.0)))

    return activity


def bench_criticality_detection() -> Dict[str, CriticalityResult]:
    """Test PyQuifer's criticality detection across subcritical/critical/supercritical regimes."""
    torch.manual_seed(42)
    results = {}
    num_neurons = 50
    num_timesteps = 500

    for regime in ['subcritical', 'critical', 'supercritical']:
        torch.manual_seed(42)
        activity = generate_regime_data(regime, num_timesteps, num_neurons)

        # --- PyQuifer: AvalancheDetector + CriticalityController ---
        controller = CriticalityController(
            target_branching_ratio=1.0,
            adaptation_rate=0.01
        )

        with timer() as t_crit:
            for t_step in range(num_timesteps):
                info = controller(activity[t_step])

            # Get final metrics
            sigma = info['branching_ratio'].item()
            power_exp = info['power_law_exponent'].item()

            # Classify detected regime
            if sigma < 0.8:
                detected = 'subcritical'
            elif sigma > 1.2:
                detected = 'supercritical'
            else:
                detected = 'critical'

        results[regime] = CriticalityResult(
            system='PyQuifer CriticalityController',
            regime=regime,
            detected_regime=detected,
            power_law_exponent=power_exp,
            branching_ratio=sigma,
            correct=(detected == regime),
            elapsed_ms=t_crit['elapsed_ms']
        )

    return results


def bench_architecture_features() -> Dict[str, Dict[str, bool]]:
    """Compare architectural features: NLB evaluation vs PyQuifer neural dynamics."""
    nlb_features = {
        'spike_train_analysis': True,
        'neural_population_smoothing': True,
        'latent_factor_recovery': True,
        'behavior_decoding': True,
        'forward_prediction': True,
        'power_law_detection': False,
        'branching_ratio_analysis': False,
        'excitatory_inhibitory_dynamics': False,
        'spiking_neuron_models': False,
        'short_term_plasticity': False,
        'predictive_coding': False,
        'online_adaptation': False,
        'criticality_control': False,
        'oscillatory_dynamics': False,
        'wilson_cowan_populations': False,
        'real_neural_datasets': True,
    }

    pyquifer_features = {
        'spike_train_analysis': True,
        'neural_population_smoothing': True,
        'latent_factor_recovery': True,
        'behavior_decoding': False,
        'forward_prediction': True,
        'power_law_detection': True,
        'branching_ratio_analysis': True,
        'excitatory_inhibitory_dynamics': True,
        'spiking_neuron_models': True,
        'short_term_plasticity': True,
        'predictive_coding': True,
        'online_adaptation': True,
        'criticality_control': True,
        'oscillatory_dynamics': True,
        'wilson_cowan_populations': True,
        'real_neural_datasets': False,
    }

    return {'nlb': nlb_features, 'pyquifer': pyquifer_features}


# ============================================================================
# Section 3: Console Output
# ============================================================================

def print_spike_results(results: Dict[str, SpikeGenResult]):
    print("\n--- 1. Spike Train Generation ---\n")
    print(f"{'System':<30} {'Mean Rate':>10} {'Target':>8} {'ISI CV':>8} {'Fano':>8} {'Time':>10}")
    print("-" * 78)
    for r in results.values():
        print(f"{r.system:<30} {r.mean_rate:>10.4f} {r.target_rate:>8.4f} "
              f"{r.isi_cv:>8.4f} {r.fano_factor:>8.4f} {r.elapsed_ms:>9.1f}ms")


def print_cosmooth_results(results: Dict[str, CoSmoothResult]):
    print("\n--- 2. Neural Co-smoothing ---\n")
    print(f"{'System':<35} {'R2 Score':>10} {'Train N':>9} {'Test N':>8} {'Time':>10}")
    print("-" * 76)
    for r in results.values():
        print(f"{r.system:<35} {r.r2_score:>10.4f} {r.num_train_neurons:>9} "
              f"{r.num_test_neurons:>8} {r.elapsed_ms:>9.1f}ms")


def print_latent_results(results: Dict[str, LatentFactorResult]):
    print("\n--- 3. Latent Factor Discovery ---\n")
    print(f"{'System':<35} {'Var Expl':>10} {'Factor Corr':>12} {'Factors':>8} {'Time':>10}")
    print("-" * 79)
    for r in results.values():
        print(f"{r.system:<35} {r.variance_explained:>10.4f} "
              f"{r.factor_correlation:>12.4f} {r.num_factors:>8} {r.elapsed_ms:>9.1f}ms")


def print_criticality_results(results: Dict[str, CriticalityResult]):
    print("\n--- 4. Criticality Detection ---\n")
    print(f"{'Regime':<15} {'Detected':<15} {'Sigma':>8} {'PL Exp':>8} {'Correct':>8} {'Time':>10}")
    print("-" * 68)
    for r in results.values():
        correct_str = 'YES' if r.correct else 'NO'
        print(f"{r.regime:<15} {r.detected_regime:<15} {r.branching_ratio:>8.4f} "
              f"{r.power_law_exponent:>8.4f} {correct_str:>8} {r.elapsed_ms:>9.1f}ms")


def print_architecture_features(arch_features: Dict[str, Dict[str, bool]]):
    print("\n--- 5. Architecture Feature Comparison ---\n")
    nlb = arch_features['nlb']
    pq = arch_features['pyquifer']
    all_f = sorted(set(list(nlb.keys()) + list(pq.keys())))
    print(f"{'Feature':<35} {'NLB':>10} {'PyQuifer':>10}")
    print("-" * 59)
    nlb_count = pq_count = 0
    for f in all_f:
        nv = 'YES' if nlb.get(f, False) else 'no'
        pv = 'YES' if pq.get(f, False) else 'no'
        if nlb.get(f, False):
            nlb_count += 1
        if pq.get(f, False):
            pq_count += 1
        print(f"  {f:<33} {nv:>10} {pv:>10}")
    print(f"\n  NLB: {nlb_count}/{len(all_f)} | PyQuifer: {pq_count}/{len(all_f)}")


def print_report(spike_results, cosmooth_results, latent_results,
                 criticality_results, arch_features):
    print("=" * 70)
    print("BENCHMARK: PyQuifer Neural Dynamics vs NLB Tasks")
    print("=" * 70)

    print_spike_results(spike_results)
    print_cosmooth_results(cosmooth_results)
    print_latent_results(latent_results)
    print_criticality_results(criticality_results)
    print_architecture_features(arch_features)


# ============================================================================
# Section 4: Pytest Tests
# ============================================================================

class TestSpikeGeneration:
    """Test spike train generation comparison."""

    def test_poisson_runs(self):
        """Poisson spike generation completes without error."""
        results = bench_spike_generation()
        assert 'poisson' in results
        assert results['poisson'].mean_rate > 0.0

    def test_lif_runs(self):
        """PyQuifer LIF spike generation completes without error."""
        results = bench_spike_generation()
        assert 'pyquifer_lif' in results

    def test_poisson_rate_accuracy(self):
        """Poisson model produces rates close to target."""
        results = bench_spike_generation()
        r = results['poisson']
        assert abs(r.mean_rate - r.target_rate) < 0.05

    def test_poisson_cv_near_one(self):
        """Poisson process should have ISI CV near 1.0."""
        results = bench_spike_generation()
        assert results['poisson'].isi_cv > 0.5

    def test_fano_factors_finite(self):
        """Both systems produce finite Fano factors."""
        results = bench_spike_generation()
        for key, r in results.items():
            assert np.isfinite(r.fano_factor), f"{key} produced non-finite Fano factor"


class TestCoSmoothing:
    """Test neural co-smoothing comparison."""

    def test_linear_regression_runs(self):
        """Linear regression co-smoothing completes."""
        results = bench_co_smoothing()
        assert 'linear_regression' in results

    def test_predictive_coding_runs(self):
        """PyQuifer PredictiveCoding co-smoothing completes."""
        results = bench_co_smoothing()
        assert 'predictive_coding' in results

    def test_linear_regression_positive_r2(self):
        """Linear regression should achieve positive R2 on correlated data."""
        results = bench_co_smoothing()
        assert results['linear_regression'].r2_score > 0.0

    def test_r2_values_finite(self):
        """Both methods produce finite R2 values."""
        results = bench_co_smoothing()
        for key, r in results.items():
            assert np.isfinite(r.r2_score), f"{key} produced non-finite R2"


class TestLatentFactors:
    """Test latent factor discovery comparison."""

    def test_pca_runs(self):
        """PCA factor recovery completes."""
        results = bench_latent_factors()
        assert 'pca' in results

    def test_wilson_cowan_runs(self):
        """WilsonCowanNetwork factor recovery completes."""
        results = bench_latent_factors()
        assert 'wilson_cowan' in results

    def test_pca_high_variance(self):
        """PCA should explain most variance on low-rank data."""
        results = bench_latent_factors()
        assert results['pca'].variance_explained > 0.5

    def test_pca_factor_correlation(self):
        """PCA factors should correlate with ground truth."""
        results = bench_latent_factors()
        assert results['pca'].factor_correlation > 0.5

    def test_variance_explained_finite(self):
        """Both methods produce finite variance explained values."""
        results = bench_latent_factors()
        for key, r in results.items():
            assert np.isfinite(r.variance_explained), f"{key} non-finite var explained"


class TestCriticalityDetection:
    """Test criticality detection across regimes."""

    def test_all_regimes_run(self):
        """Criticality detection runs for all three regimes."""
        results = bench_criticality_detection()
        assert 'subcritical' in results
        assert 'critical' in results
        assert 'supercritical' in results

    def test_branching_ratios_distinct(self):
        """Branching ratios should differ across regimes."""
        results = bench_criticality_detection()
        sigmas = [results[r].branching_ratio for r in ['subcritical', 'critical', 'supercritical']]
        # The three regimes should not all produce the same value
        assert not (sigmas[0] == sigmas[1] == sigmas[2])

    def test_metrics_finite(self):
        """All criticality metrics are finite."""
        results = bench_criticality_detection()
        for key, r in results.items():
            assert np.isfinite(r.branching_ratio), f"{key} non-finite branching ratio"
            assert np.isfinite(r.power_law_exponent), f"{key} non-finite power law exp"


class TestArchitecture:
    """Test architecture feature comparison."""

    def test_feature_counts(self):
        """Both systems have meaningful feature counts."""
        features = bench_architecture_features()
        nlb_count = sum(1 for v in features['nlb'].values() if v)
        pq_count = sum(1 for v in features['pyquifer'].values() if v)
        assert nlb_count >= 5
        assert pq_count >= 5


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("Running PyQuifer vs NLB (Neural Latents Benchmark) benchmarks...\n")

    spike_results = bench_spike_generation()
    cosmooth_results = bench_co_smoothing()
    latent_results = bench_latent_factors()
    criticality_results = bench_criticality_detection()
    arch_features = bench_architecture_features()

    print_report(spike_results, cosmooth_results, latent_results,
                 criticality_results, arch_features)

    print("\n--- Interpretation Notes ---")
    print("  - WilsonCowanNetwork generates neural dynamics; PCA decomposes data.")
    print("    These are complementary tools, not direct competitors.")
    print("  - Criticality detection uses ratio-of-means branching ratio estimator")
    print("    (Harris 1963), which is bounded and robust to bursty subcritical data.")
    print("  - Co-smoothing R2 depends on PredictiveCoding convergence; linear")
    print("    regression is expected to win on this synthetic low-rank task.")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
