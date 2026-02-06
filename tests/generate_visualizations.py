"""Generate test output visualizations for all Phase 2 modules.

Run: python PyQuifer/tests/generate_visualizations.py
Output: PyQuifer/tests/test_output/*.png

Updated based on external review comparing to research literature.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'test_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_metastability():
    """Winnerless competition dynamics + heteroclinic channel + power spectrum."""
    from pyquifer.metastability import MetastabilityIndex

    torch.manual_seed(42)
    mi = MetastabilityIndex(num_populations=6)

    steps = 1000
    activations_history = []
    dominant_history = []
    entropy_history = []
    chimera_history = []

    for _ in range(steps):
        r = mi()
        activations_history.append(r['activations'].detach().numpy().copy())
        dominant_history.append(r['dominant'].item())
        entropy_history.append(r['coalition_entropy'].item())
        chimera_history.append(r['chimera_index'].item())

    acts = np.array(activations_history)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Metastability: Winnerless Competition Dynamics', fontsize=14)

    # Panel 1: Population activations
    for i in range(6):
        axes[0, 0].plot(acts[:, i], alpha=0.7, label=f'Pop {i}')
    axes[0, 0].set_ylabel('Activation')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_title('Population Activations (Lotka-Volterra)')
    axes[0, 0].legend(loc='upper right', fontsize=7, ncol=3)

    # Panel 2: Dominant state (stream of consciousness)
    axes[0, 1].plot(dominant_history, '.', markersize=1, color='navy')
    axes[0, 1].set_ylabel('Dominant State')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_title('Heteroclinic Orbit (Stream of Consciousness)')
    axes[0, 1].set_yticks(range(6))

    # Panel 3: Power spectrum of population activations
    for i in range(6):
        freqs = np.fft.rfftfreq(steps, d=1.0)
        spectrum = np.abs(np.fft.rfft(acts[:, i] - acts[:, i].mean()))
        axes[1, 0].semilogy(freqs[1:50], spectrum[1:50], alpha=0.6, label=f'Pop {i}')
    axes[1, 0].set_xlabel('Frequency')
    axes[1, 0].set_ylabel('Power (log)')
    axes[1, 0].set_title('Frequency-Domain: Population Power Spectra')
    axes[1, 0].legend(fontsize=7, ncol=3)

    # Panel 4: Metastability metrics with smoothing
    window = 50
    entropy_smooth = np.convolve(entropy_history, np.ones(window)/window, mode='valid')
    chimera_smooth = np.convolve(chimera_history, np.ones(window)/window, mode='valid')
    axes[1, 1].plot(entropy_smooth, label='Coalition Entropy', color='green', linewidth=2)
    ax2r = axes[1, 1].twinx()
    ax2r.plot(chimera_smooth, label='Chimera Index', color='orange', linewidth=2)
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Coalition Entropy', color='green')
    ax2r.set_ylabel('Chimera Index', color='orange')
    axes[1, 1].set_title('Metastability Metrics (smoothed)')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'metastability_dynamics.png'), dpi=150)
    plt.close()
    print('  [OK] metastability_dynamics.png')


def plot_hierarchical_predictive():
    """Hierarchical predictive coding — error now DECREASES with online learning."""
    from pyquifer.hierarchical_predictive import HierarchicalPredictiveCoding

    torch.manual_seed(42)
    hpc = HierarchicalPredictiveCoding(level_dims=[16, 12, 8], lr=0.05, gen_lr=0.01)
    pattern = torch.sin(torch.linspace(0, 2 * math.pi, 16)).unsqueeze(0)

    errors_per_level = [[] for _ in range(3)]
    total_errors = []

    for _ in range(300):
        r = hpc(pattern)
        total_errors.append(r['total_error'].item())
        for i, err in enumerate(r['errors']):
            errors_per_level[i].append(err.norm().item())

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('Hierarchical Predictive Coding (Friston)', fontsize=14)

    # Panel 1: Total error DECREASING
    axes[0, 0].plot(total_errors, color='crimson', linewidth=2)
    axes[0, 0].set_title('Total Prediction Error (Decreasing)')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Error')

    # Panel 2: Per-level errors with confidence band
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    labels = ['Level 0 (Sensory)', 'Level 1 (Context)', 'Level 2 (Abstract)']
    for i in range(3):
        errs_arr = np.array(errors_per_level[i])
        axes[0, 1].plot(errs_arr, color=colors[i], label=labels[i], alpha=0.8)
        # Rolling std as confidence band
        window = 20
        if len(errs_arr) > window:
            rolling_mean = np.convolve(errs_arr, np.ones(window)/window, mode='valid')
            rolling_std = np.array([errs_arr[max(0,j-window):j+1].std()
                                    for j in range(window-1, len(errs_arr))])
            x = np.arange(window-1, len(errs_arr))
            axes[0, 1].fill_between(x, rolling_mean - rolling_std,
                                     rolling_mean + rolling_std,
                                     color=colors[i], alpha=0.15)
    axes[0, 1].set_title('Per-Level Prediction Errors')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].set_xlabel('Step')

    # Panel 3: Final beliefs at each level
    r_final = hpc(pattern)
    for i, beliefs in enumerate(r_final['beliefs']):
        axes[1, 0].bar(np.arange(len(beliefs)) + i * 0.25, beliefs.detach().numpy(),
                       width=0.25, label=labels[i], color=colors[i], alpha=0.7)
    axes[1, 0].set_title('Final Beliefs at Each Level')
    axes[1, 0].legend(fontsize=8)

    # Panel 4: Bottom-level prediction vs input (should now track better)
    pred = r_final['bottom_prediction'].detach().squeeze().numpy()
    inp = pattern.squeeze().numpy()
    axes[1, 1].plot(inp, 'b-', label='Input', linewidth=2)
    axes[1, 1].plot(pred, 'r--', label='Prediction', linewidth=2)
    axes[1, 1].fill_between(range(len(inp)), inp, pred, alpha=0.2, color='gray')
    axes[1, 1].set_title('Bottom Prediction vs Input')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'hierarchical_predictive.png'), dpi=150)
    plt.close()
    print('  [OK] hierarchical_predictive.png')


def plot_causal_flow():
    """Causal flow analysis with significance threshold."""
    from pyquifer.causal_flow import CausalFlowMap, TransferEntropyEstimator

    torch.manual_seed(42)
    cfm = CausalFlowMap(num_populations=4, buffer_size=500)

    # Create a chain: pop0 -> pop1 -> pop2, pop3 independent
    for _ in range(500):
        s = torch.zeros(4)
        s[0] = torch.randn(1).item()
        s[1] = 0.7 * s[0] + 0.3 * torch.randn(1).item()
        s[2] = 0.7 * s[1] + 0.3 * torch.randn(1).item()
        s[3] = torch.randn(1).item()
        cfm.record(s)

    flow = cfm.compute_flow()

    # Compute surrogate baseline (independent signals)
    te = TransferEntropyEstimator(num_bins=8, history_length=2)
    surrogate_tes = []
    for _ in range(20):
        x_surr = torch.randn(400)
        y_surr = torch.randn(400)
        r_surr = te(x_surr, y_surr)
        surrogate_tes.append(abs(r_surr['net_flow'].item()))
    significance_threshold = np.mean(surrogate_tes) + 2 * np.std(surrogate_tes)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Causal Flow Analysis (Transfer Entropy)', fontsize=14)

    # Panel 1: Flow matrix heatmap
    fm = flow['flow_matrix'].detach().numpy()
    im = axes[0].imshow(fm, cmap='RdBu_r', interpolation='nearest')
    axes[0].set_title('Net Flow Matrix\n(positive = row drives column)')
    axes[0].set_xlabel('Target')
    axes[0].set_ylabel('Source')
    axes[0].set_xticks(range(4))
    axes[0].set_yticks(range(4))
    axes[0].set_xticklabels(['Pop 0\n(driver)', 'Pop 1', 'Pop 2', 'Pop 3\n(indep)'])
    axes[0].set_yticklabels(['Pop 0', 'Pop 1', 'Pop 2', 'Pop 3'])
    plt.colorbar(im, ax=axes[0])

    # Panel 2: Driver scores with significance threshold
    scores = flow['driver_scores'].detach().numpy()
    bar_colors = ['#e74c3c' if abs(s) > significance_threshold else '#95a5a6' for s in scores]
    bars = axes[1].bar(range(4), scores, color=bar_colors)
    axes[1].axhline(y=significance_threshold, color='gray', linestyle='--',
                     alpha=0.7, label=f'Sig. threshold ({significance_threshold:.3f})')
    axes[1].axhline(y=-significance_threshold, color='gray', linestyle='--', alpha=0.7)
    axes[1].set_title('Driver Scores\n(colored = significant)')
    axes[1].set_xlabel('Population')
    axes[1].set_ylabel('Driver Score')
    axes[1].set_xticks(range(4))
    axes[1].set_xticklabels(['Pop 0\n(driver)', 'Pop 1', 'Pop 2', 'Pop 3\n(indep)'])
    axes[1].legend(fontsize=8)

    # Panel 3: Causal chain diagram
    axes[2].set_xlim(-0.5, 3.5)
    axes[2].set_ylim(-1, 1)
    axes[2].set_aspect('equal')
    for i, (x, label) in enumerate(zip([0, 1, 2, 3],
                                        ['Pop 0\n(Driver)', 'Pop 1', 'Pop 2', 'Pop 3\n(Indep)'])):
        color = '#e74c3c' if i == 0 else '#3498db' if i < 3 else '#95a5a6'
        circle = plt.Circle((x, 0), 0.3, color=color, alpha=0.6)
        axes[2].add_patch(circle)
        axes[2].text(x, 0, label, ha='center', va='center', fontsize=7)
    for i in range(2):
        axes[2].annotate('', xy=(i + 0.7, 0), xytext=(i + 0.3, 0),
                         arrowprops=dict(arrowstyle='->', color='black', lw=2))
    axes[2].set_title('True Causal Structure')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'causal_flow_analysis.png'), dpi=150)
    plt.close()
    print('  [OK] causal_flow_analysis.png')


def plot_memory_consolidation():
    """Memory consolidation pipeline: store -> replay -> consolidate."""
    from pyquifer.memory_consolidation import (
        EpisodicBuffer, SharpWaveRipple, ConsolidationEngine
    )

    torch.manual_seed(42)
    buf = EpisodicBuffer(state_dim=16, capacity=100)
    swr = SharpWaveRipple(state_dim=16)
    engine = ConsolidationEngine(state_dim=16, semantic_dim=8)

    rewards_stored = []
    for i in range(80):
        reward = float(np.sin(i * 0.1) * 5 + np.random.randn() * 0.5)
        buf.store(torch.randn(16), reward=reward)
        rewards_stored.append(reward)

    # Replay at different sleep levels
    replay_counts = []
    sleep_levels = np.linspace(0, 1, 20)
    for sl in sleep_levels:
        swr_fresh = SharpWaveRipple(state_dim=16)
        r = swr_fresh(buf, sleep_signal=float(sl))
        replay_counts.append(r['replayed_states'].shape[0])

    # Consolidation over multiple rounds
    trace_counts = []
    for round_i in range(10):
        replay = swr(buf, sleep_signal=0.9)
        if replay['replayed_states'].shape[0] > 0:
            engine(replay['replayed_states'],
                   torch.ones(replay['replayed_states'].shape[0]) * (round_i + 1))
        trace_counts.append(engine.num_traces.item())

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('Memory Consolidation Pipeline', fontsize=14)

    axes[0, 0].plot(rewards_stored, color='#e74c3c')
    axes[0, 0].fill_between(range(len(rewards_stored)), rewards_stored, alpha=0.2, color='#e74c3c')
    axes[0, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 0].set_title(f'Episodic Buffer ({buf.num_stored.item()} memories)')
    axes[0, 0].set_xlabel('Experience #')
    axes[0, 0].set_ylabel('Reward')

    axes[0, 1].bar(range(len(sleep_levels)), replay_counts, color='#3498db', alpha=0.7)
    axes[0, 1].axvline(x=5.5, color='red', linestyle='--', alpha=0.5, label='Sleep threshold')
    axes[0, 1].set_title('Sharp-Wave Ripple Replay')
    axes[0, 1].set_xlabel('Sleep Level Index')
    axes[0, 1].set_ylabel('# Replayed Memories')
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(trace_counts, 'o-', color='#2ecc71', linewidth=2)
    axes[1, 0].set_title('Semantic Trace Accumulation')
    axes[1, 0].set_xlabel('Consolidation Round')
    axes[1, 0].set_ylabel('# Semantic Traces')

    sample = buf.sample_prioritized(50)
    if sample['rewards'].shape[0] > 0:
        axes[1, 1].hist(sample['rewards'].numpy(), bins=15, color='#9b59b6', alpha=0.7,
                        edgecolor='white')
    axes[1, 1].set_title('Prioritized Sample Reward Distribution')
    axes[1, 1].set_xlabel('Reward')
    axes[1, 1].set_ylabel('Count')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'memory_consolidation.png'), dpi=150)
    plt.close()
    print('  [OK] memory_consolidation.png')


def plot_neural_darwinism():
    """Neural darwinism: group selection and resource competition."""
    from pyquifer.neural_darwinism import SelectionArena, SymbiogenesisDetector

    torch.manual_seed(42)
    arena = SelectionArena(num_groups=6, group_dim=16,
                           selection_pressure=0.15, total_budget=12.0)
    symb = SymbiogenesisDetector(num_groups=6, group_dim=16,
                                  mi_threshold=0.2, buffer_size=200)

    target = torch.sin(torch.linspace(0, 2 * math.pi, 16))
    fitness_history = [[] for _ in range(6)]
    resource_history = [[] for _ in range(6)]

    steps = 300
    for _ in range(steps):
        result = arena(target + torch.randn(16) * 0.2, global_coherence=target)
        symb(result['group_outputs'])
        for i in range(6):
            fitness_history[i].append(arena.groups[i].fitness.item())
            resource_history[i].append(arena.groups[i].resources.item())

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('Neural Darwinism: Neuronal Group Selection', fontsize=14)

    colors = plt.cm.Set2(np.linspace(0, 1, 6))

    for i in range(6):
        # Smooth fitness
        window = 20
        f_arr = np.array(fitness_history[i])
        f_smooth = np.convolve(f_arr, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(f_smooth, color=colors[i], label=f'Group {i}', alpha=0.8)
    axes[0, 0].set_title('Group Fitness Over Time (smoothed)')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Fitness')
    axes[0, 0].legend(fontsize=7, ncol=2)

    for i in range(6):
        axes[0, 1].plot(resource_history[i], color=colors[i], label=f'Group {i}', alpha=0.7)
    axes[0, 1].set_title('Resource Competition (Replicator Dynamics)')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Resources')
    axes[0, 1].legend(fontsize=7, ncol=2)

    final_resources = [resource_history[i][-1] for i in range(6)]
    axes[1, 0].bar(range(6), final_resources, color=colors)
    axes[1, 0].set_title('Final Resource Distribution')
    axes[1, 0].set_xlabel('Group')
    axes[1, 0].set_ylabel('Resources')
    axes[1, 0].axhline(y=12.0/6, color='red', linestyle='--', label='Equal share')
    axes[1, 0].legend()

    sym_result = symb([torch.randn(16) for _ in range(6)])
    mi_mat = sym_result['mi_matrix'].detach().numpy()
    im = axes[1, 1].imshow(mi_mat, cmap='YlOrRd', interpolation='nearest')
    axes[1, 1].set_title(f'Mutual Information (Symbiogenesis)\nBonds: {sym_result["num_bonds"].item()}')
    axes[1, 1].set_xlabel('Group')
    axes[1, 1].set_ylabel('Group')
    plt.colorbar(im, ax=axes[1, 1])

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'neural_darwinism.png'), dpi=150)
    plt.close()
    print('  [OK] neural_darwinism.png')


def plot_self_model():
    """Self-model: Markov blanket + narrative identity convergence."""
    from pyquifer.self_model import MarkovBlanket, SelfModel, NarrativeIdentity

    torch.manual_seed(42)
    mb = MarkovBlanket(internal_dim=16, sensory_dim=8, active_dim=8)
    sm = SelfModel(self_dim=16, lr=0.05)
    ni = NarrativeIdentity(dim=16, tau=0.05)

    identity_strength = []
    deviation = []
    self_errors = []

    steps = 300
    for i in range(steps):
        sensory = torch.sin(torch.linspace(0, math.pi, 8) + i * 0.02) * 0.5
        sensory += torch.randn(8) * 0.1
        blanket = mb(sensory)
        self_r = sm(blanket['internal_state'])
        narr = ni(self_r['self_summary'])

        identity_strength.append(narr['identity_strength'].item())
        deviation.append(narr['deviation'].item())
        self_errors.append(self_r['self_prediction_error_magnitude'].item())

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('Self-Model: Phenomenal Self & Narrative Identity', fontsize=14)

    axes[0, 0].plot(identity_strength, color='#8e44ad', linewidth=2)
    axes[0, 0].set_title('Identity Strength (Accumulated Experience)')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Strength')

    # Deviation with rolling mean
    dev_arr = np.array(deviation)
    axes[0, 1].plot(dev_arr, color='#e67e22', alpha=0.3)
    window = 20
    dev_smooth = np.convolve(dev_arr, np.ones(window)/window, mode='valid')
    axes[0, 1].plot(range(window-1, len(dev_arr)), dev_smooth, color='#e67e22', linewidth=2)
    axes[0, 1].set_title('Deviation from Narrative Identity')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Deviation')

    # Self-prediction error with rolling mean
    err_arr = np.array(self_errors)
    axes[1, 0].plot(err_arr, color='#e74c3c', alpha=0.3)
    err_smooth = np.convolve(err_arr, np.ones(window)/window, mode='valid')
    axes[1, 0].plot(range(window-1, len(err_arr)), err_smooth, color='#e74c3c', linewidth=2)
    axes[1, 0].set_title('Self-Prediction Error')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Error Magnitude')

    final_blanket = mb(torch.randn(8))
    blanket_state = final_blanket['blanket_state'].detach().squeeze().numpy()
    x = np.arange(blanket_state.shape[-1])
    n_sensory = 8
    axes[1, 1].bar(x[:n_sensory], blanket_state[:n_sensory], color='#3498db', alpha=0.7, label='Sensory')
    axes[1, 1].bar(x[n_sensory:], blanket_state[n_sensory:], color='#e74c3c', alpha=0.7, label='Active')
    axes[1, 1].set_title('Markov Blanket State')
    axes[1, 1].set_xlabel('Dimension')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'self_model.png'), dpi=150)
    plt.close()
    print('  [OK] self_model.png')


def plot_precision_and_resonance():
    """Precision weighting (dynamic) + stochastic resonance (inverted-U)."""
    from pyquifer.precision_weighting import PrecisionEstimator
    from pyquifer.stochastic_resonance import AdaptiveStochasticResonance

    torch.manual_seed(42)

    # Precision dynamics — now shows dynamic response
    estimator = PrecisionEstimator(num_channels=8, tau=10.0)
    precisions = []
    variances = []
    for i in range(300):
        # Three regimes: stable -> noisy -> stable again
        if i < 100:
            error = torch.ones(8) * 0.1 + torch.randn(8) * 0.05
        elif i < 200:
            error = torch.randn(8) * 2.0
        else:
            error = torch.ones(8) * 0.1 + torch.randn(8) * 0.05
        r = estimator(error)
        precisions.append(r['precision'].detach().numpy().mean())
        variances.append(r['running_var'].numpy().mean())

    # Stochastic Resonance — inverted-U curve
    noise_levels_sweep = np.linspace(0.01, 1.5, 50)
    snr_at_noise = []
    asr_probe = AdaptiveStochasticResonance(dim=16, threshold=0.5)
    signal = torch.ones(1, 16) * 0.4
    for nl in noise_levels_sweep:
        snr_val = asr_probe._measure_snr(signal, float(nl), num_trials=20)
        snr_at_noise.append(snr_val)

    # Adaptive noise trajectory
    asr = AdaptiveStochasticResonance(dim=16, threshold=0.5, initial_noise=0.01)
    noise_trajectory = []
    snr_trajectory = []
    for _ in range(200):
        r = asr(torch.ones(16) * 0.4)
        noise_trajectory.append(asr.noise_level.item())
        snr_trajectory.append(r['snr'].item())

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('Precision Weighting & Stochastic Resonance', fontsize=14)

    # Panel 1: Dynamic precision (3 regimes)
    axes[0, 0].plot(precisions, color='#2ecc71', linewidth=2)
    axes[0, 0].axvline(x=100, color='red', linestyle='--', alpha=0.7, label='Noise onset')
    axes[0, 0].axvline(x=200, color='blue', linestyle='--', alpha=0.7, label='Noise offset')
    axes[0, 0].set_title('Precision: Dynamic Response to Noise Changes')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].legend(fontsize=8)

    # Panel 2: Running variance (3 regimes)
    axes[0, 1].plot(variances, color='#e74c3c', linewidth=2)
    axes[0, 1].axvline(x=100, color='red', linestyle='--', alpha=0.7, label='Noise onset')
    axes[0, 1].axvline(x=200, color='blue', linestyle='--', alpha=0.7, label='Noise offset')
    axes[0, 1].set_title('Running Variance (Tracks Signal Changes)')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Variance')
    axes[0, 1].legend(fontsize=8)

    # Panel 3: INVERTED-U CURVE (the key SR signature)
    axes[1, 0].plot(noise_levels_sweep, snr_at_noise, 'o-', color='#3498db', linewidth=2, markersize=3)
    peak_idx = np.argmax(snr_at_noise)
    axes[1, 0].axvline(x=noise_levels_sweep[peak_idx], color='red', linestyle='--',
                        alpha=0.7, label=f'Optimal noise = {noise_levels_sweep[peak_idx]:.2f}')
    axes[1, 0].set_title('Stochastic Resonance: Classic Inverted-U')
    axes[1, 0].set_xlabel('Noise Level')
    axes[1, 0].set_ylabel('SNR')
    axes[1, 0].legend(fontsize=8)

    # Panel 4: Adaptive noise trajectory
    axes[1, 1].plot(noise_trajectory, color='#9b59b6', linewidth=2)
    axes[1, 1].axhline(y=noise_levels_sweep[peak_idx], color='red', linestyle='--',
                        alpha=0.5, label=f'Optimal = {noise_levels_sweep[peak_idx]:.2f}')
    axes[1, 1].set_title('Adaptive Noise: Tracking Optimal Level')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Noise Level')
    axes[1, 1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'precision_and_resonance.png'), dpi=150)
    plt.close()
    print('  [OK] precision_and_resonance.png')


def plot_module_dashboard():
    """Summary dashboard of all 8 Phase 2 modules."""
    from pyquifer.metastability import MetastabilityIndex
    from pyquifer.hierarchical_predictive import HierarchicalPredictiveCoding
    from pyquifer.precision_weighting import AttentionAsPrecision
    from pyquifer.stochastic_resonance import AdaptiveStochasticResonance
    from pyquifer.causal_flow import DominanceDetector
    from pyquifer.memory_consolidation import EpisodicBuffer, SharpWaveRipple
    from pyquifer.self_model import NarrativeIdentity
    from pyquifer.neural_darwinism import SelectionArena

    torch.manual_seed(42)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle('Phase 2 Module Dashboard (8 New Modules)', fontsize=16, y=1.02)

    # 1. Precision Weighting — gradual changes
    aap = AttentionAsPrecision(num_channels=8, tau=10.0)
    attn_values = []
    for i in range(100):
        signal = torch.randn(8) * 0.5
        # Vary error statistics over time
        noise_scale = 0.1 + 0.5 * abs(np.sin(i * 0.05))
        error = torch.randn(8) * noise_scale
        r = aap(signal, error)
        attn_values.append(r['attention_map'].detach().numpy().copy())
    attn_arr = np.array(attn_values)
    axes[0, 0].imshow(attn_arr.T, aspect='auto', cmap='viridis')
    axes[0, 0].set_title('1. Precision\nWeighting', fontsize=10, fontweight='bold')
    axes[0, 0].set_ylabel('Channel')
    axes[0, 0].set_xlabel('Step')

    # 2. HPC — error decreasing
    hpc = HierarchicalPredictiveCoding(level_dims=[8, 6, 4], lr=0.05, gen_lr=0.01)
    pattern = torch.sin(torch.linspace(0, math.pi, 8)).unsqueeze(0)
    errs = []
    for _ in range(200):
        r = hpc(pattern)
        errs.append(r['total_error'].item())
    axes[0, 1].plot(errs, color='crimson')
    axes[0, 1].set_title('2. Predictive\nCoding', fontsize=10, fontweight='bold')
    axes[0, 1].set_ylabel('Total Error')

    # 3. Metastability
    mi = MetastabilityIndex(num_populations=4)
    acts = []
    for _ in range(200):
        r = mi()
        acts.append(r['activations'].detach().numpy().copy())
    acts_arr = np.array(acts)
    for i in range(4):
        axes[0, 2].plot(acts_arr[:, i], alpha=0.7)
    axes[0, 2].set_title('3. Metastability', fontsize=10, fontweight='bold')
    axes[0, 2].set_ylabel('Activation')

    # 4. SR — inverted-U
    asr_probe = AdaptiveStochasticResonance(dim=8, threshold=0.5)
    noise_sweep = np.linspace(0.01, 1.5, 30)
    snrs = [asr_probe._measure_snr(torch.ones(1, 8) * 0.4, float(nl), num_trials=15)
            for nl in noise_sweep]
    axes[0, 3].plot(noise_sweep, snrs, 'o-', color='#3498db', markersize=3)
    axes[0, 3].set_title('4. Stochastic\nResonance', fontsize=10, fontweight='bold')
    axes[0, 3].set_ylabel('SNR')
    axes[0, 3].set_xlabel('Noise Level')

    # 5. Causal Flow
    dd = DominanceDetector(num_levels=4, buffer_size=200)
    dom_ratios = []
    for _ in range(200):
        r = dd(torch.randn(4), compute_every=50)
        dom_ratios.append(r['dominance_ratio'].item())
    axes[1, 0].plot(dom_ratios, color='#e67e22')
    axes[1, 0].axhline(y=0.5, color='gray', linestyle='--')
    axes[1, 0].set_title('5. Causal Flow', fontsize=10, fontweight='bold')
    axes[1, 0].set_ylabel('Dominance Ratio')
    axes[1, 0].set_xlabel('Step')

    # 6. Memory Consolidation
    buf = EpisodicBuffer(state_dim=8, capacity=50)
    for i in range(50):
        buf.store(torch.randn(8), reward=float(np.sin(i * 0.2) * 3))
    replay_n = []
    for sl in np.linspace(0, 1, 20):
        swr_t = SharpWaveRipple(state_dim=8)
        rr = swr_t(buf, sleep_signal=float(sl))
        replay_n.append(rr['replayed_states'].shape[0])
    axes[1, 1].bar(range(20), replay_n, color='#9b59b6', alpha=0.7)
    axes[1, 1].set_title('6. Memory\nConsolidation', fontsize=10, fontweight='bold')
    axes[1, 1].set_ylabel('Replayed')
    axes[1, 1].set_xlabel('Sleep Level')

    # 7. Self Model
    ni = NarrativeIdentity(dim=8, tau=0.05)
    strength = []
    for _ in range(200):
        r = ni(torch.randn(8) * 0.3 + torch.ones(8) * 0.5)
        strength.append(r['identity_strength'].item())
    axes[1, 2].plot(strength, color='#8e44ad')
    axes[1, 2].set_title('7. Self Model', fontsize=10, fontweight='bold')
    axes[1, 2].set_ylabel('Identity Strength')
    axes[1, 2].set_xlabel('Step')

    # 8. Neural Darwinism
    arena = SelectionArena(num_groups=4, group_dim=8, selection_pressure=0.2)
    target = torch.sin(torch.linspace(0, math.pi, 8))
    res_h = [[] for _ in range(4)]
    for _ in range(100):
        rr = arena(target + torch.randn(8) * 0.2, global_coherence=target)
        for j in range(4):
            res_h[j].append(arena.groups[j].resources.item())
    for j in range(4):
        axes[1, 3].plot(res_h[j], alpha=0.7, label=f'G{j}')
    axes[1, 3].set_title('8. Neural\nDarwinism', fontsize=10, fontweight='bold')
    axes[1, 3].set_ylabel('Resources')
    axes[1, 3].set_xlabel('Step')
    axes[1, 3].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'phase2_module_dashboard.png'), dpi=150,
                bbox_inches='tight')
    plt.close()
    print('  [OK] phase2_module_dashboard.png')


if __name__ == '__main__':
    print('Generating Phase 2 test output visualizations...\n')

    plot_metastability()
    plot_hierarchical_predictive()
    plot_causal_flow()
    plot_memory_consolidation()
    plot_neural_darwinism()
    plot_self_model()
    plot_precision_and_resonance()
    plot_module_dashboard()

    print(f'\nAll visualizations saved to: {OUTPUT_DIR}')
