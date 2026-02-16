"""
Consciousness Metrics Module for PyQuifer

Implements computational measures inspired by theories of consciousness:
- Perturbational Complexity Index (PCI): Response complexity to perturbation
- Integration measures: How much the system acts as a unified whole
- Differentiation measures: How many distinct states the system can achieve
- Phi-inspired measures: Integrated information approximations

These metrics enable PyQuifer to:
1. Monitor its own "consciousness-like" properties
2. Detect disorders of consciousness (breakdown of integration)
3. Serve as intrinsic signals for self-organization

Based on work by Tononi, Massimini, Koch, and Sporns.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Dict, List


class PerturbationalComplexity(nn.Module):
    """
    Computes Perturbational Complexity Index (PCI) inspired measures.

    PCI measures the complexity of the system's response to perturbation.
    High PCI = high integration AND high differentiation = conscious-like.
    Low PCI = either too uniform (coma) or too fragmented (anesthesia).

    Simplified computational version suitable for neural network states.
    """

    def __init__(self,
                 state_dim: int,
                 num_perturbations: int = 10,
                 perturbation_strength: float = 0.5,
                 response_window: int = 20):
        """
        Args:
            state_dim: Dimension of state to perturb
            num_perturbations: Number of perturbation sites to test
            perturbation_strength: Magnitude of perturbations
            response_window: Time steps to observe response
        """
        super().__init__()
        self.state_dim = state_dim
        self.num_perturbations = num_perturbations
        self.perturbation_strength = perturbation_strength
        self.response_window = response_window

        # Fixed perturbation patterns (like standardized TMS coil positions)
        # These are probes, not learnable — PCI requires consistent stimulation
        self.register_buffer('perturbation_patterns',
            torch.randn(num_perturbations, state_dim) * 0.1
        )

    def perturb(self, state: torch.Tensor, pattern_idx: int) -> torch.Tensor:
        """Apply a specific perturbation pattern to the state."""
        pattern = self.perturbation_patterns[pattern_idx]
        pattern = pattern / (pattern.norm() + 1e-6)  # Normalize
        return state + self.perturbation_strength * pattern

    def compute_response_complexity(self, responses: torch.Tensor) -> torch.Tensor:
        """
        Compute complexity of response pattern.

        Uses spatial variance, temporal change rate, and SVD entropy as a
        proxy for complexity. Not true Lempel-Ziv compression.

        High complexity = response is neither too simple nor too random.

        Args:
            responses: Response tensor (time, state_dim)

        Returns:
            Complexity score (scalar)
        """
        # Binarize responses (above/below mean)
        binary = (responses > responses.mean()).float()

        # Compute spatial complexity (differentiation)
        # How different are different dimensions?
        spatial_var = binary.var(dim=1).mean()

        # Compute temporal complexity
        # How much does the pattern change over time?
        temporal_changes = (binary[1:] != binary[:-1]).float().mean()

        # Compute compressibility via singular value entropy
        # Low entropy = compressible = low complexity
        if responses.shape[0] > 1 and responses.shape[1] > 1:
            U, S, Vh = torch.linalg.svd(responses, full_matrices=False)
            S_norm = S / (S.sum() + 1e-6)
            sv_entropy = -(S_norm * torch.log(S_norm + 1e-10)).sum()
            sv_entropy = sv_entropy / math.log(min(responses.shape))  # Normalize
        else:
            sv_entropy = torch.tensor(0.5, device=responses.device)

        # Combined complexity: high when both spatial and temporal are moderate
        # and when SVD entropy is high (not too compressible)
        complexity = spatial_var * temporal_changes * sv_entropy

        return complexity

    def forward(self,
                dynamics_fn,
                initial_state: torch.Tensor,
                baseline_steps: int = 10) -> Dict[str, torch.Tensor]:
        """
        Compute PCI-like measure by perturbing and measuring response complexity.

        Args:
            dynamics_fn: Function that takes state and returns next state
            initial_state: Starting state (state_dim,) or (batch, state_dim)
            baseline_steps: Steps to run before perturbation (establish baseline)

        Returns:
            Dictionary with:
            - pci: Perturbational Complexity Index
            - complexities: Individual complexity scores per perturbation
            - responses: All response patterns
        """
        if initial_state.dim() == 1:
            initial_state = initial_state.unsqueeze(0)

        # Run baseline
        state = initial_state.clone()
        for _ in range(baseline_steps):
            state = dynamics_fn(state)

        baseline_state = state.clone()

        # Perturb and measure responses
        complexities = []
        all_responses = []

        for i in range(self.num_perturbations):
            # Apply perturbation
            perturbed = self.perturb(baseline_state, i)

            # Collect response
            responses = [perturbed.squeeze(0)]
            state = perturbed
            for _ in range(self.response_window):
                state = dynamics_fn(state)
                responses.append(state.squeeze(0))

            responses = torch.stack(responses)
            all_responses.append(responses)

            # Measure complexity
            complexity = self.compute_response_complexity(responses)
            complexities.append(complexity)

        complexities = torch.stack(complexities)

        return {
            'pci': complexities.mean(),
            'complexities': complexities,
            'responses': torch.stack(all_responses)
        }


class IntegrationMeasure(nn.Module):
    """
    Measures how integrated (unified) a system's dynamics are.

    Integration = the whole is more than the sum of parts.
    High integration means perturbation to one part affects the whole.
    Low integration means parts operate independently.

    Related to Phi in Integrated Information Theory.
    """

    def __init__(self, state_dim: int, num_partitions: int = 4):
        """
        Args:
            state_dim: Dimension of state
            num_partitions: Number of ways to partition the system
        """
        super().__init__()
        self.state_dim = state_dim
        self.num_partitions = num_partitions

        # Generate random bipartitions
        self.register_buffer(
            'partitions',
            torch.rand(num_partitions, state_dim) > 0.5
        )

    def compute_mutual_information(self,
                                   x: torch.Tensor,
                                   y: torch.Tensor,
                                   num_bins: int = 10) -> torch.Tensor:
        """
        Estimate mutual information between two random variables.
        Uses histogram-based estimation.
        """
        # Discretize
        x_bins = torch.clamp((x * num_bins).long(), 0, num_bins - 1)
        y_bins = torch.clamp((y * num_bins).long(), 0, num_bins - 1)

        # Joint and marginal histograms
        joint = torch.zeros(num_bins, num_bins, device=x.device)
        for i in range(len(x_bins)):
            joint[x_bins[i], y_bins[i]] += 1

        joint = joint / (joint.sum() + 1e-10)
        px = joint.sum(dim=1)
        py = joint.sum(dim=0)

        # MI = sum p(x,y) log(p(x,y) / (p(x)p(y)))
        outer = torch.outer(px, py)
        mi = (joint * torch.log((joint + 1e-10) / (outer + 1e-10))).sum()

        return torch.clamp(mi, min=0)

    def forward(self, states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute integration measure from state time series.

        Args:
            states: State trajectory (time, state_dim)

        Returns:
            Dictionary with:
            - integration: Overall integration score
            - partition_mis: MI for each partition
        """
        partition_mis = []

        for partition in self.partitions:
            # Split into two parts
            part_a = states[:, partition]
            part_b = states[:, ~partition]

            if part_a.shape[1] == 0 or part_b.shape[1] == 0:
                continue

            # Summarize each part (mean activity)
            summary_a = part_a.mean(dim=1)
            summary_b = part_b.mean(dim=1)

            # Normalize to [0, 1]
            summary_a = (summary_a - summary_a.min()) / (summary_a.max() - summary_a.min() + 1e-6)
            summary_b = (summary_b - summary_b.min()) / (summary_b.max() - summary_b.min() + 1e-6)

            # Compute MI between parts
            mi = self.compute_mutual_information(summary_a, summary_b)
            partition_mis.append(mi)

        if len(partition_mis) == 0:
            return {'integration': torch.tensor(0.0, device=states.device), 'partition_mis': torch.tensor([], device=states.device)}

        partition_mis = torch.stack(partition_mis)

        # Integration = minimum MI across all partitions
        # (weakest link determines integration)
        integration = partition_mis.min()

        return {
            'integration': integration,
            'partition_mis': partition_mis
        }


class DifferentiationMeasure(nn.Module):
    """
    Measures how differentiated (diverse) a system's states are.

    High differentiation = many distinct states possible.
    Low differentiation = limited repertoire of states.

    Conscious systems have high differentiation (many possible experiences).
    """

    def __init__(self, state_dim: int, memory_size: int = 100):
        """
        Args:
            state_dim: Dimension of state
            memory_size: How many past states to remember for diversity
        """
        super().__init__()
        self.state_dim = state_dim
        self.memory_size = memory_size

        # State memory
        self.register_buffer('state_memory', torch.zeros(memory_size, state_dim))
        self.register_buffer('memory_ptr', torch.tensor(0))
        self.register_buffer('memory_filled', torch.tensor(False))

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Update memory and compute differentiation.

        Args:
            state: Current state (state_dim,) or (batch, state_dim)

        Returns:
            Dictionary with:
            - differentiation: Current differentiation score
            - num_states: Effective number of distinct states
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Add to memory
        with torch.no_grad():
            for s in state:
                self.state_memory[self.memory_ptr] = s
                self.memory_ptr = (self.memory_ptr + 1) % self.memory_size
                if self.memory_ptr == 0:
                    self.memory_filled.fill_(True)

        # Get valid memory
        if self.memory_filled:
            memory = self.state_memory
        else:
            memory = self.state_memory[:self.memory_ptr]

        if len(memory) < 2:
            return {
                'differentiation': torch.tensor(0.0, device=state.device),
                'num_states': torch.tensor(1.0, device=state.device)
            }

        # Compute differentiation via state space coverage
        # 1. Variance in each dimension
        dim_variance = memory.var(dim=0)
        total_variance = dim_variance.sum()

        # 2. Effective dimensionality (participation ratio)
        if total_variance > 0:
            normalized_var = dim_variance / total_variance
            eff_dim = 1.0 / (normalized_var.pow(2).sum() + 1e-6)
        else:
            eff_dim = torch.tensor(1.0, device=memory.device)

        # 3. State distinctiveness (average pairwise distance)
        if len(memory) > 10:
            # Sample for efficiency
            idx = torch.randperm(len(memory))[:10]
            sample = memory[idx]
        else:
            sample = memory

        dists = torch.cdist(sample, sample)
        avg_dist = dists.sum() / (len(sample) * (len(sample) - 1) + 1e-6)

        # Combined differentiation
        differentiation = torch.sqrt(total_variance) * avg_dist

        return {
            'differentiation': differentiation,
            'num_states': eff_dim,
            'variance': total_variance
        }

    def reset(self):
        """Reset state memory."""
        self.state_memory.zero_()
        self.memory_ptr.zero_()
        self.memory_filled.fill_(False)


class ConsciousnessMonitor(nn.Module):
    """
    Combines all consciousness metrics into a unified monitoring system.

    Tracks integration, differentiation, and perturbational complexity
    to provide an overall "consciousness level" estimate.

    This can be used as:
    1. A diagnostic tool to detect "disorders" in the system
    2. An intrinsic reward signal for self-organization
    3. A way to compare different model configurations
    """

    def __init__(self,
                 state_dim: int,
                 integration_weight: float = 1.0,
                 differentiation_weight: float = 1.0,
                 complexity_weight: float = 1.0):
        """
        Args:
            state_dim: Dimension of monitored state
            integration_weight: Weight for integration in combined score
            differentiation_weight: Weight for differentiation
            complexity_weight: Weight for complexity
        """
        super().__init__()
        self.state_dim = state_dim

        self.integration = IntegrationMeasure(state_dim)
        self.differentiation = DifferentiationMeasure(state_dim)
        # PCI requires dynamics function, computed separately

        self.integration_weight = integration_weight
        self.differentiation_weight = differentiation_weight
        self.complexity_weight = complexity_weight

        # History for tracking
        self.register_buffer('consciousness_history', torch.zeros(100))
        self.register_buffer('history_ptr', torch.tensor(0))

    def forward(self,
                state: torch.Tensor,
                state_history: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute all consciousness metrics.

        Args:
            state: Current state (state_dim,)
            state_history: Optional past states (time, state_dim) for integration

        Returns:
            Dictionary with all metrics and combined consciousness level
        """
        results = {}

        # Differentiation (uses internal memory)
        diff_results = self.differentiation(state)
        results.update({f'diff_{k}': v for k, v in diff_results.items()})

        # Integration (needs history)
        if state_history is not None and len(state_history) > 5:
            int_results = self.integration(state_history)
            results.update({f'int_{k}': v for k, v in int_results.items()})
        else:
            results['int_integration'] = torch.tensor(0.5, device=state.device)

        # Combined consciousness level
        # High consciousness = high integration AND high differentiation
        # (This is the core insight from IIT)
        consciousness = (
            self.integration_weight * results.get('int_integration', torch.tensor(0.5, device=state.device)) +
            self.differentiation_weight * torch.tanh(results['diff_differentiation'])
        )

        results['consciousness_level'] = consciousness

        # Update history
        with torch.no_grad():
            self.consciousness_history[self.history_ptr] = consciousness
            self.history_ptr = (self.history_ptr + 1) % 100

        return results

    def detect_disorder(self,
                        current_level: torch.Tensor,
                        threshold_ratio: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        Detect if consciousness level has dropped significantly.

        Args:
            current_level: Current consciousness level
            threshold_ratio: Ratio below baseline to trigger disorder

        Returns:
            Dictionary with disorder detection results
        """
        # Get baseline (mean of history)
        valid_history = self.consciousness_history[self.consciousness_history > 0]
        if len(valid_history) < 5:
            baseline = current_level
        else:
            baseline = valid_history.mean()

        threshold = baseline * threshold_ratio
        is_disordered = current_level < threshold

        return {
            'is_disordered': is_disordered.float(),
            'baseline': baseline,
            'threshold': threshold,
            'deficit': torch.relu(threshold - current_level)
        }

    def reset(self):
        """Reset all internal state."""
        self.differentiation.reset()
        self.consciousness_history.zero_()
        self.history_ptr.zero_()


if __name__ == '__main__':
    print("--- Consciousness Metrics Examples ---")

    # Example 1: Differentiation Measure
    print("\n1. Differentiation Measure")
    diff = DifferentiationMeasure(state_dim=8)

    # Low differentiation: similar states
    for _ in range(50):
        state = torch.randn(8) * 0.1 + torch.tensor([1.0] * 8)
        result = diff(state)
    print(f"   Low variety differentiation: {result['differentiation'].item():.4f}")

    diff.reset()

    # High differentiation: diverse states
    for _ in range(50):
        state = torch.randn(8) * 2.0
        result = diff(state)
    print(f"   High variety differentiation: {result['differentiation'].item():.4f}")

    # Example 2: Integration Measure
    print("\n2. Integration Measure")
    integ = IntegrationMeasure(state_dim=8)

    # Integrated: all dimensions correlated
    t = torch.linspace(0, 10, 100)
    base_signal = torch.sin(t)
    states_integrated = base_signal.unsqueeze(1).expand(-1, 8) + torch.randn(100, 8) * 0.1
    result = integ(states_integrated)
    print(f"   Correlated system integration: {result['integration'].item():.4f}")

    # Fragmented: independent dimensions
    states_fragmented = torch.randn(100, 8)
    result = integ(states_fragmented)
    print(f"   Independent system integration: {result['integration'].item():.4f}")

    # Example 3: Consciousness Monitor
    print("\n3. Consciousness Monitor")
    monitor = ConsciousnessMonitor(state_dim=8)

    # Simulate conscious-like dynamics
    print("   Simulating dynamics...")
    history = []
    state = torch.randn(8)
    for i in range(100):
        # Simple coupled dynamics
        state = 0.9 * state + 0.1 * torch.randn(8)
        state = state + 0.05 * torch.roll(state, 1)  # Coupling
        history.append(state.clone())

        if i > 10:
            result = monitor(state, torch.stack(history[-20:]))
            if i % 25 == 0:
                print(f"   Step {i}: consciousness={result['consciousness_level'].item():.4f}, "
                      f"diff={result['diff_differentiation'].item():.4f}")

    # Example 4: Disorder Detection
    print("\n4. Disorder Detection")
    # Sudden drop in dynamics
    state = torch.zeros(8)  # Flat state
    result = monitor(state, torch.stack([state] * 20))
    disorder = monitor.detect_disorder(result['consciousness_level'])
    print(f"   Is disordered: {disorder['is_disordered'].item() > 0.5}")
    print(f"   Deficit: {disorder['deficit'].item():.4f}")

    # Example 5: PCI (requires dynamics function)
    print("\n5. Perturbational Complexity Index")
    pci = PerturbationalComplexity(state_dim=8, num_perturbations=5, response_window=15)

    def simple_dynamics(state):
        # Coupled oscillatory dynamics
        return 0.95 * state + 0.1 * torch.sin(state) + 0.02 * torch.randn_like(state)

    result = pci(simple_dynamics, torch.randn(8))
    print(f"   PCI: {result['pci'].item():.4f}")
    print(f"   Individual complexities: {result['complexities'].tolist()}")
