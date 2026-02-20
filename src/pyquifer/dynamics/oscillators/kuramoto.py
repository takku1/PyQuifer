"""Kuramoto oscillator models — LearnableKuramotoBank and SensoryCoupling.

This module contains the core Kuramoto-family oscillator implementations:
- _rk4_step: Classical 4th-order Runge-Kutta integrator
- Snake: Learnable periodic activation function
- LearnableKuramotoBank: Bank of coupled Kuramoto oscillators with
  configurable topologies, precision-weighted coupling, and attractor analysis
- SensoryCoupling: Biologically-motivated sensory-to-oscillator coupling
"""
import math
from typing import Dict, Literal, Optional

import torch
import torch.nn as nn

__all__ = [
    '_rk4_step', 'Snake', 'LearnableKuramotoBank', 'SensoryCoupling',
]


def _rk4_step(f, y, dt):
    """Classical 4th-order Runge-Kutta integrator step.

    Pre-allocated buffer style (torchode, Lienen 2022): computes k1-k4
    in-place without list allocation.

    Args:
        f: RHS function y' = f(y)
        y: Current state tensor
        dt: Timestep

    Returns:
        Updated state y_{n+1}
    """
    k1 = f(y)
    k2 = f(y + 0.5 * dt * k1)
    k3 = f(y + 0.5 * dt * k2)
    k4 = f(y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


class Snake(nn.Module):
    """
    Snake activation function.
    Proposed in "Neural Networks with Learnable Activation Functions" by Ziyin Liu et al. (2020).
    Snake(x) = x + sin(x)^2 / frequency
    """
    def __init__(self, frequency: float = 1.0):
        super().__init__()
        self.frequency = nn.Parameter(torch.tensor(frequency))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (1/self.frequency) * (torch.sin(x * self.frequency)**2)


class LearnableKuramotoBank(nn.Module):
    """
    A bank of coupled Kuramoto oscillators with learnable parameters and
    configurable network topologies.

    The Kuramoto model describes the synchronization of a population of coupled oscillators.
    d(theta_i)/dt = omega_i + (K/N) * sum_j(A_ij * sin(theta_j - theta_i))

    Features:
    - Learnable natural frequencies (omega) and coupling strength (K)
    - Configurable network topologies: global, small-world, scale-free, ring, learnable
    - Optional learnable adjacency matrix for discovering connectivity patterns
    """

    def __init__(self,
                 num_oscillators: int,
                 dt: float = 0.01,
                 initial_frequency_range: tuple = (0.5, 1.5),
                 initial_phase_range: tuple = (0.0, 2 * math.pi),
                 phase_init: Literal['uniform', 'von_mises'] = 'uniform',
                 phase_init_concentration: float = 1.0,
                 topology: Literal['global', 'small_world', 'scale_free', 'ring', 'learnable', 'modular'] = 'global',
                 topology_params: Optional[dict] = None,
                 integration_method: Literal['euler', 'rk4'] = 'euler',
                 learnable_coupling_matrix: bool = False,
                 frustration: float = 0.0):
        """
        Args:
            num_oscillators: Number of oscillators in the bank.
            dt: Time step for integration.
            initial_frequency_range: Range for random initialization of natural frequencies.
            initial_phase_range: Range for random initialization of phases (uniform mode only).
            phase_init: Phase initialization method:
                - 'uniform': Uniform random in initial_phase_range (default)
                - 'von_mises': Von Mises (circular normal) distribution centered at 0.
                  Better preserves circular statistics than uniform. Concentration
                  controls spread: 0 = uniform, large = tightly clustered.
            phase_init_concentration: Concentration parameter for Von Mises init (kappa).
                Only used when phase_init='von_mises'. Default 1.0 gives moderate spread.
            topology: Network topology for coupling. Options:
                - 'global': All-to-all coupling (original behavior)
                - 'small_world': Watts-Strogatz small-world network
                - 'scale_free': Barabasi-Albert scale-free network
                - 'ring': Ring topology with local coupling
                - 'learnable': Fully learnable adjacency matrix
            topology_params: Parameters for the topology. Depends on topology type:
                - small_world: {'k': neighbors, 'p': rewiring_prob}
                - scale_free: {'m': edges_per_new_node}
                - ring: {'k': neighbors_each_side}
                - learnable: {'sparsity': target_sparsity}
            integration_method: 'euler' (default) or 'rk4' (4th-order Runge-Kutta)
            learnable_coupling_matrix: If True, use per-connection learnable weights
                instead of a single scalar coupling_strength. Required for EP training.
            frustration: Kuramoto-Sakaguchi phase-lag parameter alpha (radians).
                Coupling becomes sin(theta_j - theta_i - alpha). Nonzero alpha
                promotes metastable chimera states in modular networks.
                (Frontiers in Network Physiology 2024; Sakaguchi & Kuramoto 1986)
                Default 0.0 recovers standard Kuramoto.
        """
        super().__init__()
        self.num_oscillators = num_oscillators
        self.dt = dt
        self.topology = topology
        self.integration_method = integration_method
        self.topology_params = topology_params or {}
        self._global_topology = False  # Will be set True for global topology
        self.learnable_coupling_matrix = learnable_coupling_matrix
        self.frustration = frustration  # Kuramoto-Sakaguchi phase-lag α

        # Learnable natural frequencies for each oscillator
        self.natural_frequencies = nn.Parameter(
            torch.rand(num_oscillators) * (initial_frequency_range[1] - initial_frequency_range[0])
            + initial_frequency_range[0]
        )

        # Learnable global coupling strength
        self.coupling_strength = nn.Parameter(torch.tensor(1.0))

        # Oscillator phases (state evolved by Kuramoto dynamics, not backprop)
        if phase_init == 'von_mises':
            # Von Mises (circular normal) — better than uniform for phase init.
            # kappa=0 → uniform, kappa=1 → moderate spread, kappa>>1 → tight cluster.
            vm_dist = torch.distributions.VonMises(
                loc=torch.tensor(0.0),
                concentration=torch.tensor(phase_init_concentration)
            )
            init_phases = vm_dist.sample((num_oscillators,)) % (2 * math.pi)
        else:
            init_phases = (
                torch.rand(num_oscillators) * (initial_phase_range[1] - initial_phase_range[0])
                + initial_phase_range[0]
            )
        self.register_buffer('phases', init_phases)

        # Precision-weighted coupling: stable oscillators drive coupling more
        # Precision = 1 / (phase_velocity_variance + epsilon)
        self.register_buffer('phase_velocity_var',
            torch.ones(num_oscillators))                    # running variance of d_theta/dt
        self.register_buffer('prev_phases',
            self.phases.clone())                            # for computing velocity
        self.register_buffer('precision',
            torch.ones(num_oscillators))                    # current precision per oscillator
        self._precision_tau = 20.0                          # EMA time constant
        self._precision_epsilon = 1e-4                      # variance floor

        # Cached order parameter from forward() — avoids redundant sin/cos
        self._cached_R: Optional[torch.Tensor] = None
        self._cached_Psi: Optional[torch.Tensor] = None

        # Scratch buffers for sparse coupling (avoid per-tick allocation)
        self.register_buffer('_scratch_norm', torch.zeros(num_oscillators))
        self.register_buffer('_scratch_interaction', torch.zeros(num_oscillators))

        # Initialize adjacency matrix based on topology
        self._init_adjacency(topology, self.topology_params)

        # Per-connection learnable coupling weights (for EP training)
        if learnable_coupling_matrix:
            self.coupling_matrix = nn.Parameter(
                torch.ones(num_oscillators, num_oscillators) * 0.1
            )

    def _init_adjacency(self, topology: str, params: dict):
        """Initialize the adjacency matrix based on the specified topology."""
        n = self.num_oscillators

        if topology == 'global':
            # All-to-all coupling - use implicit representation
            # Instead of storing n×n matrix, we'll compute on the fly
            self.register_buffer('adjacency', None)
            self._global_topology = True

        elif topology == 'small_world':
            # Watts-Strogatz small-world network
            k = params.get('k', max(2, n // 10))  # neighbors on each side
            p = params.get('p', 0.1)  # rewiring probability
            adj = self._create_small_world(n, k, p)
            self.register_buffer('adjacency', adj)
            self._build_edge_list(adj)

        elif topology == 'scale_free':
            # Barabasi-Albert scale-free network
            m = params.get('m', max(1, n // 20))  # edges per new node
            adj = self._create_scale_free(n, m)
            self.register_buffer('adjacency', adj)
            self._build_edge_list(adj)

        elif topology == 'ring':
            # Ring topology with local coupling
            k = params.get('k', 2)  # neighbors on each side
            adj = self._create_ring(n, k)
            self.register_buffer('adjacency', adj)
            self._build_edge_list(adj)

        elif topology == 'modular':
            # Hierarchical modular network.
            # Dense within modules (intra_density), sparse between (inter_density).
            # Deco et al. 2017: modularity is the strongest predictor of
            # robust metastability in connectome-derived oscillator models.
            n_modules = params.get('n_modules', 4)
            intra_density = params.get('intra_density', 0.8)
            inter_density = params.get('inter_density', 0.1)
            adj = self._create_modular(n, n_modules, intra_density, inter_density)
            self.register_buffer('adjacency', adj)
            self._build_edge_list(adj)

        elif topology == 'learnable':
            # Fully learnable adjacency (soft, between 0-1)
            # No edge list — adjacency changes each forward pass
            sparsity = params.get('sparsity', 0.5)
            # Initialize with random values, biased toward desired sparsity
            init_values = torch.randn(n, n) * 0.5 - torch.log(torch.tensor(1/sparsity - 1))
            self.adjacency_logits = nn.Parameter(init_values)
            # No self-connections
            self.register_buffer('self_mask', 1.0 - torch.eye(n))

        else:
            raise ValueError(f"Unknown topology: {topology}")

    def _build_edge_list(self, adj: torch.Tensor):
        """Convert dense adjacency to COO edge list for O(E) sparse forward pass.

        Stores:
            _edge_src: (nnz,) int64 — row indices (node receiving influence)
            _edge_dst: (nnz,) int64 — col indices (node exerting influence)
            _edge_weight: (nnz,) float — adjacency values
            _degree: (N,) float — out-degree per row (for normalization)
        """
        src, dst = adj.nonzero(as_tuple=True)
        weights = adj[src, dst]
        self.register_buffer('_edge_src', src.long())
        self.register_buffer('_edge_dst', dst.long())
        self.register_buffer('_edge_weight', weights.float())
        self.register_buffer('_degree', adj.abs().sum(dim=1).clamp(min=1.0))

    def _create_small_world(self, n: int, k: int, p: float) -> torch.Tensor:
        """Create Watts-Strogatz small-world adjacency matrix."""
        # Start with ring lattice
        adj = torch.zeros(n, n)
        for i in range(n):
            for j in range(1, k + 1):
                adj[i, (i + j) % n] = 1
                adj[i, (i - j) % n] = 1

        # Rewiring with probability p
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] == 1 and torch.rand(1).item() < p:
                    # Rewire to random node
                    candidates = [x for x in range(n) if x != i and adj[i, x] == 0]
                    if candidates:
                        new_j = candidates[torch.randint(len(candidates), (1,)).item()]
                        adj[i, j] = 0
                        adj[j, i] = 0
                        adj[i, new_j] = 1
                        adj[new_j, i] = 1

        return adj

    def _create_scale_free(self, n: int, m: int) -> torch.Tensor:
        """Create Barabasi-Albert scale-free adjacency matrix."""
        adj = torch.zeros(n, n)

        # Start with a small complete graph
        m0 = max(m, 2)
        for i in range(m0):
            for j in range(i + 1, m0):
                adj[i, j] = 1
                adj[j, i] = 1

        # Add nodes with preferential attachment
        for i in range(m0, n):
            # Calculate degree of existing nodes
            degrees = adj[:i, :i].sum(dim=1) + 1  # +1 to avoid zero probability
            probs = degrees / degrees.sum()

            # Select m nodes to connect to (preferential attachment)
            selected = torch.multinomial(probs, min(m, i), replacement=False)
            for j in selected:
                adj[i, j] = 1
                adj[j, i] = 1

        return adj

    def _create_ring(self, n: int, k: int) -> torch.Tensor:
        """Create ring topology with k neighbors on each side."""
        adj = torch.zeros(n, n)
        for i in range(n):
            for j in range(1, k + 1):
                adj[i, (i + j) % n] = 1
                adj[i, (i - j) % n] = 1
        return adj

    def _create_modular(self, n: int, n_modules: int,
                         intra_density: float, inter_density: float) -> torch.Tensor:
        """Create hierarchical modular network.

        Dense within modules (intra_density), sparse between (inter_density).
        Deco et al. 2017: modularity → maximal metastability.

        Args:
            n: Number of oscillators
            n_modules: Number of modules
            intra_density: Connection probability within modules (0-1)
            inter_density: Connection probability between modules (0-1)

        Returns:
            Symmetric adjacency matrix (n, n)
        """
        adj = torch.zeros(n, n)
        module_size = n // n_modules
        remainder = n % n_modules

        # Compute module boundaries (distribute remainder evenly)
        boundaries = []
        start = 0
        for m in range(n_modules):
            end = start + module_size + (1 if m < remainder else 0)
            boundaries.append((start, end))
            start = end

        # Vectorized intra-module connections
        for s, e in boundaries:
            if e - s < 2:
                continue
            block = torch.rand(e - s, e - s)
            mask = (block < intra_density).float()
            # Zero diagonal (no self-connections)
            mask.fill_diagonal_(0.0)
            # Symmetrize
            mask = (mask + mask.T).clamp(max=1.0)
            adj[s:e, s:e] = mask

        # Vectorized inter-module connections
        for i, (s1, e1) in enumerate(boundaries):
            for j, (s2, e2) in enumerate(boundaries):
                if i >= j:
                    continue
                block = torch.rand(e1 - s1, e2 - s2)
                mask = (block < inter_density).float()
                adj[s1:e1, s2:e2] = mask
                adj[s2:e2, s1:e1] = mask.T  # Symmetrize

        # Store module boundaries for per-module order parameter computation
        self._module_boundaries = boundaries

        return adj

    def get_module_order_parameters(self) -> Optional[torch.Tensor]:
        """Compute per-module order parameters for modular topology.

        Each R_m = |mean(exp(i * phase_m))| measures within-module synchrony.
        Differing R_m values across modules indicate chimera states
        (Crowe et al. 2024, Frontiers in Network Physiology).

        Returns:
            Tensor of shape (n_modules,) with R values per module,
            or None if topology is not modular.
        """
        if self.topology != 'modular' or not hasattr(self, '_module_boundaries'):
            return None
        Rs = []
        for start, end in self._module_boundaries:
            module_phases = self.phases[start:end]
            z = torch.exp(1j * module_phases.to(torch.complex64))
            Rs.append(torch.abs(z.mean()))
        return torch.stack(Rs)

    def set_frequency_bands(self, bands: list, dt: float):
        """Assign frequency bands to oscillator subgroups.

        Converts Hz → rad/s for the Kuramoto equation. The forward pass
        applies ``* self.dt`` to convert rad/s → rad/step, so we store
        in rad/s here (NOT pre-multiplied by dt).

        Args:
            bands: List of (count, freq_lo_hz, freq_hi_hz) tuples.
                   Sum of counts must equal num_oscillators.
            dt: Integration timestep (unused — kept for API compat).
        """
        total = sum(b[0] for b in bands)
        if total != self.num_oscillators:
            raise ValueError(
                f"Band counts sum to {total}, expected {self.num_oscillators}"
            )
        with torch.no_grad():
            idx = 0
            for count, lo_hz, hi_hz in bands:
                # Uniform random frequencies in Hz, converted to rad/s
                # (forward pass multiplies by self.dt to get rad/step)
                freqs_hz = torch.rand(count) * (hi_hz - lo_hz) + lo_hz
                omega = freqs_hz * 2 * math.pi
                self.natural_frequencies.data[idx:idx + count] = omega
                idx += count

    def _sparse_coupling(self, ph: torch.Tensor, use_precision: bool) -> torch.Tensor:
        """Compute Kuramoto coupling via COO edge list — O(E) instead of O(N^2).

        Args:
            ph: Current phases (N,)
            use_precision: Whether to weight by source precision

        Returns:
            Normalized interaction per oscillator (N,)
        """
        src = self._edge_src        # (nnz,) — node receiving influence
        dst = self._edge_dst        # (nnz,) — node exerting influence
        w = self._edge_weight       # (nnz,) — adjacency values

        # sin(theta_src - theta_dst - alpha) — Kuramoto-Sakaguchi coupling
        sin_diffs = torch.sin(ph[src] - ph[dst] - self.frustration)

        if use_precision:
            # Weight by influencing neighbor's precision
            prec = self.precision.detach()
            ew = w * prec[dst]
            weighted = ew * sin_diffs
            # Per-row normalization: sum of precision-weighted edges
            norm = self._scratch_norm.zero_()
            norm.scatter_add_(0, src, ew)
            interaction = self._scratch_interaction.zero_()
            interaction.scatter_add_(0, src, weighted)
            return interaction / norm.clamp(min=1e-8)
        else:
            weighted = w * sin_diffs
            interaction = self._scratch_interaction.zero_()
            interaction.scatter_add_(0, src, weighted)
            return interaction / self._degree

    # ── Phase buffers must stay fp32 for numerical stability ──
    # bfloat16 has only 8 mantissa bits → modular arithmetic near
    # 2*pi accumulates ~10% drift.  Override _apply so that device
    # changes (.cuda()) work normally but dtype casts leave phase
    # accumulators in fp32.
    _FP32_BUFFERS = frozenset({
        'phases', 'prev_phases', 'phase_velocity_var', 'precision',
    })

    def _apply(self, fn):
        """Preserve fp32 for phase-accumulator buffers."""
        super()._apply(fn)
        for name in self._FP32_BUFFERS:
            buf = self._buffers.get(name)
            if buf is not None and buf.dtype != torch.float32:
                self._buffers[name] = buf.float()
        return self

    def get_adjacency(self) -> Optional[torch.Tensor]:
        """Get the current adjacency matrix (handles learnable case)."""
        if self.topology == 'learnable':
            # Apply sigmoid to get soft adjacency, mask out self-connections
            return torch.sigmoid(self.adjacency_logits) * self.self_mask
        elif hasattr(self, '_global_topology') and self._global_topology:
            # Global topology computed on the fly, no stored matrix
            return None
        else:
            return self.adjacency

    def forward(self, external_input: torch.Tensor = None, steps: int = 1,
                use_precision: bool = True) -> torch.Tensor:
        """
        Updates the phases of the oscillators for a given number of steps.

        Coupling is precision-weighted: stable oscillators (low phase-velocity
        variance) contribute more to coupling, while noisy ones are downweighted.

        Args:
            external_input: External driving force, shape (num_oscillators,).
            steps: Number of simulation steps to perform.
            use_precision: Whether to weight coupling by source precision.

        Returns:
            torch.Tensor: The updated phases of the oscillators.
        """
        phases = self.phases
        adj = self.get_adjacency()
        # Invalidate cached order parameter — will be recomputed during step
        self._cached_R = None
        self._cached_Psi = None

        if external_input is not None:
            if external_input.shape[-1] != self.num_oscillators:
                raise ValueError(f"External input last dimension {external_input.shape[-1]} "
                                 f"must match num_oscillators {self.num_oscillators}")

        _alpha = self.frustration
        _has_frustration = _alpha != 0.0

        for _ in range(steps):
            if adj is None:
                # Global topology: O(n) computation
                sin_phases = torch.sin(phases)
                cos_phases = torch.cos(phases)

                if use_precision:
                    # Weight each oscillator's contribution by its precision
                    prec = self.precision.detach()
                    w_sin = (sin_phases * prec).sum()
                    w_cos = (cos_phases * prec).sum()
                    # Subtract self-contribution: sum_j sin(θ_j - θ_i)
                    sin_interaction = (
                        cos_phases * (w_sin - sin_phases * prec)
                        - sin_phases * (w_cos - cos_phases * prec)
                    )
                    if _has_frustration:
                        # Kuramoto-Sakaguchi: sin(θ_j - θ_i - α) =
                        #   cos(α)*sin(θ_j-θ_i) - sin(α)*cos(θ_j-θ_i)
                        cos_interaction = (
                            sin_phases * (w_sin - sin_phases * prec)
                            + cos_phases * (w_cos - cos_phases * prec)
                        )
                        ca = math.cos(_alpha)
                        sa = math.sin(_alpha)
                        interaction_sum = ca * sin_interaction - sa * cos_interaction
                    else:
                        interaction_sum = sin_interaction
                    # Normalize by sum of precisions (excluding self)
                    prec_sum = prec.sum() - prec
                    normalized_interaction = interaction_sum / prec_sum.clamp(min=1e-8)
                else:
                    sum_sin = sin_phases.sum()
                    sum_cos = cos_phases.sum()
                    sin_interaction = cos_phases * (sum_sin - sin_phases) - sin_phases * (sum_cos - cos_phases)
                    if _has_frustration:
                        cos_interaction = sin_phases * (sum_sin - sin_phases) + cos_phases * (sum_cos - cos_phases)
                        ca = math.cos(_alpha)
                        sa = math.sin(_alpha)
                        interaction_sum = ca * sin_interaction - sa * cos_interaction
                    else:
                        interaction_sum = sin_interaction
                    normalized_interaction = interaction_sum / max(self.num_oscillators - 1, 1)
            elif hasattr(self, '_edge_src'):
                # Sparse COO path: O(E) instead of O(N²)
                normalized_interaction = self._sparse_coupling(
                    phases, use_precision)
            else:
                # Dense fallback (learnable topology — changes each step)
                phase_diffs = phases.unsqueeze(1) - phases.unsqueeze(0)  # (n, n)
                sin_pd = torch.sin(phase_diffs - _alpha)  # Kuramoto-Sakaguchi
                effective_adj = adj
                if self.learnable_coupling_matrix:
                    effective_adj = adj * self.coupling_matrix
                if use_precision:
                    effective_adj = effective_adj * self.precision.detach().unsqueeze(0)
                    weighted_interaction = effective_adj * sin_pd
                    interaction_sum = torch.sum(weighted_interaction, dim=1)
                    weight_sums = effective_adj.sum(dim=1).clamp(min=1e-8)
                    normalized_interaction = interaction_sum / weight_sums
                else:
                    weighted_interaction = effective_adj * sin_pd
                    interaction_sum = torch.sum(weighted_interaction, dim=1)
                    connection_counts = effective_adj.abs().sum(dim=1).clamp(min=1)
                    normalized_interaction = interaction_sum / connection_counts

            # Kuramoto dynamics equation
            if self.integration_method == 'rk4':
                # Build RHS closure capturing current coupling state
                _ext = external_input
                _has_edges = hasattr(self, '_edge_src')

                def _kuramoto_rhs(ph):
                    # Recompute coupling for this intermediate state
                    if adj is None:
                        sp = torch.sin(ph)
                        cp = torch.cos(ph)
                        if use_precision:
                            pr = self.precision.detach()
                            ws = (sp * pr).sum()
                            wc = (cp * pr).sum()
                            si = cp * (ws - sp * pr) - sp * (wc - cp * pr)
                            if _has_frustration:
                                ci = sp * (ws - sp * pr) + cp * (wc - cp * pr)
                                si = math.cos(_alpha) * si - math.sin(_alpha) * ci
                            ps = pr.sum() - pr
                            ni = si / ps.clamp(min=1e-8)
                        else:
                            ss = sp.sum()
                            sc = cp.sum()
                            si = cp * (ss - sp) - sp * (sc - cp)
                            if _has_frustration:
                                ci = sp * (ss - sp) + cp * (sc - cp)
                                si = math.cos(_alpha) * si - math.sin(_alpha) * ci
                            ni = si / (self.num_oscillators - 1)
                    elif _has_edges:
                        ni = self._sparse_coupling(ph, use_precision)
                    else:
                        pd = ph.unsqueeze(1) - ph.unsqueeze(0)
                        spd = torch.sin(pd - _alpha)  # Kuramoto-Sakaguchi
                        ea = adj
                        if self.learnable_coupling_matrix:
                            ea = ea * self.coupling_matrix
                        if use_precision:
                            ea = ea * self.precision.detach().unsqueeze(0)
                            wi = ea * spd
                            is_ = torch.sum(wi, dim=1)
                            ws = ea.sum(dim=1).clamp(min=1e-8)
                            ni = is_ / ws
                        else:
                            wi = ea * spd
                            is_ = torch.sum(wi, dim=1)
                            cc = ea.abs().sum(dim=1).clamp(min=1)
                            ni = is_ / cc
                    rhs = self.natural_frequencies + self.coupling_strength * ni
                    if _ext is not None:
                        rhs = rhs + _ext
                    return rhs

                phases = _rk4_step(_kuramoto_rhs, phases, self.dt) % (2 * math.pi)
            else:
                d_theta_dt = self.natural_frequencies + self.coupling_strength * normalized_interaction
                if external_input is not None:
                    d_theta_dt = d_theta_dt + external_input
                phases = (phases + d_theta_dt * self.dt) % (2 * math.pi)

        # ── Fused order parameter: reuse final phases, avoid redundant sin/cos ──
        # R*e^(iPsi) = (1/N) * sum(e^(i*theta))
        _N = float(self.num_oscillators)
        _sin_final = torch.sin(phases)
        _cos_final = torch.cos(phases)
        _mean_sin = _sin_final.sum() / _N
        _mean_cos = _cos_final.sum() / _N
        self._cached_R = torch.sqrt(_mean_sin * _mean_sin + _mean_cos * _mean_cos)
        self._cached_Psi = torch.atan2(_mean_sin, _mean_cos)

        # Update precision from phase velocity variance
        with torch.no_grad():
            # Phase velocity = angular distance from previous phase
            phase_diff = phases.detach() - self.prev_phases
            # Wrap to [-pi, pi]
            phase_velocity = (phase_diff + math.pi) % (2 * math.pi) - math.pi

            # Update running variance via EMA
            alpha = 1.0 / self._precision_tau
            instant_var = phase_velocity.pow(2)
            # Mean over batch dim if present (buffers are 1-D)
            if instant_var.dim() > 1:
                instant_var = instant_var.mean(0)
            self.phase_velocity_var.mul_(1 - alpha).add_(instant_var * alpha)

            # Precision = 1 / (variance + epsilon)
            self.precision.copy_(
                (1.0 / (self.phase_velocity_var + self._precision_epsilon)).clamp(max=100.0)
            )
            # Squeeze batch dim for 1-D buffers
            _phases_1d = phases.detach().mean(0) if phases.dim() > 1 else phases.detach()
            self.prev_phases.copy_(_phases_1d)

        # Update internal state — keep 1-D buffer shape
        with torch.no_grad():
            _store = phases.detach().mean(0) if phases.dim() > 1 else phases.detach()
            self.phases.copy_(_store)
        return phases

    def compute_attractor_stability(
        self,
        perturbation_scale: float = 0.1,
        n_trials: int = 10,
        recovery_steps: int = 20,
        external_input: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Measure attractor basin stability via perturbation analysis.

        Saves current phases, perturbs them, runs recovery_steps,
        measures final R vs original R, restores original phases.

        Args:
            perturbation_scale: Gaussian noise scale (fraction of 2*pi)
            n_trials: Number of perturbation trials
            recovery_steps: Steps to run after each perturbation
            external_input: Optional external driving force

        Returns:
            stability_index: 0-1 (1 = perfectly stable attractor)
            escape_probability: fraction of trials where R dropped >50%
            mean_recovery_R: average R after recovery
            R_variance: variance of recovered R across trials
        """
        with torch.no_grad():
            # Save current state
            original_phases = self.phases.clone()
            original_prev = self.prev_phases.clone()
            original_var = self.phase_velocity_var.clone()
            original_precision = self.precision.clone()
            original_R = self.get_order_parameter().item()

            recovered_Rs = []
            escapes = 0

            for _ in range(n_trials):
                # Perturb phases
                noise = torch.randn_like(self.phases) * perturbation_scale * 2 * math.pi
                self.phases.copy_((original_phases + noise) % (2 * math.pi))
                self.prev_phases.copy_(self.phases)

                # Run recovery
                for __ in range(recovery_steps):
                    self.forward(external_input=external_input, steps=1)

                trial_R = self.get_order_parameter().item()
                recovered_Rs.append(trial_R)

                if original_R > 1e-6 and trial_R < original_R * 0.5:
                    escapes += 1

                # Restore for next trial
                self.phases.copy_(original_phases)
                self.prev_phases.copy_(original_prev)
                self.phase_velocity_var.copy_(original_var)
                self.precision.copy_(original_precision)

            Rs = torch.tensor(recovered_Rs)
            mean_R = Rs.mean()
            R_var = Rs.var()
            escape_prob = escapes / n_trials

            stability_index = torch.clamp(
                1.0 - R_var - torch.tensor(escape_prob), min=0.0, max=1.0
            )

        return {
            'stability_index': stability_index,
            'escape_probability': torch.tensor(escape_prob),
            'mean_recovery_R': mean_R,
            'R_variance': R_var,
        }

    def compute_lyapunov_exponent(
        self,
        n_steps: int = 500,
        perturbation_size: float = 1e-6,
        renorm_interval: int = 10,
        external_input: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Estimate the largest Lyapunov exponent via dual-trajectory perturbation.

        Runs two copies of the oscillator dynamics — primary (unperturbed) and
        secondary (slightly perturbed). Periodically measures divergence rate
        and renormalizes the secondary trajectory. Positive λ indicates chaos,
        zero indicates edge-of-chaos (criticality), negative indicates stability.

        Adapted from multilayer-cudamoto's Lyapunov measurement approach.

        Args:
            n_steps: Total integration steps for measurement.
            perturbation_size: Initial perturbation magnitude (radians).
            renorm_interval: Steps between renormalization events.
            external_input: Optional external driving force.

        Returns:
            lyapunov_exponent: Estimated largest Lyapunov exponent (λ).
            trajectory_R: Order parameter trajectory of primary system.
            regime: 'chaotic' (λ>0.01), 'critical' (|λ|<0.01), or 'stable' (λ<-0.01).
        """
        with torch.no_grad():
            # Save current state
            original_phases = self.phases.clone()
            original_prev = self.prev_phases.clone()
            original_var = self.phase_velocity_var.clone()
            original_prec = self.precision.clone()

            # Primary trajectory starts from current state
            primary = self.phases.clone()

            # Secondary trajectory = primary + small perturbation
            perturbation = torch.randn_like(primary) * perturbation_size
            secondary = (primary + perturbation) % (2 * math.pi)

            log_divergence_sum = 0.0
            n_renorms = 0
            R_trajectory = []

            for step in range(n_steps):
                # Evolve primary
                self.phases.copy_(primary)
                self.prev_phases.copy_(primary)
                self.forward(external_input=external_input, steps=1)
                primary = self.phases.clone()

                # Record order parameter
                R_trajectory.append(self.get_order_parameter().item())

                # Evolve secondary
                self.phases.copy_(secondary)
                self.prev_phases.copy_(secondary)
                self.forward(external_input=external_input, steps=1)
                secondary = self.phases.clone()

                # Renormalize at intervals
                if (step + 1) % renorm_interval == 0:
                    # Phase difference on the circle (wrap to [-π, π])
                    diff = (secondary - primary + math.pi) % (2 * math.pi) - math.pi
                    dist = diff.norm()

                    if dist > 1e-12:
                        log_divergence_sum += torch.log(dist / perturbation_size).item()
                        n_renorms += 1

                        # Renormalize: keep direction, reset magnitude
                        secondary = (primary + diff / dist * perturbation_size) % (2 * math.pi)

            # Restore original state
            self.phases.copy_(original_phases)
            self.prev_phases.copy_(original_prev)
            self.phase_velocity_var.copy_(original_var)
            self.precision.copy_(original_prec)

            # Compute Lyapunov exponent
            if n_renorms > 0:
                lyapunov = log_divergence_sum / (n_renorms * renorm_interval * self.dt)
            else:
                lyapunov = 0.0

            lyapunov_t = torch.tensor(lyapunov)

            if lyapunov > 0.01:
                regime = 'chaotic'
            elif lyapunov < -0.01:
                regime = 'stable'
            else:
                regime = 'critical'

        return {
            'lyapunov_exponent': lyapunov_t,
            'trajectory_R': torch.tensor(R_trajectory),
            'regime': regime,
        }

    def get_order_parameter(self, phases: torch.Tensor = None) -> torch.Tensor:
        """
        Calculates the global order parameter R for the oscillator bank.
        R = |(1/N) * sum(e^(i*theta_j))|

        Uses cached value from forward() when available (avoids redundant sin/cos).
        """
        if phases is None and self._cached_R is not None:
            return self._cached_R
        if phases is None:
            phases = self.phases

        complex_phases = torch.exp(1j * phases)
        mean_complex_phase = torch.mean(complex_phases)
        return torch.abs(mean_complex_phase)

    def get_local_order_parameters(self, phases: torch.Tensor = None) -> torch.Tensor:
        """
        Calculate local order parameters for each oscillator based on its neighbors.
        Useful for detecting local synchronization clusters.
        """
        if phases is None:
            phases = self.phases

        adj = self.get_adjacency()
        complex_phases = torch.exp(1j * phases)

        if adj is None:
            # Global topology: every oscillator sees all others
            # Local order parameter = global order parameter for all
            global_mean = complex_phases.mean()
            return torch.abs(global_mean).expand(self.num_oscillators)

        # Weighted average of neighbors' complex phases
        if hasattr(self, '_edge_src'):
            # Sparse COO path
            neighbor_sum = torch.zeros(
                self.num_oscillators, dtype=complex_phases.dtype,
                device=phases.device)
            neighbor_sum.scatter_add_(
                0, self._edge_src.to(torch.int64),
                self._edge_weight.to(complex_phases.dtype) * complex_phases[self._edge_dst])
            local_mean = neighbor_sum / self._degree.to(complex_phases.dtype)
        else:
            neighbor_sum = torch.matmul(adj.to(complex_phases.dtype), complex_phases)
            neighbor_count = adj.sum(dim=1).clamp(min=1)
            local_mean = neighbor_sum / neighbor_count

        return torch.abs(local_mean)


class SensoryCoupling(nn.Module):
    """Couples sensory input to oscillator dynamics.

    Implements three biologically-motivated coupling mechanisms:

    1. **Frequency entrainment**: Input energy modulates natural frequencies.
       Arousing stimuli speed oscillators; quiet input lets intrinsic rhythm
       dominate.  (Lakatos et al. 2008, Science)

    2. **Phase reset**: Novel stimuli trigger partial phase resets toward
       input-derived target phases (event-related desynchronization /
       resynchronization).  (Makeig et al. 2002, Science)

    3. **Coupling modulation**: Input salience scales the coupling strength K.
       Strong stimuli increase K (attentional gain), weak stimuli reduce it.
       (Fries 2005 Trends Cogn Sci — CTC hypothesis)

    All projections are **fixed random** (no backprop from LLM), respecting the
    design invariant: oscillators evolve through own dynamics, gradient flow from
    the language model is severed by design.
    """

    def __init__(
        self,
        input_dim: int,
        num_oscillators: int,
        entrainment_strength: float = 0.05,
        phase_reset_threshold: float = 1.5,
        coupling_mod_range: tuple = (0.9, 1.1),
        novelty_ema_alpha: float = 0.1,
    ):
        super().__init__()
        self.num_oscillators = num_oscillators
        self.entrainment_strength = entrainment_strength
        self.phase_reset_threshold = phase_reset_threshold
        self.coupling_mod_lo, self.coupling_mod_hi = coupling_mod_range
        self._ema_alpha = novelty_ema_alpha

        # Fixed random projections (NOT learned via backprop)
        freq_proj = torch.randn(num_oscillators, input_dim) / (input_dim ** 0.5)
        self.register_buffer('freq_projection', freq_proj)

        phase_proj = torch.randn(num_oscillators, input_dim) / (input_dim ** 0.5)
        self.register_buffer('phase_projection', phase_proj)

        # Running statistics for novelty detection
        self.register_buffer('input_ema', torch.zeros(input_dim))
        self.register_buffer('input_var_ema', torch.ones(1))
        self.register_buffer('prev_input_norm', torch.zeros(1))

    @torch.no_grad()
    def forward(
        self,
        sensory_input: torch.Tensor,
        current_phases: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute coupling signals from sensory input.

        Args:
            sensory_input: (state_dim,) or (1, state_dim) sensory vector
            current_phases: (num_oscillators,) current oscillator phases

        Returns:
            Dict with:
            - freq_modulation: (num_oscillators,) additive frequency shift
            - reset_strength: scalar 0-1 how strongly to reset phases
            - phase_targets: (num_oscillators,) target phases for reset
            - coupling_scale: scalar multiplicative K adjustment
            - input_novelty: scalar novelty measure (for diagnostics)
        """
        x = sensory_input.detach().flatten()
        in_dim = self.freq_projection.shape[1]
        if x.shape[0] > in_dim:
            x = x[:in_dim]
        elif x.shape[0] < in_dim:
            x = torch.nn.functional.pad(x, (0, in_dim - x.shape[0]))

        # ── 1. Frequency entrainment ──
        # Project input to oscillator space, scale by entrainment strength.
        # Positive projection = speed up, negative = slow down.
        freq_mod = (self.freq_projection @ x) * self.entrainment_strength

        # ── 2. Novelty detection → phase reset ──
        # Compare current input to running EMA (habituation).
        diff = x - self.input_ema
        novelty = diff.norm() / (self.input_var_ema.sqrt() + 1e-8)

        # Update running statistics
        self.input_ema.mul_(1 - self._ema_alpha).add_(x * self._ema_alpha)
        var_instant = diff.pow(2).mean()
        self.input_var_ema.mul_(1 - self._ema_alpha).add_(
            var_instant * self._ema_alpha
        )
        self.prev_input_norm.copy_(x.norm().unsqueeze(0) if x.norm().dim() == 0 else x.norm())

        # Reset strength: sigmoid around threshold
        reset_strength = torch.sigmoid(
            (novelty - self.phase_reset_threshold) * 3.0
        )

        # Target phases derived from input content
        phase_targets = (self.phase_projection @ x) % (2 * math.pi)

        # ── 3. Coupling modulation from input energy ──
        # Normalized energy: ~1.0 for typical input, >1 for strong
        energy = x.norm() / (in_dim ** 0.5 + 1e-8)
        coupling_scale = self.coupling_mod_lo + (
            self.coupling_mod_hi - self.coupling_mod_lo
        ) * torch.sigmoid(energy - 1.0)

        return {
            'freq_modulation': freq_mod,
            'reset_strength': reset_strength,
            'phase_targets': phase_targets,
            'coupling_scale': coupling_scale,
            'input_novelty': novelty,
        }
