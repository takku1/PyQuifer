import torch
import torch.nn as nn
import math
from typing import Dict, List, Literal, Optional


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
                 topology: Literal['global', 'small_world', 'scale_free', 'ring', 'learnable'] = 'global',
                 topology_params: Optional[dict] = None,
                 integration_method: Literal['euler', 'rk4'] = 'euler',
                 learnable_coupling_matrix: bool = False):
        """
        Args:
            num_oscillators: Number of oscillators in the bank.
            dt: Time step for integration.
            initial_frequency_range: Range for random initialization of natural frequencies.
            initial_phase_range: Range for random initialization of phases.
            topology: Network topology for coupling. Options:
                - 'global': All-to-all coupling (original behavior)
                - 'small_world': Watts-Strogatz small-world network
                - 'scale_free': Barabási-Albert scale-free network
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
        """
        super().__init__()
        self.num_oscillators = num_oscillators
        self.dt = dt
        self.topology = topology
        self.integration_method = integration_method
        self.topology_params = topology_params or {}
        self._global_topology = False  # Will be set True for global topology
        self.learnable_coupling_matrix = learnable_coupling_matrix

        # Learnable natural frequencies for each oscillator
        self.natural_frequencies = nn.Parameter(
            torch.rand(num_oscillators) * (initial_frequency_range[1] - initial_frequency_range[0])
            + initial_frequency_range[0]
        )

        # Learnable global coupling strength
        self.coupling_strength = nn.Parameter(torch.tensor(1.0))

        # Oscillator phases (state evolved by Kuramoto dynamics, not backprop)
        self.register_buffer('phases',
            torch.rand(num_oscillators) * (initial_phase_range[1] - initial_phase_range[0])
            + initial_phase_range[0]
        )

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
            # Barabási-Albert scale-free network
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
        """Create Barabási-Albert scale-free adjacency matrix."""
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

    def _sparse_coupling(self, ph: torch.Tensor, use_precision: bool) -> torch.Tensor:
        """Compute Kuramoto coupling via COO edge list — O(E) instead of O(N²).

        Args:
            ph: Current phases (N,)
            use_precision: Whether to weight by source precision

        Returns:
            Normalized interaction per oscillator (N,)
        """
        src = self._edge_src        # (nnz,) — node receiving influence
        dst = self._edge_dst        # (nnz,) — node exerting influence
        w = self._edge_weight       # (nnz,) — adjacency values

        # sin(theta_src - theta_dst) — matches dense convention: phases[i] - phases[j]
        sin_diffs = torch.sin(ph[src] - ph[dst])

        if use_precision:
            # Weight by influencing neighbor's precision
            prec = self.precision.detach()
            ew = w * prec[dst]
            weighted = ew * sin_diffs
            # Per-row normalization: sum of precision-weighted edges
            norm = torch.zeros(self.num_oscillators, device=ph.device)
            norm.scatter_add_(0, src, ew)
            interaction = torch.zeros(self.num_oscillators, device=ph.device)
            interaction.scatter_add_(0, src, weighted)
            return interaction / norm.clamp(min=1e-8)
        else:
            weighted = w * sin_diffs
            interaction = torch.zeros(self.num_oscillators, device=ph.device)
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
                    # Subtract self-contribution
                    interaction_sum = (
                        cos_phases * (w_sin - sin_phases * prec)
                        - sin_phases * (w_cos - cos_phases * prec)
                    )
                    # Normalize by sum of precisions (excluding self)
                    prec_sum = prec.sum() - prec
                    normalized_interaction = interaction_sum / prec_sum.clamp(min=1e-8)
                else:
                    sum_sin = sin_phases.sum()
                    sum_cos = cos_phases.sum()
                    interaction_sum = cos_phases * (sum_sin - sin_phases) - sin_phases * (sum_cos - cos_phases)
                    normalized_interaction = interaction_sum / max(self.num_oscillators - 1, 1)
            elif hasattr(self, '_edge_src'):
                # Sparse COO path: O(E) instead of O(N²)
                normalized_interaction = self._sparse_coupling(
                    phases, use_precision)
            else:
                # Dense fallback (learnable topology — changes each step)
                phase_diffs = phases.unsqueeze(1) - phases.unsqueeze(0)  # (n, n)
                effective_adj = adj
                if self.learnable_coupling_matrix:
                    effective_adj = adj * self.coupling_matrix
                if use_precision:
                    effective_adj = effective_adj * self.precision.detach().unsqueeze(0)
                    weighted_interaction = effective_adj * torch.sin(phase_diffs)
                    interaction_sum = torch.sum(weighted_interaction, dim=1)
                    weight_sums = effective_adj.sum(dim=1).clamp(min=1e-8)
                    normalized_interaction = interaction_sum / weight_sums
                else:
                    weighted_interaction = effective_adj * torch.sin(phase_diffs)
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
                            ints = cp * (ws - sp * pr) - sp * (wc - cp * pr)
                            ps = pr.sum() - pr
                            ni = ints / ps.clamp(min=1e-8)
                        else:
                            ss = sp.sum()
                            sc = cp.sum()
                            ints = cp * (ss - sp) - sp * (sc - cp)
                            ni = ints / (self.num_oscillators - 1)
                    elif _has_edges:
                        ni = self._sparse_coupling(ph, use_precision)
                    else:
                        pd = ph.unsqueeze(1) - ph.unsqueeze(0)
                        ea = adj
                        if self.learnable_coupling_matrix:
                            ea = ea * self.coupling_matrix
                        if use_precision:
                            ea = ea * self.precision.detach().unsqueeze(0)
                            wi = ea * torch.sin(pd)
                            is_ = torch.sum(wi, dim=1)
                            ws = ea.sum(dim=1).clamp(min=1e-8)
                            ni = is_ / ws
                        else:
                            wi = ea * torch.sin(pd)
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


class StuartLandauOscillator(nn.Module):
    """
    Stuart-Landau oscillator: amplitude + phase dynamics with Hopf bifurcation.

    Unlike Kuramoto (phase-only), Stuart-Landau tracks both amplitude and phase
    via complex state z:

        dz/dt = (mu + i*omega) * z - |z|^2 * z + coupling * (z_mean - z)

    Where:
    - mu > 0: limit cycle (oscillating)
    - mu < 0: fixed point (damped)
    - mu = 0: Hopf bifurcation (criticality!)

    The amplitude dynamics make this a natural fit for criticality control:
    |mu| = distance from bifurcation.

    Args:
        num_oscillators: Number of oscillators
        mu: Bifurcation parameter (>0 oscillating, <0 damped, =0 critical)
        omega_range: Range for natural frequencies
        coupling: Global coupling strength
        dt: Integration timestep
    """

    def __init__(self,
                 num_oscillators: int,
                 mu: float = 0.1,
                 omega_range: tuple = (0.5, 1.5),
                 coupling: float = 1.0,
                 dt: float = 0.01,
                 integration_method: Literal['euler', 'rk4'] = 'euler'):
        super().__init__()
        self.num_oscillators = num_oscillators
        self.dt = dt
        self.integration_method = integration_method

        # Bifurcation parameter (adapted by dynamics, not backprop)
        self.register_buffer('mu', torch.tensor(mu))

        # Natural frequencies
        self.omega = nn.Parameter(
            torch.rand(num_oscillators) * (omega_range[1] - omega_range[0]) + omega_range[0]
        )

        # Coupling strength
        self.coupling = nn.Parameter(torch.tensor(coupling))

        # Complex state z = r * exp(i * theta)
        # Initialize with small random amplitudes
        r_init = torch.rand(num_oscillators) * 0.3 + 0.1
        theta_init = torch.rand(num_oscillators) * 2 * math.pi
        self.register_buffer('z_real', r_init * torch.cos(theta_init))
        self.register_buffer('z_imag', r_init * torch.sin(theta_init))

        self.register_buffer('step_count', torch.tensor(0))

    @property
    def z(self) -> torch.Tensor:
        """Complex oscillator states."""
        return torch.complex(self.z_real, self.z_imag)

    @property
    def amplitudes(self) -> torch.Tensor:
        """Current oscillator amplitudes."""
        return torch.abs(self.z)

    @property
    def phases(self) -> torch.Tensor:
        """Current oscillator phases."""
        return torch.angle(self.z)

    def forward(self, steps: int = 1,
                external_input: Optional[torch.Tensor] = None) -> Dict:
        """
        Evolve Stuart-Landau dynamics.

        Args:
            steps: Number of integration steps
            external_input: Optional complex driving input (num_oscillators,)

        Returns:
            Dict with amplitudes, phases, order_parameter, z
        """
        z = self.z

        for _ in range(steps):
            if self.integration_method == 'rk4':
                _ext = external_input

                def _sl_rhs(zz):
                    zz_abs_sq = zz.real ** 2 + zz.imag ** 2
                    mu_iw = torch.complex(
                        self.mu.expand(self.num_oscillators),
                        self.omega
                    )
                    nl = zz_abs_sq * zz
                    zm = zz.mean()
                    ct = self.coupling * (zm - zz)
                    rhs = mu_iw * zz - nl + ct
                    if _ext is not None:
                        rhs = rhs + _ext
                    return rhs

                z = _rk4_step(_sl_rhs, z, self.dt)
            else:
                z_abs_sq = z.real ** 2 + z.imag ** 2
                mu_plus_iw = torch.complex(
                    self.mu.expand(self.num_oscillators),
                    self.omega
                )
                nonlinear = z_abs_sq * z
                z_mean = z.mean()
                coupling_term = self.coupling * (z_mean - z)
                dz = mu_plus_iw * z - nonlinear + coupling_term
                if external_input is not None:
                    dz = dz + external_input
                z = z + dz * self.dt

        # Update stored state
        with torch.no_grad():
            self.z_real.copy_(z.real.detach())
            self.z_imag.copy_(z.imag.detach())
            self.step_count.add_(steps)

        amplitudes = torch.abs(z)
        phases = torch.angle(z)
        order_param = torch.abs(z.mean())

        return {
            'amplitudes': amplitudes,
            'phases': phases,
            'order_parameter': order_param,
            'z': z,
            'mean_amplitude': amplitudes.mean(),
        }

    def get_order_parameter(self) -> torch.Tensor:
        """Global synchronization level."""
        return torch.abs(self.z.mean())

    def get_critical_coupling(self) -> torch.Tensor:
        """Estimate critical coupling Kc from frequency spread.

        For globally coupled Stuart-Landau oscillators, synchronization
        onset requires coupling to exceed the natural frequency spread.
        Kc ~ max(omega) - min(omega) (Chadwick et al. 2025).

        Returns:
            Estimated critical coupling strength
        """
        return self.omega.max() - self.omega.min()

    def get_criticality_distance(self) -> torch.Tensor:
        """Distance from Hopf bifurcation (|mu|). 0 = critical."""
        return torch.abs(self.mu)

    def set_mu(self, mu: float):
        """Set bifurcation parameter (for criticality control)."""
        with torch.no_grad():
            self.mu.fill_(mu)

    def reset(self):
        """Reset oscillators to random small-amplitude state."""
        r = torch.rand(self.num_oscillators) * 0.3 + 0.1
        theta = torch.rand(self.num_oscillators) * 2 * math.pi
        self.z_real.copy_(r * torch.cos(theta))
        self.z_imag.copy_(r * torch.sin(theta))
        self.step_count.zero_()


class KuramotoDaidoMeanField(nn.Module):
    """
    Mean-field reduction of the Kuramoto model via the Ott-Antonsen ansatz.

    Instead of tracking N individual oscillator phases (O(N^2) coupling),
    tracks the complex order parameter Z directly (O(1) per step):

        dZ/dt = (-i*w_mean - Delta_eff + K/2) * Z - K/2 * |Z|^2 * Z

    Where:
    - Z = R * exp(i*Psi) is the complex order parameter
    - w_mean is the mean natural frequency
    - Delta_eff is the effective frequency spread
    - K is coupling strength

    For Cauchy/Lorentzian distributed frequencies, the Ott-Antonsen
    ansatz is exact (Delta_eff = delta, the Cauchy half-width).

    For Gaussian distributed frequencies, a scaled approximation
    (Montbrio, Pazo & Roxin 2015) is used:
    Delta_eff = delta * sqrt(2*pi) / 2, where delta is the standard
    deviation. This gives much better predictions when frequencies
    are sampled from Uniform or Gaussian distributions.

    Args:
        omega_mean: Mean natural frequency
        delta: Frequency spread (Cauchy half-width or Gaussian std)
        coupling: Coupling strength K
        dt: Integration timestep
        distribution: 'cauchy' (exact Ott-Antonsen) or 'gaussian'
                      (Montbrio-Pazo-Roxin scaled approximation)
    """

    def __init__(self,
                 omega_mean: float = 1.0,
                 delta: float = 0.1,
                 coupling: float = 1.0,
                 dt: float = 0.01,
                 distribution: str = 'cauchy'):
        super().__init__()
        self.dt = dt
        self.distribution = distribution

        # Learnable dynamics parameters
        self.omega_mean = nn.Parameter(torch.tensor(omega_mean))
        self.coupling = nn.Parameter(torch.tensor(coupling))

        # Delta (spread) as buffer — not typically learned
        self.register_buffer('delta', torch.tensor(delta))

        # Complex order parameter Z = R * exp(i*Psi)
        # Store as (real, imag) pair for buffer compatibility
        self.register_buffer('Z_real', torch.tensor(0.1))
        self.register_buffer('Z_imag', torch.tensor(0.0))

        # Step counter
        self.register_buffer('step_count', torch.tensor(0))

    @property
    def Z(self) -> torch.Tensor:
        """Complex order parameter."""
        return torch.complex(self.Z_real, self.Z_imag)

    @property
    def R(self) -> torch.Tensor:
        """Order parameter magnitude (synchronization level)."""
        return torch.abs(self.Z)

    @property
    def Psi(self) -> torch.Tensor:
        """Order parameter phase (mean phase)."""
        return torch.angle(self.Z)

    def forward(self, steps: int = 1,
                external_field: Optional[torch.Tensor] = None) -> Dict:
        """
        Evolve the mean-field ODE for given number of steps.

        Args:
            steps: Number of integration steps
            external_field: Optional complex driving field

        Returns:
            Dict with R (magnitude), Psi (phase), Z (complex)
        """
        Z = self.Z

        # Effective spread: for Gaussian, scale delta by sqrt(2*pi)/2
        # (Montbrio, Pazo & Roxin 2015 QIF mean-field reduction)
        if self.distribution == 'gaussian':
            delta_eff = self.delta * (math.sqrt(2 * math.pi) / 2)
        else:
            delta_eff = self.delta

        for _ in range(steps):
            K = self.coupling
            w = self.omega_mean

            # Ott-Antonsen mean-field ODE
            # dZ/dt = (-i*w - Delta_eff + K/2) * Z - K/2 * |Z|^2 * Z
            Z_sq_mag = (Z.real ** 2 + Z.imag ** 2)
            linear_coeff = torch.complex(-delta_eff + K / 2, -w)
            dZ = linear_coeff * Z - (K / 2) * Z_sq_mag * Z

            if external_field is not None:
                dZ = dZ + external_field

            Z = Z + dZ * self.dt

        # Update stored state
        with torch.no_grad():
            self.Z_real.copy_(Z.real.detach())
            self.Z_imag.copy_(Z.imag.detach())
            self.step_count.add_(steps)

        R = torch.abs(Z)
        Psi = torch.angle(Z)

        return {
            'R': R,
            'Psi': Psi,
            'Z': Z,
        }

    @classmethod
    def from_frequencies(cls, omega_samples: torch.Tensor,
                         coupling: float = 1.0, dt: float = 0.01,
                         distribution: str = 'auto') -> 'KuramotoDaidoMeanField':
        """Construct mean-field model by fitting distribution to frequency samples.

        For Cauchy: uses IQR-based robust fitting (Pietras & Daffertshofer 2016):
        - omega_mean = median(samples)
        - delta = IQR / 2  (half-width at half-maximum of Cauchy)

        For Gaussian: uses mean and standard deviation directly:
        - omega_mean = mean(samples)
        - delta = std(samples)

        With distribution='auto', uses excess kurtosis to choose:
        - kurtosis < 6 → Gaussian (normal kurtosis = 3, uniform = 1.8)
        - kurtosis >= 6 → Cauchy (theoretical kurtosis = infinity)

        Args:
            omega_samples: Tensor of sampled natural frequencies (N,)
            coupling: Coupling strength K
            dt: Integration timestep
            distribution: 'auto', 'cauchy', or 'gaussian'

        Returns:
            KuramotoDaidoMeanField instance with fitted parameters
        """
        if distribution == 'auto':
            # Excess kurtosis test: Cauchy has very heavy tails
            mean_val = omega_samples.mean()
            std_val = omega_samples.std()
            if std_val > 1e-8:
                centered = omega_samples - mean_val
                kurtosis = (centered ** 4).mean() / (std_val ** 4)
            else:
                kurtosis = 3.0  # Default to Gaussian
            distribution = 'gaussian' if kurtosis < 6.0 else 'cauchy'

        if distribution == 'gaussian':
            omega_mean = omega_samples.mean().item()
            delta = max(omega_samples.std().item(), 1e-6)
        else:
            omega_sorted = omega_samples.sort().values
            n = len(omega_sorted)
            q25 = omega_sorted[max(0, int(n * 0.25))].item()
            q75 = omega_sorted[min(n - 1, int(n * 0.75))].item()
            omega_mean = omega_sorted[n // 2].item()  # median
            delta = max((q75 - q25) / 2.0, 1e-6)  # IQR-based Cauchy half-width

        return cls(omega_mean=omega_mean, delta=delta, coupling=coupling,
                   dt=dt, distribution=distribution)

    def get_order_parameter(self) -> torch.Tensor:
        """Get current synchronization level R."""
        return self.R

    def get_critical_coupling(self) -> torch.Tensor:
        """Critical coupling for onset of synchronization.

        Cauchy: Kc = 2 * delta
        Gaussian: Kc = 2 * delta * sqrt(2*pi) / pi  (Montbrio et al. 2015)
        """
        if self.distribution == 'gaussian':
            return 2.0 * self.delta * math.sqrt(2 * math.pi) / math.pi
        return 2.0 * self.delta

    def is_synchronized(self, threshold: float = 0.5) -> bool:
        """Check if population is synchronized."""
        return self.R.item() > threshold

    def reset(self):
        """Reset order parameter to low-sync state."""
        self.Z_real.fill_(0.1)
        self.Z_imag.fill_(0.0)
        self.step_count.zero_()


class PhaseTopologyCache:
    """
    Cache for phase topology patterns with outcome labels.

    Stores hashes of relative phase patterns (rotation-invariant)
    with observed outcome classes. Acts as a Bayesian prior for
    confidence estimation — NOT a lookup table.

    Plain class (not nn.Module) — no learnable parameters.

    Args:
        capacity: Maximum number of cached entries
        hash_bins: Number of quantization bins for phase differences
    """

    def __init__(self, capacity: int = 1000, hash_bins: int = 64):
        self.capacity = capacity
        self.hash_bins = hash_bins
        self._store: Dict[int, Dict] = {}  # hash -> {outcome, confidence, count}
        self._access_order: List[int] = []  # LRU eviction

    def compute_hash(self, phases: torch.Tensor) -> int:
        """
        Rotation-invariant hash of relative phase pattern.

        Sorts phase differences (removes rotation ambiguity),
        quantizes into bins, then hashes.
        """
        with torch.no_grad():
            # Compute pairwise phase differences (rotation-invariant)
            diffs = (phases.unsqueeze(0) - phases.unsqueeze(1)) % (2 * math.pi)
            # Take upper triangle to avoid redundancy
            n = phases.shape[0]
            upper_diffs = []
            for i in range(n):
                for j in range(i + 1, n):
                    upper_diffs.append(diffs[i, j].item())
            if not upper_diffs:
                return 0
            # Sort for permutation invariance
            upper_diffs.sort()
            # Quantize into bins
            binned = tuple(
                int(d / (2 * math.pi) * self.hash_bins) % self.hash_bins
                for d in upper_diffs
            )
            return hash(binned)

    def store(self, phases: torch.Tensor, outcome: str, confidence: float):
        """Store an observed outcome for a phase topology."""
        h = self.compute_hash(phases)

        if h in self._store:
            entry = self._store[h]
            # Bayesian update: weighted running average
            old_count = entry['count']
            entry['confidence'] = (entry['confidence'] * old_count + confidence) / (old_count + 1)
            entry['count'] += 1
            # If outcome differs, keep the majority
            if outcome != entry['outcome'] and confidence > entry['confidence']:
                entry['outcome'] = outcome
        else:
            # Evict if at capacity
            if len(self._store) >= self.capacity:
                oldest = self._access_order.pop(0)
                self._store.pop(oldest, None)
            self._store[h] = {'outcome': outcome, 'confidence': confidence, 'count': 1}

        # Update access order
        if h in self._access_order:
            self._access_order.remove(h)
        self._access_order.append(h)

    def query(self, phases: torch.Tensor) -> Optional[Dict]:
        """Returns {outcome, confidence, count} if similar pattern seen, else None."""
        h = self.compute_hash(phases)
        entry = self._store.get(h)
        if entry is not None:
            # Update LRU
            if h in self._access_order:
                self._access_order.remove(h)
            self._access_order.append(h)
            return dict(entry)
        return None

    def get_prior(self, phases: torch.Tensor, hypothesis: str) -> float:
        """
        Bayesian prior: P(hypothesis | similar_topology_seen_before).

        Returns stored confidence weighted by count if hash matches
        and outcome matches hypothesis; else 0.5 (uninformative).
        """
        entry = self.query(phases)
        if entry is None:
            return 0.5
        if entry['outcome'] == hypothesis:
            # Confidence grows with evidence: cap at 0.95
            count_weight = min(1.0, entry['count'] / 10.0)
            return 0.5 + (entry['confidence'] - 0.5) * count_weight
        else:
            # Evidence against this hypothesis
            count_weight = min(1.0, entry['count'] / 10.0)
            return 0.5 - (entry['confidence'] - 0.5) * count_weight * 0.5


if __name__ == '__main__':
    print("--- Snake Activation Example ---")
    snake_act = Snake(frequency=0.5)
    test_tensor = torch.linspace(-5, 5, 10)
    output_snake = snake_act(test_tensor)
    print(f"Input to Snake: {test_tensor.detach().cpu().numpy()}")
    print(f"Output from Snake: {output_snake.detach().cpu().numpy()}")

    print("\n--- LearnableKuramotoBank with Different Topologies ---")

    topologies = ['global', 'small_world', 'scale_free', 'ring']
    num_oscillators = 50

    for topo in topologies:
        print(f"\nTopology: {topo}")
        bank = LearnableKuramotoBank(
            num_oscillators, dt=0.01, topology=topo,
            topology_params={'k': 4, 'p': 0.3, 'm': 3}
        )

        # Initial state
        initial_r = bank.get_order_parameter()
        print(f"  Initial R: {initial_r.item():.4f}")

        # Simulate
        bank(steps=500)
        final_r = bank.get_order_parameter()
        print(f"  Final R (500 steps): {final_r.item():.4f}")

        # Show connectivity stats
        adj = bank.get_adjacency()
        avg_degree = adj.sum(dim=1).mean().item()
        print(f"  Avg degree: {avg_degree:.1f}")

    print("\n--- Learnable Adjacency Example ---")
    learnable_bank = LearnableKuramotoBank(
        20, dt=0.01, topology='learnable',
        topology_params={'sparsity': 0.3}
    )

    optimizer = torch.optim.Adam(learnable_bank.parameters(), lr=0.1)
    target_r = torch.tensor(0.9)

    for epoch in range(10):
        learnable_bank.phases.data = torch.rand(20) * 2 * math.pi
        optimizer.zero_grad()

        phases = learnable_bank(steps=100)
        current_r = learnable_bank.get_order_parameter(phases)

        # Loss: synchronization + sparsity regularization
        adj = learnable_bank.get_adjacency()
        sparsity_loss = adj.mean()  # Encourage sparse connections
        sync_loss = (target_r - current_r) ** 2
        loss = sync_loss + 0.1 * sparsity_loss

        loss.backward()
        optimizer.step()

        if epoch % 3 == 0:
            print(f"Epoch {epoch}: R={current_r.item():.3f}, Sparsity={adj.mean().item():.3f}")

    print(f"Final learned adjacency sparsity: {learnable_bank.get_adjacency().mean().item():.3f}")
