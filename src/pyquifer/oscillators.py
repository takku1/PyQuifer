import torch
import torch.nn as nn
import math
from typing import Dict, Literal, Optional


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
                 integration_method: Literal['euler', 'rk4'] = 'euler'):
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
        """
        super().__init__()
        self.num_oscillators = num_oscillators
        self.dt = dt
        self.topology = topology
        self.integration_method = integration_method
        self.topology_params = topology_params or {}
        self._global_topology = False  # Will be set True for global topology

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

        # Initialize adjacency matrix based on topology
        self._init_adjacency(topology, self.topology_params)

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

        elif topology == 'scale_free':
            # Barabási-Albert scale-free network
            m = params.get('m', max(1, n // 20))  # edges per new node
            adj = self._create_scale_free(n, m)
            self.register_buffer('adjacency', adj)

        elif topology == 'ring':
            # Ring topology with local coupling
            k = params.get('k', 2)  # neighbors on each side
            adj = self._create_ring(n, k)
            self.register_buffer('adjacency', adj)

        elif topology == 'learnable':
            # Fully learnable adjacency (soft, between 0-1)
            sparsity = params.get('sparsity', 0.5)
            # Initialize with random values, biased toward desired sparsity
            init_values = torch.randn(n, n) * 0.5 - torch.log(torch.tensor(1/sparsity - 1))
            self.adjacency_logits = nn.Parameter(init_values)
            # No self-connections
            self.register_buffer('self_mask', 1.0 - torch.eye(n))

        else:
            raise ValueError(f"Unknown topology: {topology}")

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
                    normalized_interaction = interaction_sum / (self.num_oscillators - 1)
            else:
                # Calculate phase differences: theta_j - theta_i
                phase_diffs = phases.unsqueeze(1) - phases.unsqueeze(0)  # (n, n)

                if use_precision:
                    # Precision-weighted adjacency: effective_adj = adj * source_precision
                    effective_adj = adj * self.precision.detach().unsqueeze(0)  # (n, n) * (1, n) = broadcast over rows
                    weighted_interaction = effective_adj * torch.sin(phase_diffs)
                    interaction_sum = torch.sum(weighted_interaction, dim=1)
                    # Normalize by sum of effective weights per row
                    weight_sums = effective_adj.sum(dim=1).clamp(min=1e-8)
                    normalized_interaction = interaction_sum / weight_sums
                else:
                    weighted_interaction = adj * torch.sin(phase_diffs)
                    interaction_sum = torch.sum(weighted_interaction, dim=1)
                    connection_counts = adj.sum(dim=1).clamp(min=1)
                    normalized_interaction = interaction_sum / connection_counts

            # Kuramoto dynamics equation
            if self.integration_method == 'rk4':
                # Build RHS closure capturing current coupling state
                _ext = external_input

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
                    else:
                        pd = ph.unsqueeze(1) - ph.unsqueeze(0)
                        if use_precision:
                            ea = adj * self.precision.detach().unsqueeze(0)
                            wi = ea * torch.sin(pd)
                            is_ = torch.sum(wi, dim=1)
                            ws = ea.sum(dim=1).clamp(min=1e-8)
                            ni = is_ / ws
                        else:
                            wi = adj * torch.sin(pd)
                            is_ = torch.sum(wi, dim=1)
                            cc = adj.sum(dim=1).clamp(min=1)
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

        # Update precision from phase velocity variance
        with torch.no_grad():
            # Phase velocity = angular distance from previous phase
            phase_diff = phases.detach() - self.prev_phases
            # Wrap to [-pi, pi]
            phase_velocity = (phase_diff + math.pi) % (2 * math.pi) - math.pi

            # Update running variance via EMA
            alpha = 1.0 / self._precision_tau
            instant_var = phase_velocity.pow(2)
            self.phase_velocity_var.mul_(1 - alpha).add_(instant_var * alpha)

            # Precision = 1 / (variance + epsilon)
            self.precision.copy_(
                (1.0 / (self.phase_velocity_var + self._precision_epsilon)).clamp(max=100.0)
            )
            self.prev_phases.copy_(phases.detach())

        # Update internal state
        self.phases.data = phases.detach()
        return phases

    def get_order_parameter(self, phases: torch.Tensor = None) -> torch.Tensor:
        """
        Calculates the global order parameter R for the oscillator bank.
        R = |(1/N) * sum(e^(i*theta_j))|
        """
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
        neighbor_sum = torch.matmul(adj.to(complex_phases.dtype), complex_phases)
        neighbor_count = adj.sum(dim=1).clamp(min=1)
        local_mean = neighbor_sum / neighbor_count

        return torch.abs(local_mean)


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

        dZ/dt = (-i*w_mean - Delta + K/2) * Z - K/2 * |Z|^2 * Z

    Where:
    - Z = R * exp(i*Psi) is the complex order parameter
    - w_mean is the mean natural frequency
    - Delta is the frequency spread (Cauchy half-width)
    - K is coupling strength

    This is exact for infinite-N Cauchy-distributed frequencies and
    provides an excellent O(1) approximation for large finite N.

    Args:
        omega_mean: Mean natural frequency
        delta: Frequency spread (Cauchy half-width at half-maximum)
        coupling: Coupling strength K
        dt: Integration timestep
    """

    def __init__(self,
                 omega_mean: float = 1.0,
                 delta: float = 0.1,
                 coupling: float = 1.0,
                 dt: float = 0.01):
        super().__init__()
        self.dt = dt

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

        for _ in range(steps):
            K = self.coupling
            w = self.omega_mean

            # Ott-Antonsen mean-field ODE
            # dZ/dt = (-i*w - Delta + K/2) * Z - K/2 * |Z|^2 * Z
            Z_sq_mag = (Z.real ** 2 + Z.imag ** 2)
            linear_coeff = torch.complex(-self.delta + K / 2, -w)
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
                         coupling: float = 1.0, dt: float = 0.01) -> 'KuramotoDaidoMeanField':
        """Construct mean-field model by fitting Cauchy distribution to frequency samples.

        Uses IQR-based robust Cauchy fitting (Pietras & Daffertshofer 2016):
        - omega_mean = median(samples)
        - delta = IQR / 2  (half-width at half-maximum of Cauchy)

        This gives a much better Ott-Antonsen match than naive mean/std mapping.

        Args:
            omega_samples: Tensor of sampled natural frequencies (N,)
            coupling: Coupling strength K
            dt: Integration timestep

        Returns:
            KuramotoDaidoMeanField instance with fitted parameters
        """
        omega_sorted = omega_samples.sort().values
        n = len(omega_sorted)
        q25 = omega_sorted[max(0, int(n * 0.25))].item()
        q75 = omega_sorted[min(n - 1, int(n * 0.75))].item()
        omega_mean = omega_sorted[n // 2].item()  # median
        delta = max((q75 - q25) / 2.0, 1e-6)  # IQR-based Cauchy half-width
        return cls(omega_mean=omega_mean, delta=delta, coupling=coupling, dt=dt)

    def get_order_parameter(self) -> torch.Tensor:
        """Get current synchronization level R."""
        return self.R

    def get_critical_coupling(self) -> torch.Tensor:
        """Critical coupling Kc = 2 * Delta (onset of synchronization)."""
        return 2.0 * self.delta

    def is_synchronized(self, threshold: float = 0.5) -> bool:
        """Check if population is synchronized."""
        return self.R.item() > threshold

    def reset(self):
        """Reset order parameter to low-sync state."""
        self.Z_real.fill_(0.1)
        self.Z_imag.fill_(0.0)
        self.step_count.zero_()


if __name__ == '__main__':
    print("--- Snake Activation Example ---")
    snake_act = Snake(frequency=0.5)
    test_tensor = torch.linspace(-5, 5, 10)
    output_snake = snake_act(test_tensor)
    print(f"Input to Snake: {test_tensor.detach().numpy()}")
    print(f"Output from Snake: {output_snake.detach().numpy()}")

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
