"""
Thermodynamic Computing - Learning via energy minimization and phase transitions.

Extracted from generative_thermodynamic_computing and Neuroca patterns.
Key concepts:
- System temperature controls exploration vs exploitation
- Phase transitions when temperature crosses critical points
- Simulated annealing for optimization
- Energy-based learning objectives

This enables oscillatory systems to exhibit thermodynamic behavior.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Callable, Dict, List
import math


class TemperatureSchedule:
    """
    Temperature schedule for annealing processes.

    Controls how temperature decreases over time,
    affecting exploration-exploitation tradeoff.
    """

    def __init__(self,
                 initial_temp: float = 1.0,
                 final_temp: float = 0.01,
                 schedule_type: str = "exponential"):
        """
        Args:
            initial_temp: Starting temperature
            final_temp: Ending temperature
            schedule_type: "linear", "exponential", or "cosine"
        """
        self.T0 = initial_temp
        self.Tf = final_temp
        self.schedule_type = schedule_type

    def __call__(self, t: float, total_steps: int) -> float:
        """
        Get temperature at step t.

        Args:
            t: Current step (0 to total_steps)
            total_steps: Total number of steps

        Returns:
            Temperature at step t
        """
        progress = t / max(total_steps, 1)
        return self._SCHEDULES.get(self.schedule_type, TemperatureSchedule._const)(self, progress)

    def _linear(self, progress: float) -> float:
        return self.T0 + (self.Tf - self.T0) * progress

    def _exponential(self, progress: float) -> float:
        ratio = self.Tf / (self.T0 + 1e-10)
        return self.T0 * (ratio ** progress)

    def _cosine(self, progress: float) -> float:
        return self.Tf + 0.5 * (self.T0 - self.Tf) * (1 + math.cos(math.pi * progress))

    def _const(self, progress: float) -> float:
        return self.T0

    _SCHEDULES = {
        "linear": _linear,
        "exponential": _exponential,
        "cosine": _cosine,
    }


class LangevinDynamics(nn.Module):
    """
    Langevin dynamics for sampling from energy landscapes.

    Combines gradient descent on energy with thermal noise,
    enabling exploration of energy landscape.

    dx/dt = -dE/dx + sqrt(2T) * noise
    """

    def __init__(self,
                 dim: int,
                 temperature: float = 1.0,
                 friction: float = 1.0,
                 dt: float = 0.01):
        """
        Args:
            dim: State dimension
            temperature: System temperature (noise scale)
            friction: Friction coefficient
            dt: Time step
        """
        super().__init__()

        self.dim = dim
        self.temperature = temperature
        self.friction = friction
        self.dt = dt

        # Noise scale: sqrt(2 * T * dt / friction)
        self._update_noise_scale()

    def _update_noise_scale(self):
        """Update noise scale based on temperature."""
        self.noise_scale = math.sqrt(2 * self.temperature * self.dt / self.friction)

    def set_temperature(self, T: float):
        """Set system temperature."""
        self.temperature = T
        self._update_noise_scale()

    def step(self,
            x: torch.Tensor,
            energy_grad: torch.Tensor) -> torch.Tensor:
        """
        Take one Langevin step.

        Args:
            x: Current state [B, dim]
            energy_grad: Gradient of energy w.r.t. x [B, dim]

        Returns:
            New state after Langevin step
        """
        # Deterministic drift (gradient descent)
        drift = -energy_grad * self.dt / self.friction

        # Stochastic noise
        noise = torch.randn_like(x) * self.noise_scale

        return x + drift + noise

    def forward(self,
               x: torch.Tensor,
               energy_fn: Callable[[torch.Tensor], torch.Tensor],
               num_steps: int = 100) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Run Langevin dynamics for multiple steps.

        Args:
            x: Initial state [B, dim]
            energy_fn: Function that returns energy given state
            num_steps: Number of integration steps

        Returns:
            (final_state, trajectory) final state and list of intermediate states
        """
        trajectory = [x.clone()]

        for _ in range(num_steps):
            # Compute energy gradient
            x.requires_grad_(True)
            energy = energy_fn(x)
            energy_grad = torch.autograd.grad(energy.sum(), x)[0]
            x = x.detach()

            # Langevin step
            x = self.step(x, energy_grad)
            trajectory.append(x.clone())

        return x, trajectory


class SimulatedAnnealing(nn.Module):
    """
    Simulated Annealing optimizer.

    From Neuroca's annealing patterns. Uses temperature-dependent
    acceptance probability to escape local minima.

    Good for finding global optima in oscillator configurations.
    """

    def __init__(self,
                 dim: int,
                 initial_temp: float = 10.0,
                 final_temp: float = 0.01,
                 schedule: str = "exponential",
                 neighbor_scale: float = 0.1):
        """
        Args:
            dim: State dimension
            initial_temp: Starting temperature
            final_temp: Final temperature
            schedule: Temperature schedule type
            neighbor_scale: Scale for neighbor generation
        """
        super().__init__()

        self.dim = dim
        self.schedule = TemperatureSchedule(initial_temp, final_temp, schedule)
        self.neighbor_scale = neighbor_scale

        # Best state tracking
        self.register_buffer('best_state', torch.zeros(dim))
        self.register_buffer('best_energy', torch.tensor(float('inf')))

    def generate_neighbor(self,
                         x: torch.Tensor,
                         temperature: float) -> torch.Tensor:
        """
        Generate neighbor state.

        Neighbor distance scales with temperature for adaptive exploration.

        Args:
            x: Current state
            temperature: Current temperature

        Returns:
            Neighbor state
        """
        # Scale perturbation with temperature
        scale = self.neighbor_scale * math.sqrt(temperature)
        noise = torch.randn_like(x) * scale
        return x + noise

    def acceptance_probability(self,
                              current_energy: float,
                              new_energy: float,
                              temperature: float) -> float:
        """
        Compute Boltzmann acceptance probability.

        Args:
            current_energy: Energy of current state
            new_energy: Energy of proposed state
            temperature: Current temperature

        Returns:
            Probability of accepting new state
        """
        if new_energy < current_energy:
            return 1.0
        else:
            delta = new_energy - current_energy
            return math.exp(-delta / (temperature + 1e-10))

    def optimize(self,
                initial_state: torch.Tensor,
                energy_fn: Callable[[torch.Tensor], torch.Tensor],
                num_steps: int = 1000,
                patience: int = 100) -> Dict[str, torch.Tensor]:
        """
        Run simulated annealing optimization.

        Args:
            initial_state: Starting state [dim] or [B, dim]
            energy_fn: Function returning energy (lower is better)
            num_steps: Total optimization steps
            patience: Steps without improvement before early stopping

        Returns:
            Dict with best_state, best_energy, trajectory, temperatures
        """
        x = initial_state.clone()
        current_energy = energy_fn(x).item()

        self.best_state = x.clone()
        self.best_energy = torch.tensor(current_energy)

        trajectory = [x.clone()]
        energies = [current_energy]
        temperatures = []
        steps_without_improvement = 0

        for step in range(num_steps):
            # Get temperature
            T = self.schedule(step, num_steps)
            temperatures.append(T)

            # Generate neighbor
            neighbor = self.generate_neighbor(x, T)
            neighbor_energy = energy_fn(neighbor).item()

            # Accept or reject
            accept_prob = self.acceptance_probability(current_energy, neighbor_energy, T)

            if torch.rand(1).item() < accept_prob:
                x = neighbor
                current_energy = neighbor_energy
                trajectory.append(x.clone())
                energies.append(current_energy)

                # Update best
                if current_energy < self.best_energy.item():
                    self.best_state = x.clone()
                    self.best_energy = torch.tensor(current_energy)
                    steps_without_improvement = 0
                else:
                    steps_without_improvement += 1
            else:
                steps_without_improvement += 1

            # Early stopping
            if steps_without_improvement >= patience:
                break

        return {
            'best_state': self.best_state,
            'best_energy': self.best_energy,
            'final_state': x,
            'trajectory': trajectory,
            'energies': energies,
            'temperatures': temperatures,
            'steps': step + 1
        }


class PhaseTransitionDetector(nn.Module):
    """
    Detect phase transitions in oscillator systems.

    Phase transitions occur when the system undergoes
    qualitative changes (e.g., from disordered to synchronized).

    Monitors order parameters and susceptibility.
    """

    def __init__(self,
                 window_size: int = 50,
                 critical_threshold: float = 0.5):
        """
        Args:
            window_size: Window for computing statistics
            critical_threshold: Threshold for detecting transitions
        """
        super().__init__()

        self.window_size = window_size
        self.threshold = critical_threshold

        # History buffers
        self.order_history: List[float] = []
        self.susceptibility_history: List[float] = []

    def compute_susceptibility(self, order_params: torch.Tensor) -> float:
        """
        Compute susceptibility (variance of order parameter).

        High susceptibility indicates proximity to phase transition.

        Args:
            order_params: Recent order parameter values

        Returns:
            Susceptibility value
        """
        return order_params.var().item()

    def update(self, order_parameter: float) -> Dict[str, float]:
        """
        Update with new order parameter measurement.

        Args:
            order_parameter: Current order parameter value

        Returns:
            Dict with susceptibility, trend, is_transitioning
        """
        self.order_history.append(order_parameter)

        # Keep window
        if len(self.order_history) > self.window_size:
            self.order_history = self.order_history[-self.window_size:]

        result = {
            'order_parameter': order_parameter,
            'susceptibility': 0.0,
            'trend': 0.0,
            'is_transitioning': False
        }

        if len(self.order_history) >= 10:
            recent = torch.tensor(self.order_history[-10:])

            # Susceptibility
            susceptibility = self.compute_susceptibility(recent)
            result['susceptibility'] = susceptibility
            self.susceptibility_history.append(susceptibility)

            # Trend (is order increasing or decreasing?)
            trend = (recent[-1] - recent[0]).item()
            result['trend'] = trend

            # Detect transition (high susceptibility + significant trend)
            if susceptibility > self.threshold and abs(trend) > 0.1:
                result['is_transitioning'] = True

        return result

    def reset(self):
        """Reset history."""
        self.order_history = []
        self.susceptibility_history = []


class ThermodynamicOscillatorSystem(nn.Module):
    """
    Oscillator system with thermodynamic properties.

    Combines Kuramoto-like dynamics with temperature-dependent
    noise and phase transition detection.

    The system can "freeze" into synchronized states or
    "melt" into disordered states based on temperature.
    """

    def __init__(self,
                 num_oscillators: int,
                 coupling_strength: float = 1.0,
                 initial_temperature: float = 1.0):
        """
        Args:
            num_oscillators: Number of oscillators
            coupling_strength: Coupling strength K
            initial_temperature: Starting temperature
        """
        super().__init__()

        self.num_oscillators = num_oscillators
        self.K = coupling_strength
        self.temperature = initial_temperature

        # Oscillator phases
        self.register_buffer(
            'phases',
            torch.rand(num_oscillators) * 2 * math.pi
        )

        # Natural frequencies (drawn from Lorentzian)
        self.register_buffer(
            'frequencies',
            torch.randn(num_oscillators) * 0.5
        )

        # Coupling matrix
        self.coupling = nn.Parameter(
            torch.ones(num_oscillators, num_oscillators) / num_oscillators
        )

        # Phase transition detector
        self.transition_detector = PhaseTransitionDetector()

        # Critical temperature (theoretical for mean-field Kuramoto)
        # Tc = K / 2 for Lorentzian frequency distribution
        self.critical_temperature = coupling_strength / 2

    def compute_order_parameter(self) -> Tuple[float, float]:
        """
        Compute Kuramoto order parameter.

        Returns:
            (r, psi) magnitude and mean phase
        """
        z = torch.exp(1j * self.phases).mean()
        r = z.abs().item()
        psi = z.angle().item()
        return r, psi

    def compute_energy(self) -> float:
        """
        Compute system energy (negative of synchronization).

        E = -K/N * sum_{i,j} cos(theta_i - theta_j)
        """
        phase_diff = self.phases.unsqueeze(1) - self.phases.unsqueeze(0)
        energy = -self.K / self.num_oscillators * torch.cos(phase_diff).sum()
        return energy.item()

    def step(self, dt: float = 0.01) -> Dict[str, float]:
        """
        Take one thermodynamic Kuramoto step.

        Args:
            dt: Time step

        Returns:
            Dict with order_parameter, energy, temperature
        """
        # Kuramoto coupling force
        phase_diff = self.phases.unsqueeze(1) - self.phases.unsqueeze(0)
        coupling_force = (self.coupling * torch.sin(phase_diff)).sum(dim=1)

        # Natural frequency
        freq_force = self.frequencies

        # Total deterministic force
        force = freq_force + self.K * coupling_force

        # Thermal noise (scaled by sqrt(2T))
        noise = torch.randn_like(self.phases) * math.sqrt(2 * self.temperature * dt)

        # Update phases
        self.phases = self.phases + force * dt + noise
        self.phases = self.phases % (2 * math.pi)

        # Compute observables
        r, psi = self.compute_order_parameter()
        energy = self.compute_energy()

        # Update transition detector
        transition_info = self.transition_detector.update(r)

        return {
            'order_parameter': r,
            'mean_phase': psi,
            'energy': energy,
            'temperature': self.temperature,
            'susceptibility': transition_info['susceptibility'],
            'is_transitioning': transition_info['is_transitioning']
        }

    def anneal(self,
              target_temperature: float,
              num_steps: int = 100,
              dt: float = 0.01) -> List[Dict[str, float]]:
        """
        Anneal system to target temperature.

        Args:
            target_temperature: Final temperature
            num_steps: Steps for annealing
            dt: Time step

        Returns:
            List of state dicts at each step
        """
        schedule = TemperatureSchedule(
            self.temperature, target_temperature, "exponential"
        )

        history = []
        for i in range(num_steps):
            self.temperature = schedule(i, num_steps)
            state = self.step(dt)
            history.append(state)

        return history

    def find_synchronized_state(self,
                               cooling_steps: int = 200,
                               equilibration_steps: int = 100) -> Dict[str, float]:
        """
        Cool system to find synchronized state.

        Args:
            cooling_steps: Steps for cooling
            equilibration_steps: Steps at low temperature

        Returns:
            Final state dict
        """
        # Cool down
        self.anneal(0.01, cooling_steps)

        # Equilibrate
        for _ in range(equilibration_steps):
            state = self.step()

        return state

    def forward(self, num_steps: int = 100, dt: float = 0.01) -> Dict[str, torch.Tensor]:
        """
        Run dynamics and return trajectory.

        Args:
            num_steps: Number of steps
            dt: Time step

        Returns:
            Dict with trajectories of observables
        """
        order_params = []
        energies = []
        temperatures = []

        for _ in range(num_steps):
            state = self.step(dt)
            order_params.append(state['order_parameter'])
            energies.append(state['energy'])
            temperatures.append(state['temperature'])

        return {
            'order_parameters': torch.tensor(order_params),
            'energies': torch.tensor(energies),
            'temperatures': torch.tensor(temperatures),
            'final_order': order_params[-1],
            'final_phases': self.phases.clone()
        }
