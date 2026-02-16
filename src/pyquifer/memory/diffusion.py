import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Literal
from pyquifer.memory.perturbation import PerturbationLayer
from pyquifer.memory.potentials import MultiAttractorPotential


class MindEyeActualization(nn.Module):
    """
    Implements the "Mind's Eye" actualization loop, denoising harmonics
    into a coherent potential guided by attractors and perturbed by noise.
    This module combines PerturbationLayer (noise) and MultiAttractorPotential (intent/self).

    Key features:
    - Force scheduling: Balance between exploration (potential) and convergence (target)
    - Exponential noise decay: Faster settling than harmonic decay
    - Training mode: Allows gradients to flow through iterations for learning
    """

    def __init__(self,
                 dim: int,
                 num_attractors: int,
                 noise_params: Optional[Dict[str, Any]] = None,
                 actualization_strength: float = 0.1,
                 target_vector: Optional[torch.Tensor] = None,
                 noise_decay: Literal['exponential', 'harmonic'] = 'exponential',
                 noise_decay_rate: float = 0.9):
        super().__init__()
        self.dim = dim
        self.num_attractors = num_attractors
        self.actualization_strength = actualization_strength
        self.noise_decay = noise_decay
        self.noise_decay_rate = noise_decay_rate

        # Initialize the PerturbationLayer for adding creative jitter/noise
        self.perturbation_layer = PerturbationLayer(dim=dim, **(noise_params if noise_params else {}))

        # Initialize the MultiAttractorPotential for guiding the actualization
        if num_attractors == 1 and target_vector is not None:
            self.potential_field = MultiAttractorPotential(
                num_attractors=num_attractors, dim=dim,
                attractor_positions=target_vector.unsqueeze(0).float()
            )
        else:
            self.potential_field = MultiAttractorPotential(num_attractors=num_attractors, dim=dim)

        # The Base Vector or "Archetype" - learnable if not provided
        if target_vector is not None:
            if target_vector.shape[0] != dim:
                raise ValueError(f"Target vector dimension ({target_vector.shape[0]}) must match space dim ({dim})")
            self.target_vector = nn.Parameter(target_vector.float().unsqueeze(0), requires_grad=False)
        else:
            self.target_vector = nn.Parameter(torch.randn(1, dim))

    def forward(self,
                initial_state: torch.Tensor,
                generated_noise_field: torch.Tensor,
                iterations: int,
                noise_amplitude: float = 1.0,
                time_offset: float = 0.0,
                training_mode: bool = False,
                force_schedule: bool = True) -> torch.Tensor:
        """
        Runs the actualization (denoising) process.

        Args:
            initial_state: Starting state tensor of shape (batch_size, dim).
            generated_noise_field: Pre-generated noise field to sample jitter from.
            iterations: Number of steps to run the denoising loop.
            noise_amplitude: Initial amplitude of the creative jitter.
            time_offset: For 4D noise, represents the current 'time' slice.
            training_mode: If True, don't detach state - allows gradients to flow
                          through all iterations for learning. Memory intensive.
            force_schedule: If True, gradually shift weight from potential_force
                           (exploration) to target_pull (convergence).

        Returns:
            torch.Tensor: The final actualized state.
        """
        batch_size = initial_state.shape[0]
        state_dim = initial_state.shape[1]

        # Initialize state
        if training_mode:
            state = initial_state.clone()
        else:
            state = initial_state.clone().detach().requires_grad_(True)

        # Pre-compute expanded noise field OUTSIDE the loop (efficiency fix)
        noise_input = self._prepare_noise_input(generated_noise_field, batch_size, state_dim)

        for i in range(iterations):
            # Calculate progress ratio for scheduling (0.0 -> 1.0)
            progress = i / max(iterations - 1, 1)

            # Force scheduling: start with exploration, end with convergence
            if force_schedule:
                # Early: potential_force dominates (exploration)
                # Late: target_pull dominates (convergence)
                potential_weight = 1.0 - progress  # 1.0 -> 0.0
                target_weight = progress  # 0.0 -> 1.0
            else:
                potential_weight = 1.0
                target_weight = 1.0

            # 1. Calculate forces
            target_pull = -(state - self.target_vector)  # Linear pull toward archetype
            potential_force = self.potential_field.get_force(state)  # Landscape gradient

            # Weighted combination of forces
            combined_force = target_weight * target_pull + potential_weight * potential_force

            # 2. Calculate noise decay
            if self.noise_decay == 'exponential':
                # Exponential decay: settles much faster
                current_noise_amplitude = noise_amplitude * (self.noise_decay_rate ** i)
            else:
                # Harmonic decay (original): slower settling
                current_noise_amplitude = noise_amplitude / (i + 1)

            # 3. Sample jitter from pre-computed noise input
            jitter = self._sample_jitter(noise_input, state, state_dim)
            jitter = jitter * current_noise_amplitude

            # 4. Update state
            state = state + (combined_force + jitter) * self.actualization_strength

            # Clamp state to prevent explosion
            state = torch.clamp(state, -10.0, 10.0)

            # 5. Handle gradient flow
            if not training_mode:
                # Break gradient chain to prevent memory explosion
                state = state.detach().requires_grad_(True)

        return state

    def _prepare_noise_input(self, generated_noise_field: torch.Tensor,
                             batch_size: int, state_dim: int) -> torch.Tensor:
        """
        Prepare the noise field for grid_sample, expanded to batch size.
        Called ONCE before the iteration loop for efficiency.
        """
        if state_dim == 2:
            # 4D input for 2D grid_sample: (N, C, H, W)
            noise_input = generated_noise_field.unsqueeze(0).unsqueeze(0)
            noise_input = noise_input.expand(batch_size, -1, -1, -1)
        elif state_dim == 3:
            # 5D input for 3D grid_sample: (N, C, D, H, W)
            noise_input = generated_noise_field.unsqueeze(0).unsqueeze(0)
            noise_input = noise_input.expand(batch_size, -1, -1, -1, -1)
        else:
            # Fallback: no grid_sample, will use random noise
            noise_input = None
        return noise_input

    def _sample_jitter(self, noise_input: Optional[torch.Tensor],
                       state: torch.Tensor, state_dim: int) -> torch.Tensor:
        """
        Sample jitter from the pre-computed noise input at state coordinates.
        """
        batch_size = state.shape[0]

        if noise_input is None or state_dim not in [2, 3]:
            # Fallback: random noise for unsupported dimensions
            jitter = torch.randn(batch_size, 1, device=state.device)
        elif state_dim == 2:
            grid_coordinates = state.unsqueeze(1).unsqueeze(1) * 2 - 1  # (batch, 1, 1, 2)
            sampled = torch.nn.functional.grid_sample(
                noise_input, grid_coordinates,
                mode='bilinear', padding_mode='border', align_corners=False
            )
            jitter = sampled.squeeze(-1).squeeze(-1)  # (batch, 1)
        elif state_dim == 3:
            grid_coordinates = state.unsqueeze(1).unsqueeze(1).unsqueeze(1) * 2 - 1  # (batch, 1, 1, 1, 3)
            sampled = torch.nn.functional.grid_sample(
                noise_input, grid_coordinates,
                mode='bilinear', padding_mode='border', align_corners=False
            )
            jitter = sampled.squeeze(-1).squeeze(-1).squeeze(-1)  # (batch, 1)

        # Replicate single channel across all dimensions
        if jitter.shape[1] == 1 and self.dim > 1:
            jitter = jitter.repeat(1, self.dim)

        return jitter


if __name__ == '__main__':
    print("--- MindEyeActualization Example ---")

    # Define common parameters
    space_dim = 3
    num_attractors = 1
    noise_params = {"scale": 10.0, "octaves": 4, "persistence": 0.5, "lacunarity": 2.0}
    actualization_strength = 0.1
    iterations = 100

    # Define a target vector (archetype)
    archetype_vector = torch.tensor([0.5, 0.5, 0.5])

    # Initialize the model
    mind_model = MindEyeActualization(
        dim=space_dim,
        num_attractors=num_attractors,
        noise_params=noise_params,
        actualization_strength=actualization_strength,
        target_vector=archetype_vector,
        noise_decay='exponential',
        noise_decay_rate=0.95
    )

    # Generate noise field
    perturbation = PerturbationLayer(dim=space_dim, **noise_params)
    noise_field = perturbation((16, 16, 16))

    # Initial state (batch of specimen states)
    batch_size = 5
    initial_specimen_states = torch.rand(batch_size, space_dim)

    print(f"Initial specimen states (sample):\n{initial_specimen_states[0].detach().cpu().numpy()}")
    print(f"Target archetype vector: {archetype_vector.detach().cpu().numpy()}")

    # Run actualization (inference mode)
    final_states = mind_model(
        initial_specimen_states,
        generated_noise_field=noise_field,
        iterations=iterations,
        noise_amplitude=0.5,
        training_mode=False,
        force_schedule=True
    )

    print(f"\nFinal actualized states (sample):\n{final_states[0].detach().cpu().numpy()}")
    print(f"Distance to target: {torch.norm(final_states[0] - archetype_vector).item():.4f}")

    # Demonstrate training mode with learnable target
    print("\n--- Training Mode Demo ---")
    learnable_model = MindEyeActualization(
        dim=space_dim,
        num_attractors=1,
        noise_params=noise_params,
        actualization_strength=0.1,
        target_vector=None,  # Learnable
        noise_decay='exponential'
    )

    optimizer = torch.optim.Adam(learnable_model.parameters(), lr=0.05)
    desired_state = torch.tensor([[0.2, 0.3, 0.4]])

    for epoch in range(20):
        optimizer.zero_grad()

        # Use training_mode=True for short iterations to allow gradient flow
        result = learnable_model(
            torch.rand(1, space_dim),
            generated_noise_field=noise_field,
            iterations=10,  # Fewer iterations when training
            noise_amplitude=0.1,
            training_mode=True  # Gradients flow through all steps
        )

        loss = torch.mean((result - desired_state) ** 2)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss={loss.item():.4f}, "
                  f"Target={learnable_model.target_vector.squeeze().detach().cpu().numpy()}")

    print(f"\nFinal learned target: {learnable_model.target_vector.squeeze().detach().cpu().numpy()}")
    print(f"Desired state: {desired_state.squeeze().detach().cpu().numpy()}")
