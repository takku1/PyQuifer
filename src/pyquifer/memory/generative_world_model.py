from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from pyquifer.dynamics.oscillators.frequency_bank import FrequencyBank
from pyquifer.memory.diffusion import MindEyeActualization
from pyquifer.memory.perturbation import PerturbationLayer
from pyquifer.memory.potentials import MultiAttractorPotential


class GenerativeWorldModel(nn.Module):
    """
    A comprehensive model integrating noise, multi-frequency oscillators, potential fields,
    and a diffusion-like actualization process to simulate a "Mind's Eye" or Generative World Model.

    This model embodies the predictive processing feedback loop and leverages a Multi-Frequency Clock.
    """
    def __init__(self,
                 space_dim: int,
                 bank_configs: List[Dict[str, Any]], # Configuration for FrequencyBank
                 num_attractors: int,
                 noise_params: dict = None,
                 actualization_strength: float = 0.1):
        super().__init__()
        self.space_dim = space_dim
        self.num_attractors = num_attractors
        self.actualization_strength = actualization_strength  # Store for external access

        # Core Components
        self.perturbation_layer = PerturbationLayer(dim=space_dim, **(noise_params if noise_params else {}))
        self.potential_field = MultiAttractorPotential(num_attractors=num_attractors, dim=space_dim)

        # Replace single Kuramoto bank with a FrequencyBank
        self.frequency_bank = FrequencyBank(bank_configs=bank_configs)

        # Calculate total number of oscillators from all banks
        total_oscillators = sum(config.get("num_oscillators", 0) for config in bank_configs)

        # The Mind's Eye Actualization process, potentially guided by learnable archetypes
        self.mind_eye_actualization = MindEyeActualization(
            dim=space_dim,
            num_attractors=num_attractors, # Actualization can use its own attractors or be guided by self.potential_field
            noise_params=noise_params,
            actualization_strength=actualization_strength,
            target_vector=None # Let MindEyeActualization manage its target or make it learnable
        )

        # A central "Self" or Archetype vector that can be learned and used for top-down predictions
        self.archetype_vector = nn.Parameter(torch.randn(1, space_dim)) # Represents the "Goal"

        # Linear layer to map aggregated oscillator state (e.g., phases) to an influence on the actualization
        # or a prediction of the noise/potential.
        self.oscillator_to_prediction = nn.Linear(total_oscillators, space_dim)

        # Per-bank archetype projections (S-07): different archetype dimensions
        # drive different frequency banks with learned per-bank influence vectors
        self.archetype_to_bank = nn.ModuleList([
            nn.Linear(space_dim, config.get("num_oscillators", 0), bias=False)
            for config in bank_configs
        ])

        # Feedback gain for oscillator → archetype modulation (S-06)
        self.feedback_gain = nn.Parameter(torch.tensor(0.01))


    def forward(self,
                input_noise_shape: tuple,
                initial_specimen_state: torch.Tensor,
                actualization_iterations: int,
                oscillator_steps: int = 100,
                noise_amplitude: float = 1.0,
                time_offset: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, torch.Tensor]:
        """
        Runs one cycle of the generative world model, demonstrating the feedback loop.

        Args:
            input_noise_shape (tuple): The shape for the perturbation layer to generate noise.
            initial_specimen_state (torch.Tensor): The starting "SpecimenState" for actualization.
            actualization_iterations (int): Steps for the Mind's Eye Actualization process.
            oscillator_steps (int): Number of steps to simulate Kuramoto oscillators.
            noise_amplitude (float): Amplitude of noise for the actualization process.
            time_offset (float): Time offset for 4D noise generation.

        Returns:
            torch.Tensor: The final actualized state after one full cycle.
        """
        # --- Bottom-Up: Noise and Harmonics Coming In ---
        # 1. Generate Noise (The Canvas/Perturbation)
        generated_noise = self.perturbation_layer(input_noise_shape, time_offset=time_offset)

        # For demonstration, let's take a simplified "summary" of the noise
        # This can be more sophisticated (e.g., Fourier transform, feature extraction)
        # noise_summary = generated_noise.mean(dim=tuple(range(1, len(generated_noise.shape)))) # Mean over spatial dims

        # 2. Update Oscillators (Harmonics/Resonance) with Multi-Frequency Clock
        # S-07: Per-bank archetype projections — each bank gets a distinct,
        # learned projection of the archetype vector, scaled for stability
        external_inputs_for_banks = [
            0.1 * self.archetype_to_bank[i](self.archetype_vector).squeeze(0)
            for i, bank in enumerate(self.frequency_bank.banks)
        ]

        self.frequency_bank(external_inputs=external_inputs_for_banks, steps=oscillator_steps)

        # Get the concatenated current phases of all oscillators from all banks
        all_oscillator_phases = torch.cat(self.frequency_bank.get_all_phases(), dim=0)

        # Get the aggregated Kuramoto Order Parameter
        kuramoto_r = self.frequency_bank.get_aggregated_order_parameter()

        # S-06: Oscillator → archetype feedback (closed loop)
        # Oscillator dynamics shape the actualization target, creating
        # bidirectional coupling: archetype drives oscillators (S-07),
        # oscillators modulate archetype (S-06)
        oscillator_archetype_influence = self.oscillator_to_prediction(
            all_oscillator_phases.unsqueeze(0)
        )  # (1, space_dim)

        # Apply feedback with stability clipping
        feedback = self.feedback_gain * oscillator_archetype_influence
        feedback = feedback.clamp(-0.5, 0.5)  # Prevent runaway

        # Modulate the actualization target (not archetype_vector itself,
        # which is a slow-learning parameter; the target is per-forward)
        with torch.no_grad():
            modulated_target = self.archetype_vector + feedback.detach()
            self.mind_eye_actualization.target_vector.copy_(modulated_target)

        # --- Top-Down: The "Self" Vector Predicting ---
        # 3. "Mind's Eye" Actualization with oscillator-modulated target
        actualized_state = self.mind_eye_actualization(
            initial_state=initial_specimen_state,
            generated_noise_field=generated_noise,
            iterations=actualization_iterations,
            noise_amplitude=noise_amplitude,
            time_offset=time_offset
        )

        return actualized_state, generated_noise, all_oscillator_phases, kuramoto_r, oscillator_archetype_influence

if __name__ == '__main__':
    print("--- GenerativeWorldModel Example with FrequencyBank ---")

    # Model parameters
    space_dim = 3
    num_attractors = 3
    noise_params = {
        "scale": 15.0,
        "octaves": 3,
        "persistence": 0.7,
        "lacunarity": 2.0
    }
    actualization_strength = 0.05

    # Configuration for FrequencyBank - define fast and slow heartbeats
    bank_configs = [
        {"num_oscillators": 50, "dt": 0.005, "initial_frequency_range": (1.0, 2.0)}, # Fast heartbeat
        {"num_oscillators": 20, "dt": 0.05, "initial_frequency_range": (0.1, 0.5)}, # Slow heartbeat
    ]

    # Calculate total oscillators for context
    total_oscillators_in_banks = sum(config["num_oscillators"] for config in bank_configs)


    # Initialize the Generative World Model
    world_model = GenerativeWorldModel(
        space_dim=space_dim,
        bank_configs=bank_configs, # Pass bank_configs
        num_attractors=num_attractors,
        noise_params=noise_params,
        actualization_strength=actualization_strength
    )

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world_model.to(device)


    # Input for one cycle
    input_noise_shape = (16, 16, 16) # For 3D noise
    initial_specimen_state = torch.rand(1, space_dim).to(device) # One specimen
    actualization_iterations = 50
    oscillator_steps = 200
    noise_amplitude = 0.3
    time_offset = 0.0 # For 3D noise, can vary this for 4D animation

    print(f"Initial Archetype Vector: {world_model.archetype_vector.squeeze().detach().cpu().numpy()}")
    print(f"Initial Specimen State: {initial_specimen_state.squeeze().detach().cpu().numpy()}")

    # Run one forward pass
    final_state, noise_output, final_oscillator_phases, kuramoto_r, osc_influence = world_model(
        input_noise_shape,
        initial_specimen_state,
        actualization_iterations,
        oscillator_steps,
        noise_amplitude,
        time_offset
    )

    print(f"\nFinal Actualized State: {final_state.squeeze().detach().cpu().numpy()}")
    print(f"Generated Noise (mean over spatial dims): {noise_output.mean().item():.4f}")
    print(f"Total Oscillator Phases shape: {final_oscillator_phases.shape}")
    print(f"Aggregated Kuramoto Order Parameter R: {kuramoto_r:.4f}")
    print(f"Oscillator Archetype Influence shape: {osc_influence.shape}")
    print(f"Learned Archetype Vector in Actualization Module: {world_model.mind_eye_actualization.target_vector.squeeze().detach().cpu().numpy()}")
    print(f"Potential Field Attractor Positions (first): {world_model.potential_field.attractor_positions[0].detach().cpu().numpy()}")

    # Demonstrate a training step
    print("\n--- Demonstrating a Training Step ---")
    optimizer = torch.optim.Adam(world_model.parameters(), lr=0.005)

    # Define a goal: Make the actualized state close to [0.2, 0.4, 0.6]
    target_goal_state = torch.tensor([[0.2, 0.4, 0.6]]).to(device)

    for epoch in range(5):
        optimizer.zero_grad()

        # Generate some random initial state for this epoch
        current_initial_state = torch.rand(1, space_dim).to(device)

        # Run the model
        actualized_output, _, _, _, _ = world_model(
            input_noise_shape,
            current_initial_state,
            actualization_iterations=5,
            oscillator_steps=10,
            noise_amplitude=0.1,
            time_offset=epoch * 0.1
        )

        # Loss: how far is the actualized output from our target goal
        loss = torch.mean((actualized_output - target_goal_state)**2)

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}, "
              f"Actualized State: {actualized_output.squeeze().detach().cpu().numpy()}, "
              f"Archetype: {world_model.archetype_vector.squeeze().detach().cpu().numpy()}")
