import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from pyquifer.models import GenerativeWorldModel

def run_simple_actualization_example():
    print("--- Running Simple Actualization Example ---")

    # 1. Model Parameters
    space_dim = 3
    num_attractors = 5
    noise_params = {
        "scale": 20.0,
        "octaves": 4,
        "persistence": 0.6,
        "lacunarity": 2.0,
        "seed": 123 # For reproducibility
    }
    actualization_strength = 0.08
    simulation_cycles = 20 # Number of full model cycles

    # Define bank configs for the FrequencyBank
    bank_configs = [
        {"num_oscillators": 70, "dt": 0.01, "initial_frequency_range": (0.5, 1.5)}, # 'Normal' heartbeat
        {"num_oscillators": 30, "dt": 0.005, "initial_frequency_range": (1.0, 2.0)}, # 'Fast' heartbeat
    ]

    # 2. Initialize the Generative World Model
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
    print(f"Model moved to device: {device}")

    # 3. Simulation Inputs
    input_noise_shape = (32, 32, 32) # For 3D noise (depth, height, width)
    # The 'specimen state' that gets actualized. Can be a batch.
    initial_specimen_state = torch.rand(1, space_dim).to(device) # One specimen starting randomly
    
    actualization_iterations_per_cycle = 50 # Internal steps for MindEyeActualization
    oscillator_steps_per_cycle = 100 # Internal steps for KuramotoBank
    noise_amplitude = 0.5
    time_offset = 0.0 # Initial time offset for 4D noise, will be incremented

    print(f"\nInitial Specimen State: {initial_specimen_state.squeeze().cpu().numpy()}")
    print(f"Initial Archetype Vector: {world_model.archetype_vector.squeeze().detach().cpu().numpy()}")

    # Store states for visualization
    specimen_state_history = [initial_specimen_state.squeeze().detach().cpu().numpy()]
    archetype_history = [world_model.archetype_vector.squeeze().detach().cpu().numpy()]
    order_parameter_history = []
    
    # 4. Run the Simulation Loop
    print("\n--- Running Simulation Cycles ---")
    for cycle in range(simulation_cycles):
        # We can evolve the time_offset to see dynamic 4D noise
        current_time_offset = time_offset + cycle * 0.1 # Example: time increases each cycle

        final_state, generated_noise, final_oscillator_phases, kuramoto_r = world_model(
            input_noise_shape,
            initial_specimen_state, # Note: using the *initial* state for each cycle, actualization is stateless
                                    # For a truly continuous simulation, feed previous final_state as new initial_state
            actualization_iterations_per_cycle,
            oscillator_steps_per_cycle,
            noise_amplitude,
            current_time_offset
        )
        
        # For a continuous simulation, update initial_specimen_state with the final_state from this cycle
        initial_specimen_state = final_state.detach()

        specimen_state_history.append(final_state.squeeze().detach().cpu().numpy())
        archetype_history.append(world_model.archetype_vector.squeeze().detach().cpu().numpy())
        order_parameter_history.append(kuramoto_r) # kuramoto_r is already a float from get_aggregated_order_parameter()

        print(f"Cycle {cycle+1:2d}: Final State={final_state.squeeze().detach().cpu().numpy()}, "
              f"Kuramoto R={kuramoto_r:.4f}, " # kuramoto_r is already a float
              f"Archetype={world_model.archetype_vector.squeeze().detach().cpu().numpy()}")

    print("\n--- Simulation Complete ---")
    
    # 5. Visualization (Basic)
    specimen_state_history = np.array(specimen_state_history)
    archetype_history = np.array(archetype_history)

    fig = plt.figure(figsize=(12, 6))

    # Plot 3D Specimen State Trajectory
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(specimen_state_history[:, 0], specimen_state_history[:, 1], specimen_state_history[:, 2], 
             marker='o', linestyle='-', color='blue', label='Specimen State Trajectory')
    ax1.plot(archetype_history[:, 0], archetype_history[:, 1], archetype_history[:, 2], 
             marker='x', linestyle='--', color='red', label='Archetype Evolution')
    ax1.scatter(specimen_state_history[0, 0], specimen_state_history[0, 1], specimen_state_history[0, 2], 
                color='green', s=100, label='Start State', zorder=5)
    ax1.scatter(specimen_state_history[-1, 0], specimen_state_history[-1, 1], specimen_state_history[-1, 2], 
                color='purple', s=100, label='End State', zorder=5)
    ax1.set_title("Specimen State and Archetype Evolution in 3D Space")
    ax1.set_xlabel("Dimension 1")
    ax1.set_ylabel("Dimension 2")
    ax1.set_zlabel("Dimension 3")
    ax1.legend()
    ax1.grid(True)

    # Plot Kuramoto Order Parameter over Cycles
    ax2 = fig.add_subplot(122)
    ax2.plot(range(1, simulation_cycles + 1), order_parameter_history, marker='o', linestyle='-', color='orange')
    ax2.set_title("Kuramoto Order Parameter (R) Evolution")
    ax2.set_xlabel("Simulation Cycle")
    ax2.set_ylabel("Order Parameter R")
    ax2.set_ylim(0, 1)
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def run_learning_example():
    print("\n--- Running Learning Example ---")

    # 1. Model Parameters (can be different from simple actualization)
    space_dim = 3
    num_attractors = 2
    noise_params = {
        "scale": 15.0,
        "octaves": 3,
        "persistence": 0.7,
        "lacunarity": 2.0,
        "seed": 42
    }
    actualization_strength = 0.05
    learning_epochs = 50
    # Define a target for the archetype to learn towards
    target_archetype_value = torch.tensor([0.8, -0.5, 1.2])

    # Define bank configs for the FrequencyBank
    bank_configs = [
        {"num_oscillators": 40, "dt": 0.01, "initial_frequency_range": (0.5, 1.5)}, # 'Normal' heartbeat
        {"num_oscillators": 10, "dt": 0.005, "initial_frequency_range": (1.0, 2.0)}, # 'Fast' heartbeat
    ]

    # 2. Initialize the Generative World Model
    # Here, we will allow the archetype_vector to be learned
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
    target_archetype_value = target_archetype_value.to(device)
    print(f"Model and target moved to device: {device}")

    # 3. Setup Optimizer
    # We want to optimize the archetype_vector of the model
    optimizer = torch.optim.Adam(world_model.parameters(), lr=0.01)

    # 4. Learning Loop
    archetype_learning_history = [world_model.archetype_vector.squeeze().detach().cpu().numpy()]
    loss_history = []

    print(f"\nInitial Archetype Vector: {world_model.archetype_vector.squeeze().detach().cpu().numpy()}")
    print(f"Target Archetype Value:   {target_archetype_value.cpu().numpy()}")

    for epoch in range(learning_epochs):
        optimizer.zero_grad() 
        
        # In a real scenario, input_noise_shape and initial_specimen_state might change
        # For simplicity, we use fixed ones here.
        input_noise_shape = (16, 16, 16)
        initial_specimen_state = torch.rand(1, space_dim).to(device)
        
        # Forward pass: Run the model
        final_actualized_state, _, _, kuramoto_r = world_model( # kuramoto_r is needed for printing
            input_noise_shape,
            initial_specimen_state,
            actualization_iterations=30, # Fewer iterations for faster training steps
            oscillator_steps=50,
            noise_amplitude=0.2,
            time_offset=epoch * 0.1 # Vary time offset for dynamic noise across epochs
        )
        
        # Calculate Loss: How far is the current archetype_vector from our target
        # We are training the archetype to converge to `target_archetype_value`
        loss = torch.mean((world_model.archetype_vector - target_archetype_value.unsqueeze(0))**2)
        
        loss.backward()
        optimizer.step()

        archetype_learning_history.append(world_model.archetype_vector.squeeze().detach().cpu().numpy())
        loss_history.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}, Loss: {loss.item():.6f}, Current Archetype: {world_model.archetype_vector.squeeze().detach().cpu().numpy()}, Aggregated R: {kuramoto_r:.4f}") # Also print R
                                                                        
    print("\n--- Learning Complete ---")
    print(f"Final Learned Archetype: {world_model.archetype_vector.squeeze().detach().cpu().numpy()}")
    
    # 5. Visualization of Learning
    archetype_learning_history = np.array(archetype_learning_history)

    fig = plt.figure(figsize=(14, 7))

    # Plot Archetype Evolution during Learning
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(archetype_learning_history[:, 0], archetype_learning_history[:, 1], archetype_learning_history[:, 2], 
             marker='.', linestyle='-', color='blue', label='Learned Archetype Trajectory')
    ax1.scatter(archetype_learning_history[0, 0], archetype_learning_history[0, 1], archetype_learning_history[0, 2], 
                color='green', s=100, label='Start Archetype', zorder=5)
    ax1.scatter(target_archetype_value.cpu().numpy()[0], target_archetype_value.cpu().numpy()[1], target_archetype_value.cpu().numpy()[2], 
                color='red', marker='X', s=150, label='Target Archetype', zorder=5)
    ax1.set_title("Archetype Evolution during Learning")
    ax1.set_xlabel("Dimension 1")
    ax1.set_ylabel("Dimension 2")
    ax1.set_zlabel("Dimension 3")
    ax1.legend()
    ax1.grid(True)

    # Plot Loss Evolution during Learning
    ax2 = fig.add_subplot(122)
    ax2.plot(range(1, learning_epochs + 1), loss_history, marker='.', linestyle='-', color='purple')
    ax2.set_title("Loss Evolution during Learning")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Mean Squared Error Loss")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run_simple_actualization_example()
    # Uncomment the line below to run the learning example
    run_learning_example()
