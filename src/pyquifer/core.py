import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, Any, List, Callable, Literal
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA

from .models import GenerativeWorldModel
from .frequency_bank import FrequencyBank

class PyQuifer(nn.Module):
    """
    The official PyQuifer API.
    This class serves as the 'Laminar Bridge' connecting raw data to the Generative World Model,
    embodying the philosophy of 'Physical Intelligence' meeting 'Cognitive Architecture'.
    """
    def __init__(self,
                 space_dim: int = 3,
                 bank_configs: Optional[List[Dict[str, Any]]] = None,
                 num_attractors: int = 5,
                 perturbation_params: Optional[Dict[str, Any]] = None,
                 actualization_strength: float = 0.08,
                 viscosity_control_enabled: bool = True,
                 viscosity_constant: float = 0.01,
                 kuramoto_threshold: Optional[float] = None,
                 dim_reduction: Literal['none', 'pca', 'truncate'] = 'pca',
                 device: Optional[Union[str, torch.device]] = None):
        """
        Initializes the PyQuifer system.

        Args:
            space_dim (int): The dimensionality of the latent space for actualization.
            bank_configs (List[Dict[str, Any]], optional): Configurations for the FrequencyBank.
                                                           If None, a default single bank config is used.
            num_attractors (int): Number of attractor points in the potential field.
            perturbation_params (dict, optional): Parameters for the PerturbationLayer
                                                 (scale, octaves, persistence, lacunarity, seed).
            actualization_strength (float): The step size for the actualization process.
            viscosity_control_enabled (bool): If True, automatically adjusts actualization_strength
                                               based on data characteristics (Viscosity Control).
            viscosity_constant (float): Constant 'C' for the heuristic: actualization_strength = C / (data_variance + epsilon).
            kuramoto_threshold (float, optional): Kuramoto Order Parameter (R) threshold for early termination.
            dim_reduction (str): Dimensionality reduction method when data exceeds space_dim.
                - 'none': Raise error if dims exceed space_dim
                - 'pca': Use PCA to reduce to space_dim (default)
                - 'truncate': Simply take first space_dim dimensions
            device (str or torch.device, optional): The device to run the model on (e.g., 'cpu', 'cuda').
        """
        super().__init__()
        self.space_dim = space_dim
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(self.device)

        # Default bank configuration if not provided
        if bank_configs is None:
            bank_configs = [{"num_oscillators": 100, "dt": 0.01, "initial_frequency_range": (0.5, 1.5)}] # Default 'slow' bank
            bank_configs.append({"num_oscillators": 50, "dt": 0.005, "initial_frequency_range": (1.0, 2.0)}) # Default 'fast' bank

        # Initialize the core Generative World Model
        self.model = GenerativeWorldModel(
            space_dim=space_dim,
            bank_configs=bank_configs, # Pass bank_configs here
            num_attractors=num_attractors,
            noise_params=perturbation_params,
            actualization_strength=actualization_strength # Initial value, potentially overridden by viscosity control
        ).to(self.device)

        print(f"PyQuifer initialized on device: {self.device}")
        
        # Internal state for data handling (Automated Sieve)
        self._data_schema: Optional[Dict[str, Any]] = None
        self._feature_scalers: Dict[str, Any] = {}
        self._one_hot_encoders: Dict[str, Any] = {}
        self._raw_data_storage: Optional[Any] = None
        self._processed_data_storage: Optional[torch.Tensor] = None
        self._dim_reduction = dim_reduction
        self._pca_model: Optional[PCA] = None

        # Viscosity Control parameters
        self._viscosity_control_enabled = viscosity_control_enabled
        self._viscosity_constant = viscosity_constant

        # Kuramoto Live Feed parameters
        self._kuramoto_threshold = kuramoto_threshold
        self._kuramoto_callback: Optional[Callable[[float, int, str], None]] = None # Callback (R, cycle/epoch, context)

    def _preprocess_data(self, data: Union[pd.DataFrame, np.ndarray, List[Dict]], is_training_data: bool = True) -> np.ndarray:
        """
        Internal method for data preparation and feature mapping - the 'Automated Sieve'.
        Scans, maps, and scales data to fit the model's space_dim.
        """
        if isinstance(data, List):
            # Attempt to convert list of dicts to DataFrame
            data = pd.DataFrame(data)

        if isinstance(data, np.ndarray):
            # If NumPy array, assume all numerical for now
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            raise TypeError("Unsupported data type for preprocessing. Please provide pandas DataFrame, numpy array, or list of dicts.")

        processed_features = []
        
        for col in df.columns:
            feature_data = df[col].values.reshape(-1, 1)

            if pd.api.types.is_numeric_dtype(df[col]):
                # Numerical data: Min-Max Scaling
                scaler_key = f"numerical_scaler_{col}"
                if is_training_data:
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    processed = scaler.fit_transform(feature_data)
                    self._feature_scalers[scaler_key] = scaler
                else:
                    scaler = self._feature_scalers.get(scaler_key)
                    if scaler is None:
                        raise ValueError(f"Scaler for numerical column '{col}' not found. Ensure training data was ingested first or provide is_training_data=True.")
                    processed = scaler.transform(feature_data)
                processed_features.append(processed)

            elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                # Categorical data: One-Hot Encoding (basic 'phase space' mapping)
                encoder_key = f"one_hot_encoder_{col}"
                if is_training_data:
                    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                    processed = encoder.fit_transform(feature_data)
                    self._one_hot_encoders[encoder_key] = encoder
                else:
                    encoder = self._one_hot_encoders.get(encoder_key)
                    if encoder is None:
                        raise ValueError(f"OneHotEncoder for categorical column '{col}' not found. Ensure training data was ingested first or provide is_training_data=True.")
                    processed = encoder.transform(feature_data)
                processed_features.append(processed)

            else:
                print(f"Warning: Column '{col}' has an unsupported data type ({df[col].dtype}) and will be ignored.")

        if not processed_features:
            raise ValueError("No processable features found in the ingested data.")

        # Concatenate all processed features
        processed_data_np = np.hstack(processed_features)

        # Dimensionality Alignment
        if processed_data_np.shape[1] < self.space_dim:
            # Pad with zeros if data has fewer dimensions than space_dim
            padding_needed = self.space_dim - processed_data_np.shape[1]
            padding = np.zeros((processed_data_np.shape[0], padding_needed))
            processed_data_np = np.hstack([processed_data_np, padding])
            print(f"Data padded with {padding_needed} dimensions to match space_dim ({self.space_dim}).")
        elif processed_data_np.shape[1] > self.space_dim:
            # Apply dimensionality reduction based on configured method
            original_dims = processed_data_np.shape[1]

            if self._dim_reduction == 'none':
                raise ValueError(f"Processed data dimensions ({original_dims}) "
                                 f"exceed PyQuifer's space_dim ({self.space_dim}). "
                                 f"Set dim_reduction='pca' or 'truncate' to handle this automatically.")

            elif self._dim_reduction == 'truncate':
                # Simply take first space_dim dimensions
                processed_data_np = processed_data_np[:, :self.space_dim]
                print(f"Data truncated from {original_dims} to {self.space_dim} dimensions.")

            elif self._dim_reduction == 'pca':
                # Use PCA to reduce dimensions while preserving variance
                if is_training_data:
                    self._pca_model = PCA(n_components=self.space_dim)
                    processed_data_np = self._pca_model.fit_transform(processed_data_np)
                    explained_var = sum(self._pca_model.explained_variance_ratio_) * 100
                    print(f"PCA reduced {original_dims} dims to {self.space_dim} "
                          f"(explained variance: {explained_var:.1f}%).")
                else:
                    if self._pca_model is None:
                        raise ValueError("PCA model not fitted. Call with is_training_data=True first.")
                    processed_data_np = self._pca_model.transform(processed_data_np)

        return processed_data_np

    def _calculate_viscosity_params(self, processed_data: torch.Tensor):
        """
        Internal method to calculate hyperparameters based on data characteristics.
        This is the core of the 'Viscosity Control'.
        """
        if processed_data.numel() == 0:
            print("Warning: Processed data is empty, cannot calculate viscosity parameters.")
            return

        # Calculate variance of the processed data (across all features for simplicity)
        data_variance = torch.var(processed_data).item()
        
        # Heuristic: actualization_strength = C / (data_variance + epsilon)
        # Higher variance means more spread-out data, suggesting we might need a lower
        # actualization strength (more 'friction') to prevent overshooting.
        # Conversely, lower variance might allow for higher actualization strength (less 'friction').
        epsilon = 1e-6 # To prevent division by zero for very uniform data
        new_actualization_strength = self._viscosity_constant / (data_variance + epsilon)

        # Clamp the actualization_strength to a reasonable range to prevent extreme values
        min_strength, max_strength = 0.01, 0.5
        new_actualization_strength = max(min_strength, min(max_strength, new_actualization_strength))
        
        self.model.mind_eye_actualization.actualization_strength = new_actualization_strength
        self.model.actualization_strength = new_actualization_strength  # Keep in sync
        print(f"Viscosity Control: Adjusted actualization_strength to {new_actualization_strength:.4f} "
              f"based on data variance ({data_variance:.4f}).")

    def ingest_data(self, data: Union[pd.DataFrame, np.ndarray, List[Dict]], feature_mapping: Optional[Dict[str, str]] = None):
        """
        Ingests raw data into the PyQuifer system. This method now utilizes the 'Automated Sieve'
        for preparing the data and triggers 'Viscosity Control'.

        Args:
            data (Union[pd.DataFrame, np.ndarray, List[Dict]]): The raw input data.
            feature_mapping (dict, optional): Manual mapping of raw features to
                                              PyQuifer's internal dimensions/concepts.
                                              The 'Sieve' aims to automate this, but explicit
                                              mapping can be provided here for future use.
        """
        print("Ingesting data...")
        # Store raw data (optional, for debugging or later re-processing)
        self._raw_data_storage = data 
        
        # Use the Automated Sieve to preprocess the data
        processed_np = self._preprocess_data(data, is_training_data=True)
        
        self._processed_data_storage = torch.from_numpy(processed_np).float().to(self.device)
        print(f"Data ingested and processed. Shape: {self._processed_data_storage.shape}")

        # Apply Viscosity Control after data ingestion and processing
        if self._viscosity_control_enabled:
            self._calculate_viscosity_params(self._processed_data_storage)


    def actualize_vision(self, 
                         input_noise_shape: tuple,
                         actualization_iterations: int = 50,
                         oscillator_steps: int = 100,
                         noise_amplitude: float = 0.5,
                         time_offset: float = 0.0,
                         num_cycles: int = 1) -> Dict[str, Union[torch.Tensor, np.ndarray, float]]:
        """
        Runs the 'Mind's Eye' actualization process for the ingested data,
        generating a coherent internal representation.

        Args:
            input_noise_shape (tuple): The shape for the perturbation layer to generate noise (e.g., (32,32,32)).
            actualization_iterations (int): Steps for the MindEyeActualization process per cycle.
            oscillator_steps (int): Number of steps to simulate Kuramoto oscillators per cycle.
            noise_amplitude (float): Amplitude of noise for the actualization process.
            time_offset (float): Time offset for 4D noise generation, can be evolved.
            num_cycles (int): Number of simulation cycles to run.

        Returns:
            dict: A dictionary containing the final actualized states, Kuramoto order parameter,
                  and potentially other metrics.
        """
        if self._processed_data_storage is None:
            raise ValueError("No data ingested. Please call ingest_data() first.")

        # Use the ingested and processed data as the initial specimen states
        initial_specimen_states = self._processed_data_storage
        
        final_actualized_states = []
        kuramoto_rs = []
        archetypes = []
        
        print(f"\n--- Running Actualization for {num_cycles} cycles ---")
        for cycle in range(num_cycles):
            current_time_offset = time_offset + cycle * 0.1 # Example: time evolves

            final_state, generated_noise, final_oscillator_phases, kuramoto_r = self.model(
                input_noise_shape,
                initial_specimen_states,
                actualization_iterations,
                oscillator_steps,
                noise_amplitude,
                current_time_offset
            )
            
            # For continuous simulation, feed previous final_state as new initial_specimen_states
            initial_specimen_states = final_state.detach()

            final_actualized_states.append(final_state.cpu())
            kuramoto_rs.append(kuramoto_r) # Removed .item()
            archetypes.append(self.model.archetype_vector.cpu().detach())

            # Kuramoto Live Feed
            if self._kuramoto_callback:
                self._kuramoto_callback(kuramoto_r, cycle, "actualize_vision")
            if self._kuramoto_threshold is not None and kuramoto_r >= self._kuramoto_threshold:
                print(f"Kuramoto Live Feed: R ({kuramoto_r:.4f}) exceeded threshold ({self._kuramoto_threshold:.4f}). Early termination.")
                break

            print(f"Cycle {cycle+1:2d}: Final State Sample={final_state[0].squeeze().detach().cpu().numpy()}, Kuramoto R={kuramoto_r:.4f}")

        return {
            "final_actualized_states": torch.stack(final_actualized_states),
            "kuramoto_order_parameter_history": kuramoto_rs,
            "archetype_history": torch.stack(archetypes)
        }

    def fit(self,
            data: Union[pd.DataFrame, np.ndarray, List[Dict]],
            target_archetype_value: Union[np.ndarray, List[float], torch.Tensor],
            input_noise_shape: tuple,
            learning_epochs: int = 50,
            actualization_iterations: int = 30,
            oscillator_steps: int = 50,
            noise_amplitude: float = 0.2,
              lr: float = 0.01) -> Dict[str, Union[List[float], np.ndarray]]:
        """
        Trains the PyQuifer model, specifically optimizing the archetype_vector
        to converge towards a target value. This demonstrates the 'learning' aspect.

        Args:
            data (Union[pd.DataFrame, np.ndarray, List[Dict]]): Data to be ingested for training.
            target_archetype_value (Union[np.ndarray, List[float], torch.Tensor]): The target
                                                                                   3D vector the archetype
                                                                                   should learn.
            input_noise_shape (tuple): Shape for perturbation layer.
            learning_epochs (int): Number of epochs to train for.
            actualization_iterations (int): Iterations for MindEyeActualization per epoch.
            oscillator_steps (int): Steps for Kuramoto oscillators per epoch.
            noise_amplitude (float): Noise amplitude for actualization.
            lr (float): Learning rate for the optimizer.

        Returns:
            dict: History of loss and learned archetype during training.
        """
        # Ingest data using the updated Automated Sieve and trigger Viscosity Control
        self.ingest_data(data) # This will store processed data in _processed_data_storage
        
        if not isinstance(target_archetype_value, torch.Tensor):
            target_archetype_value = torch.tensor(target_archetype_value, dtype=torch.float32)
        target_archetype_value = target_archetype_value.to(self.device)

        # We need to ensure the archetype_vector is learnable.
        # It is by default in GenerativeWorldModel.
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        archetype_learning_history = [self.model.archetype_vector.squeeze().detach().cpu().numpy()]
        loss_history = []

        print(f"\n--- Starting Training for {learning_epochs} Epochs ---")
        print(f"Initial Archetype Vector: {self.model.archetype_vector.squeeze().detach().cpu().numpy()}")
        print(f"Target Archetype Value:   {target_archetype_value.cpu().numpy()}")

        for epoch in range(learning_epochs):
            optimizer.zero_grad()
            
            # Use ingested data as initial specimen states for training
            initial_specimen_states = self._processed_data_storage 

            # Forward pass: Run the model
            final_actualized_state, _, _, _ = self.model(
                input_noise_shape,
                initial_specimen_states,
                actualization_iterations,
                oscillator_steps,
                noise_amplitude,
                time_offset=epoch * 0.1 # Vary time offset for dynamic noise across epochs
            )
            
            # Loss: How far is the current archetype_vector from our target
            # Note(S-14): Only archetype_vector receives backprop gradients here.
            # Oscillators evolve through their own Kuramoto dynamics (called in forward()),
            # not through this loss — this is by design (see SOMATIC_STEERING.md).
            # The actualization layer could benefit from a reconstruction loss in the future.
            loss = torch.mean((self.model.archetype_vector - target_archetype_value.unsqueeze(0))**2)
            
            # Get Kuramoto Order Parameter for Live Feed
            kuramoto_r = self.model.frequency_bank.get_aggregated_order_parameter() # Use aggregated R

            # Kuramoto Live Feed
            if self._kuramoto_callback:
                self._kuramoto_callback(kuramoto_r, epoch, "train")
            if self._kuramoto_threshold is not None and kuramoto_r >= self._kuramoto_threshold:
                print(f"Kuramoto Live Feed: R ({kuramoto_r:.4f}) exceeded threshold ({self._kuramoto_threshold:.4f}). Early termination.")
                break


            loss.backward()
            optimizer.step()

            archetype_learning_history.append(self.model.archetype_vector.squeeze().detach().cpu().numpy())
            loss_history.append(loss.item())

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}, Loss: {loss.item():.6f}, Current Archetype: {self.model.archetype_vector.squeeze().detach().cpu().numpy()}")

        print("\n--- Training Complete ---")
        print(f"Final Learned Archetype: {self.model.archetype_vector.squeeze().detach().cpu().numpy()}")
        
        return {
            "archetype_history": np.array(archetype_learning_history),
            "loss_history": loss_history
        }

    def set_viscosity_control(self, enable: bool, viscosity_constant: Optional[float] = None):
        """
        Enables or disables automatic Viscosity Control.
        If enabled, actualization_strength is adjusted based on data characteristics.
        """
        self._viscosity_control_enabled = enable
        if viscosity_constant is not None:
            self._viscosity_constant = viscosity_constant
        print(f"Viscosity Control: {'Enabled' if enable else 'Disabled'}. Constant set to {self._viscosity_constant}.")

    def set_kuramoto_live_feed(self, threshold: Optional[float] = None, callback: Optional[Callable[[float, int, str], None]] = None):
        """
        Configures the Kuramoto Live Feed for real-time monitoring and early termination.

        Args:
            threshold (float, optional): Kuramoto Order Parameter (R) threshold for early termination.
                                         If R >= threshold, the current simulation/training loop will stop.
            callback (Callable[[float, int, str], None], optional): A function to call with
                                                                      (kuramoto_r, cycle_or_epoch_num, context_string)
                                                                      at each monitoring point.
        """
        self._kuramoto_threshold = threshold
        self._kuramoto_callback = callback
        print(f"Kuramoto Live Feed: Threshold set to {threshold if threshold is not None else 'None'}, Callback {'set' if callback else 'not set'}.")

    def get_realtime_metrics(self) -> Dict[str, float]:
        """Provides real-time metrics like Kuramoto Order Parameter."""
        if hasattr(self.model, 'frequency_bank'): # Changed from kuramoto_bank to frequency_bank
            return {"kuramoto_order_parameter": self.model.frequency_bank.get_aggregated_order_parameter()}
        return {}


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Example Usage of the PyQuifer API

    # 1. Initialize PyQuifer
    # With viscosity control enabled by default
    # Define bank configs for the FrequencyBank
    default_bank_configs = [
        {"num_oscillators": 100, "dt": 0.01, "initial_frequency_range": (0.5, 1.5)}, # Default 'slow' bank
        {"num_oscillators": 50, "dt": 0.005, "initial_frequency_range": (1.0, 2.0)}, # Default 'fast' bank
    ]

    pq = PyQuifer(
        space_dim=3,
        bank_configs=default_bank_configs, # Pass bank configs
        num_attractors=3,
        perturbation_params={"seed": 789},
        viscosity_control_enabled=True
    )

    # Configure Kuramoto Live Feed
    def my_kuramoto_callback(r_value, step_num, context):
        if (step_num + 1) % 5 == 0: # Only print every 5 steps
            print(f"    CALLBACK: {context} - Step {step_num+1}: Aggregated Kuramoto R = {r_value:.4f}")

    pq.set_kuramoto_live_feed(threshold=0.8, callback=my_kuramoto_callback)


    # 2. Ingest Data (Simulated for now) - This will trigger Viscosity Control
    # Shape: (num_specimens, space_dim)
    sample_data = np.random.rand(10, 3) # 10 specimens in 3D space
    pq.ingest_data(sample_data)

    # 3. Run Actualization
    actualization_results = pq.actualize_vision(
        input_noise_shape=(20, 20, 20),
        actualization_iterations=75,
        num_cycles=10,
        noise_amplitude=0.6
    )

    print("\nActualization Results:")
    print(f"Final Actualized State Sample: {actualization_results['final_actualized_states'][-1, 0].squeeze().numpy()}")
    print(f"Kuramoto R History (last 5): {actualization_results['kuramoto_order_parameter_history'][-5:]}")

    # 4. Train PyQuifer (e.g., to learn a specific archetype)
    target_archetype = [0.1, 0.2, 0.3]
    training_data = np.random.rand(5, 3) # Some training data
    learning_results = pq.fit(
        data=training_data,
        target_archetype_value=target_archetype,
        input_noise_shape=(10, 10, 10),
        learning_epochs=20,
        lr=0.05
    )

    print("\nLearning Results:")
    print(f"Final Learned Archetype: {learning_results['archetype_history'][-1]}")
    print(f"Final Loss: {learning_results['loss_history'][-1]:.6f}")

    # Basic plot for learning (similar to simple_actualization example)
    archetype_learning_history = learning_results['archetype_history']
    loss_history = learning_results['loss_history']
    target_archetype_value_tensor = torch.tensor(target_archetype)

    fig = plt.figure(figsize=(14, 7))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(archetype_learning_history[:, 0], archetype_learning_history[:, 1], archetype_learning_history[:, 2], 
             marker='.', linestyle='-', color='blue', label='Learned Archetype Trajectory')
    ax1.scatter(archetype_learning_history[0, 0], archetype_learning_history[0, 1], archetype_learning_history[0, 2], 
                color='green', s=100, label='Start Archetype', zorder=5)
    ax1.scatter(target_archetype_value_tensor[0], target_archetype_value_tensor[1], target_archetype_value_tensor[2], 
                color='red', marker='X', s=150, label='Target Archetype', zorder=5)
    ax1.set_title("Archetype Evolution during Learning (PyQuifer API)")
    ax1.set_xlabel("Dimension 1")
    ax1.set_ylabel("Dimension 2")
    ax1.set_zlabel("Dimension 3")
    ax1.legend()
    ax1.grid(True)

    ax2 = fig.add_subplot(122)
    ax2.plot(range(1, len(loss_history) + 1), loss_history, marker='.', linestyle='-', color='purple')
    ax2.set_title("Loss Evolution during Learning (PyQuifer API)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Mean Squared Error Loss")
    ax2.grid(True)
    plt.tight_layout()
    plt.show()
