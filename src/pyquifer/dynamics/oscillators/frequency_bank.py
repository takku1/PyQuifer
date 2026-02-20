import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from pyquifer.dynamics.oscillators.kuramoto import LearnableKuramotoBank

logger = logging.getLogger(__name__)

class FrequencyBank(nn.Module):
    """
    Manages multiple LearnableKuramotoBank instances, each potentially operating
    at a different characteristic frequency or with different configurations,
    representing different "speeds of thought" or processing streams.
    This acts as the "Multi-Frequency Clock" for PyQuifer.
    """
    def __init__(self, bank_configs: List[Dict[str, Any]]):
        """
        Initializes the FrequencyBank with a list of configurations for each
        LearnableKuramotoBank.

        Args:
            bank_configs (List[Dict[str, Any]]): A list of dictionaries, where each
                                                  dictionary contains parameters
                                                  for initializing a LearnableKuramotoBank.
                                                  Example:
                                                  [
                                                      {"num_oscillators": 50, "dt": 0.01, "initial_frequency_range": (0.5, 1.5)}, # Fast heartbeat
                                                      {"num_oscillators": 20, "dt": 0.1, "initial_frequency_range": (0.1, 0.3)}, # Slow heartbeat
                                                  ]
        """
        super().__init__()
        if not bank_configs:
            raise ValueError("FrequencyBank must be initialized with at least one bank configuration.")

        self.banks = nn.ModuleList()
        for i, config in enumerate(bank_configs):
            # Extract config parameters, providing defaults where necessary
            num_oscillators = config.get("num_oscillators", 100)
            dt = config.get("dt", 0.01)
            initial_frequency_range = config.get("initial_frequency_range", (0.5, 1.5))
            initial_phase_range = config.get("initial_phase_range", (0.0, 2 * torch.pi))

            bank = LearnableKuramotoBank(
                num_oscillators=num_oscillators,
                dt=dt,
                initial_frequency_range=initial_frequency_range,
                initial_phase_range=initial_phase_range
            )
            self.banks.append(bank)
            logger.info(f"FrequencyBank: Initialized bank {i+1} with {num_oscillators} oscillators, dt={dt}.")

    def forward(self, external_inputs: Optional[List[torch.Tensor]] = None, steps: int = 1):
        """
        Updates the phases of all Kuramoto banks for a given number of steps.

        Args:
            external_inputs (List[torch.Tensor], optional): A list of external inputs,
                                                            one for each bank. If None,
                                                            no external input is applied.
                                                            If provided, its length must
                                                            match the number of banks.
            steps (int): Number of simulation steps to perform for each bank.
        """
        if external_inputs is not None and len(external_inputs) != len(self.banks):
            raise ValueError(f"Number of external inputs ({len(external_inputs)}) must "
                             f"match number of banks ({len(self.banks)}).")

        for i, bank in enumerate(self.banks):
            current_input = external_inputs[i] if external_inputs is not None else None
            bank(external_input=current_input, steps=steps)

    def get_all_order_parameters(self) -> List[torch.Tensor]:
        """
        Retrieves the global order parameter R for each bank.

        Returns:
            List[torch.Tensor]: A list of R values, one for each bank.
        """
        return [bank.get_order_parameter() for bank in self.banks]

    def get_all_phases(self) -> List[torch.Tensor]:
        """
        Retrieves the current phases for all banks.

        Returns:
            List[torch.Tensor]: A list of phase tensors, one for each bank.
        """
        return [bank.phases for bank in self.banks]

    def get_aggregated_order_parameter(self) -> torch.Tensor:
        """
        Calculates an aggregated order parameter across all banks.
        Returns a tensor (diagnostic; oscillator state is detached by design).
        """
        all_rs = self.get_all_order_parameters()
        if not all_rs:
            return torch.tensor(0.0)
        return torch.stack(all_rs).mean()

    def get_num_banks(self) -> int:
        """Returns the number of Kuramoto banks in the FrequencyBank."""
        return len(self.banks)

if __name__ == '__main__':
    print("--- FrequencyBank Example ---")

    # Define configurations for different frequency banks
    bank_configs = [
        {"num_oscillators": 50, "dt": 0.005, "initial_frequency_range": (1.0, 2.0)}, # Fast
        {"num_oscillators": 30, "dt": 0.05, "initial_frequency_range": (0.1, 0.5)}, # Medium
        {"num_oscillators": 20, "dt": 0.2, "initial_frequency_range": (0.01, 0.1)}, # Slow
    ]

    freq_bank = FrequencyBank(bank_configs)

    print(f"\nNumber of banks initialized: {freq_bank.get_num_banks()}")

    # Simulate for a few steps
    simulation_steps = 100
    print("\nSimulating FrequencyBank...")
    for i in range(simulation_steps):
        # Example: apply a different external input to each bank
        external_inputs = [
            torch.sin(torch.tensor(i * bank.dt * 10)) * 0.1 * torch.ones(bank.num_oscillators)
            for bank in freq_bank.banks
        ]
        freq_bank(external_inputs=external_inputs, steps=1)

        if (i + 1) % 20 == 0:
            rs = freq_bank.get_all_order_parameters()
            aggregated_r = freq_bank.get_aggregated_order_parameter()
            print(f"Step {i+1}: R values = [{', '.join([f'{r:.4f}' for r in rs])}], Aggregated R = {aggregated_r:.4f}")

    final_rs = freq_bank.get_all_order_parameters()
    final_aggregated_r = freq_bank.get_aggregated_order_parameter()
    print(f"\nFinal R values: {final_rs}, Final Aggregated R: {final_aggregated_r:.4f}")

    # Demonstrate learnable parameters (e.g., of the first bank)
    print("\nDemonstrating learnable parameters of a bank within FrequencyBank...")
    first_bank_coupling = freq_bank.banks[0].coupling_strength.item()
    print(f"Initial coupling of first bank: {first_bank_coupling:.4f}")

    # For a real training loop, you'd optimize freq_bank.parameters()
    # This just shows that optimization is possible.
    optimizer = torch.optim.Adam(freq_bank.parameters(), lr=0.01)
    target_r_for_first_bank = torch.tensor(0.95)

    for epoch in range(5):
        optimizer.zero_grad()
        # Simulate the first bank (or all) to get its R
        # Resetting phases for each epoch for a clean start
        with torch.no_grad():
            freq_bank.banks[0].phases.copy_(torch.rand(freq_bank.banks[0].num_oscillators) * 2 * torch.pi)
        freq_bank.banks[0](steps=20)
        current_r = freq_bank.banks[0].get_order_parameter()

        loss = (target_r_for_first_bank - current_r)**2
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}, First Bank Coupling: {freq_bank.banks[0].coupling_strength.item():.4f}")

    print("\nLearning finished for first bank's coupling strength.")
