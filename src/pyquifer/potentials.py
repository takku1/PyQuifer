import torch
import torch.nn as nn
from typing import Optional

class MultiAttractorPotential(nn.Module):
    """
    Defines a potential field with multiple learnable attractor points.

    The potential at any given point in the space is a function of its distance
    to the various attractors. This creates a landscape that a "specimen" can navigate.
    """
    def __init__(self, num_attractors: int, dim: int, 
                 initial_strength_range=(0.5, 1.5), 
                 attractor_positions: Optional[torch.Tensor] = None): # New parameter
        """
        Initializes the potential field.

        Args:
            num_attractors (int): The number of attractor points in the field.
            dim (int): The dimensionality of the space (e.g., 2 for 2D, 3 for 3D).
            initial_strength_range (tuple, optional): The range for the initial random
                                                     strength of each attractor. Defaults to (0.5, 1.5).
            attractor_positions (torch.Tensor, optional): Pre-defined positions for the attractors.
                                                          If None, positions are initialized randomly.
        """
        super().__init__()
        self.num_attractors = num_attractors
        self.dim = dim

        if attractor_positions is not None:
            if attractor_positions.shape != (num_attractors, dim):
                raise ValueError(f"Provided attractor_positions shape {attractor_positions.shape} "
                                 f"does not match (num_attractors, dim) = ({num_attractors}, {dim})")
            self.attractor_positions = nn.Parameter(attractor_positions.float())
        else:
            # Learnable positions of the attractors
            # We initialize them randomly within a unit hypercube
            self.attractor_positions = nn.Parameter(torch.rand(num_attractors, dim))

        # Learnable strength of each attractor
        self.attractor_strengths = nn.Parameter(
            torch.rand(num_attractors) * (initial_strength_range[1] - initial_strength_range[0]) + initial_strength_range[0]
        )

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Calculates the potential energy at given positions in the space.

        The potential is calculated as the sum of the influences of all attractors.
        The influence of an attractor is its strength divided by the squared distance.
        A lower potential value means the position is "more favorable" or "closer to an attractor".

        Args:
            positions (torch.Tensor): A tensor of shape (num_positions, dim) representing
                                      the points at which to calculate the potential.

        Returns:
            torch.Tensor: A tensor of shape (num_positions,) containing the potential
                          energy at each input position.
        """
        if positions.dim() != 2 or positions.shape[1] != self.dim:
            raise ValueError(f"Input positions must be a tensor of shape (num_positions, {self.dim})")

        # Expand dimensions for broadcasting:
        # positions: (num_positions, 1, dim)
        # attractors: (1, num_attractors, dim)
        expanded_positions = positions.unsqueeze(1)
        expanded_attractors = self.attractor_positions.unsqueeze(0)

        # Calculate squared distances between all positions and all attractors
        # Resulting shape: (num_positions, num_attractors)
        dist_sq = torch.sum((expanded_positions - expanded_attractors)**2, dim=2)

        # Avoid division by zero by adding a small epsilon
        dist_sq = dist_sq + 1e-8

        # Potential is inversely proportional to squared distance, scaled by strength.
        # We use a negative sign so that lower potential is better (closer to attractor).
        # Shape: (num_attractors,) -> (1, num_attractors)
        strengths = self.attractor_strengths.unsqueeze(0)
        
        # Shape: (num_positions, num_attractors)
        potential_per_attractor = -strengths / dist_sq

        # Sum the potentials from all attractors for each position
        # Resulting shape: (num_positions,)
        total_potential = torch.sum(potential_per_attractor, dim=1)

        return total_potential

    def get_force(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Calculates the force (negative gradient of the potential) at given positions.
        The force vector points in the direction of the steepest descent of the potential field.

        Args:
            positions (torch.Tensor): A tensor of shape (num_positions, dim) for which to calculate the force.
                                      Requires gradient tracking to be enabled.

        Returns:
            torch.Tensor: A tensor of shape (num_positions, dim) representing the force vector at each position.
        """
        if not positions.requires_grad:
            positions.requires_grad_(True)
        
        # Calculate potential
        potential = self.forward(positions)

        # The gradient of the potential is the "anti-force".
        # We need to sum the potential to get a scalar for autograd to work on multiple positions.
        # The gradient is computed for each element of `positions` that contributed to the sum.
        grad_outputs = torch.ones_like(potential)
        gradient = torch.autograd.grad(
            outputs=potential,
            inputs=positions,
            grad_outputs=grad_outputs,
            create_graph=True  # Create graph for higher-order derivatives if needed
        )[0]

        # Force is the negative of the gradient
        return -gradient

if __name__ == '__main__':
    print("--- MultiAttractorPotential Example ---")
    num_attractors = 3
    dim = 2
    potential_field = MultiAttractorPotential(num_attractors, dim)

    print(f"Initialized a potential field in {dim}D with {num_attractors} attractors.")
    print("Learnable Attractor Positions:\n", potential_field.attractor_positions.data)
    print("Learnable Attractor Strengths:\n", potential_field.attractor_strengths.data)

    # Create a grid of points to visualize the potential
    grid_size = 10
    x = torch.linspace(-0.5, 1.5, grid_size)
    y = torch.linspace(-0.5, 1.5, grid_size)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    test_positions = torch.stack([grid_x.ravel(), grid_y.ravel()], dim=1)

    # Calculate potential on the grid
    potential_values = potential_field(test_positions)
    print(f"\nCalculated potential for {test_positions.shape[0]} points.")
    print("Potential values shape:", potential_values.shape)
    
    # Calculate force at a specific point
    single_position = torch.tensor([[0.5, 0.5]], requires_grad=True)
    force_vector = potential_field.get_force(single_position)
    print(f"\nForce at position {single_position.data.numpy()}: {force_vector.data.numpy()}")

    # Demonstrate optimization
    print("\n--- Optimizing Attractor Positions ---")
    optimizer = torch.optim.Adam(potential_field.parameters(), lr=0.1)
    
    # Let's say we want to move the attractors to attract a point at [0,0]
    target_position = torch.tensor([[0.0, 0.0]])
    
    for epoch in range(5):
        optimizer.zero_grad()
        
        # The goal is to maximize the potential's pull at the target position.
        # We can do this by minimizing the potential value at that point (making it more negative).
        loss = potential_field(target_position)
        
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1:2d}, Loss (Potential at [0,0]): {loss.item():.4f}")
    
    print("\nOptimization finished. Attractor positions should have moved closer to [0,0].")
    print("Final Attractor Positions:\n", potential_field.attractor_positions.data)
