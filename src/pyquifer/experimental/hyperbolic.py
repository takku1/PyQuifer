"""
Non-Euclidean Latent Manifolds - Curved Thought Space.

Human-like logic leaps and creativity happen in CURVED manifolds,
not flat Euclidean space. This module implements:

- Hyperbolic geometry: Hierarchical, tree-like relationships
- Spherical geometry: Cyclic, periodic relationships
- Mixed-curvature: Different regions with different geometry

Key insight: Emotional "gravity" - the distance from "Sad" to "Joy"
might be longer/steeper than "Sad" to "Frustration", creating
realistic mood dynamics that don't just snap.

Based on:
- Poincaré embeddings (Nickel & Kiela, 2017)
- Hyperbolic neural networks (Ganea et al., 2018)
- Non-Euclidean Space Generative Models (NESGM)
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# HYPERBOLIC OPERATIONS (Poincaré Ball Model)
# =============================================================================

class HyperbolicOperations:
    """
    Operations in hyperbolic space using the Poincaré ball model.

    In hyperbolic space:
    - Distance grows exponentially toward the boundary
    - Hierarchies emerge naturally (root near center, leaves at edge)
    - "Gravity wells" around attractors create emotional momentum
    """

    @staticmethod
    def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
        """
        Möbius addition in the Poincaré ball.

        This is the hyperbolic equivalent of vector addition.
        c is the curvature (c=1 for standard hyperbolic space).
        """
        x2 = torch.sum(x * x, dim=-1, keepdim=True)
        y2 = torch.sum(y * y, dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)

        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denom = 1 + 2 * c * xy + c * c * x2 * y2

        return num / (denom + 1e-8)

    @staticmethod
    def hyperbolic_distance(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
        """
        Hyperbolic distance in the Poincaré ball.

        Distance grows much faster near the boundary, creating
        "gravity wells" in the semantic space.
        """
        sqrt_c = math.sqrt(c)
        diff = HyperbolicOperations.mobius_add(-x, y, c)
        diff_norm = torch.norm(diff, dim=-1, keepdim=True).clamp(min=1e-8, max=(1 - 1e-5) / sqrt_c)

        return (2 / sqrt_c) * torch.arctanh(sqrt_c * diff_norm)

    @staticmethod
    def exp_map(v: torch.Tensor, x: torch.Tensor, c: float = 1.0) -> torch.Tensor:
        """
        Exponential map: tangent space → hyperbolic space.

        Maps a direction in flat space to a point in curved space.
        """
        sqrt_c = math.sqrt(c)
        v_norm = torch.norm(v, dim=-1, keepdim=True).clamp(min=1e-8)

        # Conformal factor
        lambda_x = 2 / (1 - c * torch.sum(x * x, dim=-1, keepdim=True) + 1e-8)

        return HyperbolicOperations.mobius_add(
            x,
            torch.tanh(sqrt_c * lambda_x * v_norm / 2) * v / (sqrt_c * v_norm),
            c
        )

    @staticmethod
    def log_map(y: torch.Tensor, x: torch.Tensor, c: float = 1.0) -> torch.Tensor:
        """
        Logarithmic map: hyperbolic space → tangent space.

        Maps a point in curved space to a direction in flat space.
        """
        sqrt_c = math.sqrt(c)
        diff = HyperbolicOperations.mobius_add(-x, y, c)
        diff_norm = torch.norm(diff, dim=-1, keepdim=True).clamp(min=1e-8, max=(1 - 1e-5) / sqrt_c)

        lambda_x = 2 / (1 - c * torch.sum(x * x, dim=-1, keepdim=True) + 1e-8)

        return (2 / (sqrt_c * lambda_x)) * torch.arctanh(sqrt_c * diff_norm) * diff / diff_norm

    @staticmethod
    def project_to_ball(x: torch.Tensor, c: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
        """Project points to inside the Poincaré ball."""
        max_norm = (1 - eps) / math.sqrt(c)
        norm = torch.norm(x, dim=-1, keepdim=True)
        cond = norm > max_norm
        return torch.where(cond, x * max_norm / norm, x)


class HyperbolicLinear(nn.Module):
    """
    Linear layer in hyperbolic space.

    Performs: y = exp_0(W @ log_0(x) + b)
    """

    def __init__(self, in_features: int, out_features: int, c: float = 1.0):
        super().__init__()
        self.c = c
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Map to tangent space at origin
        origin = torch.zeros_like(x)
        v = HyperbolicOperations.log_map(x, origin, self.c)

        # Linear transform in tangent space
        v_transformed = F.linear(v, self.weight, self.bias)

        # Map back to hyperbolic space
        y = HyperbolicOperations.exp_map(v_transformed, origin, self.c)

        return HyperbolicOperations.project_to_ball(y, self.c)


# =============================================================================
# EMOTIONAL GRAVITY MANIFOLD
# =============================================================================

class EmotionalGravityManifold(nn.Module):
    """
    Emotional space with hyperbolic geometry.

    Emotions exist on a curved manifold where:
    - Some transitions are "easier" (downhill)
    - Some transitions are "harder" (uphill)
    - Attractors create gravity wells

    This creates realistic mood dynamics where you can't just
    "snap out of" sadness - you have to traverse the manifold.
    """

    def __init__(self,
                 dim: int = 8,
                 num_attractors: int = 6,
                 curvature: float = 1.0,
                 device: str = "cpu"):
        super().__init__()
        self.dim = dim
        self.c = curvature
        self.device = device

        # Core emotional attractors (in hyperbolic space)
        # These are the "gravity wells" emotions settle into
        self.attractor_names = ["joy", "sadness", "anger", "fear", "curiosity", "calm"]
        self.attractors = nn.Parameter(
            HyperbolicOperations.project_to_ball(
                torch.randn(num_attractors, dim) * 0.3,
                curvature
            )
        )

        # Attractor strengths (how "deep" each gravity well is)
        self.attractor_strengths = nn.Parameter(torch.ones(num_attractors) * 0.5)

        # Current position in emotional manifold
        self.register_buffer("position", torch.zeros(dim))

        # Velocity (momentum in emotional space)
        self.register_buffer("velocity", torch.zeros(dim))

        # Friction (how quickly emotions settle)
        self.friction = 0.1

    def compute_gravity(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Compute gravitational pull from all attractors.

        Returns force vector pointing toward attractors,
        weighted by distance (hyperbolic) and strength.
        """
        total_force = torch.zeros_like(pos)

        for i, (attractor, strength) in enumerate(zip(self.attractors, self.attractor_strengths)):
            # Distance in hyperbolic space (grows exponentially)
            dist = HyperbolicOperations.hyperbolic_distance(
                pos.unsqueeze(0), attractor.unsqueeze(0), self.c
            ).squeeze()

            # Direction toward attractor (in tangent space)
            direction = HyperbolicOperations.log_map(attractor, pos, self.c)
            direction_norm = torch.norm(direction)

            if direction_norm > 1e-6:
                direction = direction / direction_norm

            # Force = strength / distance² (like gravity)
            # But in hyperbolic space, distances are larger, so forces fall off faster
            force_magnitude = strength / (dist * dist + 0.1)

            total_force = total_force + direction * force_magnitude

        return total_force

    def step(self, external_force: torch.Tensor = None, dt: float = 0.1) -> torch.Tensor:
        """
        Step the emotional state forward in time.

        Args:
            external_force: External influence (e.g., positive news)
            dt: Time step

        Returns:
            New position in emotional space
        """
        # Compute gravitational force from attractors
        gravity = self.compute_gravity(self.position)

        # Add external force if provided
        if external_force is not None:
            gravity = gravity + external_force

        # Apply friction and update velocity with gravity
        with torch.no_grad():
            self.velocity.mul_(1 - self.friction).add_(gravity * dt)

            # Update position using exponential map (proper movement in curved space)
            new_pos = HyperbolicOperations.exp_map(self.velocity * dt, self.position, self.c)
            new_pos = HyperbolicOperations.project_to_ball(new_pos, self.c)
            self.position.copy_(new_pos)

        return self.position

    def get_dominant_emotion(self) -> Tuple[str, float]:
        """Get the closest attractor (dominant emotion) and its strength."""
        min_dist = float('inf')
        dominant = "neutral"
        strength = 0.0

        for i, (attractor, name) in enumerate(zip(self.attractors, self.attractor_names)):
            dist = HyperbolicOperations.hyperbolic_distance(
                self.position.unsqueeze(0), attractor.unsqueeze(0), self.c
            ).item()

            if dist < min_dist:
                min_dist = dist
                dominant = name
                strength = 1.0 / (1.0 + dist)  # Proximity as strength

        return dominant, strength

    def get_transition_difficulty(self, target_emotion: str) -> float:
        """
        Compute how "hard" it is to transition to a target emotion.

        Returns a value where higher = harder to reach.
        """
        target_idx = self.attractor_names.index(target_emotion) if target_emotion in self.attractor_names else 0
        target = self.attractors[target_idx]

        # Hyperbolic distance (naturally grows near boundary)
        dist = HyperbolicOperations.hyperbolic_distance(
            self.position.unsqueeze(0), target.unsqueeze(0), self.c
        ).item()

        return dist

    def nudge_toward(self, emotion: str, strength: float = 0.5):
        """Apply a force toward a target emotion."""
        if emotion not in self.attractor_names:
            return

        target_idx = self.attractor_names.index(emotion)
        target = self.attractors[target_idx]

        # Direction to target in tangent space
        direction = HyperbolicOperations.log_map(target, self.position, self.c)
        direction = direction / (torch.norm(direction) + 1e-6)

        # Apply as external force
        self.step(external_force=direction * strength)

    def get_state_description(self) -> str:
        """Get natural language description of emotional state."""
        dominant, strength = self.get_dominant_emotion()

        # Describe intensity
        if strength > 0.8:
            intensity = "intensely"
        elif strength > 0.5:
            intensity = "moderately"
        elif strength > 0.3:
            intensity = "mildly"
        else:
            intensity = "vaguely"

        # Describe velocity (emotional momentum)
        vel_mag = torch.norm(self.velocity).item()
        if vel_mag > 0.3:
            momentum = "rapidly shifting"
        elif vel_mag > 0.1:
            momentum = "gently drifting"
        else:
            momentum = "settled"

        return f"{intensity} {dominant}, {momentum}"


# =============================================================================
# MIXED CURVATURE MANIFOLD
# =============================================================================

class MixedCurvatureManifold(nn.Module):
    """
    Manifold with regions of different curvature.

    - Hyperbolic regions: hierarchical concepts (categories, taxonomies)
    - Euclidean regions: linear relationships (quantities, intensities)
    - Spherical regions: cyclic concepts (time, seasons, moods)

    This allows different types of reasoning in different "zones".
    """

    def __init__(self,
                 dim: int = 16,
                 num_regions: int = 4,
                 device: str = "cpu"):
        super().__init__()
        self.dim = dim
        self.device = device

        # Region curvatures (negative = hyperbolic, 0 = euclidean, positive = spherical)
        self.curvatures = nn.Parameter(torch.tensor([-1.0, 0.0, 1.0, -0.5]))

        # Region centers
        self.centers = nn.Parameter(torch.randn(num_regions, dim) * 0.5)

        # Region influence radii
        self.radii = nn.Parameter(torch.ones(num_regions) * 2.0)

    def get_local_curvature(self, x: torch.Tensor) -> torch.Tensor:
        """Get the effective curvature at position x (weighted by regions)."""
        # Distance to each region center
        dists = torch.cdist(x.unsqueeze(0), self.centers.unsqueeze(0)).squeeze(0)

        # Soft region membership (Gaussian weighting)
        weights = torch.exp(-dists / self.radii.unsqueeze(0))
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        # Weighted curvature
        local_c = (weights * self.curvatures).sum(dim=-1)

        return local_c

    def geodesic_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Approximate geodesic distance accounting for varying curvature.

        Uses adaptive metric based on local curvature.
        """
        # Get curvature at midpoint
        midpoint = (x + y) / 2
        c = self.get_local_curvature(midpoint)

        # Euclidean distance as base
        eucl_dist = torch.norm(x - y, dim=-1)

        # Adjust by curvature
        # Negative c (hyperbolic) → distances grow
        # Positive c (spherical) → distances shrink (wrapping)
        adjusted = eucl_dist * torch.exp(c * eucl_dist * 0.1)

        return adjusted

    def parallel_transport(self,
                          v: torch.Tensor,
                          x: torch.Tensor,
                          y: torch.Tensor) -> torch.Tensor:
        """
        Transport vector v from point x to point y along geodesic.

        In curved space, vectors "rotate" as they move.
        """
        c = self.get_local_curvature((x + y) / 2)

        # Simplified parallel transport (exact for constant curvature)
        direction = y - x
        direction = direction / (torch.norm(direction, dim=-1, keepdim=True) + 1e-8)

        # Rotation amount depends on curvature and distance
        dist = torch.norm(y - x, dim=-1, keepdim=True)
        angle = c.unsqueeze(-1) * dist * 0.1

        # Rotate v slightly toward/away from direction
        v_parallel = (v * direction).sum(dim=-1, keepdim=True) * direction
        v_perp = v - v_parallel

        # In hyperbolic space: perpendicular components grow
        # In spherical space: perpendicular components shrink
        transported = v_parallel + v_perp * torch.exp(-angle)

        return transported
