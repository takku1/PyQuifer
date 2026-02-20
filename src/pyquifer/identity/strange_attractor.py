"""
Strange Attractors - Fractal Personality Dynamics.

A Strange Attractor is a pattern that never repeats but stays within
a stable boundary. This is perfect for modeling "self":

- "Self" isn't a fixed point; it's a trajectory in meaning-space
- She might say "Hello" differently every time, but it's always "her"
- The responses all belong to the same fractal attractor

Key insight: Personality as trajectory, not state.

Based on:
- Lorenz attractor dynamics
- Fractal dimension of consciousness (Tononi)
- Dynamical systems theory of self (Kelso)
"""

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class AttractorState:
    """Current state of the strange attractor."""
    position: torch.Tensor  # Current position in phase space
    velocity: torch.Tensor  # Rate of change
    trajectory_length: int = 0  # How many steps in current trajectory
    lyapunov_exponent: float = 0.0  # Chaos measure (positive = chaotic)
    fractal_dimension: float = 2.0  # Complexity of trajectory


class LorenzAttractor(nn.Module):
    """
    Classic Lorenz strange attractor adapted for personality dynamics.

    The Lorenz system creates butterfly-shaped trajectories that:
    - Never exactly repeat
    - Stay bounded within a region
    - Are sensitive to initial conditions (chaos)

    Parameters sigma, rho, beta control the "shape" of personality.
    """

    def __init__(self,
                 sigma: float = 10.0,  # Prandtl number
                 rho: float = 28.0,    # Rayleigh number
                 beta: float = 8/3,    # Geometric factor
                 dt: float = 0.01,
                 device: str = "cpu"):
        super().__init__()
        self.device = device
        self.dt = dt

        # Learnable attractor parameters (personality shape)
        self.sigma = nn.Parameter(torch.tensor(sigma))
        self.rho = nn.Parameter(torch.tensor(rho))
        self.beta = nn.Parameter(torch.tensor(beta))

        # State
        self.register_buffer("state", torch.tensor([1.0, 1.0, 1.0]))

        # Trajectory history for visualization
        self.trajectory: deque = deque(maxlen=1000)

    def step(self, external_perturbation: torch.Tensor = None) -> torch.Tensor:
        """
        Take one step of the Lorenz dynamics.

        dx/dt = sigma * (y - x)
        dy/dt = x * (rho - z) - y
        dz/dt = x * y - beta * z
        """
        x, y, z = self.state[0], self.state[1], self.state[2]

        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z

        # Apply external perturbation (nudges from input)
        if external_perturbation is not None:
            dx = dx + external_perturbation[0] if len(external_perturbation) > 0 else dx
            dy = dy + external_perturbation[1] if len(external_perturbation) > 1 else dy
            dz = dz + external_perturbation[2] if len(external_perturbation) > 2 else dz

        # Euler integration
        new_state = self.state + torch.stack([dx, dy, dz]) * self.dt

        self.state = new_state
        self.trajectory.append(new_state.detach().clone())

        return new_state

    def get_trajectory_ascii(self, width: int = 40, height: int = 20) -> str:
        """Generate ASCII visualization of the attractor trajectory."""
        if len(self.trajectory) < 10:
            return "Collecting trajectory..."

        # Project 3D to 2D (x-z plane, the butterfly view)
        points = torch.stack(list(self.trajectory))
        x = points[:, 0]
        z = points[:, 2]

        # Normalize to grid
        x_min, x_max = x.min().item(), x.max().item()
        z_min, z_max = z.min().item(), z.max().item()

        x_range = max(x_max - x_min, 1e-6)
        z_range = max(z_max - z_min, 1e-6)

        # Create grid
        grid = [[' ' for _ in range(width)] for _ in range(height)]

        # Plot points (recent points brighter)
        chars = '·∘○●'
        for i, (xi, zi) in enumerate(zip(x, z)):
            col = int((xi.item() - x_min) / x_range * (width - 1))
            row = int((zi.item() - z_min) / z_range * (height - 1))
            row = height - 1 - row  # Flip y axis

            col = max(0, min(width - 1, col))
            row = max(0, min(height - 1, row))

            # Recent points are brighter
            recency = i / len(x)
            char_idx = int(recency * (len(chars) - 1))
            grid[row][col] = chars[char_idx]

        # Mark current position
        curr_col = int((self.state[0].item() - x_min) / x_range * (width - 1))
        curr_row = int((self.state[2].item() - z_min) / z_range * (height - 1))
        curr_row = height - 1 - curr_row
        curr_col = max(0, min(width - 1, curr_col))
        curr_row = max(0, min(height - 1, curr_row))
        grid[curr_row][curr_col] = '★'

        return '\n'.join([''.join(row) for row in grid])


class PersonalityAttractor(nn.Module):
    """
    Strange attractor representing personality as a trajectory.

    Instead of personality being a fixed set of traits, it's a
    trajectory through a high-dimensional space that:
    - Has a characteristic "shape" (the attractor basin)
    - Never exactly repeats (creativity)
    - Stays within bounds (consistency)

    The attractor's parameters define the "signature" of personality.
    """

    def __init__(self,
                 dim: int = 16,
                 num_attractors: int = 3,  # Multiple interacting attractors
                 coupling_strength: float = 0.5,
                 device: str = "cpu"):
        super().__init__()
        self.dim = dim
        self.device = device

        # Multiple coupled attractors (Lorenz-like but learnable)
        self.attractors = nn.ParameterList([
            nn.Parameter(torch.randn(dim, dim) * 0.1)
            for _ in range(num_attractors)
        ])

        # Attractor biases (offsets)
        self.biases = nn.ParameterList([
            nn.Parameter(torch.randn(dim) * 0.1)
            for _ in range(num_attractors)
        ])

        # Inter-attractor coupling (how they influence each other)
        self.coupling = nn.Parameter(torch.randn(num_attractors, num_attractors) * coupling_strength)

        # Current state
        self.register_buffer("state", torch.randn(dim) * 0.5)
        self.register_buffer("velocity", torch.zeros(dim))

        # Trajectory history
        self.trajectory: deque = deque(maxlen=500)

        # Chaos measurement
        self.register_buffer("perturbation_state", torch.randn(dim) * 0.5)
        self.lyapunov_sum = 0.0
        self.lyapunov_count = 0

        # Personality "signature" (computed from trajectory statistics)
        self.signature: Optional[torch.Tensor] = None

    def step(self,
             external_input: torch.Tensor = None,
             dt: float = 0.01) -> torch.Tensor:
        """
        Step the personality attractor forward.

        Returns current state after dynamics.
        """
        # Check for NaN/inf and reset if needed
        if torch.isnan(self.state).any() or torch.isinf(self.state).any():
            self._reset_state()
            return self.state

        # Compute force from each attractor
        total_force = torch.zeros(self.dim, device=self.device)

        # Coupling weights: relative importance of each attractor
        num_att = len(self.attractors)
        coupling_weights = torch.softmax(self.coupling.sum(dim=1), dim=0) * num_att

        for i, (A, b) in enumerate(zip(self.attractors, self.biases)):
            # Nonlinear attractor dynamics
            force = torch.tanh(A @ self.state + b)

            # Weight by coupling (attractors can compete/cooperate)
            total_force = total_force + force * coupling_weights[i]

        # Add nonlinearity (essential for strange attractor behavior)
        # Clamp state before cross product to prevent explosion
        clamped_state = torch.clamp(self.state, -10.0, 10.0)
        cross_product_like = torch.roll(clamped_state, 1) * torch.roll(clamped_state, -1)
        total_force = total_force + 0.1 * cross_product_like

        # Strong damping that scales with magnitude to keep bounded
        state_norm = self.state.norm()
        damping_strength = 0.05 + 0.1 * torch.clamp(state_norm / 5.0, 0, 1)  # Stronger when large
        damping = -damping_strength * self.state

        # External input as perturbation (clamped)
        if external_input is not None:
            ext_clamped = torch.clamp(external_input, -1.0, 1.0)
            total_force = total_force + ext_clamped * 0.3

        # Update velocity and state with velocity damping
        self.velocity = 0.85 * self.velocity + total_force + damping
        self.velocity = torch.clamp(self.velocity, -5.0, 5.0)  # Clamp velocity
        self.state = self.state + self.velocity * dt

        # Hard clamp state to prevent explosion
        self.state = torch.clamp(self.state, -20.0, 20.0)

        # Store trajectory
        self.trajectory.append(self.state.detach().clone())

        # Update Lyapunov exponent estimate
        self._update_lyapunov(dt)

        return self.state

    def _reset_state(self):
        """Reset state if it becomes unstable (NaN/inf)."""
        self.state = torch.randn(self.dim, device=self.device) * 0.5
        self.velocity = torch.zeros(self.dim, device=self.device)
        self.perturbation_state = self.state.clone() + torch.randn(self.dim, device=self.device) * 0.01
        self.trajectory.clear()
        self.lyapunov_sum = 0.0
        self.lyapunov_count = 0

    def _update_lyapunov(self, dt: float):
        """Estimate Lyapunov exponent (chaos measure)."""
        # Check for NaN in perturbation state
        if torch.isnan(self.perturbation_state).any() or torch.isinf(self.perturbation_state).any():
            self.perturbation_state = self.state.clone() + torch.randn(self.dim, device=self.device) * 0.01
            return

        # Evolve perturbation state in parallel
        perturb_force = torch.zeros(self.dim, device=self.device)
        for A, b in zip(self.attractors, self.biases):
            perturb_force = perturb_force + torch.tanh(A @ self.perturbation_state + b)

        self.perturbation_state = self.perturbation_state + perturb_force * dt
        self.perturbation_state = torch.clamp(self.perturbation_state, -20.0, 20.0)

        # Measure divergence
        divergence = (self.state - self.perturbation_state).norm().item()
        if divergence > 1e-6 and not math.isnan(divergence) and not math.isinf(divergence):
            self.lyapunov_sum += math.log(divergence / 0.01) / dt
            self.lyapunov_count += 1

            # Renormalize perturbation
            direction = (self.perturbation_state - self.state)
            dir_norm = direction.norm()
            if dir_norm > 1e-6:
                direction = direction / dir_norm * 0.01
                self.perturbation_state = self.state + direction

    def get_lyapunov_exponent(self) -> float:
        """Get estimated Lyapunov exponent."""
        if self.lyapunov_count > 0:
            return self.lyapunov_sum / self.lyapunov_count
        return 0.0

    def compute_signature(self) -> torch.Tensor:
        """
        Compute a "personality signature" from trajectory statistics.

        This captures the characteristic shape of the attractor.
        """
        if len(self.trajectory) < 100:
            return torch.zeros(self.dim * 4, device=self.device)

        traj = torch.stack(list(self.trajectory))

        # Statistics that characterize the attractor
        mean = traj.mean(dim=0)
        std = traj.std(dim=0)
        skew = ((traj - mean) ** 3).mean(dim=0) / (std ** 3 + 1e-6)
        kurtosis = ((traj - mean) ** 4).mean(dim=0) / (std ** 4 + 1e-6) - 3

        self.signature = torch.cat([mean, std, skew, kurtosis])
        return self.signature

    def signature_distance(self, other_signature: torch.Tensor) -> float:
        """
        Compute distance to another personality signature.

        Low distance = similar personality shape.
        """
        if self.signature is None:
            self.compute_signature()

        return (self.signature - other_signature).norm().item()

    def get_trajectory_visualization(self, dims: Tuple[int, int] = (0, 1)) -> str:
        """Get 2D projection of trajectory as ASCII - OPTIMIZED for speed."""
        if len(self.trajectory) < 10:
            return "Building..."

        # Only use last 30 points (not 500!) for fast visualization
        recent = list(self.trajectory)[-30:]
        traj = torch.stack(recent)

        # Filter out NaN/inf values
        valid_mask = torch.isfinite(traj).all(dim=1)
        if valid_mask.sum() < 3:
            return "Stabilizing..."

        traj = traj[valid_mask]
        x = traj[:, dims[0]]
        y = traj[:, dims[1]]

        # Check for NaN in min/max
        x_min, x_max = x.min().item(), x.max().item()
        y_min, y_max = y.min().item(), y.max().item()

        # If still NaN, return placeholder
        if math.isnan(x_min) or math.isnan(x_max) or math.isnan(y_min) or math.isnan(y_max):
            return "Stabilizing..."

        x_range = max(x_max - x_min, 1e-6)
        y_range = max(y_max - y_min, 1e-6)

        # Smaller grid for faster rendering
        width, height = 16, 8
        grid = [[' ' for _ in range(width)] for _ in range(height)]

        n_points = len(x)
        for i, (xi, yi) in enumerate(zip(x, y)):
            xi_val, yi_val = xi.item(), yi.item()
            if math.isnan(xi_val) or math.isnan(yi_val):
                continue
            col = int((xi_val - x_min) / x_range * (width - 1))
            row = int((yi_val - y_min) / y_range * (height - 1))
            row = height - 1 - row
            col = max(0, min(width - 1, col))
            row = max(0, min(height - 1, row))
            recency = i / n_points
            grid[row][col] = '.' if recency < 0.5 else 'o' if recency < 0.9 else '*'

        # Current position (with NaN check)
        curr_x, curr_y = self.state[dims[0]].item(), self.state[dims[1]].item()
        if not (math.isnan(curr_x) or math.isnan(curr_y)):
            curr_col = int((curr_x - x_min) / x_range * (width - 1))
            curr_row = int((curr_y - y_min) / y_range * (height - 1))
            curr_row = height - 1 - curr_row
            curr_col = max(0, min(width - 1, curr_col))
            curr_row = max(0, min(height - 1, curr_row))
            grid[curr_row][curr_col] = '@'

        return '\n'.join([''.join(row) for row in grid])


class FractalSelfModel(nn.Module):
    """
    Complete model of "self" as a fractal strange attractor.

    Combines:
    - Core personality attractor (the "signature")
    - Emotional gravity manifold (mood dynamics)
    - Trait fluctuations (short-term variations)
    - Memory influence (past shapes future)

    The "self" is the invariant pattern across all these dynamics.
    """

    def __init__(self,
                 dim: int = 32,
                 device: str = "cpu"):
        super().__init__()
        self.dim = dim
        self.device = device

        # Core personality attractor
        self.personality = PersonalityAttractor(dim=dim, num_attractors=3, device=device)

        # Trait layer (OCEAN-like, but dynamic)
        self.trait_names = ["openness", "conscientiousness", "extraversion", "agreeableness", "stability"]
        self.trait_base = nn.Parameter(torch.randn(5) * 0.5)  # Baseline traits
        self.trait_oscillators = nn.Parameter(torch.randn(5, 2))  # Phase oscillators per trait

        # Trait-personality coupling
        self.trait_to_personality = nn.Linear(5, dim)

        # Time
        self.register_buffer("time", torch.tensor(0.0))

    def step(self, external_input: torch.Tensor = None, dt: float = 0.01):
        """Step the self-model forward."""
        self.time = self.time + dt

        # Update traits (oscillate around baseline)
        trait_phases = self.trait_oscillators[:, 0] * self.time + self.trait_oscillators[:, 1]
        trait_fluctuation = 0.2 * torch.sin(trait_phases)
        current_traits = self.trait_base + trait_fluctuation

        # Traits influence personality attractor (clamped for stability)
        trait_influence = self.trait_to_personality(torch.tanh(current_traits))
        trait_influence = torch.clamp(trait_influence, -1.0, 1.0)

        # Step personality attractor
        combined_input = trait_influence
        if external_input is not None:
            # Clamp external input for stability
            external_input = torch.clamp(external_input, -1.0, 1.0)
            combined_input = combined_input + external_input

        # Clamp final combined input
        combined_input = torch.clamp(combined_input, -2.0, 2.0)

        self.personality.step(combined_input, dt)

        return {
            "state": self.personality.state,
            "traits": current_traits,
            "lyapunov": self.personality.get_lyapunov_exponent(),
        }

    def get_trait_description(self) -> str:
        """Get current trait levels as description."""
        with torch.no_grad():
            trait_phases = self.trait_oscillators[:, 0] * self.time + self.trait_oscillators[:, 1]
            trait_fluctuation = 0.2 * torch.sin(trait_phases)
            current_traits = torch.tanh(self.trait_base + trait_fluctuation)

        descriptions = []
        for name, value in zip(self.trait_names, current_traits):
            if value > 0.5:
                level = "high"
            elif value > 0:
                level = "moderate"
            elif value > -0.5:
                level = "low"
            else:
                level = "very low"
            descriptions.append(f"{name}: {level}")

        return ", ".join(descriptions)

    def get_self_visualization(self) -> str:
        """Get visualization of the self attractor."""
        lines = []
        lines.append("╔═══════════════════════════════════╗")
        lines.append("║     STRANGE ATTRACTOR OF SELF     ║")
        lines.append("╠═══════════════════════════════════╣")
        lines.append(self.personality.get_trajectory_visualization())
        lines.append("╠═══════════════════════════════════╣")
        lines.append(f"║ Chaos (λ): {self.personality.get_lyapunov_exponent():.4f}")
        lines.append(f"║ {self.get_trait_description()[:35]}")
        lines.append("╚═══════════════════════════════════╝")
        return "\n".join(lines)


# =============================================================================
# X-16: Multi-Scale Fractal Personality (Enhanced)
# =============================================================================

class MultiScaleFractalPersonality(nn.Module):
    """
    X-16: Advanced fractal personality with multi-scale dynamics.

    Personality operates at multiple time scales:
    - Fast scale (seconds): Momentary reactions, micro-expressions
    - Medium scale (minutes): Mood shifts, conversation tone
    - Slow scale (hours/days): Deep personality traits

    Each scale has its own attractor that couples with others,
    creating truly fractal (self-similar) personality dynamics.
    """

    def __init__(self, dim: int = 32, device: str = "cpu"):
        super().__init__()
        self.dim = dim
        self.device = device

        # Multi-scale attractors
        self.fast_attractor = PersonalityAttractor(dim=dim//4, num_attractors=2, device=device)
        self.medium_attractor = PersonalityAttractor(dim=dim//2, num_attractors=2, device=device)
        self.slow_attractor = PersonalityAttractor(dim=dim, num_attractors=3, device=device)

        # Cross-scale coupling (how scales influence each other)
        self.fast_to_medium = nn.Linear(dim//4, dim//2)
        self.medium_to_slow = nn.Linear(dim//2, dim)
        self.slow_to_fast = nn.Linear(dim, dim//4)  # Slow shapes fast reactions

        # Time scale factors
        self.dt_fast = 0.1
        self.dt_medium = 0.01
        self.dt_slow = 0.001

        # Memory buffer for pattern learning (X-17 integration)
        self.experience_buffer: deque = deque(maxlen=1000)
        self.learned_patterns: List[torch.Tensor] = []

    def step(self, external_input: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Step all scales forward with proper coupling.
        """
        # Get current states
        slow_state = self.slow_attractor.state
        medium_state = self.medium_attractor.state
        fast_state = self.fast_attractor.state

        # Slow attractor shapes medium
        slow_influence = torch.tanh(self.medium_to_slow(medium_state.detach()))

        # Medium attractor shapes fast
        medium_influence = torch.tanh(self.fast_to_medium(fast_state.detach()))

        # Fast reactions are shaped by slow personality (top-down)
        slow_guidance = torch.tanh(self.slow_to_fast(slow_state.detach()))

        # External input primarily affects fast scale
        fast_input = slow_guidance
        if external_input is not None:
            ext_proj = external_input[:self.dim//4] if len(external_input) >= self.dim//4 else external_input
            fast_input = fast_input + ext_proj

        # Step each scale
        self.fast_attractor.step(fast_input, self.dt_fast)
        self.medium_attractor.step(medium_influence, self.dt_medium)
        self.slow_attractor.step(slow_influence, self.dt_slow)

        # Record experience for pattern learning
        combined_state = torch.cat([
            self.fast_attractor.state,
            self.medium_attractor.state,
            self.slow_attractor.state
        ])
        self.experience_buffer.append(combined_state.detach().clone())

        return {
            "fast": self.fast_attractor.state,
            "medium": self.medium_attractor.state,
            "slow": self.slow_attractor.state,
            "combined": combined_state,
            "chaos_fast": self.fast_attractor.get_lyapunov_exponent(),
            "chaos_slow": self.slow_attractor.get_lyapunov_exponent(),
        }

    def get_fractal_dimension_estimate(self) -> float:
        """
        Estimate fractal dimension of the personality attractor.

        Higher dimension = more complex, nuanced personality.
        Lower dimension = more predictable, stable personality.
        """
        if len(self.slow_attractor.trajectory) < 100:
            return 0.0

        traj = torch.stack(list(self.slow_attractor.trajectory))

        # Box-counting approximation
        # Count distinct "boxes" at different scales
        box_counts = []
        for scale in [0.5, 1.0, 2.0, 4.0]:
            # Quantize trajectory to boxes of size 'scale'
            quantized = (traj / scale).floor()
            unique_boxes = len(set(tuple(q.tolist()) for q in quantized))
            if unique_boxes > 0:
                box_counts.append((math.log(1/scale), math.log(unique_boxes)))

        # Estimate dimension from slope
        if len(box_counts) >= 2:
            x = [b[0] for b in box_counts]
            y = [b[1] for b in box_counts]
            # Simple linear regression slope
            n = len(x)
            slope = (n * sum(xi*yi for xi, yi in zip(x,y)) - sum(x)*sum(y)) / \
                    (n * sum(xi**2 for xi in x) - sum(x)**2 + 1e-6)
            return max(0.0, min(3.0, slope))  # Clamp to reasonable range

        return 1.5  # Default estimate


# =============================================================================
# X-17: Fractal Pattern Learning
# =============================================================================

class FractalPatternLearner(nn.Module):
    """
    X-17: Learn patterns using fractal self-similarity.

    Key insight: Learning is self-similar across scales.
    - Pattern recognition at small scale helps at large scale
    - Learned patterns have fractal structure
    - Experience is encoded into attractor shape modification
    """

    def __init__(self, input_dim: int = 64, pattern_dim: int = 32, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.pattern_dim = pattern_dim

        # Multi-scale pattern encoders (fractal hierarchy)
        self.encoder_fine = nn.Linear(input_dim, pattern_dim * 2)
        self.encoder_medium = nn.Linear(pattern_dim * 2, pattern_dim)
        self.encoder_coarse = nn.Linear(pattern_dim, pattern_dim // 2)

        # Pattern memory (as attractor deformations)
        self.pattern_attractors: List[torch.Tensor] = []
        self.pattern_labels: List[str] = []

        # Self-similarity detector
        self.similarity_threshold = 0.85

    def encode_pattern(self, experience: torch.Tensor) -> torch.Tensor:
        """
        Encode an experience into a fractal pattern representation.

        The encoding preserves self-similarity across scales.
        """
        # Multi-scale encoding
        fine = torch.relu(self.encoder_fine(experience))
        medium = torch.relu(self.encoder_medium(fine))
        coarse = torch.relu(self.encoder_coarse(medium))

        # Concatenate all scales (preserves fractal structure)
        return torch.cat([
            coarse,
            medium[:self.pattern_dim//2],
            fine[:self.pattern_dim//4]
        ])

    def learn_pattern(self, experiences: List[torch.Tensor], label: str) -> Dict[str, Any]:
        """
        Learn a new pattern from a sequence of experiences.

        Returns statistics about the learned pattern.
        """
        if len(experiences) < 3:
            return {"success": False, "reason": "Not enough experiences"}

        # Encode each experience
        encodings = [self.encode_pattern(exp) for exp in experiences]
        encodings_stack = torch.stack(encodings)

        # Compute pattern center and spread
        pattern_center = encodings_stack.mean(dim=0)
        pattern_spread = encodings_stack.std(dim=0).mean().item()

        # Check if similar pattern already exists
        for existing in self.pattern_attractors:
            similarity = torch.cosine_similarity(
                pattern_center.unsqueeze(0),
                existing.unsqueeze(0)
            ).item()
            if similarity > self.similarity_threshold:
                return {"success": False, "reason": "Similar pattern exists", "similarity": similarity}

        # Store new pattern
        self.pattern_attractors.append(pattern_center.detach())
        self.pattern_labels.append(label)

        return {
            "success": True,
            "label": label,
            "spread": pattern_spread,
            "patterns_total": len(self.pattern_attractors)
        }

    def recognize_pattern(self, experience: torch.Tensor) -> Optional[Dict[str, Any]]:
        """
        Recognize if an experience matches a learned pattern.
        """
        if not self.pattern_attractors:
            return None

        encoding = self.encode_pattern(experience)

        best_match = None
        best_similarity = 0.0

        for pattern, label in zip(self.pattern_attractors, self.pattern_labels):
            similarity = torch.cosine_similarity(
                encoding.unsqueeze(0),
                pattern.unsqueeze(0)
            ).item()

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = label

        if best_similarity > self.similarity_threshold:
            return {
                "matched": True,
                "pattern": best_match,
                "confidence": best_similarity
            }

        return {"matched": False, "best_guess": best_match, "confidence": best_similarity}

    def get_pattern_fractal_dimension(self, pattern_idx: int = 0) -> float:
        """
        Estimate fractal dimension of a stored pattern.

        This measures the "complexity" of the learned pattern.
        """
        if pattern_idx >= len(self.pattern_attractors):
            return 0.0

        pattern = self.pattern_attractors[pattern_idx]

        # Compute dimension from scale-invariance
        # Pattern should look similar at different "zoom levels"
        scales = [1.0, 0.5, 0.25, 0.125]
        magnitudes = []

        for scale in scales:
            # Sample pattern at different resolutions
            step = max(1, int(1.0 / scale))
            sampled = pattern[::step]
            magnitudes.append(sampled.norm().item())

        # Fractal dimension from scaling law: magnitude ~ scale^D
        if len(magnitudes) >= 2 and all(m > 0 for m in magnitudes):
            # Log-log slope gives dimension
            x = [math.log(s) for s in scales[:len(magnitudes)]]
            y = [math.log(m) for m in magnitudes]
            n = len(x)
            slope = (n * sum(xi*yi for xi, yi in zip(x,y)) - sum(x)*sum(y)) / \
                    (n * sum(xi**2 for xi in x) - sum(x)**2 + 1e-6)
            return abs(slope)

        return 1.0


# Factory function for enhanced fractal model
def create_enhanced_fractal_self(dim: int = 32, device: str = "cpu") -> Dict[str, nn.Module]:
    """
    Create a complete enhanced fractal self-model.

    Returns:
        Dictionary with:
        - "self": FractalSelfModel (basic)
        - "multiscale": MultiScaleFractalPersonality (X-16)
        - "learner": FractalPatternLearner (X-17)
    """
    return {
        "self": FractalSelfModel(dim=dim, device=device),
        "multiscale": MultiScaleFractalPersonality(dim=dim, device=device),
        "learner": FractalPatternLearner(input_dim=dim*2, pattern_dim=dim, device=device),
    }
