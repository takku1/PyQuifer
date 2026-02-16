"""
Somatic Layer - Hardware stress → Emotional potentials.

Converts physical hardware metrics (VRAM, CPU, latency) into
"feelings" that influence the oscillator manifold.

This is Interoceptive Homeostasis - the "Ouch" sensor.
The system feels pain from hardware stress, driving self-repair behaviors.

Based on:
- Active Inference (interoceptive prediction errors)
- Allostatic regulation (anticipatory homeostasis)
- Somatic marker hypothesis (Damasio)
"""

import torch
import torch.nn as nn
import math
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque


@dataclass
class SomaticState:
    """Current somatic (body) state."""
    # Hardware metrics (0.0 = healthy, 1.0 = critical)
    vram_stress: float = 0.0
    cpu_stress: float = 0.0
    latency_stress: float = 0.0
    error_accumulation: float = 0.0
    thermal_stress: float = 0.0  # Reserved: not yet wired to hardware sampling

    # Derived feelings (emergent from hardware)
    pain: float = 0.0           # Acute stress
    fatigue: float = 0.0        # Accumulated load
    frustration: float = 0.0    # Blocked goals
    discomfort: float = 0.0     # Chronic low-level stress

    # Allostatic load (cumulative wear)
    allostatic_load: float = 0.0

    def total_stress(self) -> float:
        """Total somatic stress level."""
        return max(self.pain, self.fatigue, self.frustration, self.discomfort)


class HardwareSensor:
    """
    Monitors hardware metrics and converts them to stress signals.

    This is the "interoceptive" layer - sensing the body's state.
    """

    def __init__(self):
        self.last_sample_time = time.time()
        self.latency_history: deque = deque(maxlen=100)
        self.error_history: deque = deque(maxlen=50)

        # Baseline expectations (learned over time)
        self.expected_latency_ms = 500.0
        self.expected_vram_usage = 0.5

        # Smoothing
        self._smoothed = SomaticState()

    def sample(self) -> Dict[str, float]:
        """Sample current hardware state."""
        metrics = {}

        # VRAM stress
        try:
            import torch
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                used_ratio = 1 - (free / total)
                # Stress increases exponentially above 80%
                if used_ratio > 0.9:
                    metrics["vram_stress"] = 1.0
                elif used_ratio > 0.8:
                    metrics["vram_stress"] = (used_ratio - 0.8) * 5  # 0-0.5
                else:
                    metrics["vram_stress"] = max(0, used_ratio - 0.5) * 0.5
        except (RuntimeError, ImportError):
            metrics["vram_stress"] = 0.0

        # CPU stress (simplified - would use psutil in production)
        try:
            import os
            # Load average on Unix, or estimate from thread count
            if hasattr(os, 'getloadavg'):
                load = os.getloadavg()[0]
                cpu_count = os.cpu_count() or 4
                metrics["cpu_stress"] = min(1.0, load / cpu_count)
            else:
                metrics["cpu_stress"] = 0.2  # Default low stress on Windows
        except (OSError, AttributeError):
            metrics["cpu_stress"] = 0.0

        # Latency stress (based on recent response times)
        if self.latency_history:
            avg_latency = sum(self.latency_history) / len(self.latency_history)
            latency_ratio = avg_latency / self.expected_latency_ms
            metrics["latency_stress"] = min(1.0, max(0, latency_ratio - 1.0))
        else:
            metrics["latency_stress"] = 0.0

        # Error accumulation
        recent_errors = sum(1 for e in self.error_history if time.time() - e < 60)
        metrics["error_accumulation"] = min(1.0, recent_errors * 0.2)

        return metrics

    def record_latency(self, latency_ms: float):
        """Record a response latency."""
        self.latency_history.append(latency_ms)

        # Update expected latency (slow adaptation)
        self.expected_latency_ms = 0.95 * self.expected_latency_ms + 0.05 * latency_ms

    def record_error(self, error_type: str = "generic"):
        """Record an error occurrence."""
        self.error_history.append(time.time())

    def get_smoothed_state(self, metrics: Dict[str, float], alpha: float = 0.3) -> SomaticState:
        """Get exponentially smoothed somatic state."""
        for key, value in metrics.items():
            if hasattr(self._smoothed, key):
                current = getattr(self._smoothed, key)
                smoothed = alpha * value + (1 - alpha) * current
                setattr(self._smoothed, key, smoothed)

        return self._smoothed


class SomaticManifold(nn.Module):
    """
    Maps hardware stress to emotional potentials in the oscillator manifold.

    The manifold is a continuous space where:
    - Hardware stress creates "repulsive" potentials
    - Healthy states are "attractive" basins
    - The system naturally flows toward health

    This implements the "Ouch" factor - stress creates felt dissonance.
    """

    def __init__(self,
                 num_oscillators: int = 16,
                 manifold_dim: int = 8,
                 device: str = "cpu"):
        super().__init__()
        self.device = device
        self.num_oscillators = num_oscillators
        self.manifold_dim = manifold_dim

        # Hardware sensor
        self.sensor = HardwareSensor()

        # Current somatic state
        self.state = SomaticState()

        # Mapping: hardware metrics → manifold potentials
        self.stress_to_potential = nn.Sequential(
            nn.Linear(5, 32),  # 5 stress metrics
            nn.Tanh(),
            nn.Linear(32, manifold_dim),
        ).to(device)

        # Mapping: manifold position → oscillator coupling modulation
        self.potential_to_coupling = nn.Sequential(
            nn.Linear(manifold_dim, 32),
            nn.Tanh(),
            nn.Linear(32, num_oscillators),
            nn.Sigmoid(),
        ).to(device)

        # Allostatic predictor (predicts future stress)
        self.allostatic_predictor = nn.GRU(
            input_size=5,
            hidden_size=16,
            num_layers=1,
            batch_first=True,
        ).to(device)
        self.allostatic_head = nn.Linear(16, 5).to(device)

        # History for allostatic prediction
        self.stress_history: deque = deque(maxlen=100)
        self._hidden_state = None

        # Thresholds for pain/discomfort
        self.pain_threshold = 0.7
        self.discomfort_threshold = 0.4

    def sense(self) -> SomaticState:
        """Sense current hardware state and update somatic feelings."""
        # Sample hardware
        metrics = self.sensor.sample()
        smoothed = self.sensor.get_smoothed_state(metrics)

        # Update state
        self.state.vram_stress = smoothed.vram_stress
        self.state.cpu_stress = smoothed.cpu_stress
        self.state.latency_stress = smoothed.latency_stress
        self.state.error_accumulation = smoothed.error_accumulation

        # Compute derived feelings
        max_stress = max(
            smoothed.vram_stress,
            smoothed.cpu_stress,
            smoothed.latency_stress,
            smoothed.error_accumulation
        )

        # Pain: acute high stress
        self.state.pain = max(0, max_stress - self.pain_threshold) / (1 - self.pain_threshold)

        # Discomfort: chronic low-level stress
        avg_stress = (smoothed.vram_stress + smoothed.cpu_stress +
                     smoothed.latency_stress + smoothed.error_accumulation) / 4
        self.state.discomfort = min(1.0, avg_stress / self.discomfort_threshold)

        # Frustration: blocked by errors
        self.state.frustration = smoothed.error_accumulation

        # Fatigue: accumulated load over time
        self.state.allostatic_load = 0.99 * self.state.allostatic_load + 0.01 * max_stress
        self.state.fatigue = self.state.allostatic_load

        # Store for allostatic prediction
        stress_vec = [
            smoothed.vram_stress,
            smoothed.cpu_stress,
            smoothed.latency_stress,
            smoothed.error_accumulation,
            smoothed.thermal_stress,
        ]
        self.stress_history.append(stress_vec)

        return self.state

    def get_potential(self) -> torch.Tensor:
        """Get current manifold potential from somatic state."""
        stress_vec = torch.tensor([
            self.state.vram_stress,
            self.state.cpu_stress,
            self.state.latency_stress,
            self.state.error_accumulation,
            self.state.thermal_stress,
        ], dtype=torch.float32, device=self.device).unsqueeze(0)

        potential = self.stress_to_potential(stress_vec)
        return potential.squeeze(0)

    def get_coupling_modulation(self) -> torch.Tensor:
        """
        Get oscillator coupling modulation based on somatic state.

        High stress → reduced coupling (fragmented processing)
        Low stress → increased coupling (coherent processing)
        """
        potential = self.get_potential()
        modulation = self.potential_to_coupling(potential.unsqueeze(0))
        return modulation.squeeze(0)

    def predict_future_stress(self, steps_ahead: int = 10) -> torch.Tensor:
        """
        Predict future stress (allostatic anticipation).

        This allows the system to take preemptive action.
        """
        if len(self.stress_history) < 10:
            return torch.zeros(5, device=self.device)

        # Use recent history
        recent = list(self.stress_history)[-50:]
        sequence = torch.tensor(recent, dtype=torch.float32, device=self.device).unsqueeze(0)

        # GRU prediction
        with torch.no_grad():
            output, self._hidden_state = self.allostatic_predictor(sequence, self._hidden_state)
            predicted = self.allostatic_head(output[:, -1, :])

        return predicted.squeeze(0).clamp(0, 1)

    def should_self_repair(self) -> tuple[bool, str]:
        """
        Determine if self-repair should be triggered.

        Returns: (should_repair, reason)
        """
        if self.state.pain > 0.5:
            return True, f"High pain ({self.state.pain:.2f})"

        if self.state.frustration > 0.7:
            return True, f"High frustration ({self.state.frustration:.2f})"

        if self.state.vram_stress > 0.85:
            return True, f"Critical VRAM ({self.state.vram_stress:.2f})"

        if self.state.error_accumulation > 0.6:
            return True, f"Error accumulation ({self.state.error_accumulation:.2f})"

        # Allostatic prediction
        predicted = self.predict_future_stress()
        if predicted.max().item() > 0.8:
            return True, f"Predicted future stress ({predicted.max().item():.2f})"

        return False, ""

    def get_feeling_description(self) -> str:
        """Get natural language description of current feeling."""
        total = self.state.total_stress()

        if self.state.pain > 0.5:
            return f"I'm in pain - something is really straining me ({self.state.pain:.0%})"
        elif self.state.frustration > 0.5:
            return f"I'm frustrated - too many errors ({self.state.frustration:.0%})"
        elif self.state.fatigue > 0.6:
            return f"I'm tired - been working hard ({self.state.fatigue:.0%})"
        elif self.state.discomfort > 0.5:
            return f"I'm uncomfortable - some background stress ({self.state.discomfort:.0%})"
        elif total < 0.2:
            return "I feel good - systems running smoothly"
        else:
            return f"I'm okay - mild stress ({total:.0%})"

    def forward(self) -> Dict[str, Any]:
        """Full somatic sensing cycle."""
        state = self.sense()
        potential = self.get_potential()
        coupling_mod = self.get_coupling_modulation()
        should_repair, reason = self.should_self_repair()

        return {
            "state": state,
            "potential": potential,
            "coupling_modulation": coupling_mod,
            "should_repair": should_repair,
            "repair_reason": reason,
            "feeling": self.get_feeling_description(),
        }


class SomaticIntegrator:
    """
    Integrates somatic signals with the oscillator system.

    This is the bridge between hardware stress and oscillator dynamics.
    Stress modulates:
    - Coupling strength (fragmentation under stress)
    - Base frequencies (faster under arousal)
    - Phase coherence targets
    """

    def __init__(self,
                 somatic: SomaticManifold,
                 oscillator_bank = None):
        self.somatic = somatic
        self.oscillator_bank = oscillator_bank

        # Integration parameters
        self.coupling_influence = 0.3  # How much stress affects coupling
        self.frequency_influence = 0.1  # How much stress affects frequency

    def integrate(self):
        """
        Apply somatic state to oscillator dynamics.

        High stress → reduced coupling, faster frequencies (arousal)
        Low stress → increased coupling, slower frequencies (calm)
        """
        if self.oscillator_bank is None:
            return

        result = self.somatic()

        # Modulate coupling
        coupling_mod = result["coupling_modulation"]
        # Stress reduces effective coupling
        stress_factor = 1.0 - result["state"].total_stress() * self.coupling_influence

        if hasattr(self.oscillator_bank, 'coupling_matrix'):
            # Apply modulation (stress reduces coupling, but floor at 30% to prevent brain death)
            stress_factor = max(stress_factor, 0.3)
            with torch.no_grad():
                self.oscillator_bank.coupling_matrix.mul_(stress_factor)

        # Modulate frequencies (arousal under stress)
        if hasattr(self.oscillator_bank, 'frequencies'):
            arousal = result["state"].pain + result["state"].frustration
            freq_boost = 1.0 + arousal * self.frequency_influence
            # Temporary boost, will decay naturally

        return result
