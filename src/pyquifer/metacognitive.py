"""
Metacognitive Loop (MCL) - Thinking about thinking.

The "Thinking Governor" that watches the thinking process itself.
Enables:
- Confidence awareness ("I'm 40% sure about this")
- Reasoning quality monitoring
- Uncertainty vocalization
- Self-correction triggers

This prevents the "impersonal" feeling by letting the system
vocalize its uncertainty and reasoning before acting.

Based on:
- Metacognition research (Flavell, Nelson & Narens)
- Confidence calibration (Bayesian approaches)
- Epistemic emotions (curiosity, confusion, aha moments)
"""

import torch
import torch.nn as nn
import math
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum


class ConfidenceLevel(Enum):
    """Discrete confidence levels for reasoning."""
    CERTAIN = "certain"         # 90%+ confidence
    CONFIDENT = "confident"     # 70-90%
    MODERATE = "moderate"       # 50-70%
    UNCERTAIN = "uncertain"     # 30-50%
    GUESSING = "guessing"       # <30%


@dataclass
class ReasoningStep:
    """A single step in the reasoning process."""
    step_id: int
    content: str
    confidence: float  # 0.0 to 1.0
    reasoning_type: str  # "deduction", "induction", "analogy", "retrieval", etc.
    evidence: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def confidence_level(self) -> ConfidenceLevel:
        if self.confidence >= 0.9:
            return ConfidenceLevel.CERTAIN
        elif self.confidence >= 0.7:
            return ConfidenceLevel.CONFIDENT
        elif self.confidence >= 0.5:
            return ConfidenceLevel.MODERATE
        elif self.confidence >= 0.3:
            return ConfidenceLevel.UNCERTAIN
        else:
            return ConfidenceLevel.GUESSING


@dataclass
class MetacognitiveState:
    """Current metacognitive state."""
    # Confidence tracking
    overall_confidence: float = 0.5
    confidence_trend: str = "stable"  # "rising", "falling", "stable"

    # Reasoning quality
    coherence_score: float = 0.5  # How well steps connect
    completeness_score: float = 0.5  # How complete the reasoning is

    # Epistemic emotions
    curiosity: float = 0.0
    confusion: float = 0.0
    surprise: float = 0.0
    satisfaction: float = 0.0

    # Control signals
    should_verify: bool = False
    should_elaborate: bool = False
    should_simplify: bool = False
    should_seek_info: bool = False


class ConfidenceEstimator(nn.Module):
    """
    Estimates confidence in reasoning steps.

    Uses multiple signals:
    - Embedding consistency (do concepts align?)
    - Evidence count and quality
    - Historical accuracy calibration
    """

    def __init__(self,
                 embedding_dim: int = 256,
                 device: str = "cpu"):
        super().__init__()
        self.device = device

        # Confidence estimation network
        self.estimator = nn.Sequential(
            nn.Linear(embedding_dim + 4, 128),  # embedding + meta features
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        ).to(device)

        # Historical calibration
        self.prediction_history: deque = deque(maxlen=1000)
        self.calibration_bins = {i: {"predicted": [], "actual": []} for i in range(10)}

    def estimate(self,
                 step_embedding: torch.Tensor,
                 evidence_count: int,
                 reasoning_type: str,
                 context_coherence: float) -> float:
        """Estimate confidence for a reasoning step."""

        # Meta features
        type_encoding = {
            "deduction": 0.9,
            "retrieval": 0.8,
            "induction": 0.6,
            "analogy": 0.5,
            "speculation": 0.3,
        }.get(reasoning_type, 0.5)

        evidence_factor = min(1.0, evidence_count / 5)

        meta_features = torch.tensor([
            type_encoding,
            evidence_factor,
            context_coherence,
            0.5,  # Placeholder for historical accuracy
        ], dtype=torch.float32, device=self.device)

        # Combine
        combined = torch.cat([step_embedding.flatten()[:252], meta_features])
        if combined.shape[0] < 260:
            combined = torch.nn.functional.pad(combined, (0, 260 - combined.shape[0]))

        confidence = self.estimator(combined.unsqueeze(0))
        return confidence.item()

    def calibrate(self, predicted_confidence: float, was_correct: bool):
        """Update calibration with outcome."""
        bin_idx = min(9, int(predicted_confidence * 10))
        self.calibration_bins[bin_idx]["predicted"].append(predicted_confidence)
        self.calibration_bins[bin_idx]["actual"].append(1.0 if was_correct else 0.0)

    def get_calibration_error(self) -> float:
        """Calculate Expected Calibration Error."""
        total_error = 0.0
        total_samples = 0

        for bin_idx, data in self.calibration_bins.items():
            if len(data["actual"]) > 0:
                avg_predicted = sum(data["predicted"]) / len(data["predicted"])
                avg_actual = sum(data["actual"]) / len(data["actual"])
                error = abs(avg_predicted - avg_actual)
                total_error += error * len(data["actual"])
                total_samples += len(data["actual"])

        return total_error / max(1, total_samples)


class ReasoningMonitor:
    """
    Monitors the reasoning process and detects issues.

    Tracks:
    - Logical coherence (do steps follow?)
    - Circular reasoning detection
    - Missing evidence
    - Overconfidence patterns

    Note: Plain class (not nn.Module) — no learnable parameters or buffers.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

        # Reasoning chain
        self.current_chain: List[ReasoningStep] = []

        # Issue detection thresholds
        self.coherence_threshold = 0.4
        self.confidence_variance_threshold = 0.3

    def add_step(self, step: ReasoningStep):
        """Add a reasoning step to the chain."""
        self.current_chain.append(step)

    def clear_chain(self):
        """Clear the current reasoning chain."""
        self.current_chain = []

    def analyze_chain(self) -> Dict[str, Any]:
        """Analyze the current reasoning chain for issues."""
        if not self.current_chain:
            return {"issues": [], "quality": 0.5}

        issues = []
        quality_factors = []

        # 1. Check confidence variance (wild swings = unstable reasoning)
        confidences = [s.confidence for s in self.current_chain]
        if len(confidences) > 1:
            variance = sum((c - sum(confidences)/len(confidences))**2 for c in confidences) / len(confidences)
            if variance > self.confidence_variance_threshold:
                issues.append("High confidence variance - reasoning may be unstable")
            quality_factors.append(1 - min(1, variance))

        # 2. Check for confidence decay (getting less sure = might be lost)
        if len(confidences) >= 3:
            trend = confidences[-1] - confidences[0]
            if trend < -0.3:
                issues.append("Confidence declining - may need to reconsider approach")
            quality_factors.append(0.5 + trend * 0.5)

        # 3. Check for very low confidence steps
        low_confidence_count = sum(1 for c in confidences if c < 0.3)
        if low_confidence_count > len(confidences) / 2:
            issues.append("Many uncertain steps - consider seeking more information")
            quality_factors.append(1 - low_confidence_count / len(confidences))

        # 4. Check evidence coverage
        evidence_counts = [len(s.evidence) for s in self.current_chain]
        avg_evidence = sum(evidence_counts) / len(evidence_counts) if evidence_counts else 0
        if avg_evidence < 1:
            issues.append("Low evidence - claims may be unsupported")
            quality_factors.append(min(1, avg_evidence))

        # 5. Detect repetition (circular reasoning)
        contents = [s.content.lower()[:50] for s in self.current_chain]
        unique_ratio = len(set(contents)) / len(contents)
        if unique_ratio < 0.7:
            issues.append("Repetitive reasoning - may be going in circles")
            quality_factors.append(unique_ratio)

        # Compute overall quality
        quality = sum(quality_factors) / len(quality_factors) if quality_factors else 0.5

        return {
            "issues": issues,
            "quality": quality,
            "step_count": len(self.current_chain),
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0,
            "confidence_trend": "declining" if len(confidences) >= 2 and confidences[-1] < confidences[0] - 0.2 else "stable",
        }


class MetacognitiveLoop(nn.Module):
    """
    The complete Metacognitive Loop - thinking about thinking.

    Provides:
    - Real-time confidence tracking
    - Reasoning quality monitoring
    - Epistemic emotion generation
    - Control signal generation (when to verify, elaborate, etc.)
    - Natural language uncertainty vocalization
    """

    def __init__(self,
                 embedding_dim: int = 256,
                 device: str = "cpu"):
        super().__init__()
        self.device = device

        # Components
        self.confidence_estimator = ConfidenceEstimator(embedding_dim, device)
        self.reasoning_monitor = ReasoningMonitor(device)

        # State
        self.state = MetacognitiveState()

        # Confidence history for trend detection
        self.confidence_history: deque = deque(maxlen=20)

        # Epistemic emotion dynamics
        self.curiosity_decay = 0.95
        self.confusion_decay = 0.9
        self.satisfaction_boost = 0.3

    def begin_reasoning(self, query: str):
        """Begin a new reasoning episode."""
        self.reasoning_monitor.clear_chain()
        self.confidence_history.clear()
        self.state = MetacognitiveState()

        # Initial curiosity from question
        if "?" in query or any(w in query.lower() for w in ["what", "how", "why"]):
            self.state.curiosity = 0.6

    def observe_step(self,
                    content: str,
                    embedding: torch.Tensor,
                    evidence: List[str] = None,
                    reasoning_type: str = "deduction") -> ReasoningStep:
        """Observe and evaluate a reasoning step."""

        # Estimate confidence
        context_coherence = self.reasoning_monitor.analyze_chain()["quality"]
        confidence = self.confidence_estimator.estimate(
            embedding,
            len(evidence) if evidence else 0,
            reasoning_type,
            context_coherence,
        )

        # Create step
        step = ReasoningStep(
            step_id=len(self.reasoning_monitor.current_chain),
            content=content,
            confidence=confidence,
            reasoning_type=reasoning_type,
            evidence=evidence or [],
        )

        # Add to chain
        self.reasoning_monitor.add_step(step)
        self.confidence_history.append(confidence)

        # Update overall confidence
        self.state.overall_confidence = 0.7 * self.state.overall_confidence + 0.3 * confidence

        # Update epistemic emotions
        self._update_epistemic_emotions(step)

        # Generate control signals
        self._update_control_signals()

        return step

    def _update_epistemic_emotions(self, step: ReasoningStep):
        """Update epistemic emotions based on step."""

        # Curiosity decays but can be reignited
        self.state.curiosity *= self.curiosity_decay
        if step.reasoning_type == "speculation":
            self.state.curiosity = min(1.0, self.state.curiosity + 0.2)

        # Confusion increases with low confidence
        if step.confidence < 0.4:
            self.state.confusion = min(1.0, self.state.confusion + 0.2)
        else:
            self.state.confusion *= self.confusion_decay

        # Surprise from unexpected confidence
        if len(self.confidence_history) > 2:
            expected = sum(list(self.confidence_history)[-3:-1]) / 2
            surprise = abs(step.confidence - expected)
            self.state.surprise = min(1.0, surprise * 2)

        # Satisfaction from high confidence conclusions
        if step.confidence > 0.8 and len(self.reasoning_monitor.current_chain) > 2:
            self.state.satisfaction = min(1.0, self.state.satisfaction + self.satisfaction_boost)

    def _update_control_signals(self):
        """Generate control signals based on metacognitive state."""

        analysis = self.reasoning_monitor.analyze_chain()

        # Should verify: low confidence or many issues
        self.state.should_verify = (
            self.state.overall_confidence < 0.5 or
            len(analysis["issues"]) > 2
        )

        # Should elaborate: too few steps for complex question
        self.state.should_elaborate = (
            analysis["step_count"] < 3 and
            self.state.curiosity > 0.3
        )

        # Should simplify: high confusion
        self.state.should_simplify = self.state.confusion > 0.6

        # Should seek info: low evidence
        self.state.should_seek_info = (
            "Low evidence" in str(analysis["issues"]) or
            analysis["avg_confidence"] < 0.4
        )

        # Update confidence trend
        if len(self.confidence_history) >= 3:
            recent = list(self.confidence_history)[-3:]
            if recent[-1] > recent[0] + 0.1:
                self.state.confidence_trend = "rising"
            elif recent[-1] < recent[0] - 0.1:
                self.state.confidence_trend = "falling"
            else:
                self.state.confidence_trend = "stable"

    def get_uncertainty_statement(self) -> str:
        """Generate natural language uncertainty statement."""
        conf = self.state.overall_confidence
        analysis = self.reasoning_monitor.analyze_chain()

        # Confidence statement
        if conf >= 0.85:
            conf_stmt = "I'm quite confident about this"
        elif conf >= 0.7:
            conf_stmt = "I'm fairly confident, but let me double-check"
        elif conf >= 0.5:
            conf_stmt = "I think this is right, but I'm not certain"
        elif conf >= 0.3:
            conf_stmt = "I'm uncertain about this"
        else:
            conf_stmt = "This is mostly a guess"

        # Add specific concerns
        concerns = []
        if self.state.should_verify:
            concerns.append("I should verify this")
        if self.state.should_seek_info:
            concerns.append("I might need more information")
        if self.state.confusion > 0.5:
            concerns.append("parts of this are confusing me")

        if concerns:
            return f"{conf_stmt} - {', '.join(concerns)}."
        return f"{conf_stmt}."

    def get_inner_monologue(self) -> str:
        """Generate inner monologue showing reasoning state."""
        lines = []

        # Confidence status
        conf_bar = "█" * int(self.state.overall_confidence * 10) + "░" * (10 - int(self.state.overall_confidence * 10))
        lines.append(f"Confidence: [{conf_bar}] {self.state.overall_confidence:.0%} ({self.state.confidence_trend})")

        # Epistemic emotions
        emotions = []
        if self.state.curiosity > 0.3:
            emotions.append(f"curious ({self.state.curiosity:.0%})")
        if self.state.confusion > 0.3:
            emotions.append(f"confused ({self.state.confusion:.0%})")
        if self.state.surprise > 0.3:
            emotions.append(f"surprised ({self.state.surprise:.0%})")
        if self.state.satisfaction > 0.3:
            emotions.append(f"satisfied ({self.state.satisfaction:.0%})")

        if emotions:
            lines.append(f"Feeling: {', '.join(emotions)}")

        # Control signals
        signals = []
        if self.state.should_verify:
            signals.append("verify")
        if self.state.should_elaborate:
            signals.append("elaborate")
        if self.state.should_simplify:
            signals.append("simplify")
        if self.state.should_seek_info:
            signals.append("seek info")

        if signals:
            lines.append(f"Should: {', '.join(signals)}")

        return "\n".join(lines)

    def forward(self,
                content: str,
                embedding: torch.Tensor,
                evidence: List[str] = None,
                reasoning_type: str = "deduction") -> Dict[str, Any]:
        """Process a reasoning step and return metacognitive analysis."""

        step = self.observe_step(content, embedding, evidence, reasoning_type)
        analysis = self.reasoning_monitor.analyze_chain()

        return {
            "step": step,
            "state": self.state,
            "analysis": analysis,
            "uncertainty": self.get_uncertainty_statement(),
            "monologue": self.get_inner_monologue(),
        }
