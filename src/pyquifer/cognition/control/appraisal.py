"""
Cognitive Appraisal Model — structured emotion generation from evaluation.

Implements appraisal theory: emotions arise from evaluating events along
structured dimensions (goal relevance, coping potential, norm compatibility,
novelty, certainty). This gives emotions REASONS, not just values.

Key classes:
- AppraisalDimension: Individual evaluation axis
- AppraisalChain: Sequential evaluation pipeline
- OCCModel: Ortony/Clore/Collins emotion generation
- EmotionAttribution: Explain WHY an emotion was generated

References:
- Ortony, Clore & Collins (1988). The Cognitive Structure of Emotions.
- Scherer (2001). Appraisal considered as a process of multilevel sequential checking.
- Lazarus (1991). Emotion and adaptation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class AppraisalResult:
    """Result of appraising a stimulus along one dimension."""
    dimension: str
    value: float  # [-1, 1] for bipolar, [0, 1] for unipolar
    confidence: float = 1.0
    reasoning: str = ""


@dataclass
class EmotionState:
    """Generated emotion with attribution."""
    emotion_type: str  # e.g., "joy", "anger", "fear", "surprise"
    intensity: float  # [0, 1]
    valence: float  # [-1, 1] negative to positive
    arousal: float  # [0, 1] calm to excited
    appraisals: List[AppraisalResult] = field(default_factory=list)
    attribution: str = ""  # Why this emotion was generated


class AppraisalDimension(nn.Module):
    """Individual evaluation axis for stimulus appraisal.

    Each dimension evaluates one aspect of a stimulus:
    - goal_relevance: Does this affect my goals?
    - coping_potential: Can I deal with this?
    - norm_compatibility: Does this align with my values?
    - novelty: How unexpected is this?
    - certainty: How sure am I about the outcome?

    Args:
        dim: Input feature dimension.
        name: Name of this appraisal dimension.
        bipolar: If True, output in [-1, 1]; if False, [0, 1].
    """

    def __init__(self, dim: int, name: str, bipolar: bool = True):
        super().__init__()
        self.name = name
        self.bipolar = bipolar

        self.evaluator = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate stimulus on this dimension.

        Args:
            x: (..., dim) stimulus features

        Returns:
            (value, confidence): dimension evaluation and confidence
        """
        raw = self.evaluator(x).squeeze(-1)
        if self.bipolar:
            value = torch.tanh(raw)
        else:
            value = torch.sigmoid(raw)
        confidence = self.confidence_head(x).squeeze(-1)
        return value, confidence


class AppraisalChain(nn.Module):
    """Sequential multi-dimension appraisal pipeline.

    Evaluates a stimulus along multiple dimensions in sequence,
    where each evaluation can be influenced by previous results
    (Scherer's sequential checking model).

    Args:
        dim: Input feature dimension.
        dimensions: List of dimension names to evaluate.
    """

    STANDARD_DIMENSIONS = [
        ("goal_relevance", True),   # Does this matter? [-1, 1]
        ("coping_potential", True),  # Can I handle it? [-1, 1]
        ("norm_compatibility", True),  # Does it fit my values? [-1, 1]
        ("novelty", False),          # How unexpected? [0, 1]
        ("certainty", False),        # How sure am I? [0, 1]
    ]

    def __init__(self, dim: int, dimensions: Optional[List[Tuple[str, bool]]] = None):
        super().__init__()
        self.dim = dim

        if dimensions is None:
            dimensions = self.STANDARD_DIMENSIONS

        self.dimension_names = [d[0] for d in dimensions]

        self.dimensions = nn.ModuleDict({
            name: AppraisalDimension(dim, name, bipolar)
            for name, bipolar in dimensions
        })

        # Context integration: previous appraisals influence later ones
        self.context_proj = nn.Linear(len(dimensions), dim)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run sequential appraisal.

        Args:
            x: (..., dim) stimulus features

        Returns:
            Dict with dimension names as keys, each containing
            'value' and 'confidence' tensors
        """
        results = {}
        accumulated = torch.zeros(
            *x.shape[:-1], len(self.dimension_names),
            device=x.device, dtype=x.dtype,
        )

        for i, name in enumerate(self.dimension_names):
            # Integrate context from previous appraisals
            if i > 0:
                context = self.context_proj(accumulated)
                x_ctx = x + context
            else:
                x_ctx = x

            value, confidence = self.dimensions[name](x_ctx)
            results[name] = {'value': value, 'confidence': confidence}
            accumulated[..., i] = value.detach() if value.dim() > 0 else value

        return results


class OCCModel(nn.Module):
    """Ortony/Clore/Collins emotion generation from appraisal dimensions.

    Maps appraisal patterns to specific emotions using learned
    prototypical appraisal profiles for each emotion type.

    Standard OCC emotions:
    - Joy/Distress: goal-relevant, controllable outcomes
    - Hope/Fear: uncertain future outcomes
    - Anger: goal-blocking + other-attributed
    - Surprise: novel, uncertain
    - Gratitude/Resentment: other-attributed outcomes
    - Pride/Shame: self-attributed outcomes

    Args:
        num_dimensions: Number of appraisal dimensions.
        num_emotions: Number of emotion types.
        emotion_names: Optional list of emotion type names.
    """

    DEFAULT_EMOTIONS = [
        "joy", "distress", "hope", "fear", "anger",
        "surprise", "gratitude", "shame", "pride",
        "relief", "disappointment", "interest",
    ]

    def __init__(
        self,
        num_dimensions: int = 5,
        num_emotions: int = 12,
        emotion_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.num_dimensions = num_dimensions
        self.num_emotions = num_emotions
        self.emotion_names = emotion_names or self.DEFAULT_EMOTIONS[:num_emotions]

        # Prototypical appraisal profiles for each emotion
        self.emotion_profiles = nn.Parameter(
            torch.randn(num_emotions, num_dimensions) * 0.3
        )

        # Intensity modulation
        self.intensity_net = nn.Sequential(
            nn.Linear(num_dimensions, num_dimensions * 2),
            nn.ReLU(),
            nn.Linear(num_dimensions * 2, num_emotions),
            nn.Sigmoid(),
        )

        # Valence and arousal mapping
        self.valence_map = nn.Linear(num_emotions, 1)
        self.arousal_map = nn.Linear(num_emotions, 1)

    def forward(
        self, appraisal_values: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Generate emotions from appraisal values.

        Args:
            appraisal_values: (..., num_dimensions) concatenated appraisal values

        Returns:
            Dict with 'emotion_scores', 'dominant_emotion', 'valence',
            'arousal', 'intensity'
        """
        # Match against emotion profiles (cosine similarity)
        normed_input = appraisal_values / (appraisal_values.norm(dim=-1, keepdim=True) + 1e-8)
        normed_profiles = self.emotion_profiles / (
            self.emotion_profiles.norm(dim=-1, keepdim=True) + 1e-8
        )

        # Similarity to each emotion prototype
        scores = torch.matmul(normed_input, normed_profiles.T)  # (..., num_emotions)

        # Intensity modulation
        intensity = self.intensity_net(appraisal_values)
        modulated = scores * intensity

        # Softmax for dominant emotion
        emotion_probs = torch.softmax(modulated, dim=-1)
        dominant = emotion_probs.argmax(dim=-1)

        # Valence and arousal
        valence = torch.tanh(self.valence_map(emotion_probs)).squeeze(-1)
        arousal = torch.sigmoid(self.arousal_map(emotion_probs)).squeeze(-1)

        return {
            'emotion_scores': modulated,
            'emotion_probs': emotion_probs,
            'dominant_emotion': dominant,
            'valence': valence,
            'arousal': arousal,
            'intensity': intensity,
        }


# Backward compat alias
OCC_Model = OCCModel


class EmotionAttribution(nn.Module):
    """Explain WHY an emotion was generated.

    Traces the causal chain from stimulus through appraisal dimensions
    to the generated emotion, producing a structured explanation.

    Args:
        dim: Feature dimension.
        num_dimensions: Number of appraisal dimensions.
        num_emotions: Number of emotion types.
    """

    def __init__(
        self,
        dim: int,
        num_dimensions: int = 5,
        num_emotions: int = 12,
    ):
        super().__init__()
        self.dim = dim

        # Attribution network: which dimensions most contributed?
        self.attribution_net = nn.Sequential(
            nn.Linear(num_dimensions + num_emotions, num_dimensions * 2),
            nn.ReLU(),
            nn.Linear(num_dimensions * 2, num_dimensions),
            nn.Softmax(dim=-1),
        )

        # Cause encoder: map stimulus features to cause representation
        self.cause_encoder = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim // 4),
        )

    def forward(
        self,
        stimulus: torch.Tensor,
        appraisal_values: torch.Tensor,
        emotion_scores: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute emotion attribution.

        Args:
            stimulus: (..., dim) original stimulus features
            appraisal_values: (..., num_dimensions) appraisal results
            emotion_scores: (..., num_emotions) emotion scores

        Returns:
            Dict with 'dimension_importance', 'cause_embedding',
            'top_dimension'
        """
        combined = torch.cat([appraisal_values, emotion_scores], dim=-1)
        importance = self.attribution_net(combined)

        cause = self.cause_encoder(stimulus)
        top_dim = importance.argmax(dim=-1)

        return {
            'dimension_importance': importance,
            'cause_embedding': cause,
            'top_dimension': top_dim,
        }
