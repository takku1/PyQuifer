"""
ConfidenceTracker — Online ECE computation for calibrated action selection.

Records (predicted_confidence, outcome_success) pairs and computes
Expected Calibration Error (ECE) with 10-bin histogram.

ECE = sum_b |avg_conf_b - avg_acc_b| * (|bin_b| / N)

Lower ECE = better calibrated. Well-calibrated system: predicted 0.8 confidence
should be correct ~80% of the time.

Usage:
    tracker = ConfidenceTracker()
    tracker.record(predicted_conf=0.82, success=True)
    tracker.record(predicted_conf=0.45, success=False)
    print(tracker.ece())            # -> float, lower is better
    print(tracker.calibration_summary())  # -> dict for logging/API
"""
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class _Bin:
    total_conf: float = 0.0
    total_success: float = 0.0
    count: int = 0

    @property
    def avg_conf(self) -> float:
        return self.total_conf / self.count if self.count else 0.0

    @property
    def avg_acc(self) -> float:
        return self.total_success / self.count if self.count else 0.0


class ConfidenceTracker:
    """
    Online ECE tracker. Thread-safe via GIL-protected list appends.

    Records one (confidence, success) pair per workspace competition winner.
    Call record() after each tick where an outcome is observable.
    Call ece() to get current calibration quality.
    """

    N_BINS = 10
    MIN_SAMPLES_FOR_ECE = 10

    def __init__(self, max_history: int = 1000):
        self._pairs: List[Tuple[float, bool]] = []
        self._max_history = max_history

    def record(self, predicted_conf: float, success: bool) -> None:
        """Record one (confidence, outcome) pair.

        Args:
            predicted_conf: Confidence score from GlobalWorkspace ignition [0, 1].
            success: Whether the action/generation was successful (caller-defined).
        """
        self._pairs.append((float(predicted_conf), bool(success)))
        if len(self._pairs) > self._max_history:
            self._pairs = self._pairs[-self._max_history:]

    def ece(self) -> float:
        """Compute Expected Calibration Error over recorded pairs.

        Returns 0.0 if fewer than MIN_SAMPLES_FOR_ECE samples are recorded.
        """
        if len(self._pairs) < self.MIN_SAMPLES_FOR_ECE:
            return 0.0

        bins = [_Bin() for _ in range(self.N_BINS)]
        for conf, success in self._pairs:
            b = min(int(conf * self.N_BINS), self.N_BINS - 1)
            bins[b].total_conf += conf
            bins[b].total_success += float(success)
            bins[b].count += 1

        n = len(self._pairs)
        return sum(
            abs(b.avg_conf - b.avg_acc) * (b.count / n)
            for b in bins if b.count > 0
        )

    def calibration_summary(self) -> dict:
        """Return a summary dict for logging/API exposure."""
        n = len(self._pairs)
        recent = self._pairs[-100:]
        return {
            "n_samples": n,
            "ece": round(self.ece(), 4),
            "recent_success_rate": (
                sum(1 for _, s in recent if s) / len(recent)
                if recent else 0.0
            ),
            "calibrated": self.ece() < 0.1 if n >= self.MIN_SAMPLES_FOR_ECE else None,
        }

    def __len__(self) -> int:
        return len(self._pairs)
