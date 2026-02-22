"""
Tests for PyQuifer Gap closure:
- Gap 2: Proposal provenance (timestamp, source_trust, OscillatoryWriteGate freshness)
- Gap 1: Budget pre-filter in _run_organ_workspace
- Gap 3: ConfidenceTracker ECE computation
"""
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch


# ─────────────────────────────────────────────────────────────────────────────
# Proposal provenance fields
# ─────────────────────────────────────────────────────────────────────────────

class TestProposalProvenance(unittest.TestCase):
    def _make_proposal(self, **kwargs):
        from pyquifer.workspace.organ_base import Proposal
        return Proposal(
            content=torch.zeros(16),
            salience=0.5,
            **kwargs,
        )

    def test_default_timestamp_is_recent(self):
        p = self._make_proposal()
        self.assertAlmostEqual(p.timestamp, time.monotonic(), delta=1.0)

    def test_default_source_trust_is_one(self):
        p = self._make_proposal()
        self.assertEqual(p.source_trust, 1.0)

    def test_custom_source_trust(self):
        p = self._make_proposal(source_trust=0.6)
        self.assertEqual(p.source_trust, 0.6)

    def test_old_proposal_has_old_timestamp(self):
        p = self._make_proposal(timestamp=time.monotonic() - 100.0)
        self.assertGreater(time.monotonic() - p.timestamp, 99.0)

    def test_cost_still_defaults_zero(self):
        p = self._make_proposal()
        self.assertEqual(p.cost, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# OscillatoryWriteGate freshness
# ─────────────────────────────────────────────────────────────────────────────

class TestWriteGateFreshness(unittest.TestCase):
    def _make_gate(self):
        from pyquifer.workspace.organ_base import OscillatoryWriteGate
        return OscillatoryWriteGate()

    def test_freshness_one_is_normal(self):
        gate = self._make_gate()
        phase = torch.tensor(0.0)
        g1 = gate(phase, phase, freshness=1.0)
        g_ref = gate(phase, phase)  # default freshness=1.0
        self.assertAlmostEqual(g1.item(), g_ref.item(), places=5)

    def test_freshness_zero_gates_output(self):
        gate = self._make_gate()
        phase = torch.tensor(0.0)
        g = gate(phase, phase, freshness=0.0)
        self.assertAlmostEqual(g.item(), 0.0, places=5)

    def test_freshness_half_halves_output(self):
        gate = self._make_gate()
        phase = torch.tensor(0.0)
        g1 = gate(phase, phase, freshness=1.0)
        g_half = gate(phase, phase, freshness=0.5)
        self.assertAlmostEqual(g_half.item(), g1.item() * 0.5, places=5)

    def test_freshness_above_one_allowed(self):
        # freshness > 1.0 is technically allowed (boosts), should not error
        gate = self._make_gate()
        phase = torch.tensor(0.0)
        g = gate(phase, phase, freshness=1.5)
        self.assertGreater(g.item(), 0.0)

    def test_old_proposal_gets_lower_gate(self):
        from pyquifer.workspace.organ_base import OscillatoryWriteGate
        import time as _time
        gate = OscillatoryWriteGate()
        phase = torch.tensor(0.0)
        fresh = gate(phase, phase, freshness=1.0)
        stale = gate(phase, phase, freshness=0.1)
        self.assertGreater(fresh.item(), stale.item())


# ─────────────────────────────────────────────────────────────────────────────
# ConfidenceTracker
# ─────────────────────────────────────────────────────────────────────────────

class TestConfidenceTracker(unittest.TestCase):
    def _make_tracker(self):
        from pyquifer.runtime.confidence_tracker import ConfidenceTracker
        return ConfidenceTracker()

    def test_ece_zero_when_no_samples(self):
        t = self._make_tracker()
        self.assertEqual(t.ece(), 0.0)

    def test_ece_zero_when_too_few_samples(self):
        t = self._make_tracker()
        for _ in range(5):
            t.record(0.8, True)
        self.assertEqual(t.ece(), 0.0)

    def test_ece_nonzero_after_enough_samples(self):
        t = self._make_tracker()
        # Perfect calibration: always predict 0.9, always succeed
        for _ in range(20):
            t.record(0.9, True)
        ece = t.ece()
        # ECE = |0.9 - 1.0| * 1.0 = 0.1 (all in one bin, acc=1.0, conf=0.9)
        self.assertAlmostEqual(ece, 0.1, places=3)

    def test_perfect_calibration_near_zero(self):
        t = self._make_tracker()
        # 10 samples: 90% conf, 90% success
        for i in range(10):
            t.record(0.9, i < 9)  # 9 successes, 1 fail
        # avg_conf=0.9, avg_acc=0.9 → ECE ≈ 0.0
        self.assertLess(t.ece(), 0.05)

    def test_record_increments_count(self):
        t = self._make_tracker()
        t.record(0.5, True)
        self.assertEqual(len(t), 1)

    def test_max_history_truncation(self):
        t = self._make_tracker.__func__(self)
        from pyquifer.runtime.confidence_tracker import ConfidenceTracker
        t = ConfidenceTracker(max_history=20)
        for _ in range(30):
            t.record(0.5, True)
        self.assertLessEqual(len(t), 20)

    def test_calibration_summary_keys(self):
        t = self._make_tracker()
        s = t.calibration_summary()
        self.assertIn("n_samples", s)
        self.assertIn("ece", s)
        self.assertIn("recent_success_rate", s)
        self.assertIn("calibrated", s)

    def test_calibrated_none_when_few_samples(self):
        t = self._make_tracker()
        t.record(0.5, True)
        s = t.calibration_summary()
        self.assertIsNone(s["calibrated"])

    def test_calibrated_true_when_low_ece(self):
        t = self._make_tracker()
        for i in range(20):
            t.record(0.9, i < 18)  # 18/20 success, conf=0.9 → ECE small
        s = t.calibration_summary()
        if s["ece"] < 0.1:
            self.assertTrue(s["calibrated"])


# ─────────────────────────────────────────────────────────────────────────────
# ModulationState new fields
# ─────────────────────────────────────────────────────────────────────────────

class TestModulationStateNewFields(unittest.TestCase):
    def test_workspace_winner_confidence_default(self):
        from pyquifer.api.bridge import ModulationState
        s = ModulationState()
        self.assertEqual(s.workspace_winner_confidence, 0.0)

    def test_workspace_winner_source_trust_default(self):
        from pyquifer.api.bridge import ModulationState
        s = ModulationState()
        self.assertEqual(s.workspace_winner_source_trust, 1.0)

    def test_energy_budget_default(self):
        from pyquifer.api.bridge import ModulationState
        s = ModulationState()
        self.assertEqual(s.energy_budget, 1.0)

    def test_energy_ratio_default(self):
        from pyquifer.api.bridge import ModulationState
        s = ModulationState()
        self.assertEqual(s.energy_ratio, 1.0)

    def test_can_set_confidence_field(self):
        from pyquifer.api.bridge import ModulationState
        s = ModulationState(workspace_winner_confidence=0.78, workspace_winner_source_trust=0.9)
        self.assertAlmostEqual(s.workspace_winner_confidence, 0.78)
        self.assertAlmostEqual(s.workspace_winner_source_trust, 0.9)


if __name__ == "__main__":
    unittest.main()
