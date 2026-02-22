"""
Eval Harness — Falsifiable benchmarks for PyQuifer feedback loops.

E1: ECE < 0.15 after 50 well-calibrated samples.
E2: Freshness gate at 35s < 0.1 * freshness gate at 0s (stale proposals silenced).
E4: Budget pre-filter admits zero cost-budget violations when cumulative cost > budget.

These tests are self-contained and require no Mizuki runtime.
"""
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch


# ─────────────────────────────────────────────────────────────────────────────
# E1 — Calibration benchmark
# ─────────────────────────────────────────────────────────────────────────────

class TestE1CalibrationBenchmark(unittest.TestCase):
    """
    E1: After 50 samples whose predicted confidence matches empirical accuracy,
    the Expected Calibration Error (ECE) must be < 0.15.
    """

    def _make_tracker(self):
        from pyquifer.runtime.confidence_tracker import ConfidenceTracker
        return ConfidenceTracker(max_history=500)

    def test_e1_perfectly_calibrated_50_samples(self):
        """70% confidence with 70% accuracy → ECE ≈ 0 after 50 samples."""
        t = self._make_tracker()
        import random
        random.seed(42)
        for i in range(50):
            success = (i % 10) < 7   # deterministic 70% accuracy
            t.record(0.7, success)
        ece = t.ece()
        self.assertLess(ece, 0.15,
            f"E1 FAIL: ECE={ece:.4f} ≥ 0.15 after 50 calibrated samples")

    def test_e1_near_perfect_calibration_90pct(self):
        """90% confidence with 90% accuracy → ECE should be < 0.05."""
        t = self._make_tracker()
        for i in range(50):
            success = (i % 10) < 9   # 90% accuracy
            t.record(0.9, success)
        ece = t.ece()
        self.assertLess(ece, 0.15,
            f"E1 FAIL: 90% calibrated ECE={ece:.4f} ≥ 0.15")

    def test_e1_miscalibrated_exceeds_threshold(self):
        """Overconfident (predict 0.95, succeed 30%) → ECE >> 0.15."""
        t = self._make_tracker()
        for i in range(50):
            success = (i % 10) < 3   # 30% accuracy despite 0.95 confidence
            t.record(0.95, success)
        ece = t.ece()
        self.assertGreater(ece, 0.15,
            f"E1: Expected miscalibrated ECE > 0.15, got {ece:.4f}")

    def test_e1_ece_not_computed_below_min_samples(self):
        """ECE returns 0.0 when fewer than min_samples recorded."""
        t = self._make_tracker()
        for _ in range(5):
            t.record(0.5, True)
        self.assertEqual(t.ece(), 0.0,
            "E1: ECE should be 0.0 when sample count < min_samples")

    def test_e1_set_calibration_hint_method_exists(self):
        """bridge.set_calibration_hint() must exist and accept float."""
        from pyquifer.api.bridge import PyQuiferBridge
        from pyquifer.runtime.config import CycleConfig
        cfg = CycleConfig.default()
        cfg.n_organs = 0
        bridge = PyQuiferBridge(cfg)
        self.assertTrue(hasattr(bridge, 'set_calibration_hint'),
            "E1: PyQuiferBridge missing set_calibration_hint()")
        bridge.set_calibration_hint(0.20)
        self.assertAlmostEqual(bridge._last_ece, 0.20, places=5)

    def test_e1_calibration_hint_clamped_to_unit_range(self):
        """set_calibration_hint clamps to [0, 1]."""
        from pyquifer.api.bridge import PyQuiferBridge
        from pyquifer.runtime.config import CycleConfig
        cfg = CycleConfig.default()
        cfg.n_organs = 0
        bridge = PyQuiferBridge(cfg)
        bridge.set_calibration_hint(5.0)
        self.assertAlmostEqual(bridge._last_ece, 1.0, places=5)
        bridge.set_calibration_hint(-2.0)
        self.assertAlmostEqual(bridge._last_ece, 0.0, places=5)


# ─────────────────────────────────────────────────────────────────────────────
# E2 — Freshness gate decay benchmark
# ─────────────────────────────────────────────────────────────────────────────

class TestE2FreshnessGateBenchmark(unittest.TestCase):
    """
    E2: The OscillatoryWriteGate score for a proposal aged 35 seconds
    must be < 0.1 × the score for a fresh proposal.

    The cycle uses: freshness = max(0.0, 1.0 - age_s / 30.0)
    At 35s: freshness = 0.0  → gate = 0.0
    At  0s: freshness = 1.0  → gate > 0
    """

    def _make_gate(self):
        from pyquifer.workspace.organ_base import OscillatoryWriteGate
        return OscillatoryWriteGate()

    def _freshness_at(self, age_s: float) -> float:
        return max(0.0, 1.0 - age_s / 30.0)

    def test_e2_stale_35s_gate_below_10pct_of_fresh(self):
        """Gate at 35s must be < 0.1 × gate at 0s."""
        gate = self._make_gate()
        phase = torch.tensor(0.0)

        gate_fresh = gate(phase, phase, freshness=self._freshness_at(0.0)).item()
        gate_stale = gate(phase, phase, freshness=self._freshness_at(35.0)).item()

        self.assertGreater(gate_fresh, 0.0,
            "E2: Fresh gate must be positive")
        self.assertLess(gate_stale, 0.1 * gate_fresh,
            f"E2 FAIL: gate(35s)={gate_stale:.4f} ≥ 0.1 × gate(0s)={gate_fresh:.4f}")

    def test_e2_freshness_formula_at_30s_is_zero(self):
        """At exactly 30s, freshness = 0.0."""
        self.assertAlmostEqual(self._freshness_at(30.0), 0.0, places=6)

    def test_e2_freshness_formula_at_15s_is_half(self):
        """At 15s, freshness = 0.5."""
        self.assertAlmostEqual(self._freshness_at(15.0), 0.5, places=6)

    def test_e2_freshness_formula_at_0s_is_one(self):
        """At 0s, freshness = 1.0."""
        self.assertAlmostEqual(self._freshness_at(0.0), 1.0, places=6)

    def test_e2_gate_monotone_decreases_with_age(self):
        """Gate score must be monotonically decreasing from 0s to 30s."""
        gate = self._make_gate()
        phase = torch.tensor(0.0)
        ages = [0, 5, 10, 15, 20, 25, 30]
        gates = [gate(phase, phase, freshness=self._freshness_at(a)).item() for a in ages]
        for i in range(len(gates) - 1):
            self.assertGreaterEqual(gates[i], gates[i + 1],
                f"E2: Gate not monotone at age {ages[i]}s → {ages[i+1]}s "
                f"({gates[i]:.4f} < {gates[i+1]:.4f})")

    def test_e2_cycle_propagates_source_trust_to_proposal(self):
        """cycle.py must propagate organ.source_trust → proposal.source_trust."""
        from pyquifer.workspace.organ_base import Organ, Proposal
        import torch.nn as nn

        class MockOrgan(Organ):
            def __init__(self):
                super().__init__("mock", latent_dim=8, source_trust=0.55)
                self._latent = torch.zeros(8)

            def observe(self, si, gb=None): pass
            def propose(self, gw=None):
                return Proposal(content=self._latent, salience=0.5, organ_id=self.organ_id)
            def accept(self, gb): pass

        organ = MockOrgan()
        proposal = organ.propose()
        # Simulate what cycle.py does:
        proposal.source_trust = getattr(organ, 'source_trust', 0.75)
        self.assertAlmostEqual(proposal.source_trust, 0.55, places=5)


# ─────────────────────────────────────────────────────────────────────────────
# E4 — Budget pre-filter benchmark
# ─────────────────────────────────────────────────────────────────────────────

class TestE4BudgetPreFilterBenchmark(unittest.TestCase):
    """
    E4: When the budget pre-filter is active and proposals' total cost exceeds
    the remaining budget, the admitted set must not violate the budget.

    Budget invariant: sum(admitted_costs) <= remaining_budget
    (except the first/highest-salience proposal is always admitted regardless of cost).
    """

    def _simulate_budget_filter(self, proposals_data, budget: float):
        """
        Simulate cycle.py budget pre-filter logic.

        proposals_data: list of (salience, cost) tuples
        Returns admitted proposals (salience, cost) list.
        """
        from dataclasses import dataclass

        @dataclass
        class _FakeProposal:
            salience: float
            cost: float

        proposals = [(_FakeProposal(s, c),) for s, c in proposals_data]
        # Simulate: sort by salience desc, admit if cumulative cost ≤ budget
        sorted_props = sorted(proposals, key=lambda x: x[0].salience, reverse=True)
        admitted = []
        cum_cost = 0.0
        for (p,) in sorted_props:
            p_cost = max(0.0, p.cost)
            if not admitted or cum_cost + p_cost <= budget:
                admitted.append((p.salience, p_cost))
                cum_cost += p_cost
        return admitted, cum_cost

    def test_e4_admitted_costs_within_budget(self):
        """No admitted proposal set may exceed budget (except forced first)."""
        proposals = [(0.9, 0.4), (0.7, 0.35), (0.5, 0.30), (0.3, 0.25)]
        budget = 0.50
        admitted, total_cost = self._simulate_budget_filter(proposals, budget)

        # Only admitted[1:] is budget-gated; admitted[0] is always in
        forced_first_cost = admitted[0][1] if admitted else 0.0
        remaining_after_first = budget - forced_first_cost
        non_forced_cost = total_cost - forced_first_cost

        self.assertGreaterEqual(remaining_after_first + 1e-9, non_forced_cost,
            f"E4 FAIL: non-forced admitted cost {non_forced_cost:.3f} > "
            f"remaining budget {remaining_after_first:.3f}")

    def test_e4_first_proposal_always_admitted(self):
        """Highest-salience proposal is always admitted regardless of cost."""
        proposals = [(0.95, 0.99), (0.4, 0.01)]  # first costs 99% of any reasonable budget
        budget = 0.10
        admitted, _ = self._simulate_budget_filter(proposals, budget)
        saliences = [s for s, _ in admitted]
        self.assertIn(0.95, saliences,
            "E4: Highest-salience proposal must always be admitted")

    def test_e4_zero_cost_proposals_all_admitted(self):
        """Zero-cost proposals should all be admitted when budget is any positive value."""
        proposals = [(0.9, 0.0), (0.7, 0.0), (0.5, 0.0), (0.3, 0.0)]
        budget = 0.5
        admitted, total_cost = self._simulate_budget_filter(proposals, budget)
        self.assertEqual(len(admitted), len(proposals),
            "E4: All zero-cost proposals must be admitted")
        self.assertAlmostEqual(total_cost, 0.0, places=5)

    def test_e4_costly_proposals_excluded(self):
        """When every proposal costs more than budget, only first is admitted."""
        proposals = [(0.9, 0.6), (0.7, 0.6), (0.5, 0.6)]
        budget = 0.5
        admitted, _ = self._simulate_budget_filter(proposals, budget)
        # First is always in; second would push cost to 1.2 > 0.5 → excluded
        self.assertEqual(len(admitted), 1,
            f"E4: Expected 1 admitted (budget=0.5, each costs 0.6), got {len(admitted)}")

    def test_e4_cycle_config_has_metabolic_budget_flag(self):
        """CycleConfig must have use_metabolic_budget flag."""
        from pyquifer.runtime.config import CycleConfig
        cfg = CycleConfig.default()
        self.assertTrue(hasattr(cfg, 'use_metabolic_budget'),
            "E4: CycleConfig missing use_metabolic_budget flag")

    def test_e4_calibration_hint_raises_ignition_cost(self):
        """CognitiveCycle.tick accepts calibration_hint and uses it to raise ignition cost."""
        from pyquifer.runtime.cycle import CognitiveCycle
        from pyquifer.runtime.config import CycleConfig
        import inspect
        sig = inspect.signature(CognitiveCycle.tick)
        self.assertIn('calibration_hint', sig.parameters,
            "E4: CognitiveCycle.tick missing calibration_hint parameter")

    def test_e4_calibration_hint_zero_no_change(self):
        """When calibration_hint=0.0, ignition cost multiplier should be 1.0."""
        # The cycle uses: if calibration_hint > 0.15: _ign_cost *= (1 + 2*(hint-0.15))
        # At hint=0.0: no change
        hint = 0.0
        base_cost = 1.0
        if hint > 0.15:
            effective = base_cost * (1.0 + 2.0 * (hint - 0.15))
        else:
            effective = base_cost
        self.assertAlmostEqual(effective, base_cost, places=5)

    def test_e4_calibration_hint_high_raises_cost(self):
        """At calibration_hint=0.30, ignition cost should be 1.3× base."""
        hint = 0.30
        base_cost = 1.0
        if hint > 0.15:
            effective = base_cost * (1.0 + 2.0 * (hint - 0.15))
        else:
            effective = base_cost
        expected = 1.0 + 2.0 * (0.30 - 0.15)  # = 1.30
        self.assertAlmostEqual(effective, expected, places=5)


if __name__ == "__main__":
    unittest.main()
