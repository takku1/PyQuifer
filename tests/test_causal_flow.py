"""Tests for causal_flow module."""
import torch
import pytest
from pyquifer.causal_flow import TransferEntropyEstimator, CausalFlowMap, DominanceDetector


class TestTransferEntropyEstimator:
    def test_causal_signal_detected(self):
        torch.manual_seed(42)
        te = TransferEntropyEstimator(num_bins=8, history_length=2)
        T = 400
        x = torch.randn(T)
        y = torch.zeros(T)
        for t in range(1, T):
            y[t] = 0.8 * x[t-1] + 0.2 * torch.randn(1).item()
        r = te(x, y)
        assert r['te_source_to_target'].item() > 0
        assert r['net_flow'].item() > 0

    def test_independent_signals_low_flow(self):
        torch.manual_seed(42)
        te = TransferEntropyEstimator(num_bins=8, history_length=2)
        # Causal
        T = 400
        x = torch.randn(T)
        y_causal = torch.zeros(T)
        for t in range(1, T):
            y_causal[t] = 0.8 * x[t-1] + 0.2 * torch.randn(1).item()
        causal_flow = abs(te(x, y_causal)['net_flow'].item())
        # Independent
        y_indep = torch.randn(T)
        indep_flow = abs(te(x, y_indep)['net_flow'].item())
        assert indep_flow < causal_flow

    def test_short_series_returns_zero(self):
        te = TransferEntropyEstimator(num_bins=4, history_length=2)
        r = te(torch.randn(5), torch.randn(5))
        assert r['te_source_to_target'].item() == 0.0


class TestCausalFlowMap:
    def test_flow_matrix_shape(self):
        cfm = CausalFlowMap(num_populations=4, buffer_size=200)
        for _ in range(200):
            cfm.record(torch.randn(4))
        flow = cfm.compute_flow()
        assert flow['flow_matrix'].shape == (4, 4)

    def test_chain_driver_detection(self):
        cfm = CausalFlowMap(num_populations=3, buffer_size=300)
        for _ in range(300):
            s = torch.zeros(3)
            s[0] = torch.randn(1).item()
            s[1] = 0.7 * s[0] + 0.3 * torch.randn(1).item()
            s[2] = 0.7 * s[1] + 0.3 * torch.randn(1).item()
            cfm.record(s)
        flow = cfm.compute_flow()
        assert flow['driver_scores'][0] >= flow['driver_scores'][2]

    def test_reset(self):
        cfm = CausalFlowMap(num_populations=3)
        cfm.record(torch.randn(3))
        cfm.reset()
        assert cfm.ts_ptr.item() == 0


class TestDominanceDetector:
    def test_output_keys(self):
        dd = DominanceDetector(num_levels=3, buffer_size=100)
        for _ in range(100):
            r = dd(torch.randn(3), compute_every=50)
        assert all(k in r for k in ['dominance_ratio', 'bottom_up_te', 'top_down_te', 'mode'])

    def test_reset(self):
        dd = DominanceDetector(num_levels=3)
        dd(torch.randn(3))
        dd.reset()
        assert dd.hist_ptr.item() == 0
        assert dd.dominance_ratio.item() == 0.5
