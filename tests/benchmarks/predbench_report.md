# Benchmark Report: PyQuifer Temporal Prediction vs PredBench

**Date:** 2026-02-06
**PyQuifer version:** Phase 5 (479 tests, 12 enhancements)
**Benchmark repo:** `tests/benchmarks/PredBench/` -- OpenEarthLab
**Script:** `tests/benchmarks/bench_predbench.py`

---

## Executive Summary

PyQuifer's oscillators and population models are **not designed for sequence prediction** -- they model internal neural dynamics, not external time series. A trained linear AR model achieves MSE=0.42 (corr=0.73) on sine sequences, while untrained Kuramoto oscillators achieve MSE=0.89 (corr=0.18) and WilsonCowan achieves MSE=0.50 (corr=-0.03). This is expected: PredBench methods (CNN, RNN, Transformer, diffusion) are trained on target data, while PyQuifer dynamics evolve according to their own equations. The comparison validates that PyQuifer modules produce bounded, finite predictions but aren't competitive as raw forecasters. PredBench: 9/14 features (ML prediction). PyQuifer: 5/14 (biological dynamics). No new gaps.

## Results

### Temporal Prediction (Sine Sequences)

| System | MSE | MAE | Correlation | Time |
|--------|:---:|:---:|:----------:|:----:|
| Linear AR (trained) | 0.417 | 0.538 | 0.725 | 1660 ms |
| Kuramoto (untrained) | 0.885 | 0.746 | 0.183 | 28 ms |
| WilsonCowan (untrained) | 0.498 | 0.632 | -0.027 | 36 ms |

PyQuifer's oscillators are 50x faster but produce uncorrelated predictions because they aren't trained on the target sequence. This is by design -- oscillators evolve through their own dynamics, not supervised training.

## Gaps Identified

No new gaps. PyQuifer's purpose is cognitive oscillatory dynamics, not time series forecasting.

## Pytest Results

```
5/5 passed (3.45s)
```

## Verdict

**PyQuifer temporal dynamics: PASS -- modules produce bounded, finite outputs; not designed for competitive sequence prediction.**
