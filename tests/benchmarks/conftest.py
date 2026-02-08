"""
Benchmark conftest.py — Prevent pytest from crawling vendored third-party repos.

All directories listed in collect_ignore_glob are third-party repos cloned
for comparison benchmarking. They are NOT part of PyQuifer's test suite.
"""

collect_ignore_glob = [
    # Third-party vendored repos (alphabetical)
    "arXiv-*/**",
    "bifurcation/**",
    "chessqa-benchmark/**",
    "EGG/**",
    "emergent_communication_at_scale/**",
    "Gymnasium/**",
    "hanabi-learning-environment/**",
    "KuraNet/**",
    "lava/**",
    "MD-Bench/**",
    "meltingpot/**",
    "neurobench/**",
    "neurodiscoverybench/**",
    "nlb_tools/**",
    "open_spiel/**",
    "osbenchmarks/**",
    "Plaskett_puzzle/**",
    "PredBench/**",
    "SciMLBenchmarks.jl/**",
    "searchless_chess/**",
    "Torch2PC/**",
    "torchdiffeq/**",
    "torchdyn/**",
    # Runtime artifacts
    "data/**",
    "results/**",
]
