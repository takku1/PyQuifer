"""Unified Benchmark Report Generator.

Collects all JSON results from tests/benchmarks/results/ and produces
a unified markdown report with three-column comparison tables,
weighted scoring, environment info, and Cohen's d effect sizes.

Usage:
  python tests/benchmarks/generate_report.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from harness import (
    BenchmarkResult, MetricCollector, cohens_d, compute_scenario_score,
    generate_report, get_environment_block,
)


def enhanced_report(results_dir: str, output_path: str) -> str:
    """Generate enhanced report with environment block, weighted scoring,
    and Cohen's d where multi-seed data is available."""
    results_path = Path(results_dir)
    output = Path(output_path)
    json_files = sorted(results_path.glob("*.json"))

    lines = [
        "# PyQuifer Comprehensive Benchmark Report\n",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        get_environment_block(),
    ]

    scenario_scores = []

    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        suite_name = data.get("suite", jf.stem)
        lines.append(f"\n## {suite_name}\n")

        for scenario in data.get("scenarios", []):
            mc = MetricCollector(scenario["scenario"])
            for r in scenario.get("results", []):
                mc.add_result(BenchmarkResult(**r))
            lines.append(mc.to_markdown_table())
            lines.append("")

            # Extract accuracy metrics for scoring
            acc_c, acc_b = None, None
            speed_b, speed_c = None, None
            lat_b, lat_c = None, None
            mem_b, mem_c = None, None

            for r in mc.results:
                if r.column == "C_pyquifer":
                    if "accuracy" in r.metrics:
                        acc_c = r.metrics["accuracy"]
                    elif "avg_accuracy" in r.metrics:
                        acc_c = r.metrics["avg_accuracy"]
                    # Look for speed/latency/memory
                    for k, v in r.metrics.items():
                        if "steps_per_sec" in k and speed_c is None:
                            speed_c = v
                        if "p50_ms" in k and lat_c is None:
                            lat_c = v
                        if "peak_memory" in k and mem_c is None:
                            mem_c = v
                elif r.column == "B_pytorch":
                    if "accuracy" in r.metrics:
                        acc_b = r.metrics["accuracy"]
                    elif "avg_accuracy" in r.metrics:
                        acc_b = r.metrics["avg_accuracy"]
                    for k, v in r.metrics.items():
                        if "steps_per_sec" in k and speed_b is None:
                            speed_b = v
                        if "p50_ms" in k and lat_b is None:
                            lat_b = v
                        if "peak_memory" in k and mem_b is None:
                            mem_b = v

            # Compute weighted score if we have accuracy
            if acc_c is not None and acc_c > 0:
                speed_ratio = (speed_c / speed_b) if speed_b and speed_c else 1.0
                lat_ratio = (lat_b / lat_c) if lat_b and lat_c and lat_c > 0 else 1.0
                mem_ratio = (mem_b / mem_c) if mem_b and mem_c and mem_c > 0 else 1.0
                score = compute_scenario_score(
                    accuracy=acc_c,
                    speed_ratio=speed_ratio,
                    latency_ratio=lat_ratio,
                    memory_ratio=mem_ratio,
                )
                scenario_scores.append({
                    "scenario": scenario["scenario"],
                    "score": round(score, 4),
                    "accuracy_c": acc_c,
                    "accuracy_b": acc_b,
                })

            # Cohen's d from CI metadata (multi-seed)
            accs_b_list, accs_c_list = [], []
            for r in mc.results:
                if r.column == "B_pytorch" and "ci_95" in r.metadata:
                    if "accuracy" in r.metrics:
                        accs_b_list.append(r.metrics["accuracy"])
                if r.column == "C_pyquifer" and "ci_95" in r.metadata:
                    if "accuracy" in r.metrics:
                        accs_c_list.append(r.metrics["accuracy"])

            if len(accs_b_list) >= 2 and len(accs_c_list) >= 2:
                d = cohens_d(accs_c_list, accs_b_list)
                lines.append(f"**Cohen's d (C vs B):** {d:.3f}\n")

    # Aggregate scoring
    if scenario_scores:
        lines.append("\n## Aggregate Scores\n")
        lines.append("| Scenario | Score | Accuracy (C) | Accuracy (B) |")
        lines.append("|---|---|---|---|")
        total_score = 0.0
        for s in scenario_scores:
            acc_b_str = f"{s['accuracy_b']:.4f}" if s["accuracy_b"] else "—"
            lines.append(
                f"| {s['scenario']} | {s['score']:.4f} "
                f"| {s['accuracy_c']:.4f} | {acc_b_str} |"
            )
            total_score += s["score"]

        mean_score = total_score / len(scenario_scores)
        lines.append(f"\n**Mean weighted score:** {mean_score:.4f}")
        lines.append(f"**Number of scored scenarios:** {len(scenario_scores)}")

        # Also compute geometric mean of C/B accuracy ratios
        ratios = [s["accuracy_c"] / s["accuracy_b"]
                  for s in scenario_scores
                  if s["accuracy_b"] and s["accuracy_b"] > 0]
        if ratios:
            geo_mean = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
            lines.append(f"**Geometric mean of C/B accuracy ratios:** {geo_mean:.4f}")

    lines.append("")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return str(output)


def main():
    results_dir = Path(__file__).parent / "results"
    output_path = results_dir / "BENCHMARK_REPORT.md"

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        print("Run benchmark scripts first to generate JSON results.")
        return

    json_files = sorted(results_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON result files found in {results_dir}")
        print("Run benchmark scripts first:")
        print("  python tests/benchmarks/bench_ep_training.py")
        print("  python tests/benchmarks/bench_local_rules.py")
        print("  python tests/benchmarks/bench_continual.py")
        print("  python tests/benchmarks/bench_modules.py")
        print("  python tests/benchmarks/bench_cycle.py")
        print("  python tests/benchmarks/bench_consciousness.py")
        print("  python tests/benchmarks/bench_chess.py")
        print("  python tests/benchmarks/bench_predictive_coding.py")
        print("  python tests/benchmarks/bench_efficiency.py")
        print("  python tests/benchmarks/bench_robustness.py")
        print("  python tests/benchmarks/bench_llm_ab.py")
        return

    print(f"Found {len(json_files)} result files:")
    for f in json_files:
        print(f"  - {f.name}")

    # Generate enhanced report
    report_path = enhanced_report(str(results_dir), str(output_path))
    print(f"\nReport generated: {report_path}")

    # Print summary
    with open(report_path, encoding="utf-8") as f:
        content = f.read()

    n_scenarios = content.count("###")
    print(f"Total scenarios reported: {n_scenarios}")

    for line in content.split("\n"):
        if "Mean weighted score" in line:
            print(f"\n{line.strip()}")
        if "Geometric mean" in line:
            print(line.strip())

    print(f"\nFull report: {report_path}")


if __name__ == "__main__":
    main()
