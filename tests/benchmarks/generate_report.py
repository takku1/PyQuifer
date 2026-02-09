"""Unified Benchmark Report Generator.

Collects all JSON results from tests/benchmarks/results/ and produces
a unified markdown report with three-column comparison tables,
weighted scoring, environment info, and Cohen's d effect sizes.

Categories (from COMPREHENSIVE_BENCHMARK_PLAN):
  Cat 1: LLM A/B Testing
  Cat 2: Microbenchmarks — Modules, Cycle, Efficiency
  Cat 3: Training — EP, Predictive Coding, Local Rules, Continual
  Cat 4: Chess — Chess, Penrose Chess, ChessQA, Searchless Chess
  Cat 5: Consciousness — PCI, metastability, coherence-complexity
  Cat 6: Efficiency — dtype, compile, scaling (overlap with Cat 2)
  Cat 7: Robustness — Noise tolerance, perturbation recovery
  Cat 8: AKOrN — (future, placeholder)

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

# ═══════════════════════════════════════════════════════════════════════════════
# Category Mappings
# ═══════════════════════════════════════════════════════════════════════════════

# Map JSON result file stems to benchmark plan categories
FILE_TO_CATEGORY = {
    # Cat 1: LLM A/B
    "llm_ab": "Cat 1: LLM A/B Testing",
    # Cat 2: Microbenchmarks
    "modules": "Cat 2: Microbenchmarks",
    "cycle": "Cat 2: Microbenchmarks",
    # Cat 3: Training
    "ep_training": "Cat 3: Training",
    "ep_training_quick": "Cat 3: Training",
    "predictive_coding": "Cat 3: Training",
    "local_rules": "Cat 3: Training",
    "local_rules_quick": "Cat 3: Training",
    "continual": "Cat 3: Training",
    "continual_quick": "Cat 3: Training",
    # Cat 4: Chess
    "chess": "Cat 4: Chess & Strategy",
    "penrose_chess": "Cat 4: Chess & Strategy",
    "chessqa": "Cat 4: Chess & Strategy",
    "searchless_chess": "Cat 4: Chess & Strategy",
    # Cat 5: Consciousness
    "consciousness": "Cat 5: Consciousness",
    # Cat 6: Efficiency
    "efficiency": "Cat 6: Efficiency",
    # Cat 7: Robustness
    "robustness": "Cat 7: Robustness",
}

# Ordered list for report sections
CATEGORY_ORDER = [
    "Cat 1: LLM A/B Testing",
    "Cat 2: Microbenchmarks",
    "Cat 3: Training",
    "Cat 4: Chess & Strategy",
    "Cat 5: Consciousness",
    "Cat 6: Efficiency",
    "Cat 7: Robustness",
]


def _classify_file(stem: str) -> str:
    """Classify a JSON result file into a category."""
    return FILE_TO_CATEGORY.get(stem, "Uncategorized")


# ═══════════════════════════════════════════════════════════════════════════════
# Report Generation
# ═══════════════════════════════════════════════════════════════════════════════

def enhanced_report(results_dir: str, output_path: str) -> str:
    """Generate enhanced report with environment block, weighted scoring,
    category grouping, and Cohen's d where multi-seed data is available."""
    results_path = Path(results_dir)
    output = Path(output_path)
    json_files = sorted(results_path.glob("*.json"))

    lines = [
        "# PyQuifer Comprehensive Benchmark Report\n",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        get_environment_block(),
    ]

    # Group files by category
    categorized: dict[str, list[Path]] = {cat: [] for cat in CATEGORY_ORDER}
    categorized["Uncategorized"] = []
    for jf in json_files:
        cat = _classify_file(jf.stem)
        if cat not in categorized:
            categorized[cat] = []
        categorized[cat].append(jf)

    scenario_scores = []

    # Generate report sections per category
    for cat in CATEGORY_ORDER + ["Uncategorized"]:
        files = categorized.get(cat, [])
        if not files:
            continue

        lines.append(f"\n---\n\n## {cat}\n")

        for jf in files:
            with open(jf) as f:
                data = json.load(f)
            suite_name = data.get("suite", jf.stem)
            lines.append(f"\n### {suite_name}\n")

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
                        "category": cat,
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
        lines.append("\n---\n\n## Aggregate Scores\n")

        # Per-category subtotals
        cats_seen = []
        for cat in CATEGORY_ORDER:
            cat_scores = [s for s in scenario_scores if s["category"] == cat]
            if not cat_scores:
                continue
            cats_seen.append(cat)
            cat_mean = sum(s["score"] for s in cat_scores) / len(cat_scores)
            lines.append(f"### {cat}")
            lines.append(f"**Mean score:** {cat_mean:.4f} ({len(cat_scores)} scenarios)\n")

        # Overall table
        lines.append("\n### All Scenarios\n")
        lines.append("| Category | Scenario | Score | Accuracy (C) | Accuracy (B) |")
        lines.append("|---|---|---|---|---|")
        total_score = 0.0
        for s in scenario_scores:
            acc_b_str = f"{s['accuracy_b']:.4f}" if s["accuracy_b"] else "--"
            lines.append(
                f"| {s['category']} | {s['scenario']} | {s['score']:.4f} "
                f"| {s['accuracy_c']:.4f} | {acc_b_str} |"
            )
            total_score += s["score"]

        mean_score = total_score / len(scenario_scores)
        lines.append(f"\n**Mean weighted score:** {mean_score:.4f}")
        lines.append(f"**Number of scored scenarios:** {len(scenario_scores)}")

        # Geometric mean of C/B accuracy ratios
        ratios = [s["accuracy_c"] / s["accuracy_b"]
                  for s in scenario_scores
                  if s["accuracy_b"] and s["accuracy_b"] > 0]
        if ratios:
            geo_mean = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
            lines.append(f"**Geometric mean of C/B accuracy ratios:** {geo_mean:.4f}")

    # Coverage summary
    lines.append("\n---\n\n## Coverage Summary\n")
    for cat in CATEGORY_ORDER:
        files = categorized.get(cat, [])
        n_files = len(files)
        n_scored = len([s for s in scenario_scores if s["category"] == cat])
        mark = "done" if n_files > 0 else "missing"
        lines.append(f"- **{cat}**: {n_files} result files, {n_scored} scored scenarios ({mark})")

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
        print("  python tests/benchmarks/run_quick.py")
        print()
        print("Or run individual benchmarks:")
        print("  python tests/benchmarks/bench_ep_training.py")
        print("  python tests/benchmarks/bench_local_rules.py")
        print("  python tests/benchmarks/bench_continual.py")
        print("  python tests/benchmarks/bench_modules.py")
        print("  python tests/benchmarks/bench_cycle.py")
        print("  python tests/benchmarks/bench_consciousness.py")
        print("  python tests/benchmarks/bench_chess.py")
        print("  python tests/benchmarks/bench_penrose_chess.py")
        print("  python tests/benchmarks/bench_chessqa.py")
        print("  python tests/benchmarks/bench_searchless_chess.py")
        print("  python tests/benchmarks/bench_predictive_coding.py")
        print("  python tests/benchmarks/bench_efficiency.py")
        print("  python tests/benchmarks/bench_robustness.py")
        print("  python tests/benchmarks/bench_llm_ab.py")
        return

    print(f"Found {len(json_files)} result files:")
    for f in json_files:
        cat = _classify_file(f.stem)
        print(f"  - {f.name}  [{cat}]")

    # Generate enhanced report
    report_path = enhanced_report(str(results_dir), str(output_path))
    print(f"\nReport generated: {report_path}")

    # Print summary
    with open(report_path, encoding="utf-8") as f:
        content = f.read()

    n_scenarios = content.count("###")
    print(f"Total sections reported: {n_scenarios}")

    for line in content.split("\n"):
        if "Mean weighted score" in line:
            print(f"\n{line.strip()}")
        if "Geometric mean" in line:
            print(line.strip())

    print(f"\nFull report: {report_path}")


if __name__ == "__main__":
    main()
