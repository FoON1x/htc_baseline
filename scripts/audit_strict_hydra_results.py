"""
Audit strict HYDRA SciBERT reproduction artifacts.

This script is intentionally strict: it succeeds only when the full
3-architecture x 5-seed experiment is complete and all required metrics,
configuration fields, and summary files are present.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


EXPECTED_ARCHES = ["local", "local_global", "local_nested"]
EXPECTED_SEEDS = [1, 2, 3, 4, 42]

EXPECTED_CONFIG = {
    "model_name": "pretrained_models/scibert",
    "batch_size": 32,
    "max_length": 512,
    "padding_strategy": "max_length",
    "num_epochs": 50,
    "learning_rate": 3.5e-5,
    "warmup_steps": 500,
    "early_stopping_patience": 5,
    "threshold": 0.5,
    "loss_alpha": 1.0,
    "selection_metric": "official",
}

COMMON_REQUIRED_METRICS = [
    "training_mode_level_0_accuracy",
    "training_mode_level_0_micro_f1",
    "training_mode_level_0_macro_f1",
    "training_mode_level_1_accuracy",
    "training_mode_level_1_micro_f1",
    "training_mode_level_1_macro_f1",
    "training_mode_overall_accuracy",
    "training_mode_overall_precision",
    "training_mode_overall_recall",
    "training_mode_overall_micro_f1",
    "training_mode_overall_macro_f1",
    "training_mode_overall_macro_precision",
    "training_mode_overall_macro_recall",
    "inference_mode_overall_accuracy",
    "inference_mode_overall_micro_f1",
    "inference_mode_overall_macro_f1",
    "parent_acc_argmax",
    "parent_micro_f1_argmax",
    "parent_macro_f1_argmax",
    "child_acc_argmax",
    "child_micro_f1_argmax",
    "child_macro_f1_argmax",
    "overall_accuracy",
    "overall_micro_f1",
    "overall_macro_f1",
    "hierarchical_consistency",
    "best_epoch",
    "best_val_metric",
    "selection_metric",
    "protocol",
    "training_time_seconds",
    "total_params",
]

UNIFIED_REQUIRED_METRICS = [
    "unified_accuracy",
    "unified_precision",
    "unified_recall",
    "unified_micro_f1",
    "unified_macro_f1",
    "unified_macro_precision",
    "unified_macro_recall",
]


def expected_selection(arch):
    if arch in {"local_global", "local_nested"}:
        return "unified_macro_f1"
    return "training_mode_overall_macro_f1"


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def is_close(a, b, tolerance=1e-12):
    return abs(float(a) - float(b)) <= tolerance


def main():
    parser = argparse.ArgumentParser(description="Audit strict HYDRA SciBERT results")
    parser.add_argument("results_dir", nargs="?", default="results_strict")
    parser.add_argument("--docs-dir", default="docs")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Report issues without failing when 15/15 runs are not complete.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    docs_dir = Path(args.docs_dir)
    issues = []
    grouped = defaultdict(list)

    for metrics_file in sorted(results_dir.glob("hydra_*_seed*_*/test_metrics.json")):
        run_dir = metrics_file.parent
        cfg_file = run_dir / "config.json"
        history_file = run_dir / "training_history.json"
        best_val_file = run_dir / "best_val_metrics.json"
        model_file = run_dir / "best_model.pt"

        if not cfg_file.exists():
            issues.append(f"{run_dir}: missing config.json")
            continue
        cfg = load_json(cfg_file)
        metrics = load_json(metrics_file)
        arch = cfg.get("architecture")
        seed = cfg.get("seed")
        grouped[(arch, seed)].append(run_dir)

        if arch not in EXPECTED_ARCHES:
            issues.append(f"{run_dir}: unexpected architecture {arch}")
        if seed not in EXPECTED_SEEDS:
            issues.append(f"{run_dir}: unexpected seed {seed}")
        for key, expected in EXPECTED_CONFIG.items():
            actual = cfg.get(key)
            if isinstance(expected, float):
                ok = actual is not None and is_close(actual, expected)
            else:
                ok = actual == expected
            if not ok:
                issues.append(f"{run_dir}: config {key}={actual!r}, expected {expected!r}")

        if metrics.get("protocol") != "strict_hydra_scibert_official_metrics":
            issues.append(f"{run_dir}: invalid protocol marker {metrics.get('protocol')!r}")
        if metrics.get("selection_metric") != expected_selection(arch):
            issues.append(
                f"{run_dir}: selection_metric={metrics.get('selection_metric')!r}, "
                f"expected {expected_selection(arch)!r}")

        required = list(COMMON_REQUIRED_METRICS)
        if arch in {"local_global", "local_nested"}:
            required.extend(UNIFIED_REQUIRED_METRICS)
        for key in required:
            if key not in metrics:
                issues.append(f"{run_dir}: missing metric {key}")

        for artifact in [history_file, best_val_file, model_file]:
            if not artifact.exists():
                issues.append(f"{run_dir}: missing {artifact.name}")

    for arch in EXPECTED_ARCHES:
        complete_seeds = sorted(
            seed for (seen_arch, seed), run_dirs in grouped.items()
            if seen_arch == arch and seed in EXPECTED_SEEDS and run_dirs
        )
        if complete_seeds != EXPECTED_SEEDS:
            issues.append(f"{arch}: complete seeds {complete_seeds}, expected {EXPECTED_SEEDS}")
        for seed in EXPECTED_SEEDS:
            run_dirs = grouped.get((arch, seed), [])
            if len(run_dirs) > 1:
                newest = max(run_dirs, key=lambda path: path.stat().st_mtime)
                issues.append(
                    f"{arch} seed {seed}: duplicate complete runs "
                    f"{[str(path) for path in run_dirs]}, newest={newest}")

    summary_files = [
        docs_dir / "strict_hydra_scibert_results.md",
        docs_dir / "strict_hydra_scibert_summary.json",
        docs_dir / "strict_hydra_scibert_runs.csv",
    ]
    if sum(len(run_dirs) for run_dirs in grouped.values()) == 15:
        for path in summary_files:
            if not path.exists():
                issues.append(f"missing summary artifact {path}")
        report_path = docs_dir / "strict_hydra_scibert_results.md"
        if report_path.exists():
            report = report_path.read_text(encoding="utf-8")
            if "Status: **complete**" not in report:
                issues.append(f"{report_path}: protocol check is not complete")

        summary_json = docs_dir / "strict_hydra_scibert_summary.json"
        if summary_json.exists():
            summary = load_json(summary_json)
            for arch in EXPECTED_ARCHES:
                arch_summary = summary.get(arch)
                if not arch_summary:
                    issues.append(f"{summary_json}: missing architecture {arch}")
                    continue
                if arch_summary.get("runs") != 5:
                    issues.append(
                        f"{summary_json}: {arch} runs={arch_summary.get('runs')}, expected 5")
                if arch_summary.get("seeds") != EXPECTED_SEEDS:
                    issues.append(
                        f"{summary_json}: {arch} seeds={arch_summary.get('seeds')}, "
                        f"expected {EXPECTED_SEEDS}")
                metric_keys = arch_summary.get("metrics", {})
                primary_keys = (
                    ["unified_micro_f1", "unified_macro_f1"]
                    if arch in {"local_global", "local_nested"}
                    else ["training_mode_overall_micro_f1", "training_mode_overall_macro_f1"]
                )
                for key in primary_keys:
                    if key not in metric_keys:
                        issues.append(f"{summary_json}: {arch} missing summary metric {key}")

        runs_csv = docs_dir / "strict_hydra_scibert_runs.csv"
        if runs_csv.exists():
            with open(runs_csv, encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            if len(rows) != 15:
                issues.append(f"{runs_csv}: row count {len(rows)}, expected 15")
            seen = defaultdict(list)
            for row in rows:
                try:
                    seed = int(row.get("seed", ""))
                except ValueError:
                    seed = row.get("seed")
                seen[(row.get("architecture"), seed)].append(row.get("run_dir"))
            for arch in EXPECTED_ARCHES:
                seeds = sorted(seed for (seen_arch, seed), rows_for_key in seen.items()
                               if seen_arch == arch and rows_for_key)
                if seeds != EXPECTED_SEEDS:
                    issues.append(f"{runs_csv}: {arch} seeds {seeds}, expected {EXPECTED_SEEDS}")
            for key, run_dirs in seen.items():
                if len(run_dirs) > 1:
                    issues.append(f"{runs_csv}: duplicate rows for {key}: {run_dirs}")

    total = sum(len(run_dirs) for run_dirs in grouped.values())
    print(f"strict result files: {total}/15")
    for arch in EXPECTED_ARCHES:
        seeds = sorted(
            seed for (seen_arch, seed), run_dirs in grouped.items()
            if seen_arch == arch and run_dirs
        )
        print(f"{arch}: {seeds}")

    if issues:
        print("\nIssues:")
        for issue in issues:
            print(f"- {issue}")
        if args.allow_incomplete and total < 15:
            return 0
        return 1

    print("Audit passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
