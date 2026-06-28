"""
Summarize strict HYDRA SciBERT reproduction results.

The primary metric follows the official HYDRA implementation:
- local: training_mode_overall_{micro,macro}_f1
- local_global/local_nested: unified_{micro,macro}_f1
"""

import argparse
import json
import sys
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_FILE = PROJECT_DIR / "docs" / "strict_hydra_scibert_results.md"
SUMMARY_JSON = PROJECT_DIR / "docs" / "strict_hydra_scibert_summary.json"
RUNS_CSV = PROJECT_DIR / "docs" / "strict_hydra_scibert_runs.csv"

HYDRA_PAPER_WOS = {
    "local": {"micro_f1": 86.90, "macro_f1": 81.18},
    "local_global": {"micro_f1": 86.91, "macro_f1": 81.22},
    "local_nested": {"micro_f1": 86.83, "macro_f1": 81.08},
}

ARCH_DISPLAY = {
    "local": "HYDRA Local",
    "local_global": "HYDRA Local+Global",
    "local_nested": "HYDRA Local+Nested",
}


def primary_keys(architecture):
    if architecture in ("local_global", "local_nested"):
        return "unified_micro_f1", "unified_macro_f1"
    return "training_mode_overall_micro_f1", "training_mode_overall_macro_f1"


def expected_selection_key(architecture):
    return (
        "unified_macro_f1"
        if architecture in ("local_global", "local_nested")
        else "training_mode_overall_macro_f1"
    )


def is_strict_run(config, metrics):
    arch = config.get("architecture")
    return (
        arch in {"local", "local_global", "local_nested"}
        and config.get("model_name") == "pretrained_models/scibert"
        and config.get("max_length") == 512
        and config.get("padding_strategy") == "max_length"
        and config.get("num_epochs") == 50
        and config.get("selection_metric") == "official"
        and metrics.get("selection_metric") == expected_selection_key(arch)
        and metrics.get("protocol") == "strict_hydra_scibert_official_metrics"
    )


def load_runs(results_dir):
    candidates = defaultdict(list)
    ignored = []
    for run_dir in sorted(results_dir.glob("hydra_*")):
        if not run_dir.is_dir():
            continue
        config_file = run_dir / "config.json"
        metrics_file = run_dir / "test_metrics.json"
        history_file = run_dir / "training_history.json"
        if not config_file.exists():
            ignored.append({
                "run_dir": run_dir.name,
                "reason": "missing config.json",
            })
            continue
        if not metrics_file.exists():
            ignored.append({
                "run_dir": run_dir.name,
                "reason": "missing test_metrics.json; incomplete or interrupted run",
            })
            continue
        with open(config_file) as f:
            config = json.load(f)
        with open(metrics_file) as f:
            metrics = json.load(f)
        history_len = None
        if history_file.exists():
            with open(history_file) as f:
                history_len = len(json.load(f))
        record = {
            "run_dir": run_dir.name,
            "seed": config["seed"],
            "config": config,
            "metrics": metrics,
            "history_len": history_len,
            "mtime": metrics_file.stat().st_mtime,
        }
        if is_strict_run(config, metrics):
            candidates[(config["architecture"], config["seed"])].append(record)
        else:
            ignored.append({
                "run_dir": run_dir.name,
                "reason": "non-strict config or metrics",
            })

    grouped = defaultdict(list)
    duplicates = []
    for (arch, seed), records in sorted(candidates.items()):
        records = sorted(records, key=lambda record: record["mtime"], reverse=True)
        grouped[arch].append(records[0])
        for duplicate in records[1:]:
            duplicates.append({
                "architecture": arch,
                "seed": seed,
                "kept": records[0]["run_dir"],
                "ignored": duplicate["run_dir"],
            })
    return grouped, duplicates, ignored


def values(runs, key):
    vals = [run["metrics"].get(key) for run in runs]
    return [v for v in vals if isinstance(v, (float, int))]


def mean_std(runs, key):
    vals = values(runs, key)
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


def fmt(v):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v * 100:.2f}"
    return str(v)


def fmt_mean_std(mean, std):
    if mean is None:
        return "N/A"
    return f"{mean * 100:.2f} +/- {std * 100:.2f}"


def validate_protocol(grouped):
    issues = []
    expected_arches = ["local", "local_global", "local_nested"]
    expected_seeds = [1, 2, 3, 4, 42]

    for arch in expected_arches:
        runs = grouped.get(arch, [])
        seeds = sorted(run["seed"] for run in runs)
        if seeds != expected_seeds:
            issues.append(f"{arch}: expected seeds {expected_seeds}, found {seeds}")
        for run in runs:
            cfg = run["config"]
            metrics = run["metrics"]
            if cfg.get("model_name") != "pretrained_models/scibert":
                issues.append(f"{run['run_dir']}: model_name={cfg.get('model_name')}")
            if cfg.get("max_length") != 512:
                issues.append(f"{run['run_dir']}: max_length={cfg.get('max_length')} (expected 512)")
            if cfg.get("padding_strategy") != "max_length":
                issues.append(f"{run['run_dir']}: padding_strategy={cfg.get('padding_strategy')} (expected max_length)")
            if cfg.get("num_epochs") != 50:
                issues.append(f"{run['run_dir']}: num_epochs={cfg.get('num_epochs')} (expected 50)")
            if cfg.get("selection_metric") != "official":
                issues.append(f"{run['run_dir']}: selection_metric={cfg.get('selection_metric')}")
            expected_selection = expected_selection_key(arch)
            if metrics.get("selection_metric") != expected_selection:
                issues.append(
                    f"{run['run_dir']}: recorded selection_metric={metrics.get('selection_metric')} "
                    f"(expected {expected_selection})")
            micro_key, macro_key = primary_keys(arch)
            if micro_key not in metrics or macro_key not in metrics:
                issues.append(f"{run['run_dir']}: missing primary metrics {micro_key}/{macro_key}")
            if metrics.get("protocol") != "strict_hydra_scibert_official_metrics":
                issues.append(f"{run['run_dir']}: protocol={metrics.get('protocol')}")
    return issues


def generate_report(grouped, results_dir, duplicates=None, ignored=None):
    duplicates = duplicates or []
    ignored = ignored or []
    lines = []
    lines.append("# Strict HYDRA SciBERT Reproduction Results\n")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"Results directory: `{results_dir}`\n")

    issues = validate_protocol(grouped)
    lines.append("## Protocol Check\n")
    if issues:
        lines.append("Status: **INCOMPLETE / CHECK REQUIRED**\n")
        for issue in issues:
            lines.append(f"- {issue}")
    else:
        lines.append("Status: **complete**. Found all 3 architectures x 5 seeds with strict protocol settings.\n")
    if duplicates:
        lines.append("\nDuplicate complete runs were detected. The newest strict run per architecture/seed was used:\n")
        for item in duplicates:
            lines.append(
                f"- {item['architecture']} seed {item['seed']}: kept `{item['kept']}`, ignored `{item['ignored']}`")
    if ignored:
        lines.append("\nNon-strict or incomplete result directories ignored by the summary:\n")
        for item in ignored[:20]:
            lines.append(f"- `{item['run_dir']}`: {item['reason']}")
        if len(ignored) > 20:
            lines.append(f"- ... {len(ignored) - 20} more")

    lines.append("\n## Primary Official-Style Metrics\n")
    lines.append(
        "Primary metrics follow the official HYDRA evaluation mode for each architecture. "
        "`local` uses local-head training-mode overall Micro/Macro-F1. "
        "`local_global` and `local_nested` use the unified head Micro/Macro-F1, "
        "matching the validation metric used for early stopping.\n"
    )
    lines.append("| Architecture | Runs | Primary Micro-F1 | Primary Macro-F1 | Paper WOS Micro-F1 | Paper WOS Macro-F1 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for arch in ["local", "local_global", "local_nested"]:
        runs = grouped.get(arch, [])
        micro_key, macro_key = primary_keys(arch)
        micro_mean, micro_std = mean_std(runs, micro_key)
        macro_mean, macro_std = mean_std(runs, macro_key)
        paper = HYDRA_PAPER_WOS[arch]
        lines.append(
            f"| {ARCH_DISPLAY[arch]} | {len(runs)} | {fmt_mean_std(micro_mean, micro_std)} "
            f"| {fmt_mean_std(macro_mean, macro_std)} | {paper['micro_f1']:.2f} | {paper['macro_f1']:.2f} |"
        )

    detail_keys = [
        ("training_mode_overall_micro_f1", "TrainMode Micro"),
        ("training_mode_overall_macro_f1", "TrainMode Macro"),
        ("inference_mode_overall_micro_f1", "InferMode Micro"),
        ("inference_mode_overall_macro_f1", "InferMode Macro"),
        ("unified_micro_f1", "Unified Micro"),
        ("unified_macro_f1", "Unified Macro"),
        ("overall_accuracy", "Overall Acc"),
        ("child_micro_f1_argmax", "Child Argmax Micro"),
        ("child_macro_f1_argmax", "Child Argmax Macro"),
        ("child_acc_argmax", "Child Acc"),
        ("parent_acc_argmax", "Parent Acc"),
        ("hierarchical_consistency", "Hierarchy Consistency"),
    ]

    lines.append("\n## Per-Seed Details\n")
    lines.append(
        "Per-seed tables include the official HYDRA metric families plus project comparison "
        "metrics. `Argmax` metrics are single-label parent/child diagnostics for comparison "
        "with the current method; they are not replacements for the official threshold metrics.\n"
    )
    for arch in ["local", "local_global", "local_nested"]:
        runs = sorted(grouped.get(arch, []), key=lambda r: r["seed"])
        lines.append(f"### {ARCH_DISPLAY[arch]}\n")
        header = "| Seed | Best Epoch | Ran Epochs | Best Val Metric | " + " | ".join(label for _, label in detail_keys) + " |"
        lines.append(header)
        lines.append("|---:|---:|---:|---:|" + "|".join(["---:" for _ in detail_keys]) + "|")
        for run in runs:
            metrics = run["metrics"]
            vals = [fmt(metrics.get(key)) for key, _ in detail_keys]
            lines.append(
                f"| {run['seed']} | {metrics.get('best_epoch', 'N/A')} | {run['history_len'] or 'N/A'} "
                f"| {fmt(metrics.get('best_val_metric'))} | " + " | ".join(vals) + " |"
            )

        summary = []
        for key, _ in detail_keys:
            mean, std = mean_std(runs, key)
            summary.append(fmt_mean_std(mean, std))
        lines.append("| **Mean+/-Std** | - | - | - | " + " | ".join(summary) + " |")
        lines.append("")

    lines.append("## Experimental Settings\n")
    lines.append("- Encoder: SciBERT from `pretrained_models/scibert`")
    lines.append("- Data: WOS46985 raw txt files, 7 parent + 134 child labels")
    lines.append("- Split: official HYDRA WOS split procedure (`np.random.seed(7)` shuffle, two `train_test_split(..., random_state=0)` calls)")
    lines.append("- Hyperparameters: batch size 32, max length 512, max_length padding, learning rate 3.5e-5, warmup 500, threshold 0.5, loss alpha 1.0")
    lines.append("- Early stopping: official HYDRA code rule, patience 5; local uses training-mode overall macro-F1, global/nested use unified macro-F1")
    lines.append("- Paper comparison values: WOS RoBERTa-base rows from HYDRA Table 3; `local_global` uses Global Head and `local_nested` uses Nested Head because those match the unified-head primary metrics")

    return "\n".join(lines)


def write_machine_readable(grouped):
    all_metric_keys = sorted({
        key
        for runs in grouped.values()
        for run in runs
        for key, value in run["metrics"].items()
        if isinstance(value, (float, int))
    })

    summary = {}
    for arch, runs in grouped.items():
        summary[arch] = {
            "runs": len(runs),
            "seeds": sorted(run["seed"] for run in runs),
            "metrics": {},
        }
        for key in all_metric_keys:
            vals = values(runs, key)
            if vals:
                summary[arch]["metrics"][key] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    RUNS_CSV.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "architecture", "seed", "run_dir", "selection_metric",
        "best_epoch", "best_val_metric", "history_len",
    ] + all_metric_keys
    with open(RUNS_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for arch in sorted(grouped):
            for run in sorted(grouped[arch], key=lambda r: r["seed"]):
                metrics = run["metrics"]
                row = {
                    "architecture": arch,
                    "seed": run["seed"],
                    "run_dir": run["run_dir"],
                    "selection_metric": metrics.get("selection_metric"),
                    "best_epoch": metrics.get("best_epoch"),
                    "best_val_metric": metrics.get("best_val_metric"),
                    "history_len": run["history_len"],
                }
                for key in all_metric_keys:
                    row[key] = metrics.get(key)
                writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Summarize strict HYDRA SciBERT results")
    parser.add_argument("results_dir", nargs="?", default=str(PROJECT_DIR / "results_strict"))
    parser.add_argument("--output-prefix", default=str(PROJECT_DIR / "docs" / "strict_hydra_scibert"),
                        help="Output prefix for .md, _summary.json, and _runs.csv files.")
    args = parser.parse_args()

    global OUTPUT_FILE, SUMMARY_JSON, RUNS_CSV
    output_prefix = Path(args.output_prefix)
    OUTPUT_FILE = output_prefix.with_name(output_prefix.name + "_results.md")
    SUMMARY_JSON = output_prefix.with_name(output_prefix.name + "_summary.json")
    RUNS_CSV = output_prefix.with_name(output_prefix.name + "_runs.csv")

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory does not exist: {results_dir}")
        sys.exit(1)

    grouped, duplicates, ignored = load_runs(results_dir)
    total = sum(len(runs) for runs in grouped.values())
    if total == 0:
        print(f"No strict HYDRA results found in {results_dir}")
        sys.exit(1)

    print(f"Found {total} runs")
    for arch in sorted(grouped):
        print(f"  {arch}: {len(grouped[arch])}")

    report = generate_report(grouped, results_dir, duplicates, ignored)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(report)
    write_machine_readable(grouped)
    print(f"Wrote {OUTPUT_FILE}")
    print(f"Wrote {SUMMARY_JSON}")
    print(f"Wrote {RUNS_CSV}")


if __name__ == "__main__":
    main()
