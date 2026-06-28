#!/usr/bin/env bash
# Read-only status check for the strict HYDRA SciBERT experiment.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif command -v uv >/dev/null 2>&1; then
    PYTHON="uv run python"
else
    PYTHON="python3"
fi

OUTPUT_DIR="results_strict"
LOG_DIR="logs_strict"

echo "== Strict HYDRA SciBERT Status =="
date
echo

echo "== Background Jobs =="
if command -v screen >/dev/null 2>&1; then
    screen -ls || true
else
    echo "screen command not found"
fi
echo

echo "== GPU =="
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
else
    echo "nvidia-smi command not found"
fi
echo

echo "== Python Training Processes =="
ps -ef | grep -E 'train_hydra.py|run_strict_hydra_scibert|ensure_strict_hydra_complete|finalize_strict_hydra_when_done' | grep -v grep || true
echo

echo "== Completed Strict Runs =="
$PYTHON - "$OUTPUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
archs = ["local", "local_global", "local_nested"]
seeds = [1, 2, 3, 4, 42]
complete = {}

for metrics_file in out.glob("hydra_*_seed*_*/test_metrics.json"):
    run_dir = metrics_file.parent
    cfg_file = run_dir / "config.json"
    if not cfg_file.exists():
        continue
    try:
        cfg = json.loads(cfg_file.read_text())
        metrics = json.loads(metrics_file.read_text())
    except Exception:
        continue
    arch = cfg.get("architecture")
    seed = cfg.get("seed")
    if arch not in archs or seed not in seeds:
        continue
    expected_selection = "training_mode_overall_macro_f1" if arch == "local" else "unified_macro_f1"
    strict = (
        cfg.get("model_name") == "pretrained_models/scibert"
        and cfg.get("max_length") == 512
        and cfg.get("padding_strategy") == "max_length"
        and cfg.get("num_epochs") == 50
        and cfg.get("selection_metric") == "official"
        and metrics.get("selection_metric") == expected_selection
        and metrics.get("protocol") == "strict_hydra_scibert_official_metrics"
    )
    if strict:
        complete[(arch, seed)] = run_dir

for arch in archs:
    done = sorted(seed for (a, seed) in complete if a == arch)
    missing = [seed for seed in seeds if seed not in done]
    print(f"{arch}: complete {len(done)}/5 seeds={done} missing={missing}")

print(f"total: {len(complete)}/15")
if complete:
    print("latest complete:")
    for key, run_dir in sorted(complete.items(), key=lambda item: item[1].stat().st_mtime)[-3:]:
        print(f"  {key[0]} seed={key[1]} -> {run_dir}")
PY
echo

echo "== Runtime Estimate =="
$PYTHON - "$OUTPUT_DIR" <<'PY'
import json
from pathlib import Path

out = Path(__import__("sys").argv[1])
durations = []
epochs = []

for metrics_file in out.glob("hydra_*_seed*_*/test_metrics.json"):
    run_dir = metrics_file.parent
    history_file = run_dir / "training_history.json"
    try:
        metrics = json.loads(metrics_file.read_text())
    except Exception:
        continue
    duration = metrics.get("training_time_seconds")
    if isinstance(duration, (int, float)) and duration > 0:
        durations.append(float(duration))
    if history_file.exists():
        try:
            epochs.append(len(json.loads(history_file.read_text())))
        except Exception:
            pass

if not durations:
    print("No completed strict run durations yet.")
else:
    complete = len(durations)
    remaining = max(15 - complete, 0)
    avg_seconds = sum(durations) / complete
    avg_epochs = (sum(epochs) / len(epochs)) if epochs else None
    print(f"completed durations: {complete}/15")
    print(f"avg training time/run: {avg_seconds / 3600:.2f}h")
    if avg_epochs is not None:
        print(f"avg ran epochs/run: {avg_epochs:.1f}")
    print(f"projected total training time: {avg_seconds * 15 / 3600:.1f}h")
    print(f"projected remaining training time: {avg_seconds * remaining / 3600:.1f}h")
    print("estimate excludes final summarization/audit overhead and may shift for global/nested variants")
PY
echo

echo "== Latest Logs =="
ls -1t "$LOG_DIR"/*.log 2>/dev/null | head -8 || true
echo

latest_run_log="$(ls -1t "$LOG_DIR"/*_seed*.log 2>/dev/null | head -1 || true)"
if [ -n "$latest_run_log" ]; then
    echo "== Latest Epoch From $latest_run_log =="
    latest_epoch="$(perl -0ne 'while(/([0-9-]+ [0-9:]+ \[INFO\] Epoch [^\r\n]*)/g){$last=$1} END{print "$last\n" if defined $last}' "$latest_run_log" || true)"
    if [ -n "$latest_epoch" ]; then
        echo "$latest_epoch"
    else
        grep -a "\\[INFO\\]" "$latest_run_log" | tail -n 12 || true
    fi
    echo
fi

echo "When total reaches 15/15, read docs/strict_hydra_scibert_results.md."
