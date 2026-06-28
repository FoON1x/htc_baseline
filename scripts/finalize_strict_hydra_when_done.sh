#!/usr/bin/env bash
# Final summarization/audit guard for the strict HYDRA SciBERT experiment.
# It never trains. It waits for training guards to finish, then generates the
# official summary artifacts only after all 15 strict runs are present.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif command -v uv >/dev/null 2>&1; then
    PYTHON="uv run python"
else
    PYTHON="python3"
fi

OUTPUT_DIR="results_strict"
LOG_DIR="$PROJECT_DIR/logs_strict"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="$LOG_DIR/finalize_strict_hydra_${TS}.log"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$MASTER_LOG"; }

count_complete() {
    $PYTHON - "$OUTPUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
expected_arches = {"local", "local_global", "local_nested"}
expected_seeds = {1, 2, 3, 4, 42}
complete = set()

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
    expected_selection = (
        "unified_macro_f1"
        if arch in {"local_global", "local_nested"}
        else "training_mode_overall_macro_f1"
    )
    strict = (
        arch in expected_arches
        and seed in expected_seeds
        and cfg.get("model_name") == "pretrained_models/scibert"
        and cfg.get("max_length") == 512
        and cfg.get("padding_strategy") == "max_length"
        and cfg.get("num_epochs") == 50
        and cfg.get("selection_metric") == "official"
        and metrics.get("selection_metric") == expected_selection
        and metrics.get("protocol") == "strict_hydra_scibert_official_metrics"
    )
    if strict:
        complete.add((arch, seed))

print(len(complete))
PY
}

log "Final strict HYDRA summarization/audit guard started"

while screen -ls | grep -Eq '[.]hydra_strict[[:space:]]|[.]hydra_strict_guard[[:space:]]'; do
    log "Training or completion guard still running; waiting 10 minutes"
    sleep 600
done

while true; do
    complete="$(count_complete)"
    log "Strict complete runs: ${complete}/15"
    if [ "$complete" = "15" ]; then
        break
    fi
    log "Not complete yet; waiting 30 minutes before rechecking"
    sleep 1800
done

log "Generating final strict summary"
$PYTHON scripts/summarize_strict_results.py "$OUTPUT_DIR" 2>&1 | tee -a "$MASTER_LOG"

log "Running final strict audit"
$PYTHON scripts/audit_strict_hydra_results.py "$OUTPUT_DIR" 2>&1 | tee -a "$MASTER_LOG"

log "Final strict HYDRA summary and audit complete"
