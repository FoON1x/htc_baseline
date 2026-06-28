#!/usr/bin/env bash
# Wait for the primary strict HYDRA run to finish, then fill any missing
# strict SciBERT runs and regenerate summaries.

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

DATA_DIR="data/wos_raw/WOS46985"
MODEL_NAME="pretrained_models/scibert"
OUTPUT_DIR="results_strict"
LOG_DIR="$PROJECT_DIR/logs_strict"
ARCHS=("local" "local_global" "local_nested")
SEEDS=(42 1 2 3 4)

BATCH_SIZE=32
MAX_LENGTH=512
NUM_EPOCHS=50
LR=3.5e-5
WARMUP=500
PATIENCE=5
THRESHOLD=0.5
LOSS_ALPHA=1.0

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
TS=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="$LOG_DIR/ensure_strict_complete_${TS}.log"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$MASTER_LOG"; }

has_complete_run() {
    local arch="$1"
    local seed="$2"
    $PYTHON - "$OUTPUT_DIR" "$arch" "$seed" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
arch = sys.argv[2]
seed = int(sys.argv[3])
expected_selection = "unified_macro_f1" if arch in {"local_global", "local_nested"} else "training_mode_overall_macro_f1"

for run_dir in sorted(out.glob(f"hydra_{arch}_seed{seed}_*"), reverse=True):
    cfg_file = run_dir / "config.json"
    metrics_file = run_dir / "test_metrics.json"
    if not cfg_file.exists() or not metrics_file.exists():
        continue
    try:
        cfg = json.loads(cfg_file.read_text())
        metrics = json.loads(metrics_file.read_text())
    except Exception:
        continue
    checks = [
        cfg.get("architecture") == arch,
        cfg.get("seed") == seed,
        cfg.get("model_name") == "pretrained_models/scibert",
        cfg.get("max_length") == 512,
        cfg.get("padding_strategy") == "max_length",
        cfg.get("num_epochs") == 50,
        cfg.get("selection_metric") == "official",
        metrics.get("selection_metric") == expected_selection,
        metrics.get("protocol") == "strict_hydra_scibert_official_metrics",
    ]
    if all(checks):
        print(run_dir)
        sys.exit(0)
sys.exit(1)
PY
}

log "Strict HYDRA completion guard started"
log "This guard waits for hydra_strict, fills missing strict runs, then summarizes."

while screen -ls | grep -Eq '[.]hydra_strict[[:space:]]'; do
    log "hydra_strict still running; waiting 10 minutes"
    sleep 600
done

log "hydra_strict is no longer running; checking strict run completeness"

[ -f "$DATA_DIR/X.txt" ] || { log "Missing data: $DATA_DIR/X.txt"; exit 1; }
[ -f "$MODEL_NAME/config.json" ] || { log "Missing model: $MODEL_NAME/config.json"; exit 1; }

FP16_FLAG=""
if $PYTHON -c "import torch; assert torch.cuda.is_available()" >/dev/null 2>&1; then
    FP16_FLAG="--fp16"
    log "CUDA available; fp16 enabled for any rerun jobs"
else
    log "CUDA unavailable; fp16 disabled for any rerun jobs"
fi

missing=0
for arch in "${ARCHS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        if complete_dir=$(has_complete_run "$arch" "$seed"); then
            log "complete: $arch seed=$seed -> $complete_dir"
            continue
        fi

        missing=$((missing + 1))
        run_log="$LOG_DIR/ensure_${arch}_seed${seed}_${TS}.log"
        log "missing/incomplete: $arch seed=$seed; starting strict rerun"

        $PYTHON scripts/train_hydra.py \
            --data_dir "$DATA_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --model_name "$MODEL_NAME" \
            --architecture "$arch" \
            --pooling cls \
            --project_embedding \
            --batch_size "$BATCH_SIZE" \
            --max_length "$MAX_LENGTH" \
            --padding_strategy max_length \
            --num_epochs "$NUM_EPOCHS" \
            --learning_rate "$LR" \
            --warmup_steps "$WARMUP" \
            --early_stopping_patience "$PATIENCE" \
            --threshold "$THRESHOLD" \
            --loss_alpha "$LOSS_ALPHA" \
            --selection_metric official \
            --seed "$seed" \
            $FP16_FLAG \
            2>&1 | tee "$run_log"
    done
done

log "Completeness check done; filled $missing missing/incomplete runs"
log "Generating strict summary"
$PYTHON scripts/summarize_strict_results.py "$OUTPUT_DIR" 2>&1 | tee -a "$MASTER_LOG"
log "Auditing strict artifacts"
$PYTHON scripts/audit_strict_hydra_results.py "$OUTPUT_DIR" 2>&1 | tee -a "$MASTER_LOG"
log "Done. Summary files are under docs/strict_hydra_scibert_*"
