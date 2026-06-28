#!/usr/bin/env bash
# Strict HYDRA reproduction on WOS46985 with SciBERT.
# Matches the official HYDRA training protocol except for the encoder swap.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif command -v uv &> /dev/null; then
    PYTHON="uv run python"
else
    PYTHON="python3"
fi

DATA_DIR="data/wos_raw/WOS46985"
MODEL_NAME="pretrained_models/scibert"
OUTPUT_DIR="results_strict"
SEEDS="42,1,2,3,4"
ARCHS=("local" "local_global" "local_nested")

BATCH_SIZE=32
MAX_LENGTH=512
NUM_EPOCHS=50
LR=3.5e-5
WARMUP=500
PATIENCE=5
THRESHOLD=0.5
LOSS_ALPHA=1.0

[ -f "$DATA_DIR/X.txt" ] || { echo "Missing data: $DATA_DIR/X.txt"; exit 1; }
[ -f "$MODEL_NAME/config.json" ] || { echo "Missing model: $MODEL_NAME/config.json"; exit 1; }

FP16_FLAG=""
if $PYTHON -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    FP16_FLAG="--fp16"
    GPU_NAME=$($PYTHON -c "import torch; print(torch.cuda.get_device_name(0))")
    echo "CUDA: $GPU_NAME, fp16 enabled"
else
    echo "CUDA unavailable, fp16 disabled"
fi

LOG_DIR="$PROJECT_DIR/logs_strict"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
TS=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="$LOG_DIR/strict_hydra_scibert_${TS}.log"

log() { echo "$@" | tee -a "$MASTER_LOG"; }

log "============================================================"
log "Strict HYDRA SciBERT reproduction | $(date)"
log "Protocol: official HYDRA metrics/early stopping, SciBERT encoder"
log "Runs: 3 architectures x 5 seeds = 15"
log "Output: $OUTPUT_DIR"
log "============================================================"

TOTAL=15
DONE=0
FAIL=0
T0=$(date +%s)

for ARCH in "${ARCHS[@]}"; do
    log ""
    log "====== $ARCH ======"
    IFS=',' read -ra SEED_ARR <<< "$SEEDS"
    for SEED in "${SEED_ARR[@]}"; do
        DONE=$((DONE + 1))
        RUN_LOG="$LOG_DIR/${ARCH}_seed${SEED}_${TS}.log"
        log ""
        log "[$DONE/$TOTAL] architecture=$ARCH seed=$SEED @ $(date)"

        if $PYTHON scripts/train_hydra.py \
            --data_dir "$DATA_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --model_name "$MODEL_NAME" \
            --architecture "$ARCH" \
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
            --seed "$SEED" \
            $FP16_FLAG \
            2>&1 | tee "$RUN_LOG"; then

            RESULT_DIR=$(ls -dt "$OUTPUT_DIR"/hydra_${ARCH}_seed${SEED}_* 2>/dev/null | head -1)
            log "  completed: $RESULT_DIR"
            if [ -n "$RESULT_DIR" ] && [ -f "$RESULT_DIR/test_metrics.json" ]; then
                $PYTHON -c "
import json
m = json.load(open('$RESULT_DIR/test_metrics.json'))
print('  selection:', m.get('selection_metric'), 'best_val=', round(m.get('best_val_metric', 0), 4), 'best_epoch=', m.get('best_epoch'))
print('  training overall:', round(m.get('training_mode_overall_micro_f1', 0)*100, 2), round(m.get('training_mode_overall_macro_f1', 0)*100, 2))
if 'unified_micro_f1' in m:
    print('  unified:', round(m.get('unified_micro_f1', 0)*100, 2), round(m.get('unified_macro_f1', 0)*100, 2))
" | tee -a "$MASTER_LOG"
            fi
        else
            FAIL=$((FAIL + 1))
            log "  failed; see $RUN_LOG"
        fi
    done
done

TOTAL_T=$(( $(date +%s) - T0 ))
log ""
log "============================================================"
log "Done | elapsed $((TOTAL_T/3600))h$((TOTAL_T%3600/60))min | success $((TOTAL-FAIL))/$TOTAL"
log "============================================================"

log "Generating strict summary..."
$PYTHON scripts/summarize_strict_results.py "$OUTPUT_DIR" 2>&1 | tee -a "$MASTER_LOG"
log "Summary: docs/strict_hydra_scibert_results.md"
