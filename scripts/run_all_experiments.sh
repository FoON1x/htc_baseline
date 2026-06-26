#!/usr/bin/env bash
# ============================================================
# HYDRA 复现 - 全量实验脚本
#
# 3 架构 x 5 种子 = 15 次实验
# 预估用时 (4090+fp16): 6-11 小时
#
# 用法: bash scripts/run_all_experiments.sh
# ============================================================
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# 确定 python
if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif command -v uv &> /dev/null; then
    PYTHON="uv run python"
else
    PYTHON="python3"
fi

# ---------- 配置 ----------
DATA_DIR="data/wos_raw/WOS46985"
MODEL_NAME="pretrained_models/scibert"
OUTPUT_DIR="results"
SEEDS="42,1,2,3,4"
MAX_LENGTH=256
BATCH_SIZE=32
NUM_EPOCHS=30
LR=3.5e-5
WARMUP=500
PATIENCE=5
FP16_FLAG=""

# 检测 CUDA → 自动开 fp16
if $PYTHON -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    FP16_FLAG="--fp16"
    GPU_NAME=$($PYTHON -c "import torch; print(torch.cuda.get_device_name(0))")
    echo "CUDA: $GPU_NAME, fp16 已启用"
else
    echo "CUDA 不可用，fp16 未启用"
fi

# 前置检查
[ -f "$DATA_DIR/X.txt" ] || { echo "错误: 数据缺失，先运行 bash scripts/setup_server.sh"; exit 1; }
[ -f "$MODEL_NAME/config.json" ] || { echo "错误: 模型缺失，先运行 bash scripts/setup_server.sh"; exit 1; }

# ---------- 日志 ----------
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="$LOG_DIR/experiment_${TS}.log"

log() { echo "$@" | tee -a "$MASTER_LOG"; }

log "============================================================"
log "HYDRA 全量实验 | $(date)"
log "Python: $PYTHON | 3 架构 x 5 种子 = 15 次"
log "============================================================"

# ---------- 运行 ----------
ARCHS=("local" "local_global" "local_nested")
TOTAL=15
DONE=0
FAIL=0
T0=$(date +%s)

for ARCH in "${ARCHS[@]}"; do
    log ""
    log "====== HYDRA $ARCH ======"

    IFS=',' read -ra SEED_ARR <<< "$SEEDS"
    for SEED in "${SEED_ARR[@]}"; do
        DONE=$((DONE + 1))
        log ""
        log "[$DONE/$TOTAL] $ARCH seed=$SEED @ $(date)"

        T1=$(date +%s)
        RUN_LOG="$LOG_DIR/hydra_${ARCH}_seed${SEED}_${TS}.log"

        if $PYTHON scripts/train_hydra.py \
            --data_dir "$DATA_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --model_name "$MODEL_NAME" \
            --architecture "$ARCH" \
            --pooling cls \
            --project_embedding \
            --batch_size "$BATCH_SIZE" \
            --max_length "$MAX_LENGTH" \
            --num_epochs "$NUM_EPOCHS" \
            --learning_rate "$LR" \
            --warmup_steps "$WARMUP" \
            --early_stopping_patience "$PATIENCE" \
            --threshold 0.5 \
            --loss_alpha 1.0 \
            --seed "$SEED" \
            $FP16_FLAG \
            2>&1 | tee "$RUN_LOG"; then

            DT=$(( $(date +%s) - T1 ))
            log "  完成 (用时: $((DT/60))min)"

            # 提取结果
            RESULT_DIR=$(ls -dt "$OUTPUT_DIR"/hydra_${ARCH}_seed${SEED}_* 2>/dev/null | head -1)
            if [ -n "$RESULT_DIR" ] && [ -f "$RESULT_DIR/test_metrics.json" ]; then
                log "  结果:"
                $PYTHON -c "
import json
m = json.load(open('$RESULT_DIR/test_metrics.json'))
print(f'    Child Micro-F1: {m.get(\"child_micro_f1_argmax\",0)*100:.2f}')
print(f'    Child Macro-F1: {m.get(\"child_macro_f1_argmax\",0)*100:.2f}')
print(f'    Parent Micro-F1: {m.get(\"parent_micro_f1_argmax\",0)*100:.2f}')
print(f'    Hier Cons: {m.get(\"hierarchical_consistency\",0)*100:.2f}%')
print(f'    Best Epoch: {m.get(\"best_epoch\",\"?\")}')
" 2>/dev/null | tee -a "$MASTER_LOG"
            fi
        else
            FAIL=$((FAIL + 1))
            log "  ✗ 失败! 详见 $RUN_LOG"
        fi
    done
done

# ---------- 汇总 ----------
TOTAL_T=$(( $(date +%s) - T0 ))
log ""
log "============================================================"
log "全部完成 | 总用时 $((TOTAL_T/3600))h$((TOTAL_T%3600/60))min | 成功 $((TOTAL-FAIL))/$TOTAL"
log "============================================================"

log ""
log "生成结果汇总..."
$PYTHON scripts/summarize_results.py "$OUTPUT_DIR" 2>&1 | tee -a "$MASTER_LOG"

log ""
log "汇总报告: docs/experiment_results_summary.md"
log "详细日志: $LOG_DIR/"
