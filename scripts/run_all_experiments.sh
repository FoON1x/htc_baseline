#!/bin/bash
# ============================================================
# HYDRA 复现 - 全量实验脚本
# 
# 实验设计：
#   3 种架构 x 5 个种子 = 15 次实验
#   - HYDRA Local Heads Only (seeds: 42,1,2,3,4)
#   - HYDRA Local + Global   (seeds: 42,1,2,3,4)
#   - HYDRA Local + Nested   (seeds: 42,1,2,3,4)
#
# 预估用时（4090 + fp16）：
#   - 每 epoch ~2-3 min，约 10-15 epoch 收敛
#   - 每次实验 ~25-45 min
#   - 15 次实验总计 ~6-11 小时
#
# 用法：bash scripts/run_all_experiments.sh
# ============================================================
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# 确保 uv 在 PATH 中
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# 确定 python 命令
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

# 检测 GPU 类型，自动启用 fp16
if $PYTHON -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    FP16_FLAG="--fp16"
    GPU_NAME=$($PYTHON -c "import torch; print(torch.cuda.get_device_name(0))")
    echo "检测到 CUDA GPU: $GPU_NAME，启用 fp16"
elif $PYTHON -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null; then
    echo "检测到 Apple MPS，不使用 fp16"
else
    echo "警告：未检测到 GPU，将使用 CPU（速度很慢）"
fi

# 检查数据
if [ ! -f "$DATA_DIR/X.txt" ]; then
    echo "错误：数据未找到 ($DATA_DIR/X.txt)"
    echo "请先运行: bash scripts/setup_server.sh"
    exit 1
fi

# 检查模型
if [ ! -f "$MODEL_NAME/config.json" ]; then
    echo "错误：模型未找到 ($MODEL_NAME/config.json)"
    echo "请先运行: bash scripts/setup_server.sh"
    exit 1
fi

# ---------- 日志 ----------
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="$LOG_DIR/experiment_${TIMESTAMP}.log"

log() {
    echo "$@" | tee -a "$MASTER_LOG"
}

log "============================================================"
log "HYDRA 复现 - 全量实验"
log "开始时间: $(date)"
log "Python: $PYTHON"
log "配置: 3 架构 x 5 种子 = 15 次实验"
log "日志文件: $MASTER_LOG"
log "============================================================"

# ---------- 运行实验 ----------
ARCHS=("local" "local_global" "local_nested")
TOTAL=15
COMPLETED=0
FAILED=0
START_ALL=$(date +%s)

for ARCH in "${ARCHS[@]}"; do
    log ""
    log "============================================================"
    log "架构: HYDRA $ARCH"
    log "============================================================"
    
    IFS=',' read -ra SEED_ARR <<< "$SEEDS"
    for SEED in "${SEED_ARR[@]}"; do
        COMPLETED=$((COMPLETED + 1))
        log ""
        log "[$COMPLETED/$TOTAL] HYDRA $ARCH seed=$SEED"
        log "  开始: $(date)"
        
        RUN_START=$(date +%s)
        
        RUN_LOG="$LOG_DIR/hydra_${ARCH}_seed${SEED}_${TIMESTAMP}.log"
        
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
            
            RUN_END=$(date +%s)
            RUN_TIME=$((RUN_END - RUN_START))
            log "  完成: $(date) (用时: $((RUN_TIME/60))min)"
            
            # 提取关键指标
            RESULT_DIR=$(ls -dt "$OUTPUT_DIR"/hydra_${ARCH}_seed${SEED}_* 2>/dev/null | head -1)
            if [ -n "$RESULT_DIR" ] && [ -f "$RESULT_DIR/test_metrics.json" ]; then
                log "  结果:"
                $PYTHON -c "
import json
m = json.load(open('$RESULT_DIR/test_metrics.json'))
print(f'    Child Micro-F1: {m.get(\"child_micro_f1_argmax\", 0)*100:.2f}')
print(f'    Child Macro-F1: {m.get(\"child_macro_f1_argmax\", 0)*100:.2f}')
print(f'    Parent Micro-F1: {m.get(\"parent_micro_f1_argmax\", 0)*100:.2f}')
print(f'    Hier Consistency: {m.get(\"hierarchical_consistency\", 0)*100:.2f}%')
print(f'    Best Epoch: {m.get(\"best_epoch\", \"?\")}')
" 2>/dev/null | tee -a "$MASTER_LOG"
            fi
        else
            FAILED=$((FAILED + 1))
            log "  失败! 详见 $RUN_LOG"
        fi
    done
done

# ---------- 汇总 ----------
END_ALL=$(date +%s)
TOTAL_TIME=$((END_ALL - START_ALL))

log ""
log "============================================================"
log "全部实验完成"
log "============================================================"
log "总用时: $((TOTAL_TIME/3600))h $((TOTAL_TIME%3600/60))min"
log "成功: $((TOTAL - FAILED)) / $TOTAL"
log "失败: $FAILED"
log ""

# 生成汇总报告
log "生成结果汇总..."
$PYTHON scripts/summarize_results.py "$OUTPUT_DIR" 2>&1 | tee -a "$MASTER_LOG"

log ""
log "汇总报告: docs/experiment_results_summary.md"
log "详细日志: $LOG_DIR/"
log "完成时间: $(date)"
