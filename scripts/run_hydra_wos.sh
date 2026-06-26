#!/bin/bash
# HYDRA 复现实验脚本
# 使用方法: bash scripts/run_hydra_wos.sh [architecture] [seed]
# 示例: bash scripts/run_hydra_wos.sh local 42

set -e

ARCH=${1:-local}
SEED=${2:-42}
DATA_DIR="data/wos_raw/WOS46985"
MODEL="pretrained_models/scibert"
OUTPUT="results"

echo "============================================"
echo "HYDRA Training on WOS46985"
echo "Architecture: $ARCH"
echo "Seed: $SEED"
echo "Data: $DATA_DIR"
echo "Model: $MODEL"
echo "============================================"

uv run python scripts/train_hydra.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT" \
    --model_name "$MODEL" \
    --architecture "$ARCH" \
    --pooling cls \
    --project_embedding \
    --batch_size 32 \
    --max_length 256 \
    --num_epochs 30 \
    --learning_rate 3.5e-5 \
    --warmup_steps 500 \
    --early_stopping_patience 5 \
    --threshold 0.5 \
    --seed "$SEED"

echo "Done! Results saved to $OUTPUT"
