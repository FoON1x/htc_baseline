#!/bin/bash
# HYDRA 单次实验脚本
# 用法: bash scripts/run_hydra_wos.sh [architecture] [seed]
# 示例: bash scripts/run_hydra_wos.sh local 42

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

ARCH=${1:-local}
SEED=${2:-42}
DATA_DIR="data/wos_raw/WOS46985"
MODEL="pretrained_models/scibert"
OUTPUT="results"

# 确定 python
if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif command -v uv &> /dev/null; then
    PYTHON="uv run python"
else
    PYTHON="python3"
fi

# 检测 fp16
FP16_FLAG=""
if $PYTHON -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    FP16_FLAG="--fp16"
fi

echo "============================================"
echo "HYDRA Training on WOS46985"
echo "Architecture: $ARCH"
echo "Seed: $SEED"
echo "Python: $PYTHON"
echo "============================================"

$PYTHON scripts/train_hydra.py \
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
    --seed "$SEED" \
    $FP16_FLAG

echo "Done! Results saved to $OUTPUT"
