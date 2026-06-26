#!/bin/bash
# ============================================================
# HYDRA 复现 - 服务器初始化脚本
# 功能：安装 uv → 装 Python 依赖 → 下载数据集 → 下载模型
# 用法：bash scripts/setup_server.sh
# ============================================================
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "============================================================"
echo "HYDRA 复现 - 服务器初始化"
echo "项目目录: $PROJECT_DIR"
echo "============================================================"

# ---------- 0. 安装 uv ----------
echo ""
echo "[0/4] 检查 uv..."

if ! command -v uv &> /dev/null; then
    echo "  未检测到 uv，正在安装..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv &> /dev/null; then
        echo "  uv 安装失败，尝试 pip 方式..."
    fi
fi

if command -v uv &> /dev/null; then
    echo "  uv 版本: $(uv --version)"
fi

# ---------- 1. 安装 Python 依赖 ----------
echo ""
echo "[1/4] 安装 Python 依赖..."

if command -v uv &> /dev/null; then
    echo "  使用 uv sync..."
    uv sync --dev
else
    echo "  使用 pip..."
    pip install -e ".[dev]" 2>/dev/null || pip install -e .
fi

# 验证关键包
python3 -c "import torch; import transformers; import sklearn; print('  依赖安装成功')" || {
    echo "  依赖安装失败，请手动检查"
    exit 1
}

# ---------- 2. 下载 WOS46985 数据集 ----------
echo ""
echo "[2/4] 下载 WOS46985 数据集..."

DATA_DIR="$PROJECT_DIR/data/wos_raw/WOS46985"
if [ -f "$DATA_DIR/X.txt" ] && [ -f "$DATA_DIR/YL1.txt" ] && [ -f "$DATA_DIR/YL2.txt" ] && [ -f "$DATA_DIR/Y.txt" ]; then
    LINES=$(wc -l < "$DATA_DIR/X.txt")
    echo "  数据已存在 ($LINES 样本)，跳过下载"
else
    ZIP_DIR="$PROJECT_DIR/data/wos_raw"
    mkdir -p "$ZIP_DIR"

    ZIP_URL="https://data.mendeley.com/public-files/datasets/9rw3vkcfy4/files/1cb41d1e-4f7f-4d5e-a4db-65e1e50f0e5a/file_downloaded"
    ZIP_PATH="$ZIP_DIR/WebOfScience.zip"

    echo "  从 Mendeley Data 下载 WebOfScience.zip (~60MB)..."
    DOWNLOAD_OK=false

    if command -v wget &> /dev/null; then
        wget -q --show-progress -O "$ZIP_PATH" "$ZIP_URL" && DOWNLOAD_OK=true
    elif command -v curl &> /dev/null; then
        curl -L --progress-bar -o "$ZIP_PATH" "$ZIP_URL" && DOWNLOAD_OK=true
    fi

    if [ "$DOWNLOAD_OK" = true ] && [ -f "$ZIP_PATH" ] && [ -s "$ZIP_PATH" ]; then
        echo "  解压..."
        unzip -q -o "$ZIP_PATH" -d "$ZIP_DIR/"
        echo "  数据下载完成"
        rm -f "$ZIP_PATH"
    else
        rm -f "$ZIP_PATH"
        echo ""
        echo "  ============================================================"
        echo "  自动下载失败！请手动操作："
        echo "  方式1: 浏览器访问 https://data.mendeley.com/datasets/9rw3vkcfy4/6"
        echo "         下载 WebOfScience.zip 到 $ZIP_DIR/"
        echo "         然后执行: cd $ZIP_DIR && unzip -o WebOfScience.zip"
        echo "  方式2: 从本地 Mac 传输:"
        echo "         scp -r data/wos_raw/ user@server:$PROJECT_DIR/data/"
        echo "  完成后重新运行此脚本"
        echo "  ============================================================"
        exit 1
    fi
fi

# 验证
echo "  数据文件检查:"
for f in X.txt YL1.txt YL2.txt Y.txt; do
    if [ -f "$DATA_DIR/$f" ]; then
        echo "    ✓ $f"
    else
        echo "    ✗ $f 缺失!"
        exit 1
    fi
done

# ---------- 3. 下载 SciBERT 模型 ----------
echo ""
echo "[3/4] 下载 SciBERT 模型..."

MODEL_DIR="$PROJECT_DIR/pretrained_models/scibert"
if [ -f "$MODEL_DIR/config.json" ] && [ -f "$MODEL_DIR/pytorch_model.bin" ]; then
    echo "  模型已存在，跳过下载"
else
    mkdir -p "$MODEL_DIR"
    echo "  从 HuggingFace 下载 allenai/scibert_scivocab_uncased..."

    # 尝试直接下载，如果失败则用镜像
    HF_ENDPOINT="${HF_ENDPOINT:-}"
    
    python3 -c "
import os
os.environ.setdefault('HF_ENDPOINT', '$HF_ENDPOINT')
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model.save_pretrained('$MODEL_DIR')
tokenizer.save_pretrained('$MODEL_DIR')
print('  模型下载完成')
" || {
        echo "  直接下载失败，尝试 HuggingFace 镜像..."
        HF_ENDPOINT="https://hf-mirror.com" python3 -c "
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model.save_pretrained('$MODEL_DIR')
tokenizer.save_pretrained('$MODEL_DIR')
print('  模型下载完成 (通过镜像)')
" || {
            echo ""
            echo "  ============================================================"
            echo "  模型下载失败！请手动操作："
            echo "  方式1: 设置 HF_ENDPOINT 环境变量后重试:"
            echo "         HF_ENDPOINT=https://hf-mirror.com bash scripts/setup_server.sh"
            echo "  方式2: 从本地 Mac 传输:"
            echo "         scp -r pretrained_models/scibert/ user@server:$PROJECT_DIR/pretrained_models/"
            echo "  方式3: 用 huggingface-cli 下载:"
            echo "         huggingface-cli download allenai/scibert_scivocab_uncased --local-dir $MODEL_DIR"
            echo "  ============================================================"
            exit 1
        }
    }
fi

# ---------- 4. 环境验证 ----------
echo ""
echo "[4/4] 环境验证..."

python3 << 'PYCHECK'
import torch
import sys
sys.path.insert(0, '.')

print(f"  Python: {sys.version.split()[0]}")
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
    print(f"  GPU Memory: {mem:.1f} GB")
print(f"  MPS: {torch.backends.mps.is_available()}")

from src.data.wos_dataset import get_wos_datasets
from src.models.hydra import HYDRA
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Subset

tokenizer = AutoTokenizer.from_pretrained('pretrained_models/scibert')
train_ds, val_ds, test_ds, hier_info = get_wos_datasets('data/wos_raw/WOS46985', tokenizer, max_length=64)
print(f"  数据集: {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test")
print(f"  标签: {hier_info.label_dims[0]} parent / {hier_info.label_dims[1]} child")

label_dims = [hier_info.label_dims[i] for i in range(len(hier_info.label_dims))]
model = HYDRA(label_dims, hier_info, 'pretrained_models/scibert')
params = sum(p.numel() for p in model.parameters())
print(f"  模型参数: {params:,}")

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
x = torch.randint(0, 30000, (2, 32), device=device)
mask = torch.ones(2, 32, device=device)
out, _ = model(x, mask)
print(f"  前向传播: parent {out[0].shape}, child {out[1].shape}")
print("  ✓ 全部验证通过")
PYCHECK

echo ""
echo "============================================================"
echo "初始化完成！"
echo "============================================================"
echo ""
echo "运行全部实验:"
echo "  bash scripts/run_all_experiments.sh"
echo ""
echo "运行单个实验:"
echo "  bash scripts/run_hydra_wos.sh local 42"
