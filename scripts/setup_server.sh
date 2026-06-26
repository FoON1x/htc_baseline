#!/usr/bin/env bash
# ============================================================
# HYDRA 复现 - 服务器初始化脚本
# 用法: bash scripts/setup_server.sh
# ============================================================
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "============================================================"
echo "HYDRA 复现 - 服务器初始化"
echo "项目目录: $PROJECT_DIR"
echo "============================================================"

# -------------------- 1. 安装依赖 --------------------
echo ""
echo "[1/4] 安装 Python 依赖..."

# 检测 uv，没有就装
if ! command -v uv &> /dev/null; then
    echo "  安装 uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh 2>/dev/null || true
fi

# 把 uv 加入 PATH（安装后当前 shell 可能找不到）
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if command -v uv &> /dev/null; then
    echo "  uv $(uv --version 2>/dev/null || echo 'installed')"
    echo "  uv sync..."
    uv sync
else
    echo "  uv 不可用，使用 pip 安装依赖..."
    pip install torch transformers scikit-learn numpy tqdm pandas
fi

# 确定 venv python
if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="python3"
fi

echo "  Python: $($PYTHON --version 2>&1)"
echo "  路径: $(which $PYTHON 2>/dev/null || echo 'unknown')"

# 验证核心依赖
$PYTHON -c "import torch, transformers, sklearn, numpy, tqdm; print('  ✓ 依赖安装成功')" || {
    echo "  ✗ 依赖缺失，尝试 pip 补装..."
    pip install torch transformers scikit-learn numpy tqdm pandas
    $PYTHON -c "import torch, transformers, sklearn, numpy, tqdm; print('  ✓ 依赖安装成功')" || {
        echo "  ✗ 依赖安装失败，请手动检查"
        exit 1
    }
}

# -------------------- 2. 下载 WOS46985 --------------------
echo ""
echo "[2/4] 下载 WOS46985 数据集..."

DATA_DIR="$PROJECT_DIR/data/wos_raw/WOS46985"

if [ -f "$DATA_DIR/X.txt" ] && [ -f "$DATA_DIR/YL1.txt" ] && [ -f "$DATA_DIR/YL2.txt" ] && [ -f "$DATA_DIR/Y.txt" ]; then
    echo "  数据已存在，跳过下载"
else
    ZIP_DIR="$PROJECT_DIR/data/wos_raw"
    mkdir -p "$ZIP_DIR"
    ZIP_PATH="$ZIP_DIR/WebOfScience.zip"
    ZIP_URL="https://data.mendeley.com/public-files/datasets/9rw3vkcfy4/files/1cb41d1e-4f7f-4d5e-a4db-65e1e50f0e5a/file_downloaded"

    echo "  下载 WebOfScience.zip (~60MB)..."
    OK=false
    if command -v wget &> /dev/null; then
        wget -q --show-progress -O "$ZIP_PATH" "$ZIP_URL" && OK=true || rm -f "$ZIP_PATH"
    fi
    if [ "$OK" = false ] && command -v curl &> /dev/null; then
        curl -L --progress-bar -o "$ZIP_PATH" "$ZIP_URL" && OK=true || rm -f "$ZIP_PATH"
    fi

    if [ "$OK" = true ] && [ -s "$ZIP_PATH" ]; then
        echo "  解压..."
        unzip -q -o "$ZIP_PATH" -d "$ZIP_DIR/"
        rm -f "$ZIP_PATH"
        echo "  ✓ 数据下载完成"
    else
        rm -f "$ZIP_PATH"
        echo ""
        echo "  ============================================================"
        echo "  自动下载失败！手动操作："
        echo "  1. 访问 https://data.mendeley.com/datasets/9rw3vkcfy4/6"
        echo "  2. 下载 WebOfScience.zip 放到 $ZIP_DIR/"
        echo "  3. cd $ZIP_DIR && unzip -o WebOfScience.zip"
        echo "  或从 Mac 传输: scp -r data/wos_raw/ user@server:$PROJECT_DIR/data/"
        echo "  ============================================================"
        exit 1
    fi
fi

echo "  文件检查:"
for f in X.txt YL1.txt YL2.txt Y.txt; do
    if [ -f "$DATA_DIR/$f" ]; then echo "    ✓ $f"
    else echo "    ✗ $f 缺失!"; exit 1; fi
done

# -------------------- 3. 下载 SciBERT --------------------
echo ""
echo "[3/4] 下载 SciBERT 模型..."

MODEL_DIR="$PROJECT_DIR/pretrained_models/scibert"

if [ -f "$MODEL_DIR/config.json" ] && [ -f "$MODEL_DIR/pytorch_model.bin" ]; then
    echo "  模型已存在，跳过下载"
else
    mkdir -p "$MODEL_DIR"
    echo "  下载 allenai/scibert_scivocab_uncased..."

    OK=false
    # 直接下载
    $PYTHON -c "
from transformers import AutoModel, AutoTokenizer
AutoModel.from_pretrained('allenai/scibert_scivocab_uncased').save_pretrained('$MODEL_DIR')
AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased').save_pretrained('$MODEL_DIR')
print('  ✓ 模型下载完成')
" && OK=true

    # 镜像下载
    if [ "$OK" = false ]; then
        echo "  直接下载失败，尝试 hf-mirror.com..."
        HF_ENDPOINT="https://hf-mirror.com" $PYTHON -c "
import os; os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from transformers import AutoModel, AutoTokenizer
AutoModel.from_pretrained('allenai/scibert_scivocab_uncased').save_pretrained('$MODEL_DIR')
AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased').save_pretrained('$MODEL_DIR')
print('  ✓ 模型下载完成 (镜像)')
" && OK=true
    fi

    if [ "$OK" = false ]; then
        echo ""
        echo "  ============================================================"
        echo "  模型下载失败！手动操作："
        echo "  方式1: HF_ENDPOINT=https://hf-mirror.com bash scripts/setup_server.sh"
        echo "  方式2: 从 Mac 传输: scp -r pretrained_models/scibert/ user@server:$PROJECT_DIR/pretrained_models/"
        echo "  方式3: huggingface-cli download allenai/scibert_scivocab_uncased --local-dir $MODEL_DIR"
        echo "  ============================================================"
        exit 1
    fi
fi

# -------------------- 4. 验证 --------------------
echo ""
echo "[4/4] 环境验证..."

$PYTHON << 'PYCHECK'
import torch, sys
sys.path.insert(0, '.')
print(f"  PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}", end="")
if torch.cuda.is_available():
    print(f" ({torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_mem/1024**3:.0f}GB)")
else:
    print()
from src.data.wos_dataset import get_wos_datasets
from src.models.hydra import HYDRA
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('pretrained_models/scibert')
tr,va,te,hi = get_wos_datasets('data/wos_raw/WOS46985', tok, max_length=64)
print(f"  数据: {len(tr)}/{len(va)}/{len(te)} | 标签: {hi.label_dims[0]}+{hi.label_dims[1]}")
ld = [hi.label_dims[i] for i in range(len(hi.label_dims))]
m = HYDRA(ld, hi, 'pretrained_models/scibert')
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
m.to(dev)
x = torch.randint(0,30000,(2,32),device=dev)
o,_ = m(x, torch.ones(2,32,device=dev))
print(f"  前向传播: parent {o[0].shape}, child {o[1].shape} [{dev}]")
print("  ✓ 全部验证通过")
PYCHECK

echo ""
echo "============================================================"
echo "初始化完成！"
echo "============================================================"
echo ""
echo "  bash scripts/run_all_experiments.sh    # 全部实验"
echo "  bash scripts/run_hydra_wos.sh local 42 # 单次实验"
