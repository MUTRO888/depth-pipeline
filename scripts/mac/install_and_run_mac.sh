#!/usr/bin/env bash
# Mac 一键安装并启动浮雕深度估计（全效果版）
# 用法:  ./install_and_run_mac.sh
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

VENV_DIR="$PROJECT_DIR/.venv"
CONFIG_FILE="$PROJECT_DIR/config.full.yaml"
LORA_FILE="$PROJECT_DIR/models/sd-relief/lora/relief_lora_438287.safetensors"

# ── 0. 检查并自动安装系统依赖 (Mac Homebrew) ───────────────────
if ! command -v brew &>/dev/null; then
  echo "[ERROR] 未检测到 Homebrew。请先在终端运行以下命令安装 Mac 万能包管理器："
  echo '        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
  exit 1
fi

echo "[SETUP] 正在检查基础系统组件..."
BREW_DEPS=("python@3.12" "python-tk@3.12" "git" "git-lfs")
for dep in "${BREW_DEPS[@]}"; do
  if ! brew list "$dep" &>/dev/null; then
    echo "        > 未检测到或者版本过旧: $dep，正在通过 Homebrew 自动下载安装..."
    brew install "$dep"
  else
    echo "        > [OK] $dep 已安装"
  fi
done
echo ""

# ── 1. 检查 Python ──────────────────────────────────────────────
PYTHON_EXE=""
for cmd in python3.12 python3.13 python3; do
  if command -v "$cmd" &>/dev/null; then
    PYTHON_EXE="$cmd"
    break
  fi
done

if [ -z "$PYTHON_EXE" ]; then
  echo "[ERROR] 未找到 Python 3。请先安装 Python 3.12+："
  echo "        brew install python@3.12"
  exit 1
fi

PY_VER="$($PYTHON_EXE -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PY_MAJOR="${PY_VER%%.*}"
PY_MINOR="${PY_VER##*.}"
if [ "$PY_MAJOR" -lt 3 ] || [ "$PY_MINOR" -lt 12 ]; then
  echo "[ERROR] 需要 Python 3.12+，当前为 $PY_VER"
  exit 1
fi
echo "[OK] Python $PY_VER ($PYTHON_EXE)"

# ── 2. 检查 config.full.yaml ────────────────────────────────────
if [ ! -f "$CONFIG_FILE" ]; then
  echo "[ERROR] 缺少全效果配置文件: $CONFIG_FILE"
  exit 1
fi

# ── 3. 创建/激活虚拟环境 ────────────────────────────────────────
if [ ! -f "$VENV_DIR/bin/python3" ]; then
  echo "[SETUP] 检测到虚拟环境不完整，正在重新创建..."
  rm -rf "$VENV_DIR"
  "$PYTHON_EXE" -m venv "$VENV_DIR" --copies
  if [ ! -f "$VENV_DIR/bin/python3" ]; then
      echo "[ERROR] 无法正常生成 Python3 虚拟环境执行文件，请检查您的 Python3.12 安装是否完整。"
      ls -la "$VENV_DIR/bin" 2>/dev/null
      exit 1
  fi
fi
source "$VENV_DIR/bin/activate"
echo "[OK] 虚拟环境已激活"

# ── 4. 安装依赖 ─────────────────────────────────────────────────
echo "[SETUP] 安装/更新依赖 (调用清华镜像加速下载，可能需要几分钟，请耐心等待)..."
PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"

"$VENV_DIR/bin/python3" -m pip install -U pip -i "$PIP_MIRROR"

# Mac Apple Silicon: 安装 PyTorch（带 MPS 支持）
"$VENV_DIR/bin/python3" -m pip install torch torchvision -i "$PIP_MIRROR"
"$VENV_DIR/bin/python3" -m pip install -r "$PROJECT_DIR/requirements.txt" -i "$PIP_MIRROR"

echo "[OK] 依赖安装完成"

# ── 5. Git LFS 拉取 LoRA 文件 ───────────────────────────────────
if [ -d "$PROJECT_DIR/.git" ]; then
  if command -v git &>/dev/null && git lfs version &>/dev/null; then
    echo "[SETUP] 拉取 Git LFS 文件..."
    git lfs pull
  fi
fi

if [ ! -f "$LORA_FILE" ]; then
  echo "[ERROR] 缺少 LoRA 文件: $LORA_FILE"
  echo "        请安装 Git LFS 后运行: git lfs pull"
  exit 1
fi

# 检查是否为 LFS pointer 而非真实文件
if [ "$(stat -f%z "$LORA_FILE" 2>/dev/null || stat -c%s "$LORA_FILE" 2>/dev/null)" -lt 4096 ]; then
  if grep -q "git-lfs.github.com/spec/v1" "$LORA_FILE" 2>/dev/null; then
    echo "[ERROR] LoRA 文件还是 Git LFS 指针，请运行: git lfs pull"
    exit 1
  fi
fi

# ── 6. 下载 Hugging Face 模型 ───────────────────────────────────
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export DEPTH_PIPELINE_CONFIG="$CONFIG_FILE"

echo "[SETUP] 下载/检查 Hugging Face 模型（镜像: $HF_ENDPOINT）..."
"$VENV_DIR/bin/python3" scripts/download_full_models.py --config "$CONFIG_FILE"

# ── 7. 启动 ─────────────────────────────────────────────────────
echo ""
echo "========================================="
echo "  启动浮雕深度估计（Mac Apple Silicon）"
echo "========================================="
echo ""
"$VENV_DIR/bin/python3" app.py
