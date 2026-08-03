#!/usr/bin/env bash
# 通过 ModelScope 下载 SD 2.1 base 模型（避免 HF mirror gated 限制）
set -e

export MODELSCOPE_CACHE=/data2/fzx/.cache/modelscope
mkdir -p "$MODELSCOPE_CACHE"

# 确保 modelscope 装了
if ! python -c "import modelscope" 2>/dev/null; then
    echo "安装 modelscope..."
    pip install modelscope
fi

python - <<'PY'
from modelscope import snapshot_download
import os
print("开始下载 SD 2.1 base（ModelScope），约 5GB...")

path = snapshot_download(
    'AI-ModelScope/stable-diffusion-2-1-base',
    cache_dir='/data2/fzx/.cache/modelscope',
    allow_patterns=[
        '*.json',
        'tokenizer/*',
        'text_encoder/*',
        'vae/*',
        'unet/*',
        'scheduler/*',
    ],
    ignore_patterns=['*.ckpt', '*.msgpack', '*.fp16.*', '*.nonema.*'],
)
print(f"\n[OK] 模型下载到: {path}")
print(f"  symlink 路径: /data2/fzx/.cache/modelscope/AI-ModelScope/stable-diffusion-2-1-base")
PY

# 重建 HF cache 的符号链接，让 from_pretrained 找得到
HF_LINK=/data2/fzx/.cache/huggingface/hub/stable-diffusion-2-1-base
TARGET=/data2/fzx/.cache/modelscope/AI-ModelScope/stable-diffusion-2-1-base

if [ -L "$HF_LINK" ] || [ -d "$HF_LINK" ]; then
    rm -f "$HF_LINK"
fi
ln -sf "$TARGET" "$HF_LINK"

echo ""
echo "[OK] 链接已建好："
echo "  $HF_LINK -> $TARGET"
echo ""
echo "下一步：直接跑训练（不要再用 src.training.train 的相对导入路径）"
echo "  PYTHONPATH=/data2/fzx/SERUM python -m src.training.train --config configs/config_sd21_gpu2_innov.yaml"