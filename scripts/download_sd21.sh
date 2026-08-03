#!/usr/bin/env bash
# 下载 Stable Diffusion 2.1 base 模型到本地缓存
# 用于 SERUM 训练（避免模型路径断链导致 from_pretrained 失败）
set -e

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_CACHE=/data2/fzx/.cache/huggingface/hub
export HF_TOKEN="${HF_TOKEN:-}"

# 删除断链符号链接
if [ -L "/data2/fzx/.cache/huggingface/hub/stable-diffusion-2-1-base" ]; then
    rm -f /data2/fzx/.cache/huggingface/hub/stable-diffusion-2-1-base
fi

mkdir -p "$HF_HUB_CACHE"

# 直接下载到 hub cache（HF 格式，diffusers 可直接 from_pretrained）
python - <<'PY'
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_CACHE", "/data2/fzx/.cache/huggingface/hub")

from huggingface_hub import snapshot_download

# 只下载必要组件：UNet + VAE + text_encoder + tokenizer + scheduler
# 这样下载体积约 5GB，节省时间
path = snapshot_download(
    repo_id="stabilityai/stable-diffusion-2-1-base",
    cache_dir="/data2/fzx/.cache/huggingface/hub",
    allow_patterns=[
        "model_index.json",
        "*.json",
        "tokenizer/*",
        "text_encoder/*.safetensors",
        "text_encoder/config.json",
        "vae/*.safetensors",
        "vae/config.json",
        "unet/*.safetensors",
        "unet/config.json",
        "scheduler/*",
        "feature_extractor/*",
    ],
    ignore_patterns=[
        "*.ckpt",     # 跳过老格式 ckpt，只下 safetensors
        "*.fp16.*",   # 跳过 fp16 版本（我们在内存里再转换）
        "*.nonema.*",
        "v1-*",
        "*.msgpack",
    ],
)
print(f"\n[OK] SD 2.1 base downloaded to: {path}")
print(f"路径添加到 config: {path}")
PY

echo ""
echo "完成后可以跑："
echo "  PYTHONPATH=/data2/fzx/SERUM python -m src.training.train --config configs/config_sd21_gpu2_innov.yaml"
