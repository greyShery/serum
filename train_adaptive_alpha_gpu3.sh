#!/bin/bash
# ============================================================================
# SERUM 创新点 #1: 自适应 α 训练 (GPU3)
# ============================================================================
# 与 GPU0 的 SERUM_p0fix 训练是平行的, 不冲突.
# 改造点:
#   - src/models/watermark.py: forward 支持 per-sample alpha tensor
#   - src/training/train.py: 按 prompt token 长度动态算 α (sigmoid 映射)
#   - configs/config_sd21_gpu3_innov_adaptive_alpha.yaml: 新 config
#
# 预期 (按 17-SERUM_INNOVATION_QUICKSTART.md):
#   - FID 退化从 +1.06 降到 +0.3
#   - TPR 保持 99%+
#   - 改动 < 50 行, 投入产出比最高
#
# 用法:
#   bash train_adaptive_alpha_gpu3.sh
# ============================================================================

set -e
cd /data2/fzx/SERUM

# 路径
CONFIG=configs/config_sd21_gpu3_innov_adaptive_alpha.yaml
LOG_DIR=logs
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p $LOG_DIR

GPU=3

# 环境
source /data2/fzx/conda/etc/profile.d/conda.sh
conda activate serum
export HF_HOME=/data2/fzx/SERUM/.cache/huggingface
export HF_DATASETS_OFFLINE=1
export PYTHONPATH=/data2/fzx/SERUM:$PYTHONPATH

echo "============================================================"
echo "SERUM 创新点 #1: 自适应 α 训练 (GPU$GPU)"
echo "  Config:  $CONFIG"
echo "  Time:    $TS"
echo ""
echo "  Adaptive α range: [0.3, 0.7], sigmoid 拐点 = 25 tokens"
echo "  Buffer size:      4096 (快速验证)"
echo "  Epochs:           30 (快速验证)"
echo ""
echo "  ⚠ 注意: PT cache 不能复用 GPU0 的 (注入时 α=0.5)"
echo "     会重新生成 watermarked_dataset_4096_adaptive_latent.pt"
echo "============================================================"

CUDA_VISIBLE_DEVICES=$GPU python -m src.training.train \
    --config $CONFIG \
    > ${LOG_DIR}/train_adaptive_alpha_${TS}.log 2>&1

echo ""
echo "训练完成 -> ${LOG_DIR}/train_adaptive_alpha_${TS}.log"
echo ""
echo "  实验目录: results/SERUM_sd21_gpu3_innov_adaptive_alpha/"
echo "  PT cache: watermarked_dataset_4096_adaptive_latent.pt"
echo "  Best ckpt: checkpoints/latest_checkpoint.pt"