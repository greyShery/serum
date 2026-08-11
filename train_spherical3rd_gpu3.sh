#!/bin/bash
# ============================================================================
# SERUM 创新点 #2: Spherical 3阶矩精确保持 (GPU3 训练)
# ============================================================================
# 改造点:
#   - src/models/watermark.py: 新增 _spherical_normalize (1/2/3 阶矩),
#     forward 时若 config 中 use_spherical_3rd_moment=true, 会做 tanh 重映射
#   - src/training/train.py: 新增 loss_sph3 = w * skewness^2 (3 阶矩正则)
#   - configs/config_sd21_gpu3_spherical3rd.yaml: 新 config
#
# 预期 (按 16-SERUM_INNOVATION_ANALYSIS.md):
#   - FID 退化 < +0.5 (3 阶矩严格保持 → 与 N(0,I) 更高对齐)
#   - TPR 保持 99%+ (与 p0fix baseline 持平)
#   - 改动 < 50 行, 训练成本 +10% (3 阶矩正则计算)
#
# 用法:
#   bash train_spherical3rd_gpu3.sh
# ============================================================================

set -e
cd /data2/fzx/SERUM

# 路径
CONFIG=configs/config_sd21_gpu3_spherical3rd.yaml
LOG_DIR=logs
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p $LOG_DIR

GPU=1

# 环境
source /data2/fzx/conda/etc/profile.d/conda.sh
conda activate serum
export HF_HOME=/data2/fzx/SERUM/.cache/huggingface
export HF_DATASETS_OFFLINE=1
export PYTHONPATH=/data2/fzx/SERUM:$PYTHONPATH

echo "============================================================"
echo "SERUM 创新点 #2: Spherical 3阶矩精确保持 (GPU$GPU)"
echo "  Config:  $CONFIG"
echo "  Time:    $TS"
echo ""
echo "  _spherical_normalize: 2 阶归一化 + tanh 重映射 + norm 还原"
echo "  训练正则:  loss_sph3 = 0.01 * skewness(grid)^2"
echo "  Buffer size: 15000 (与 p0fix 一致)"
echo "  Epochs:      50 (与 p0fix 一致)"
echo ""
echo "  ⚠ 注意: 因 use_spherical_3rd_moment=true 改变了 A 分布,"
echo "     不能复用 GPU1_p0fix 的 PT cache. 会重新生成,"
echo "     耗时 +1.5h"
echo "============================================================"

CUDA_VISIBLE_DEVICES=$GPU python -m src.training.train \
    --config $CONFIG \
    > ${LOG_DIR}/train_spherical3rd_${TS}.log 2>&1

echo ""
echo "训练完成 -> ${LOG_DIR}/train_spherical3rd_${TS}.log"
echo ""
echo "  实验目录: results/SERUM_sd21_gpu3_spherical3rd/"
echo "  PT cache: watermarked_dataset_14992_0.5_latent.pt"
echo "  Best ckpt: checkpoints/latest_checkpoint.pt"
