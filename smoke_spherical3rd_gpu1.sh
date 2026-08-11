#!/bin/bash
# ============================================================================
# Smoke test: Spherical 3阶矩精确保持 (GPU1, 5 epoch)
# ============================================================================
# 验证目标:
#   1. 代码能跑通 (无语法错误)
#   2. loss 收敛 (loss_sph3 不发散)
#   3. watermark 生成无 NaN
#   4. 反向兼容 (现有 checkpoint 行为不变)
#
# 资源: GPU1 空闲 (5 epoch ≈ 1h)
# ============================================================================

set -e
cd /data2/fzx/SERUM

CONFIG=configs/config_sd21_gpu1_spherical3rd_smoke.yaml
LOG_DIR=logs
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p $LOG_DIR

GPU=1

source /data2/fzx/conda/etc/profile.d/conda.sh
conda activate serum
export HF_HOME=/data2/fzx/SERUM/.cache/huggingface
export HF_DATASETS_OFFLINE=1
export PYTHONPATH=/data2/fzx/SERUM:$PYTHONPATH

echo "============================================================"
echo "Smoke test: Spherical 3阶矩精确保持 (GPU$GPU)"
echo "  Config:  $CONFIG"
echo "  Epochs:  5"
echo "  Time:    $TS"
echo "============================================================"

CUDA_VISIBLE_DEVICES=$GPU python -m src.training.train \
    --config $CONFIG \
    > ${LOG_DIR}/smoke_spherical3rd_${TS}.log 2>&1

echo ""
echo "Smoke 完成 -> ${LOG_DIR}/smoke_spherical3rd_${TS}.log"
echo ""
echo "  检查:"
echo "  1. 末尾 loss 是否 < 0.1 (收敛)"
echo "  2. 有无 'sph3' log (正则项有梯度)"
echo "  3. 有无 NaN/Inf"
echo "  4. checkpoint 已保存"
echo ""
tail -40 ${LOG_DIR}/smoke_spherical3rd_${TS}.log
