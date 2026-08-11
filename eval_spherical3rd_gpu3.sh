#!/bin/bash
# ============================================================================
# SERUM 创新点 #2: Spherical 3阶矩精确保持 评估
# ============================================================================
# 训练产物:
#   - Checkpoint: results/SERUM_sd21_gpu3_spherical3rd/checkpoints/latest_checkpoint.pt
#   - Config:     configs/config_sd21_gpu3_spherical3rd.yaml
#
# 评估内容:
#   - run_full_eval: TPR eval + Augmentation Robustness eval
#   - 与 p0fix (99.395%) / 论文 SERUM (99.75%) 对比
#
# 用法:
#   bash eval_spherical3rd_gpu3.sh                  # 默认 5000 samples
#   NUM_SAMPLES=1000 bash eval_spherical3rd_gpu3.sh # 自定义样本数
# ============================================================================

set -e
cd /data2/fzx/SERUM

# 路径
CONFIG=configs/config_sd21_gpu3_spherical3rd.yaml
CKPT=results/SERUM_sd21_gpu3_spherical3rd/checkpoints/latest_checkpoint.pt
LOG_DIR=logs
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p $LOG_DIR

# 评估样本数
NUM_SAMPLES=${NUM_SAMPLES:-5000}
GPU=1

# 环境
source /data2/fzx/conda/etc/profile.d/conda.sh
conda activate serum
export HF_HOME=/data2/fzx/SERUM/.cache/huggingface
export HF_DATASETS_OFFLINE=1
export PYTHONPATH=/data2/fzx/SERUM:$PYTHONPATH

echo "============================================================"
echo "SERUM 创新点 #2 评估 — GPU$GPU"
echo "  Samples:   $NUM_SAMPLES"
echo "  Config:    $CONFIG"
echo "  Checkpoint:$CKPT"
echo "  Time:      $TS"
echo "============================================================"
echo ""
echo "TPR eval + Augmentation Robustness eval"
echo "  log: ${LOG_DIR}/eval_spherical3rd_${TS}.log"

CUDA_VISIBLE_DEVICES=$GPU python -m src.evaluation.eval \
    --config $CONFIG \
    --load-from-checkpoint \
    --checkpoint-path $CKPT \
    --num-samples $NUM_SAMPLES \
    > ${LOG_DIR}/eval_spherical3rd_${TS}.log 2>&1

echo "  done -> ${LOG_DIR}/eval_spherical3rd_${TS}.log"
echo ""
echo "============================================================"
echo "完成. 输出位置:"
echo "  生成图 cache: results/SERUM_sd21_gpu3_spherical3rd/gen_images/"
echo "    clean/      watermarked/"
echo "  eval log:    results/SERUM_sd21_gpu3_spherical3rd/eval_*.txt"
echo "  pkl 结果:    results/SERUM_sd21_gpu3_spherical3rd/eval_results_*.pkl"
echo "  ROC 图:      results/SERUM_sd21_gpu3_spherical3rd/gen_images/eval_tpr_*.png"
echo "============================================================"
