#!/bin/bash
# ============================================================================
# SERUM P0 (smooth loss) 评估脚本 — torchrun 双 GPU 并行
# ============================================================================
# 训练产物:
#   - Checkpoint: results/SERUM_sd21_gpu1_p0fix/checkpoints/latest_checkpoint.pt
#   - Config:     configs/config_sd21_gpu1_p0fix.yaml
#
# 评估内容:
#   run_full_eval 默认分支 = TPR eval + Augmentation Robustness eval
#   generate() 内部按 RANK 切片, GPU0 + GPU3 各跑一半 (rank 1 完成后自动 exit)
#   只有 rank 0 继续走 TPR 评分 + Augmentation 评分
#
# 用法:
#   bash eval_p0fix_dual_gpu.sh                  # 默认 1000 samples
#   NUM_SAMPLES=500 bash eval_p0fix_dual_gpu.sh # 自定义样本数
# ============================================================================

set -e
cd /data2/fzx/SERUM

# 路径
CONFIG=configs/config_sd21_gpu1_p0fix.yaml
CKPT=results/SERUM_sd21_gpu1_p0fix/checkpoints/latest_checkpoint.pt
LOG_DIR=logs
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p $LOG_DIR

# 评估样本数
NUM_SAMPLES=${NUM_SAMPLES:-1000}
GPU0=0
GPU3=3
MASTER_PORT=29501

# 环境
source /data2/fzx/conda/etc/profile.d/conda.sh
conda activate serum
export HF_HOME=/data2/fzx/SERUM/.cache/huggingface
export HF_DATASETS_OFFLINE=1
export PYTHONPATH=/data2/fzx/SERUM:$PYTHONPATH

echo "============================================================"
echo "SERUM P0 评估 — 双 GPU 并行 (GPU$GPU0 + GPU$GPU3)"
echo "  Samples:   $NUM_SAMPLES  (--num-samples 改这里)"
echo "  Config:    $CONFIG"
echo "  Checkpoint:$CKPT"
echo "  Time:      $TS"
echo "============================================================"
echo ""
echo "双 GPU 并行: TPR eval + Augmentation Robustness eval"
echo "  log: ${LOG_DIR}/eval_p0fix_full_${TS}.log"

CUDA_VISIBLE_DEVICES=$GPU0,$GPU3 torchrun \
    --nproc_per_node=2 \
    --master_port=$MASTER_PORT \
    -m src.evaluation.eval \
    --config $CONFIG \
    --load-from-checkpoint \
    --checkpoint-path $CKPT \
    --num-samples $NUM_SAMPLES \
    > ${LOG_DIR}/eval_p0fix_full_${TS}.log 2>&1

echo "  done -> ${LOG_DIR}/eval_p0fix_full_${TS}.log"
echo ""
echo "============================================================"
echo "完成. 输出位置:"
echo "  生成图 cache: results/SERUM_sd21_gpu1_p0fix/gen_images/"
echo "    clean/      watermarked/"
echo "  eval log:    results/SERUM_sd21_gpu1_p0fix/eval_*.txt"
echo "  pkl 结果:    results/SERUM_sd21_gpu1_p0fix/eval_results_*.pkl"
echo "  ROC 图:      results/SERUM_sd21_gpu1_p0fix/gen_images/eval_tpr_*.png"
echo "============================================================"