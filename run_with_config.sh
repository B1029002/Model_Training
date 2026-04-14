#!/bin/bash
# Launch training using JSON config file
# Uses 4x H200 GPUs with DeepSpeed ZeRO Stage 2

set -e

# ============ Configuration ============
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_PROJECT="ministral-3-finetune"

# Optimize NCCL for H200
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# Training directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Number of GPUs
NUM_GPUS=4

# Config file
CONFIG_FILE="${1:-$SCRIPT_DIR/training_config.json}"

echo "============================================"
echo "Starting Ministral-3-14B Fine-tuning"
echo "============================================"
echo "Config: $CONFIG_FILE"
echo "GPUs: $NUM_GPUS"
echo "============================================"

# Launch training with torchrun
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    train.py "$CONFIG_FILE"

echo "============================================"
echo "Training Complete!"
echo "============================================"
