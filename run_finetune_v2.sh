#!/bin/bash

set -e

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

# ============ Configuration ============
export CUDA_VISIBLE_DEVICES=2
export WANDB_PROJECT="ministral-3-finetune"
export WANDB_RUN_NAME="ministral-3-14b-finetune-v2-merged"

# Optimize NCCL for H200
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# PyTorch memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NCCL timeout (disable watchdog for single GPU - not needed)
export TORCH_NCCL_ENABLE_MONITORING=0
export NCCL_TIMEOUT=7200
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200

# Training directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Model and data paths
# 使用 ft/checkpoint-17466 作為起點（第一輪 finetune 的結果）
MODEL_NAME="/home/chris/Training/ft/checkpoint-17466"
DATASET_PATH="/home/chris/LLM-Training/final_merged_dataset"
OUTPUT_DIR="/home/chris/Training/ft_v2"

# Create output directory and logs directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$SCRIPT_DIR/logs"

# Log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$SCRIPT_DIR/logs/finetune_v2_${TIMESTAMP}.log"

# Number of GPUs
NUM_GPUS=1

# DeepSpeed config
DS_CONFIG="$SCRIPT_DIR/ds_config_zero2.json"

# ============ Training Parameters ============
EPOCHS=5
BATCH_SIZE=16
GRAD_ACCUM=1
LR=2e-5
MIN_LR=1e-6
WARMUP_RATIO=0.1
MAX_SEQ_LEN=2048
SAVE_STEPS=500
EVAL_STEPS=500
LOGGING_STEPS=50
USE_PACKING=False

# Resume training from checkpoint (set to False to start fresh, or specify checkpoint path)
# Examples:
#   RESUME=False          # 從頭開始訓練
#   RESUME=True           # 自動從最新 checkpoint 繼續
#   RESUME=/path/to/checkpoint-1000  # 從指定 checkpoint 繼續
RESUME=False

# Determine resume checkpoint
RESUME_ARG=""
if [ "$RESUME" = "True" ]; then
    LATEST_CKPT=$(ls -td "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | head -1)
    if [ -n "$LATEST_CKPT" ]; then
        RESUME_ARG="--resume_from_checkpoint $LATEST_CKPT"
        echo "Will resume from: $LATEST_CKPT"
    else
        echo "No checkpoint found in $OUTPUT_DIR, starting from scratch"
    fi
elif [ "$RESUME" != "False" ] && [ -n "$RESUME" ]; then
    RESUME_ARG="--resume_from_checkpoint $RESUME"
    echo "Will resume from: $RESUME"
fi

echo "============================================"
echo "Starting Ministral-3-14B Finetuning V2"
echo "============================================"
echo "Base Model: $MODEL_NAME"
echo "Dataset: $DATASET_PATH (local)"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $NUM_GPUS"
echo "Learning Rate: $LR -> $MIN_LR (cosine)"
echo "Epochs: $EPOCHS"
echo "Effective Batch Size: $((BATCH_SIZE * GRAD_ACCUM))"
echo "Packing: $USE_PACKING"
echo "Resume: $RESUME_ARG"
echo "Best Model: Saved based on eval_loss"
echo "Log File: $LOG_FILE"
echo "============================================"

# Launch training with torchrun (log to file and terminal)
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29501 \
    train.py \
    --model_name_or_path "$MODEL_NAME" \
    --dataset_path "$DATASET_PATH" \
    --is_continue_pretrain False \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LR \
    --min_learning_rate $MIN_LR \
    --warmup_ratio $WARMUP_RATIO \
    --max_seq_length $MAX_SEQ_LEN \
    --save_steps $SAVE_STEPS \
    --eval_steps $EVAL_STEPS \
    --logging_steps $LOGGING_STEPS \
    --bf16 True \
    --tf32 True \
    --gradient_checkpointing True \
    --use_flash_attention_2 True \
    --deepspeed "$DS_CONFIG" \
    --report_to wandb \
    --run_name "$WANDB_RUN_NAME" \
    --save_total_limit 3 \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --greater_is_better False \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory True \
    --dataloader_prefetch_factor 2 \
    --use_packing $USE_PACKING \
    $RESUME_ARG \
    2>&1 | tee -a "$LOG_FILE"

echo "============================================"
echo "Finetuning V2 Complete!"
echo "Model saved to: $OUTPUT_DIR"
echo "Log saved to: $LOG_FILE"
echo "============================================"
