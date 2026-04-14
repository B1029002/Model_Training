#!/bin/bash

set -e

# ============ Configuration ============
export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="ministral-3-continue-pretrain"
export WANDB_RUN_NAME="ministral-3-14b-continue-pretrain-ld1-hq3"

# Optimize NCCL for H200
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# PyTorch memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Training directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Model and data paths
MODEL_NAME="mistralai/Ministral-3-14B-Instruct-2512-BF16"
DATASET_NAME="TAIDE-EDU/Edu-TAIDE-PT-Data"
DATASET_REVISION="ld1-hq3"
OUTPUT_DIR="/home/chris/LLM-Training/Training/cp"

# Create output directory and logs directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$SCRIPT_DIR/logs"

# Log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$SCRIPT_DIR/logs/train_${TIMESTAMP}.log"

# Number of GPUs
NUM_GPUS=1

# DeepSpeed config (use ZeRO-3 with CPU offload for memory efficiency)
DS_CONFIG="$SCRIPT_DIR/ds_config_zero3.json"

# ============ Training Parameters ============
# Continue-pretrain 建議：較低 LR、較少 epochs、較大 batch
EPOCHS=1              # 1800萬筆資料，1 epoch 約 91萬步 (with packing)
BATCH_SIZE=1          
GRAD_ACCUM=1          
LR=1e-5              
MIN_LR=1e-6           
WARMUP_RATIO=0.05     
MAX_SEQ_LEN=1536
SAVE_STEPS=5000      
EVAL_STEPS=5000      
LOGGING_STEPS=100
USE_PACKING=True          

echo "============================================"
echo "Starting Ministral-3-14B Continue-Pretrain"
echo "============================================"
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET_NAME (revision: $DATASET_REVISION)"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $NUM_GPUS"
echo "Learning Rate: $LR -> $MIN_LR (cosine)"
echo "Epochs: $EPOCHS"
echo "Packing: $USE_PACKING"
echo "Best Model: Saved based on eval_loss"
echo "Log File: $LOG_FILE"
echo "============================================"

# Launch training with torchrun (log to file and terminal)
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    train.py \
    --model_name_or_path "$MODEL_NAME" \
    --dataset_name "$DATASET_NAME" \
    --dataset_revision "$DATASET_REVISION" \
    --is_continue_pretrain True \
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
    --dataloader_num_workers 16 \
    --dataloader_pin_memory True \
    --dataloader_prefetch_factor 4 \
    --use_packing $USE_PACKING \
    2>&1 | tee -a "$LOG_FILE"

echo "============================================"
echo "Training Complete!"
echo "Model saved to: $OUTPUT_DIR"
echo "Log saved to: $LOG_FILE"
echo "============================================"
