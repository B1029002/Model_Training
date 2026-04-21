#!/bin/bash
# Training launch script for Phi-3.5-mini fine-tuning
# FT base model: TAIDE-EDU/phi-3.5-mini-instruct_zhtw_ld1_hq3.1_b8.3-p3_st-task-1-2-3-v3_e4_ORPO_1103-1019_fix3

set -e

# ============ Configuration ============
export CUDA_VISIBLE_DEVICES=3
export WANDB_PROJECT="phi-3.5-mini-finetune"
export WANDB_RUN_NAME="phi-3.5-mini-instruct-finetune"

# Training directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Model and data paths
MODEL_NAME="TAIDE-EDU/phi-3.5-mini-instruct_zhtw_ld1_hq3.1_b8.3-p3_st-task-1-2-3-v3_e4_ORPO_1103-1019_fix3"
DATASET_PATH="/home/chris/LLM-Training/final_balanced_dataset"
OUTPUT_DIR="/home/chris/Training/outputs/phi-3.5-mini-finetune"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Number of GPUs
NUM_GPUS=1

# DeepSpeed config (use zero2 for better speed, zero3 for larger models)
DS_CONFIG="$SCRIPT_DIR/ds_config_zero2.json"

# ============ Training Parameters ============
EPOCHS=3
BATCH_SIZE=1
GRAD_ACCUM=8
LR=2e-5
WARMUP_RATIO=0.03
MAX_SEQ_LEN=4096
SAVE_STEPS=500
EVAL_STEPS=500
LOGGING_STEPS=10

echo "============================================"
echo "Starting Phi-3.5-mini Fine-tuning"
echo "============================================"
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $NUM_GPUS"
echo "DeepSpeed Config: $DS_CONFIG"
echo "============================================"

# Launch training with accelerate
accelerate launch \
    --config_file "$SCRIPT_DIR/accelerate_config.yaml" \
    --num_processes $NUM_GPUS \
    --num_machines 1 \
    --mixed_precision bf16 \
    --use_deepspeed \
    --deepspeed_config_file "$DS_CONFIG" \
    train.py \
    --model_name_or_path "$MODEL_NAME" \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LR \
    --warmup_ratio $WARMUP_RATIO \
    --max_seq_length $MAX_SEQ_LEN \
    --save_steps $SAVE_STEPS \
    --eval_steps $EVAL_STEPS \
    --logging_steps $LOGGING_STEPS \
    --bf16 True \
    --tf32 True \
    --gradient_checkpointing True \
    --use_flash_attention_2 True \
    --report_to wandb \
    --run_name "$WANDB_RUN_NAME" \
    --save_total_limit 3 \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory True

echo "============================================"
echo "Training Complete!"
echo "Model saved to: $OUTPUT_DIR"
echo "============================================"
