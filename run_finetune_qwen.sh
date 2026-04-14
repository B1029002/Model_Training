#!/bin/bash

set -e

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen

# ============ Configuration ============
export CUDA_VISIBLE_DEVICES=1,2,3
export WANDB_PROJECT="qwen3.5-35b-a3b-finetune"
export WANDB_RUN_NAME="qwen3.5-35b-a3b-ft-v1"

# Use conda's newer libstdc++ for flash-attn CXXABI_1.3.15 compatibility
# Include nvidia CUDA libs for DeepSpeed JIT compilation (cpu_adam needs libcurand)
NVIDIA_LIBS="/home/chris/miniconda3/envs/qwen/lib/python3.11/site-packages/nvidia/curand/lib"
NVIDIA_LIBS="$NVIDIA_LIBS:/home/chris/miniconda3/envs/qwen/lib/python3.11/site-packages/nvidia/cuda_runtime/lib"
NVIDIA_LIBS="$NVIDIA_LIBS:/home/chris/miniconda3/envs/qwen/lib/python3.11/site-packages/nvidia/cublas/lib"
export LD_LIBRARY_PATH=/home/chris/miniconda3/envs/qwen/lib/gcc/x86_64-conda-linux-gnu/14.3.0:$NVIDIA_LIBS:$LD_LIBRARY_PATH
export LIBRARY_PATH=$NVIDIA_LIBS:$LIBRARY_PATH

# Optimize NCCL for H200
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# PyTorch memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NCCL timeout
export TORCH_NCCL_ENABLE_MONITORING=0
export NCCL_TIMEOUT=7200
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200

# Training directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Model and data paths
MODEL_NAME="Qwen/Qwen3.5-35B-A3B"
DATASET_PATH="/home/chris/LLM-Training/final_balanced_dataset"
OUTPUT_DIR="/home/chris/LLM-Training/Training/ft_qwen"

# Create output directory and logs directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$SCRIPT_DIR/logs"

# Log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$SCRIPT_DIR/logs/finetune_qwen_${TIMESTAMP}.log"

# Number of GPUs
NUM_GPUS=3

# DeepSpeed config (ZeRO-3 + CPU offload for 35B MoE model)
DS_CONFIG="$SCRIPT_DIR/ds_config_zero3.json"

# ============ Training Parameters ============
EPOCHS=5
BATCH_SIZE=1             # 35B 總參數，比 14B 吃更多記憶體，保守設定
GRAD_ACCUM=1              # effective batch = 4 * 4 * 3 GPUs = 48
LR=3e-6                   # MoE 模型建議用較小 LR
MIN_LR=3e-7
WARMUP_RATIO=0.1
MAX_SEQ_LEN=4096
SAVE_STEPS=200
EVAL_STEPS=200
LOGGING_STEPS=50
USE_PACKING=False

# Resume training from checkpoint
RESUME=False
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
echo "Starting Qwen3.5-35B-A3B Finetuning"
echo "============================================"
echo "Base Model: $MODEL_NAME"
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $NUM_GPUS (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "DeepSpeed: $DS_CONFIG"
echo "Learning Rate: $LR -> $MIN_LR (cosine)"
echo "Epochs: $EPOCHS"
echo "Effective Batch Size: $((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS))"
echo "Packing: $USE_PACKING"
echo "Resume: $RESUME_ARG"
echo "Log File: $LOG_FILE"
echo "============================================"

# Launch training with torchrun
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
echo "Finetuning Complete!"
echo "Model saved to: $OUTPUT_DIR"
echo "Log saved to: $LOG_FILE"
echo "============================================"
