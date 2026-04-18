# Model Training

LLM fine-tuning pipeline for **Ministral-3-14B-Instruct**, supporting multi-GPU training with DeepSpeed ZeRO-2, LoRA, sequence packing, and checkpoint resumption.

## Features

- Full fine-tuning and LoRA (PEFT) for instruction tuning / continue-pretrain
- Multi-GPU distributed training via `torchrun` + DeepSpeed ZeRO Stage 2 with CPU offload
- Sequence packing for improved training efficiency
- Custom special tokens and chat template support
- Cosine learning rate scheduler with configurable minimum LR
- Checkpoint resume and best-model selection by `eval_loss`
- W&B logging integration
- Utilities for merging LoRA weights, converting DeepSpeed checkpoints, and uploading models to Hugging Face Hub

## Project Structure

```
Training/
├── train.py                    # Main training script
├── run_finetune.sh             # Launch script
├── training_config.json        # Example training config (JSON)
├── accelerate_config.yaml      # Accelerate config
├── ds_config_zero2.json        # DeepSpeed ZeRO-2 config (CPU offload)
├── chat_template_simple.jinja  # Mistral chat template
├── requirements.txt            # Python dependencies
├── setup_tokenizer.py          # Tokenizer setup utility
├── merge_and_save.py           # Merge LoRA / convert DeepSpeed checkpoints
├── upload_to_hub.py            # Upload model to Hugging Face Hub
├── analyze_dataset_length.py   # Dataset sequence length analysis
├── test_dataset.py             # Dataset loading test
└── preprocessing.ipynb         # Data preprocessing notebook
```

## Setup

```bash
#create the virtual environment
conda create -n model_training python=3.10 -y
#run the virtaul environment
conda activate model_training

#you can use the instructions to install pytorch and cuda
pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.5.1 torchvision torchaudio
conda install cuda -c "nvidia/label/cuda-12.4.0"
conda install -c conda-forge cuda-nvcc=12.4

# install after install pytorch and cuda
pip install -r requirements.txt

# Flash Attention 2 (install separately based on your CUDA version)
pip install flash-attn --no-build-isolation
```

## Usage

### Fine-tuning

```bash
# Edit run_finetune.sh to set MODEL_NAME, DATASET_NAME, OUTPUT_DIR, etc.
bash run_finetune.sh
```

### Using JSON Config

```bash
python train.py training_config.json
```

### Key Training Parameters

| Parameter | Value |
|-----------|-------|
| Base Model | Ministral-3-14B-Instruct-2512-BF16 |
| DeepSpeed | ZeRO-2 + CPU offload |
| Batch size | 16 |
| Learning rate | 2e-5 → 1e-6 (cosine) |
| Max seq length | 2048 |
| Precision | BF16 + TF32 |
| Attention | Flash Attention 2 |

### Resume from Checkpoint

In the launch script, set the `RESUME` variable:

```bash
RESUME=True                          # Auto-resume from latest checkpoint
RESUME=/path/to/checkpoint-1000      # Resume from specific checkpoint
RESUME=False                         # Train from scratch
```

## Post-Training

### Merge LoRA Weights

```bash
python merge_and_save.py \
    --mode merge_lora \
    --base_model_path mistralai/Ministral-3-14B-Instruct-2512-BF16 \
    --checkpoint_path /path/to/lora-checkpoint \
    --output_path /path/to/merged-model
```

### Convert DeepSpeed Checkpoint

```bash
python merge_and_save.py \
    --mode convert_deepspeed \
    --checkpoint_path /path/to/deepspeed-checkpoint \
    --output_path /path/to/output
```

### Upload to Hugging Face Hub

```bash
python upload_to_hub.py \
    --repo_id your-org/model-name \
    --model_path /path/to/model \
    --private
```
