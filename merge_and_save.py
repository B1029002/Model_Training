#!/usr/bin/env python3
"""
Script to merge LoRA weights (if used) and save the final model.
Also handles DeepSpeed checkpoint conversion.
"""

import os
import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Special tokens (same as in train.py)
SPECIAL_TOKENS = [
    "<|task-1-indicator|>",
    "<|task-2-indicator|>",
    "<|task-2-bow|>",
    "<|task-2-eow|>",
    "<|task-3-indicator|>",
    "<|task-4_1-indicator|>",
    "<|task-4_2-indicator|>",
]


def convert_deepspeed_checkpoint(checkpoint_dir: str, output_dir: str) -> None:
    """
    Convert DeepSpeed ZeRO checkpoint to a standard PyTorch checkpoint.
    """
    from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict

    print(f"Converting DeepSpeed checkpoint from {checkpoint_dir}...")

    # Find the latest checkpoint
    checkpoint_tag = None
    for item in os.listdir(checkpoint_dir):
        if item.startswith("global_step"):
            checkpoint_tag = item
            break

    if checkpoint_tag:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_tag)
    else:
        checkpoint_path = checkpoint_dir

    output_file = os.path.join(output_dir, "pytorch_model.bin")
    convert_zero_checkpoint_to_fp32_state_dict(checkpoint_path, output_file)
    print(f"Converted checkpoint saved to {output_file}")


def merge_lora_and_save(
    base_model_path: str,
    lora_model_path: str,
    output_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> None:
    """
    Merge LoRA adapter weights with base model and save.
    """
    print(f"Loading base model from {base_model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto",
    )

    print(f"Loading LoRA adapter from {lora_model_path}...")
    model = PeftModel.from_pretrained(model, lora_model_path)

    print("Merging LoRA weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True)

    # Also save tokenizer
    print("Loading and saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        lora_model_path,
        trust_remote_code=True,
    )
    tokenizer.save_pretrained(output_path)

    print(f"Merged model saved to {output_path}")


def save_full_model(
    model_path: str,
    output_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> None:
    """
    Load a full fine-tuned model and save in a clean format.
    """
    print(f"Loading model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto",
    )

    print(f"Saving model to {output_path}...")
    os.makedirs(output_path, exist_ok=True)

    # Save in safetensors format
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)

    print(f"Model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA weights or convert DeepSpeed checkpoints"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["merge_lora", "convert_deepspeed", "save_full"],
        required=True,
        help="Operation mode",
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="mistralai/Ministral-3-14B-Instruct-2512-BF16",
        help="Path to base model (for LoRA merge)",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to checkpoint (LoRA adapter or DeepSpeed checkpoint)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the output model",
    )

    args = parser.parse_args()

    if args.mode == "merge_lora":
        merge_lora_and_save(
            base_model_path=args.base_model_path,
            lora_model_path=args.checkpoint_path,
            output_path=args.output_path,
        )
    elif args.mode == "convert_deepspeed":
        convert_deepspeed_checkpoint(
            checkpoint_dir=args.checkpoint_path,
            output_dir=args.output_path,
        )
    elif args.mode == "save_full":
        save_full_model(
            model_path=args.checkpoint_path,
            output_path=args.output_path,
        )


if __name__ == "__main__":
    main()
