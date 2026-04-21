#!/usr/bin/env python3
"""
Script to setup tokenizer with special tokens for Ministral-3-14B-Instruct-2512-BF16.
Run this script before training to prepare the tokenizer.
"""

import os
import json
import argparse
from transformers import AutoTokenizer

# Special tokens to add
SPECIAL_TOKENS = [
    "<|task-1-indicator|>",
    "<|task-2-indicator|>",
    "<|task-2-bow|>",
    "<|task-2-eow|>",
    "<|task-3-indicator|>",
    "<|task-4_1-indicator|>",
    "<|task-4_2-indicator|>",
]


def setup_tokenizer(
    model_name: str,
    output_path: str,
    trust_remote_code: bool = True,
) -> None:
    """
    Setup tokenizer with special tokens and save to output path.

    Args:
        model_name: HuggingFace model name or local path
        output_path: Path to save the modified tokenizer
        trust_remote_code: Whether to trust remote code
    """
    print(f"Loading tokenizer from: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )

    print(f"Original vocab size: {len(tokenizer)}")
    print(f"Original special tokens: {tokenizer.special_tokens_map}")

    # Add special tokens
    print(f"\nAdding {len(SPECIAL_TOKENS)} special tokens:")
    for token in SPECIAL_TOKENS:
        print(f"  - {token}")

    special_tokens_dict = {"additional_special_tokens": SPECIAL_TOKENS}
    num_added = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"\nAdded {num_added} new tokens")

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"Set pad_token to eos_token: {tokenizer.pad_token}")

    print(f"\nNew vocab size: {len(tokenizer)}")

    # Verify special tokens
    print("\nVerifying special tokens:")
    for token in SPECIAL_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        decoded = tokenizer.decode([token_id])
        print(f"  {token} -> ID: {token_id} -> Decoded: {decoded}")

    # Save tokenizer
    os.makedirs(output_path, exist_ok=True)
    tokenizer.save_pretrained(output_path)
    print(f"\nTokenizer saved to: {output_path}")

    # Print chat template info
    print("\n" + "="*50)
    print("Chat Template Info:")
    print("="*50)
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        print("Chat template is defined")
        print(f"Template (first 500 chars):\n{tokenizer.chat_template[:500]}...")
    else:
        print("No custom chat template defined (using default)")

    # Test the tokenizer with a sample
    print("\n" + "="*50)
    print("Testing tokenizer with sample message:")
    print("="*50)

    test_messages = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": f"I'm doing well! {SPECIAL_TOKENS[0]} This is a test."},
    ]

    try:
        formatted = tokenizer.apply_chat_template(
            test_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        print(f"Formatted output:\n{formatted}")

        # Tokenize and decode
        tokens = tokenizer.encode(formatted)
        print(f"\nNumber of tokens: {len(tokens)}")
        print(f"Token IDs (first 50): {tokens[:50]}")
    except Exception as e:
        print(f"Warning: Could not apply chat template: {e}")
        print("You may need to define a custom chat template.")


def main():
    parser = argparse.ArgumentParser(
        description="Setup tokenizer with special tokens for Ministral-3-14B"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistralai/Ministral-3-14B-Instruct-2512-BF16",
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/home/chris/Training/tokenizer",
        help="Path to save the modified tokenizer",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=True,
        help="Whether to trust remote code",
    )

    args = parser.parse_args()

    setup_tokenizer(
        model_name=args.model_name,
        output_path=args.output_path,
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    main()
