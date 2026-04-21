#!/usr/bin/env python3
"""
Upload fine-tuned model to HuggingFace Hub.

Usage:
    python upload_to_hub.py --repo_id TAIDE-EDU/model-name
    python upload_to_hub.py --repo_id TAIDE-EDU/model-name --checkpoint checkpoint-2500
    python upload_to_hub.py --repo_id TAIDE-EDU/model-name --private
"""

import argparse
import os

from huggingface_hub import HfApi, create_repo


def parse_args():
    parser = argparse.ArgumentParser(description="Upload model to HuggingFace Hub")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/chris/Training/outputs",
        help="Path to the trained model directory",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Specific checkpoint to upload (e.g., 'checkpoint-2500')",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="HuggingFace repo ID (e.g., 'TAIDE-EDU/model-name')",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private",
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        default="Upload full model and tokenizer",
        help="Commit message for the upload",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine source path
    if args.checkpoint:
        folder_path = os.path.join(args.model_path, args.checkpoint)
    else:
        folder_path = args.model_path

    if not os.path.exists(folder_path):
        raise ValueError(f"Path not found: {folder_path}")

    print("=" * 50)
    print("HuggingFace Model Upload")
    print("=" * 50)
    print(f"Source: {folder_path}")
    print(f"Target: {args.repo_id}")
    print(f"Private: {args.private}")
    print("=" * 50)

    # List files to upload
    print("\nFiles to upload:")
    total_size = 0
    for f in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, f)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            total_size += size
            if size > 1e9:
                print(f"  {f}: {size/1e9:.2f} GB")
            elif size > 1e6:
                print(f"  {f}: {size/1e6:.2f} MB")
            else:
                print(f"  {f}: {size/1e3:.2f} KB")
    print(f"\nTotal size: {total_size/1e9:.2f} GB")

    # Create repo if needed
    print(f"\nCreating/verifying repository: {args.repo_id}")
    try:
        create_repo(
            repo_id=args.repo_id,
            repo_type="model",
            private=args.private,
            exist_ok=True,
        )
        print(f"  Repository ready: {'private' if args.private else 'public'}")
    except Exception as e:
        print(f"  Note: {e}")

    # Upload
    print(f"\nUploading to HuggingFace Hub...")
    print("This may take a while for large models...")

    api = HfApi()
    api.upload_folder(
        folder_path=folder_path,
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=args.commit_message,
    )

    print("\n" + "=" * 50)
    print("Upload complete!")
    print(f"Model URL: https://huggingface.co/{args.repo_id}")
    print("=" * 50)


if __name__ == "__main__":
    main()
