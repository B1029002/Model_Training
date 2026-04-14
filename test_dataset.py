#!/usr/bin/env python3
"""
Test script to verify dataset loading
"""
from datasets import load_dataset

DATASET_NAME = "TAIDE-EDU/Edu-TAIDE-PT-Data"
DATASET_REVISION = "ld1-hq3"

print("=" * 60)
print(f"Loading dataset: {DATASET_NAME}")
print(f"Revision: {DATASET_REVISION}")
print("=" * 60)

try:
    dataset = load_dataset(
        DATASET_NAME,
        revision=DATASET_REVISION,
        split="train",
    )

    print(f"\n✓ Dataset loaded successfully!")
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Column names: {dataset.column_names}")

    if len(dataset) > 0:
        print(f"\n  First example keys: {list(dataset[0].keys())}")

        # Print first example
        print(f"\n  First example preview:")
        for key, value in dataset[0].items():
            if isinstance(value, str):
                preview = value[:200] + "..." if len(value) > 200 else value
                print(f"    {key}: {preview}")
            else:
                print(f"    {key}: {value}")

except Exception as e:
    print(f"\n✗ Error loading dataset: {e}")
    import traceback
    traceback.print_exc()
