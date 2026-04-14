#!/usr/bin/env python3
"""
Analyze the token length distribution of the continue-pretrain dataset
"""
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

DATASET_NAME = "TAIDE-EDU/Edu-TAIDE-PT-Data"
DATASET_REVISION = "ld1-hq3"
MODEL_NAME = "mistralai/Ministral-3-14B-Instruct-2512-BF16"
SAMPLE_SIZE = 50000  # Sample 50k examples for analysis

print("=" * 60)
print("Loading tokenizer...")
print("=" * 60)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
print(f"Tokenizer loaded (vocab size: {len(tokenizer)})")

print("\n" + "=" * 60)
print("Loading dataset...")
print("=" * 60)
dataset = load_dataset(
    DATASET_NAME,
    revision=DATASET_REVISION,
    split="train",
)
print(f"Dataset loaded: {len(dataset)} examples")

print("\n" + "=" * 60)
print(f"Sampling {SAMPLE_SIZE} examples for analysis...")
print("=" * 60)

# Sample random examples
import random
random.seed(42)
sample_indices = random.sample(range(len(dataset)), min(SAMPLE_SIZE, len(dataset)))
sampled_dataset = dataset.select(sample_indices)

print(f"Analyzing {len(sampled_dataset)} examples...")

# Tokenize and collect lengths
lengths = []
for example in tqdm(sampled_dataset, desc="Tokenizing"):
    text = example["text"]
    tokens = tokenizer(text, truncation=False, add_special_tokens=False)
    lengths.append(len(tokens["input_ids"]))

lengths = np.array(lengths)

print("\n" + "=" * 60)
print("Token Length Statistics:")
print("=" * 60)
print(f"Total examples analyzed: {len(lengths)}")
print(f"Mean length: {lengths.mean():.1f} tokens")
print(f"Median length: {np.median(lengths):.1f} tokens")
print(f"Std deviation: {lengths.std():.1f} tokens")
print(f"Min length: {lengths.min()} tokens")
print(f"Max length: {lengths.max()} tokens")

print("\n" + "=" * 60)
print("Percentiles:")
print("=" * 60)
percentiles = [50, 75, 80, 85, 90, 95, 99]
for p in percentiles:
    value = np.percentile(lengths, p)
    coverage = (lengths <= value).sum() / len(lengths) * 100
    print(f"  {p}th percentile: {value:.0f} tokens (covers {coverage:.1f}% of data)")

print("\n" + "=" * 60)
print("Coverage Analysis for Different MAX_SEQ_LEN:")
print("=" * 60)
seq_lengths = [512, 1024, 1536, 2048, 3072, 4096, 8192]
for seq_len in seq_lengths:
    coverage = (lengths <= seq_len).sum() / len(lengths) * 100
    truncated = (lengths > seq_len).sum()
    truncated_pct = truncated / len(lengths) * 100
    avg_loss = np.mean(np.maximum(0, lengths - seq_len))
    print(f"  MAX_SEQ_LEN={seq_len:5d}: {coverage:5.2f}% coverage, "
          f"{truncated:6d} ({truncated_pct:5.2f}%) truncated, "
          f"avg loss: {avg_loss:.1f} tokens")

print("\n" + "=" * 60)
print("Recommendation:")
print("=" * 60)

# Find optimal seq_len (covers 90-95% with minimal truncation)
target_coverage = 0.90
optimal_seq_len = np.percentile(lengths, target_coverage * 100)
print(f"For 90% coverage: MAX_SEQ_LEN ≈ {optimal_seq_len:.0f}")

target_coverage = 0.95
optimal_seq_len = np.percentile(lengths, target_coverage * 100)
print(f"For 95% coverage: MAX_SEQ_LEN ≈ {optimal_seq_len:.0f}")

print("\nConsidering memory constraints:")
print("  - 1024: Very safe, ~85-90% coverage")
print("  - 2048: Safe, ~95%+ coverage (RECOMMENDED)")
print("  - 3072: Moderate risk, ~98%+ coverage")
print("  - 4096: High OOM risk, ~99%+ coverage")
print("=" * 60)
