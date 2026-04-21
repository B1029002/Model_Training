#!/usr/bin/env python3
"""
Fine-tuning script for Ministral-3-14B-Instruct-2512-BF16
Supports multi-GPU training with DeepSpeed ZeRO Stage 2/3
"""

import os
import sys
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed,
    Mistral3ForConditionalGeneration,
    EarlyStoppingCallback,
)
from datasets import load_from_disk, load_dataset, Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ============ Special Tokens ============
SPECIAL_TOKENS = [
    "<|task-1-indicator|>",
    "<|task-2-indicator|>",
    "<|task-2-bow|>",
    "<|task-2-eow|>",
    "<|task-3-indicator|>",
    "<|task-4_1-indicator|>",
    "<|task-4_2-indicator|>",
]


@dataclass
class ModelArguments:
    """Arguments for model configuration."""
    model_name_or_path: str = field(
        default="mistralai/Ministral-3-14B-Instruct-2512-BF16",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    use_flash_attention_2: bool = field(
        default=True,
        metadata={"help": "Whether to use Flash Attention 2"}
    )
    use_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use LoRA for parameter-efficient fine-tuning"}
    )
    lora_r: int = field(
        default=64,
        metadata={"help": "LoRA rank"}
    )
    lora_alpha: int = field(
        default=128,
        metadata={"help": "LoRA alpha"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout"}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code"}
    )


@dataclass
class DataArguments:
    """Arguments for data configuration."""
    dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the local preprocessed dataset (for instruction tuning)"}
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "HuggingFace dataset name (e.g., TAIDE-EDU/Edu-TAIDE-PT-Data)"}
    )
    dataset_revision: Optional[str] = field(
        default=None,
        metadata={"help": "Dataset revision/branch to use"}
    )
    data_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Data directory within the dataset (e.g., 'b8.3-p3')"}
    )
    dataset_split: str = field(
        default="train",
        metadata={"help": "Which split to use for training"}
    )
    text_column: str = field(
        default="text",
        metadata={"help": "Column name containing the text for continue-pretrain"}
    )
    is_continue_pretrain: bool = field(
        default=False,
        metadata={"help": "Whether this is continue-pretrain (raw text) vs instruction-tuning (chat format)"}
    )
    max_seq_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length"}
    )
    validation_split: float = field(
        default=0.05,
        metadata={"help": "Validation split ratio"}
    )
    use_packing: bool = field(
        default=True,
        metadata={"help": "Whether to use sequence packing for efficiency"}
    )


@dataclass
class CustomTrainingArguments(TrainingArguments):
    """Extended training arguments."""
    output_dir: str = field(
        default="/home/chris/Training/outputs",
        metadata={"help": "Output directory for model checkpoints"}
    )
    num_train_epochs: float = field(
        default=3.0,
        metadata={"help": "Number of training epochs"}
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size per GPU for training"}
    )
    per_device_eval_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size per GPU for evaluation"}
    )
    gradient_accumulation_steps: int = field(
        default=8,
        metadata={"help": "Number of gradient accumulation steps"}
    )
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "Initial learning rate"}
    )
    min_learning_rate: float = field(
        default=2e-6,
        metadata={"help": "Minimum learning rate for cosine scheduler"}
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "Weight decay"}
    )
    warmup_ratio: float = field(
        default=0.03,
        metadata={"help": "Warmup ratio"}
    )
    lr_scheduler_type: str = field(
        default="cosine",
        metadata={"help": "Learning rate scheduler type"}
    )
    lr_scheduler_kwargs: Optional[str] = field(
        default=None,
        metadata={"help": "JSON string of lr_scheduler_kwargs"}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "Logging interval"}
    )
    save_strategy: str = field(
        default="steps",
        metadata={"help": "Save strategy"}
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "Save checkpoint every X steps"}
    )
    eval_strategy: str = field(
        default="steps",
        metadata={"help": "Evaluation strategy"}
    )
    eval_steps: int = field(
        default=500,
        metadata={"help": "Evaluation interval"}
    )
    save_total_limit: int = field(
        default=3,
        metadata={"help": "Maximum number of checkpoints to keep"}
    )
    load_best_model_at_end: bool = field(
        default=True,
        metadata={"help": "Load best model at end of training"}
    )
    metric_for_best_model: str = field(
        default="eval_loss",
        metadata={"help": "Metric for best model selection"}
    )
    greater_is_better: bool = field(
        default=False,
        metadata={"help": "Whether greater metric value is better"}
    )
    bf16: bool = field(
        default=True,
        metadata={"help": "Use bfloat16 precision"}
    )
    tf32: bool = field(
        default=True,
        metadata={"help": "Use TF32 precision for matmul"}
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Enable gradient checkpointing"}
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={"help": "Path to DeepSpeed config file"}
    )
    report_to: str = field(
        default="wandb",
        metadata={"help": "Reporting integration (wandb, tensorboard, none)"}
    )
    run_name: str = field(
        default="ministral-3-14b-finetune",
        metadata={"help": "Run name for logging"}
    )
    dataloader_num_workers: int = field(
        default=4,
        metadata={"help": "Number of data loader workers"}
    )
    dataloader_pin_memory: bool = field(
        default=True,
        metadata={"help": "Pin memory in data loader"}
    )


def setup_tokenizer(model_args: ModelArguments, is_continue_pretrain: bool = False) -> AutoTokenizer:
    """Setup tokenizer with special tokens and custom chat template."""
    logger.info(f"Loading tokenizer from {model_args.model_name_or_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=True,
    )

    # Only add special tokens for instruction tuning
    if not is_continue_pretrain:
        logger.info(f"Adding {len(SPECIAL_TOKENS)} special tokens: {SPECIAL_TOKENS}")
        special_tokens_dict = {"additional_special_tokens": SPECIAL_TOKENS}
        num_added = tokenizer.add_special_tokens(special_tokens_dict)
        logger.info(f"Added {num_added} special tokens to tokenizer")

        # Set custom chat template per model family (others use their native template)
        model_name_lower = model_args.model_name_or_path.lower()
        if "mistral" in model_name_lower or "ministral" in model_name_lower:
            custom_template = get_chat_template_for_ministral()
            if custom_template:
                tokenizer.chat_template = custom_template
                logger.info("Set custom Mistral chat template (no system prompt)")
        elif "phi-3" in model_name_lower or "phi3" in model_name_lower:
            tokenizer.chat_template = get_chat_template_for_phi3()
            logger.info("Set custom Phi-3 chat template")
        else:
            logger.info("Unknown model family, keeping native chat template")
    else:
        logger.info("Continue-pretrain mode: skipping special tokens and chat template")

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info(f"Set pad_token to eos_token: {tokenizer.pad_token}")

    return tokenizer


def setup_model(model_args: ModelArguments, tokenizer: AutoTokenizer, is_continue_pretrain: bool = False):
    """Setup model with proper configuration."""
    logger.info(f"Loading model from {model_args.model_name_or_path}")

    # Determine attention implementation
    attn_implementation = "flash_attention_2" if model_args.use_flash_attention_2 else "sdpa"
    logger.info(f"Using attention implementation: {attn_implementation}")

    # Load raw config to detect model type
    if os.path.exists(model_args.model_name_or_path):
        # Local path
        config_path = os.path.join(model_args.model_name_or_path, "config.json")
        logger.info(f"Loading config from local path: {config_path}")
    else:
        # HuggingFace Hub
        from huggingface_hub import hf_hub_download
        config_path = hf_hub_download(model_args.model_name_or_path, "config.json")
        logger.info(f"Downloaded config from HuggingFace Hub: {config_path}")

    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    model_type = config_dict.get("model_type", "")

    if model_type == "mistral3":
        # Mistral-3 is a multimodal model, use Mistral3ForConditionalGeneration
        # Fix text_config.model_type if needed
        if config_dict.get("text_config", {}).get("model_type") == "ministral3":
            config_dict["text_config"]["model_type"] = "mistral"
            logger.info("Fixed config: text_config.model_type ministral3 -> mistral")

        from transformers import Mistral3Config
        config = Mistral3Config.from_dict(config_dict)

        logger.info("Detected Mistral-3 multimodal model, using Mistral3ForConditionalGeneration")
        model = Mistral3ForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            config=config,  # Use our fixed config
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
            trust_remote_code=model_args.trust_remote_code,
        )
        # Disable cache in config for training
        model.config.use_cache = False
        if hasattr(model.config, 'text_config'):
            model.config.text_config.use_cache = False
        # For Mistral-3, vocab size is in text_config
        original_vocab_size = config.text_config.vocab_size
    else:
        # Standard causal LM (Phi-3/3.5, Llama, Qwen, Mistral, etc.)
        logger.info(f"Using AutoModelForCausalLM for model_type: {model_type}")
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
            trust_remote_code=model_args.trust_remote_code,
        )
        model.config.use_cache = False  # Disable KV cache for training
        original_vocab_size = model.config.vocab_size

    # Resize token embeddings for new special tokens (only for instruction tuning)
    if not is_continue_pretrain:
        model.resize_token_embeddings(len(tokenizer))
        new_vocab_size = len(tokenizer)
        logger.info(f"Resized token embeddings: {original_vocab_size} -> {new_vocab_size}")

        # Initialize new token embeddings with mean of existing embeddings
        if new_vocab_size > original_vocab_size:
            with torch.no_grad():
                input_embeddings = model.get_input_embeddings().weight
                output_embeddings = model.get_output_embeddings().weight

                # Use mean of existing embeddings for initialization
                input_mean = input_embeddings[:original_vocab_size].mean(dim=0)
                output_mean = output_embeddings[:original_vocab_size].mean(dim=0)

                for i in range(original_vocab_size, new_vocab_size):
                    input_embeddings[i] = input_mean + torch.randn_like(input_mean) * 0.01
                    output_embeddings[i] = output_mean + torch.randn_like(output_mean) * 0.01

                logger.info(f"Initialized {new_vocab_size - original_vocab_size} new token embeddings")
    else:
        logger.info("Continue-pretrain mode: not resizing token embeddings")

    # Setup LoRA if enabled
    if model_args.use_lora:
        logger.info("Setting up LoRA...")

        # Phi-3/3.5 uses fused qkv_proj and gate_up_proj; others use separate projections
        model_name_lower = model_args.model_name_or_path.lower()
        if "phi-3" in model_name_lower or "phi3" in model_name_lower:
            target_modules = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
        else:
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]
        logger.info(f"LoRA target_modules: {target_modules}")

        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Enable gradient checkpointing
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing")

    return model


def get_chat_template_for_ministral():
    """Get simple chat template without system prompt."""
    # Simple Mistral-style template for transformers Trainer
    # Format: <s>[INST]user message[/INST]assistant response</s>[INST]user2[/INST]assistant2</s>
    template = """{%- for message in messages -%}
{%- if message['role'] == 'user' -%}
{%- if loop.index0 == 0 -%}{{ bos_token }}{%- endif -%}
[INST]{{ message['content'] }}[/INST]
{%- elif message['role'] == 'assistant' -%}
{{ message['content'] }}{{ eos_token }}
{%- endif -%}
{%- endfor -%}"""
    return template


def get_chat_template_for_phi3():
    """Phi-3 chat template matching src/llm_training/data/chat_templates/phi-3.j2.

    Format: <|system|>\\n...<|end|>\\n<|user|>\\n...<|end|>\\n<|assistant|>\\n...<|end|>\\n
    """
    template = """{%- for message in messages %}
    {%- set content = message.content | trim %}
    {%- if message.role == 'system' and content %}
        {{- '<|system|>\n' + content + '<|end|>\n' }}
    {%- elif message.role == 'user' %}
        {{- '<|user|>\n' + content + '<|end|>\n' }}
    {%- elif message.role == 'assistant' %}
        {{- '<|assistant|>\n' -}}
        {% generation %}
            {{- content + '<|end|>\n' -}}
        {% endgeneration %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|assistant|>\n' }}
{%- else %}
    {% generation %}
        {{- eos_token -}}
    {% endgeneration %}
{%- endif %}"""
    return template


def preprocess_function(
    examples: Dict[str, List],
    tokenizer: AutoTokenizer,
    max_seq_length: int,
) -> Dict[str, List]:
    """Preprocess examples using chat template for instruction tuning.
    Only assistant responses are used for loss computation (label masking).
    """

    IGNORE_INDEX = -100
    all_input_ids = []
    all_attention_mask = []
    all_labels = []

    for messages in examples["messages"]:
        # Skip empty messages
        if not messages or len(messages) == 0:
            continue

        # Build the full conversation text and track assistant response boundaries
        # Strategy: tokenize incrementally to find where each assistant turn starts/ends
        try:
            # Tokenize the full conversation
            full_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception as e:
            logger.warning(f"Skipping message due to error: {e}")
            continue

        full_tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            return_tensors=None,
        )

        input_ids = full_tokenized["input_ids"]
        attention_mask = full_tokenized["attention_mask"]
        labels = [IGNORE_INDEX] * len(input_ids)

        # Tokenize prefix up to each assistant turn to find boundaries
        prefix_messages = []
        for msg in messages:
            if msg["role"] == "assistant":
                # Tokenize everything before this assistant turn
                try:
                    prefix_text = tokenizer.apply_chat_template(
                        prefix_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception:
                    prefix_text = tokenizer.apply_chat_template(
                        prefix_messages,
                        tokenize=False,
                        add_generation_prompt=False,
                    )

                # Tokenize everything including this assistant turn
                prefix_plus_assistant = prefix_messages + [msg]
                full_up_to_here = tokenizer.apply_chat_template(
                    prefix_plus_assistant,
                    tokenize=False,
                    add_generation_prompt=False,
                )

                prefix_ids = tokenizer(
                    prefix_text, truncation=False, padding=False, return_tensors=None,
                )["input_ids"]
                full_up_to_here_ids = tokenizer(
                    full_up_to_here, truncation=False, padding=False, return_tensors=None,
                )["input_ids"]

                start_idx = len(prefix_ids)
                end_idx = min(len(full_up_to_here_ids), len(input_ids))

                # Unmask assistant tokens in labels
                for i in range(start_idx, end_idx):
                    labels[i] = input_ids[i]

            prefix_messages.append(msg)

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_labels.append(labels)

    # Handle case where all messages were skipped
    if len(all_input_ids) == 0:
        return {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "labels": all_labels,
    }


def preprocess_continue_pretrain(
    examples: Dict[str, List],
    tokenizer: AutoTokenizer,
    max_seq_length: int,
    text_column: str = "text",
) -> Dict[str, List]:
    """Preprocess raw text for continue-pretrain."""

    all_input_ids = []
    all_attention_mask = []
    all_labels = []

    texts = examples[text_column]

    for text in texts:
        # Skip empty text
        if not text or len(text.strip()) == 0:
            continue

        # Tokenize
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            return_tensors=None,
        )

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        # Create labels (same as input_ids for causal LM)
        labels = input_ids.copy()

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_labels.append(labels)

    # Handle case where all texts were skipped
    if len(all_input_ids) == 0:
        return {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "labels": all_labels,
    }


def pack_sequences_chunk(examples, max_seq_length: int, pad_token_id: int):
    """Pack sequences within a single chunk (for multiprocessing)."""
    all_input_ids = []
    all_attention_mask = []
    all_labels = []

    current_input_ids = []
    current_attention_mask = []
    current_labels = []
    current_length = 0

    # Process each example in the chunk
    for i in range(len(examples["input_ids"])):
        input_ids = examples["input_ids"][i]
        attention_mask = examples["attention_mask"][i]
        labels = examples["labels"][i]
        seq_length = len(input_ids)

        # If this sequence alone is longer than max_seq_length, truncate it
        if seq_length > max_seq_length:
            input_ids = input_ids[:max_seq_length]
            attention_mask = attention_mask[:max_seq_length]
            labels = labels[:max_seq_length]
            seq_length = max_seq_length

        # Check if adding this sequence would exceed max_seq_length
        if current_length + seq_length > max_seq_length:
            # Save current packed sequence (pad if necessary)
            if current_length > 0:
                # Pad to max_seq_length
                padding_length = max_seq_length - current_length
                if padding_length > 0:
                    current_input_ids.extend([pad_token_id] * padding_length)
                    current_attention_mask.extend([0] * padding_length)
                    current_labels.extend([-100] * padding_length)

                all_input_ids.append(current_input_ids)
                all_attention_mask.append(current_attention_mask)
                all_labels.append(current_labels)

            # Start new packed sequence
            current_input_ids = list(input_ids)
            current_attention_mask = list(attention_mask)
            current_labels = list(labels)
            current_length = seq_length
        else:
            # Add to current packed sequence
            current_input_ids.extend(input_ids)
            current_attention_mask.extend(attention_mask)
            current_labels.extend(labels)
            current_length += seq_length

    # Don't forget the last packed sequence in this chunk
    if current_length > 0:
        padding_length = max_seq_length - current_length
        if padding_length > 0:
            current_input_ids.extend([pad_token_id] * padding_length)
            current_attention_mask.extend([0] * padding_length)
            current_labels.extend([-100] * padding_length)

        all_input_ids.append(current_input_ids)
        all_attention_mask.append(current_attention_mask)
        all_labels.append(current_labels)

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "labels": all_labels,
    }


def pack_sequences(
    dataset: Dataset,
    max_seq_length: int,
    pad_token_id: int,
    num_proc: int = 64,
) -> Dataset:
    """
    Pack multiple sequences into single sequences of max_seq_length using multiprocessing.
    This significantly improves training efficiency by reducing padding waste.
    """
    logger.info(f"Packing sequences with max_seq_length={max_seq_length} using {num_proc} processes")

    # Use dataset.map with batched=True for multiprocessing
    packed_dataset = dataset.map(
        lambda examples: pack_sequences_chunk(examples, max_seq_length, pad_token_id),
        batched=True,
        batch_size=10000,  # Process 10k sequences at a time per worker
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc="Packing sequences",
    )

    logger.info(f"Packed {len(dataset)} sequences into {len(packed_dataset)} packed sequences")
    if len(packed_dataset) > 0:
        logger.info(f"Compression ratio: {len(dataset) / len(packed_dataset):.2f}x")

    return packed_dataset


def prepare_dataset(
    data_args: DataArguments,
    tokenizer: AutoTokenizer,
) -> tuple:
    """Load and prepare dataset."""

    # Load dataset from HuggingFace Hub or local disk
    if data_args.dataset_name:
        logger.info(f"Loading dataset from HuggingFace Hub: {data_args.dataset_name}")
        if data_args.data_dir:
            logger.info(f"  Using data_dir: {data_args.data_dir}")
        if data_args.dataset_revision:
            logger.info(f"  Using revision: {data_args.dataset_revision}")

        # Prepare load_dataset kwargs
        load_kwargs = {
            "path": data_args.dataset_name,
            "split": data_args.dataset_split,
        }
        if data_args.data_dir:
            load_kwargs["data_dir"] = data_args.data_dir
        if data_args.dataset_revision:
            load_kwargs["revision"] = data_args.dataset_revision

        dataset = load_dataset(**load_kwargs)
    elif data_args.dataset_path:
        logger.info(f"Loading dataset from local disk: {data_args.dataset_path}")
        dataset = load_from_disk(data_args.dataset_path)
    else:
        raise ValueError("Either dataset_name or dataset_path must be provided")

    # Handle different dataset structures
    if hasattr(dataset, 'keys') and 'train' in dataset.keys():
        # DatasetDict structure
        if 'validation' in dataset.keys() or 'test' in dataset.keys():
            train_dataset = dataset['train']
            eval_dataset = dataset.get('validation', dataset.get('test'))
        else:
            # Split the training set
            split = dataset['train'].train_test_split(
                test_size=data_args.validation_split,
                seed=42,
            )
            train_dataset = split['train']
            eval_dataset = split['test']
    else:
        # Single Dataset - need to split
        split = dataset.train_test_split(
            test_size=data_args.validation_split,
            seed=42,
        )
        train_dataset = split['train']
        eval_dataset = split['test']

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Eval dataset size: {len(eval_dataset)}")

    # Different preprocessing for continue-pretrain vs instruction tuning
    if data_args.is_continue_pretrain:
        logger.info("Continue-pretrain mode: preprocessing raw text...")

        # Filter out empty text
        train_dataset = train_dataset.filter(
            lambda x: x[data_args.text_column] is not None and len(x[data_args.text_column].strip()) > 0,
            num_proc=64,
            desc="Filtering empty train text",
        )
        eval_dataset = eval_dataset.filter(
            lambda x: x[data_args.text_column] is not None and len(x[data_args.text_column].strip()) > 0,
            num_proc=64,
            desc="Filtering empty eval text",
        )

        # Preprocess
        train_dataset = train_dataset.map(
            lambda x: preprocess_continue_pretrain(x, tokenizer, data_args.max_seq_length, data_args.text_column),
            batched=True,
            num_proc=64,
            remove_columns=train_dataset.column_names,
            desc="Preprocessing train dataset",
        )

        eval_dataset = eval_dataset.map(
            lambda x: preprocess_continue_pretrain(x, tokenizer, data_args.max_seq_length, data_args.text_column),
            batched=True,
            num_proc=64,
            remove_columns=eval_dataset.column_names,
            desc="Preprocessing eval dataset",
        )

    else:
        logger.info("Instruction tuning mode: preprocessing chat messages...")

        # Filter out samples with empty messages
        train_dataset = train_dataset.filter(
            lambda x: x["messages"] is not None and len(x["messages"]) > 0,
            num_proc=64,
            desc="Filtering empty train messages",
        )
        eval_dataset = eval_dataset.filter(
            lambda x: x["messages"] is not None and len(x["messages"]) > 0,
            num_proc=64,
            desc="Filtering empty eval messages",
        )

        # Preprocess
        train_dataset = train_dataset.map(
            lambda x: preprocess_function(x, tokenizer, data_args.max_seq_length),
            batched=True,
            num_proc=64,
            remove_columns=train_dataset.column_names,
            desc="Preprocessing train dataset",
        )

        eval_dataset = eval_dataset.map(
            lambda x: preprocess_function(x, tokenizer, data_args.max_seq_length),
            batched=True,
            num_proc=64,
            remove_columns=eval_dataset.column_names,
            desc="Preprocessing eval dataset",
        )

    # Filter out empty samples
    train_dataset = train_dataset.filter(
        lambda x: len(x["input_ids"]) > 0,
        num_proc=64,
        desc="Filtering empty train samples",
    )
    eval_dataset = eval_dataset.filter(
        lambda x: len(x["input_ids"]) > 0,
        num_proc=64,
        desc="Filtering empty eval samples",
    )

    logger.info(f"Train dataset size after filtering: {len(train_dataset)}")
    logger.info(f"Eval dataset size after filtering: {len(eval_dataset)}")

    # Apply packing if enabled
    if data_args.use_packing:
        logger.info("Applying sequence packing with multiprocessing...")
        train_dataset = pack_sequences(
            train_dataset,
            data_args.max_seq_length,
            tokenizer.pad_token_id,
            num_proc=64,
        )
        eval_dataset = pack_sequences(
            eval_dataset,
            data_args.max_seq_length,
            tokenizer.pad_token_id,
            num_proc=64,
        )
        logger.info(f"Train dataset size after packing: {len(train_dataset)}")
        logger.info(f"Eval dataset size after packing: {len(eval_dataset)}")

    return train_dataset, eval_dataset


def fix_saved_files(output_dir: str):
    """Fix config and tokenizer files for compatibility after saving."""
    import glob

    # Find all config.json files (including in checkpoints)
    config_files = glob.glob(os.path.join(output_dir, "**/config.json"), recursive=True)
    config_files.append(os.path.join(output_dir, "config.json"))

    for config_path in config_files:
        if not os.path.exists(config_path):
            continue

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            modified = False

            # Fix text_config.model_type: ministral3 -> mistral
            if config.get("text_config", {}).get("model_type") == "ministral3":
                config["text_config"]["model_type"] = "mistral"
                modified = True
                logger.info(f"Fixed text_config.model_type in {config_path}")

            if modified:
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to fix {config_path}: {e}")

    # Find all tokenizer_config.json files
    tokenizer_files = glob.glob(os.path.join(output_dir, "**/tokenizer_config.json"), recursive=True)
    tokenizer_files.append(os.path.join(output_dir, "tokenizer_config.json"))

    for tokenizer_path in tokenizer_files:
        if not os.path.exists(tokenizer_path):
            continue

        try:
            with open(tokenizer_path, 'r') as f:
                tokenizer_config = json.load(f)

            modified = False

            # Remove extra_special_tokens if it's a list (should be dict or not exist)
            if "extra_special_tokens" in tokenizer_config:
                if isinstance(tokenizer_config["extra_special_tokens"], list):
                    del tokenizer_config["extra_special_tokens"]
                    modified = True
                    logger.info(f"Removed invalid extra_special_tokens from {tokenizer_path}")

            if modified:
                with open(tokenizer_path, 'w') as f:
                    json.dump(tokenizer_config, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to fix {tokenizer_path}: {e}")

    logger.info("Finished fixing saved files for compatibility")


def main():
    """Main training function."""
    # Parse arguments
    parser = transformers.HfArgumentParser((
        ModelArguments,
        DataArguments,
        CustomTrainingArguments,
    ))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # Load from JSON config
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set seed
    set_seed(training_args.seed)

    # Setup logging
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Set up lr_scheduler for min_learning_rate support
    if hasattr(training_args, 'min_learning_rate') and training_args.min_learning_rate is not None:
        # Use cosine_with_min_lr scheduler instead of cosine
        training_args.lr_scheduler_type = "cosine_with_min_lr"
        min_lr_rate = training_args.min_learning_rate / training_args.learning_rate
        training_args.lr_scheduler_kwargs = {"min_lr_rate": min_lr_rate}
        logger.info(f"Using cosine_with_min_lr scheduler with min_learning_rate: {training_args.min_learning_rate} (min_lr_rate: {min_lr_rate:.4f})")

    # Log training configuration
    logger.info(f"Training arguments: {training_args}")
    logger.info(f"Model arguments: {model_args}")
    logger.info(f"Data arguments: {data_args}")

    # Setup tokenizer and model
    tokenizer = setup_tokenizer(model_args, data_args.is_continue_pretrain)
    model = setup_model(model_args, tokenizer, data_args.is_continue_pretrain)

    # Prepare dataset
    train_dataset, eval_dataset = prepare_dataset(data_args, tokenizer)

    # Data collator - use simpler collator for packed sequences
    if data_args.use_packing:
        # For packed sequences, all are same length, use default collator
        from transformers import default_data_collator
        data_collator = default_data_collator
        logger.info("Using default_data_collator for packed sequences")
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            pad_to_multiple_of=64,
            return_tensors="pt",
        )

    # Initialize trainer with early stopping
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,  # Changed from 'tokenizer' in transformers 5.0
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # Disabled for finetune
    )

    # Training
    # Check for resume_from_checkpoint
    resume_checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        resume_checkpoint = training_args.resume_from_checkpoint
        logger.info(f"Resuming training from checkpoint: {resume_checkpoint}")

    logger.info("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)

    # Save final model
    logger.info("Saving final model...")
    trainer.save_model()
    trainer.save_state()

    # Log metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # Evaluate
    logger.info("Running final evaluation...")
    eval_metrics = trainer.evaluate()
    eval_metrics["eval_samples"] = len(eval_dataset)
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    # Save tokenizer
    tokenizer.save_pretrained(training_args.output_dir)

    # Fix config and tokenizer files for compatibility
    fix_saved_files(training_args.output_dir)

    logger.info(f"Training complete! Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()
