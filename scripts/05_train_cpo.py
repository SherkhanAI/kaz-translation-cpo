#!/usr/bin/env python3
"""
Phase 5: CPO Training
=====================

Train the translation model using CPO (Contrastive Preference Optimization).

Usage:
    python scripts/05_train_cpo.py \
        --dataset cpo_training_data.jsonl \
        --base_model Qwen/Qwen2.5-7B-Instruct \
        --output_dir models/qwen-kaz-translator-cpo

Author: Dataset for Translator Project
Python: 3.11
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import CPOConfig, CPOTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def load_cpo_dataset(file_path: str) -> Dataset:
    """Load CPO dataset from JSONL file."""
    dataset = load_dataset("json", data_files=file_path, split="train")
    return dataset


def prepare_model_for_training(
    model_name: str,
    use_lora: bool = True,
    use_4bit: bool = False
):
    """
    Prepare model for CPO training with optional LoRA and quantization.
    """
    print(f"Loading model: {model_name}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Quantization config
    if use_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

    # LoRA config
    if use_lora:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train model with CPO")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cpo_training_data.jsonl",
        help="CPO dataset file"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model to fine-tune"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/qwen-kaz-translator-cpo",
        help="Output directory for trained model"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Per-device batch size"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Learning rate"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="CPO beta parameter (contrastive weight)"
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        default=True,
        help="Use LoRA for efficient training"
    )
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        help="Use 4-bit quantization"
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable Weights & Biases logging"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples for testing"
    )
    args = parser.parse_args()

    # Setup paths
    base_dir = Path(__file__).parent.parent
    dataset_path = base_dir / args.dataset
    output_dir = base_dir / args.output_dir

    # Load dataset
    print(f"Loading dataset from: {dataset_path}")
    dataset = load_cpo_dataset(str(dataset_path))
    print(f"Dataset size: {len(dataset):,} examples")

    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
        print(f"Limited to {len(dataset):,} samples")

    # Split dataset
    train_test = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = train_test["train"]
    eval_dataset = train_test["test"]

    print(f"Train size: {len(train_dataset):,}")
    print(f"Eval size:  {len(eval_dataset):,}")

    # Load model
    model, tokenizer = prepare_model_for_training(
        args.base_model,
        use_lora=args.use_lora,
        use_4bit=args.use_4bit
    )

    # CPO Config
    cpo_config = CPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        # Learning rate schedule
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,

        # CPO specific
        beta=args.beta,
        label_smoothing=0.0,
        loss_type="sigmoid",

        # Memory optimization
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if args.use_4bit else "adamw_torch",

        # Logging
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="epoch",

        # Mixed precision
        bf16=True,

        # Weights & Biases
        report_to="wandb" if not args.no_wandb else "none",
        run_name=f"qwen-kaz-cpo-{datetime.now().strftime('%Y%m%d-%H%M')}",

        # Other
        remove_unused_columns=False,
        max_length=1024,
        max_prompt_length=512,
    )

    # Trainer
    trainer = CPOTrainer(
        model=model,
        args=cpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting CPO Training")
    print("=" * 60)
    print(f"  Base model:     {args.base_model}")
    print(f"  Epochs:         {args.num_epochs}")
    print(f"  Batch size:     {args.batch_size} x {args.gradient_accumulation_steps} = {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  Learning rate:  {args.learning_rate}")
    print(f"  CPO beta:       {args.beta}")
    print(f"  LoRA:           {args.use_lora}")
    print(f"  4-bit:          {args.use_4bit}")
    print("=" * 60 + "\n")

    trainer.train()

    # Save final model
    print("\nSaving final model...")
    trainer.save_model(str(output_dir / "final"))

    # Save training metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "base_model": args.base_model,
        "dataset": str(dataset_path),
        "train_size": len(train_dataset),
        "eval_size": len(eval_dataset),
        "config": {
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "beta": args.beta,
            "use_lora": args.use_lora,
            "use_4bit": args.use_4bit
        }
    }

    with open(output_dir / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nTraining complete!")
    print(f"Model saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
