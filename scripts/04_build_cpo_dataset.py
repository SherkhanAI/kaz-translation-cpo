#!/usr/bin/env python3
"""
Phase 4: Build CPO Dataset
==========================

Generates rejected translations using base model and creates
CPO (Contrastive Preference Optimization) training dataset.

Usage:
    python scripts/04_build_cpo_dataset.py \
        --input_file aligned_corpus.jsonl \
        --output_file cpo_training_data.jsonl \
        --base_model Qwen/Qwen2.5-7B-Instruct \
        --directions kaz2rus kaz2eng

Author: Dataset for Translator Project
Python: 3.11
"""

import argparse
import json
import torch
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Generator
from dataclasses import dataclass
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class GenerationConfig:
    """Configuration for rejected sample generation."""
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    batch_size: int = 4


# Translation prompt templates
PROMPT_TEMPLATES = {
    "kaz2rus": """Translate the following Kazakh text to Russian. Provide only the translation without explanations.

Kazakh: {source}

Russian:""",

    "kaz2eng": """Translate the following Kazakh text to English. Provide only the translation without explanations.

Kazakh: {source}

English:""",

    "rus2kaz": """Translate the following Russian text to Kazakh. Provide only the translation without explanations.

Russian: {source}

Kazakh:""",

    "rus2eng": """Translate the following Russian text to English. Provide only the translation without explanations.

Russian: {source}

English:""",

    "eng2kaz": """Translate the following English text to Kazakh. Provide only the translation without explanations.

English: {source}

Kazakh:""",

    "eng2rus": """Translate the following English text to Russian. Provide only the translation without explanations.

English: {source}

Russian:"""
}


def load_aligned_corpus(file_path: Path) -> List[dict]:
    """Load aligned corpus from JSONL file."""
    triplets = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            triplets.append(json.loads(line))
    return triplets


def load_model_and_tokenizer(model_name: str, use_4bit: bool = False):
    """
    Load model and tokenizer with optional quantization.
    """
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if use_4bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded on: {model.device}")
    return model, tokenizer


def generate_translation(
    model,
    tokenizer,
    prompt: str,
    config: GenerationConfig
) -> str:
    """Generate translation using the model."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=config.do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode only the generated part
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    translation = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Clean up
    translation = translation.strip()

    # Remove any trailing prompts or artifacts
    for stop_word in ["\n\n", "Kazakh:", "Russian:", "English:"]:
        if stop_word in translation:
            translation = translation.split(stop_word)[0].strip()

    return translation


def generate_batch_translations(
    model,
    tokenizer,
    prompts: List[str],
    config: GenerationConfig
) -> List[str]:
    """Generate translations for a batch of prompts."""
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=config.do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    translations = []
    for i, output in enumerate(outputs):
        # Decode only the generated part
        input_length = inputs['input_ids'][i].ne(tokenizer.pad_token_id).sum()
        generated_ids = output[input_length:]
        translation = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Clean up
        translation = translation.strip()
        for stop_word in ["\n\n", "Kazakh:", "Russian:", "English:"]:
            if stop_word in translation:
                translation = translation.split(stop_word)[0].strip()

        translations.append(translation)

    return translations


def create_cpo_examples(
    triplet: dict,
    directions: List[str],
    rejected_translations: Dict[str, str]
) -> List[dict]:
    """
    Create CPO training examples from a triplet.

    Returns list of CPO examples with:
    - prompt: Translation instruction
    - chosen: Official translation (ground truth)
    - rejected: Model-generated translation (baseline)
    """
    examples = []

    for direction in directions:
        source_lang, target_lang = direction.split("2")

        # Get source and target texts
        source_text = triplet.get(source_lang)
        target_text = triplet.get(target_lang)  # This is our "chosen"

        if not source_text or not target_text:
            continue

        # Get rejected translation
        rejected_key = f"{direction}_{triplet['id']}"
        rejected_text = rejected_translations.get(rejected_key)

        if not rejected_text:
            continue

        # Skip if rejected is too similar to chosen (model already good)
        if rejected_text.strip().lower() == target_text.strip().lower():
            continue

        # Create prompt
        prompt = PROMPT_TEMPLATES[direction].format(source=source_text)

        example = {
            "id": f"{triplet['id']}_{direction}",
            "prompt": prompt,
            "chosen": target_text,
            "rejected": rejected_text,
            "direction": direction,
            "source_text": source_text,
            "metadata": triplet.get("metadata", {})
        }

        examples.append(example)

    return examples


def batch_generator(items: List, batch_size: int) -> Generator:
    """Generate batches from a list."""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def main():
    parser = argparse.ArgumentParser(description="Build CPO training dataset")
    parser.add_argument(
        "--input_file",
        type=str,
        default="aligned_corpus.jsonl",
        help="Input aligned corpus file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="cpo_training_data.jsonl",
        help="Output CPO dataset file"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model for generating rejected samples"
    )
    parser.add_argument(
        "--directions",
        nargs="+",
        default=["kaz2rus", "kaz2eng"],
        choices=list(PROMPT_TEMPLATES.keys()),
        help="Translation directions to include"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for generation"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of triplets to process (for testing)"
    )
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        help="Use 4-bit quantization to reduce memory usage"
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Resume from a partial output file"
    )
    args = parser.parse_args()

    # Setup paths
    base_dir = Path(__file__).parent.parent
    input_file = base_dir / args.input_file
    output_file = base_dir / args.output_file

    # Load aligned corpus
    print(f"Loading aligned corpus from: {input_file}")
    triplets = load_aligned_corpus(input_file)
    print(f"Loaded {len(triplets):,} aligned triplets")

    if args.max_samples:
        triplets = triplets[:args.max_samples]
        print(f"Limited to {len(triplets):,} triplets for testing")

    # Check for resume
    processed_ids = set()
    if args.resume_from and Path(args.resume_from).exists():
        print(f"Resuming from: {args.resume_from}")
        with open(args.resume_from, 'r', encoding='utf-8') as f:
            for line in f:
                example = json.loads(line)
                processed_ids.add(example['id'].rsplit('_', 1)[0])
        print(f"Already processed: {len(processed_ids):,} triplets")
        triplets = [t for t in triplets if t['id'] not in processed_ids]
        print(f"Remaining: {len(triplets):,} triplets")

    if len(triplets) == 0:
        print("No triplets to process. Exiting.")
        return

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.base_model, args.use_4bit)

    # Generation config
    gen_config = GenerationConfig(
        temperature=args.temperature,
        batch_size=args.batch_size
    )

    # Generate rejected translations and create CPO examples
    print("\n" + "=" * 60)
    print(f"Generating rejected translations for directions: {args.directions}")
    print("=" * 60)

    all_cpo_examples = []
    stats = {
        "total_triplets": len(triplets),
        "total_examples": 0,
        "by_direction": {d: 0 for d in args.directions}
    }

    # Process in batches
    for batch_triplets in tqdm(
        list(batch_generator(triplets, gen_config.batch_size)),
        desc="Processing batches"
    ):
        rejected_translations = {}

        # Generate rejected for each direction
        for direction in args.directions:
            source_lang, _ = direction.split("2")

            # Prepare prompts
            prompts = []
            triplet_ids = []

            for triplet in batch_triplets:
                source_text = triplet.get(source_lang)
                if source_text:
                    prompt = PROMPT_TEMPLATES[direction].format(source=source_text)
                    prompts.append(prompt)
                    triplet_ids.append(triplet['id'])

            if not prompts:
                continue

            # Generate translations
            translations = generate_batch_translations(
                model, tokenizer, prompts, gen_config
            )

            # Store rejected translations
            for triplet_id, translation in zip(triplet_ids, translations):
                rejected_key = f"{direction}_{triplet_id}"
                rejected_translations[rejected_key] = translation

        # Create CPO examples
        for triplet in batch_triplets:
            examples = create_cpo_examples(triplet, args.directions, rejected_translations)
            all_cpo_examples.extend(examples)

            for ex in examples:
                stats["by_direction"][ex["direction"]] += 1

        # Periodic save (every 100 batches)
        if len(all_cpo_examples) % (gen_config.batch_size * 100) < gen_config.batch_size:
            temp_file = output_file.with_suffix('.partial.jsonl')
            with open(temp_file, 'w', encoding='utf-8') as f:
                for ex in all_cpo_examples:
                    f.write(json.dumps(ex, ensure_ascii=False) + '\n')
            print(f"\n  [Checkpoint] Saved {len(all_cpo_examples):,} examples")

    stats["total_examples"] = len(all_cpo_examples)

    # Save final output
    print(f"\nSaving {len(all_cpo_examples):,} CPO examples...")

    # Append to existing file if resuming
    mode = 'a' if args.resume_from else 'w'
    with open(output_file, mode, encoding='utf-8') as f:
        for example in all_cpo_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

    # Save metadata
    metadata_file = output_file.with_suffix('.meta.json')
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "base_model": args.base_model,
            "directions": args.directions,
            "generation_config": {
                "temperature": gen_config.temperature,
                "top_p": gen_config.top_p,
                "max_new_tokens": gen_config.max_new_tokens
            },
            "stats": stats
        }, f, ensure_ascii=False, indent=2)

    # Clean up partial file
    temp_file = output_file.with_suffix('.partial.jsonl')
    if temp_file.exists():
        temp_file.unlink()

    # Print summary
    print("\n" + "=" * 60)
    print("CPO DATASET SUMMARY")
    print("=" * 60)
    print(f"  Total triplets:     {stats['total_triplets']:,}")
    print(f"  Total CPO examples: {stats['total_examples']:,}")
    print()
    print("  By direction:")
    for direction, count in stats["by_direction"].items():
        print(f"    {direction}: {count:,}")
    print(f"\n  Output file: {output_file.absolute()}")
    print(f"  Metadata:    {metadata_file.absolute()}")


if __name__ == "__main__":
    main()
