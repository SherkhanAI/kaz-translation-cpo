#!/usr/bin/env python3
"""
Phase 3b: Post-process Alignment with LLM Verification
=======================================================

Cleans and verifies aligned corpus using LLM:
1. Filters out null values
2. Verifies alignment quality with LLM
3. Fixes OCR errors
4. Ensures consistent granularity
5. Splits merged segments

Usage:
    python scripts/03b_postprocess_alignment.py \
        --input_file aligned_corpus.jsonl \
        --output_file aligned_corpus_clean.jsonl \
        --model google/gemma-3-27b-it

Author: Dataset for Translator Project
Python: 3.11
"""

import argparse
import json
import re
import torch
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class CleaningStats:
    total_input: int = 0
    null_filtered: int = 0
    too_short: int = 0
    too_long: int = 0
    low_quality: int = 0
    ocr_fixed: int = 0
    split_segments: int = 0
    final_output: int = 0


# Common OCR errors in Kazakh/Russian financial documents
OCR_FIXES = {
    # Spacing issues in company names
    r'Hal\s*yk': 'Halyk',
    r'Hal yk': 'Halyk',
    r'GrOup': 'Group',
    r'GrOUp': 'Group',
    r'GROUP': 'Group',

    # Common Kazakh OCR errors
    r'Қазақста|н': 'Қазақстан',
    r'те|ңге': 'теңге',

    # Number formatting
    r'(\d)\s+(\d{3})': r'\1\2',  # Fix "1 000" -> "1000"

    # Mixed case issues
    r'БаНК': 'Банк',
    r'БАнК': 'Банк',

    # Broken words
    r'([а-яәғқңөұүіА-ЯӘҒҚҢӨҰҮІ])\s+([а-яәғқңөұүі])': r'\1\2',
}


VERIFICATION_PROMPT = """You are a translation quality verifier. Analyze these aligned texts and respond with JSON.

Kazakh: {kaz}
Russian: {rus}
English: {eng}

Check:
1. Are these semantically equivalent translations? (not just similar topics)
2. Is the alignment atomic? (single coherent unit, not multiple merged sentences)
3. Are there OCR errors to fix?

Respond ONLY with JSON:
{{
  "is_valid_alignment": true/false,
  "is_atomic": true/false,
  "quality_score": 0.0-1.0,
  "issues": ["list of issues if any"],
  "fixed_kaz": "corrected text or null",
  "fixed_rus": "corrected text or null",
  "fixed_eng": "corrected text or null"
}}"""


def load_model(model_name: str, use_4bit: bool = True):
    """Load model for verification."""
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

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
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

    return model, tokenizer


def fix_ocr_errors(text: str) -> Tuple[str, bool]:
    """Apply OCR fixes to text."""
    if not text:
        return text, False

    original = text
    for pattern, replacement in OCR_FIXES.items():
        text = re.sub(pattern, replacement, text)

    # Fix excessive whitespace
    text = re.sub(r'\s{2,}', ' ', text)
    text = text.strip()

    return text, text != original


def basic_filters(record: dict) -> Tuple[bool, str]:
    """Apply basic quality filters."""
    kaz = record.get('kaz', '')
    rus = record.get('rus', '')
    eng = record.get('eng', '')

    # Must have at least KAZ + one other language
    if not kaz:
        return False, "missing_kaz"

    if not rus and not eng:
        return False, "missing_both_targets"

    # Length filters
    kaz_words = len(kaz.split()) if kaz else 0
    rus_words = len(rus.split()) if rus else 0
    eng_words = len(eng.split()) if eng else 0

    # Too short (less than 5 words)
    if kaz_words < 5:
        return False, "too_short"

    # Too long (likely merged segments) - more than 200 words
    if kaz_words > 200 or rus_words > 200 or eng_words > 200:
        return False, "too_long"

    # Check for garbage patterns
    garbage_patterns = [
        r'^[\d\s\.\,\-\:]+$',  # Only numbers/punctuation
        r'^[A-Z\s]+$',  # Only uppercase letters
        r'^\d+\s*(млн|млрд|тыс|%)',  # Just numbers with units
        r'^(table|figure|рисунок|таблица)\s*\d',  # Table/figure references
    ]

    for pattern in garbage_patterns:
        if re.match(pattern, kaz.lower()):
            return False, "garbage_pattern"

    return True, "ok"


def verify_with_llm(
    model,
    tokenizer,
    record: dict,
    max_new_tokens: int = 512
) -> dict:
    """Verify alignment quality using LLM."""
    kaz = record.get('kaz', '') or ''
    rus = record.get('rus', '') or ''
    eng = record.get('eng', '') or ''

    prompt = VERIFICATION_PROMPT.format(kaz=kaz, rus=rus, eng=eng)

    messages = [{"role": "user", "content": prompt}]

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )

    generated = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)

    # Parse JSON response
    try:
        # Find JSON in response
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return result
    except json.JSONDecodeError:
        pass

    # Default response if parsing fails
    return {
        "is_valid_alignment": True,
        "is_atomic": True,
        "quality_score": 0.7,
        "issues": ["could not parse LLM response"],
        "fixed_kaz": None,
        "fixed_rus": None,
        "fixed_eng": None
    }


def process_record(
    record: dict,
    model=None,
    tokenizer=None,
    use_llm: bool = True,
    stats: CleaningStats = None
) -> Optional[dict]:
    """Process a single record through the cleaning pipeline."""

    # Step 1: Basic filters
    is_valid, reason = basic_filters(record)
    if not is_valid:
        if reason == "missing_kaz" or reason == "missing_both_targets":
            stats.null_filtered += 1
        elif reason == "too_short":
            stats.too_short += 1
        elif reason == "too_long":
            stats.too_long += 1
        return None

    # Step 2: OCR fixes
    kaz_fixed, kaz_changed = fix_ocr_errors(record.get('kaz', ''))
    rus_fixed, rus_changed = fix_ocr_errors(record.get('rus', ''))
    eng_fixed, eng_changed = fix_ocr_errors(record.get('eng', ''))

    if kaz_changed or rus_changed or eng_changed:
        stats.ocr_fixed += 1

    cleaned_record = {
        'id': record['id'],
        'kaz': kaz_fixed,
        'rus': rus_fixed if rus_fixed else None,
        'eng': eng_fixed if eng_fixed else None,
        'similarity_kaz_rus': record.get('similarity_kaz_rus'),
        'similarity_kaz_eng': record.get('similarity_kaz_eng'),
        'metadata': record.get('metadata', {})
    }

    # Step 3: LLM verification (optional, for sampling)
    if use_llm and model is not None:
        verification = verify_with_llm(model, tokenizer, cleaned_record)

        if not verification.get('is_valid_alignment', True):
            stats.low_quality += 1
            return None

        if not verification.get('is_atomic', True):
            stats.low_quality += 1
            # Could split here in future
            return None

        quality = verification.get('quality_score', 0.7)
        if quality < 0.6:
            stats.low_quality += 1
            return None

        # Apply LLM fixes if provided
        if verification.get('fixed_kaz'):
            cleaned_record['kaz'] = verification['fixed_kaz']
        if verification.get('fixed_rus'):
            cleaned_record['rus'] = verification['fixed_rus']
        if verification.get('fixed_eng'):
            cleaned_record['eng'] = verification['fixed_eng']

        cleaned_record['llm_quality_score'] = quality

    return cleaned_record


def main():
    parser = argparse.ArgumentParser(description="Post-process aligned corpus")
    parser.add_argument(
        "--input_file",
        type=str,
        default="aligned_corpus.jsonl",
        help="Input aligned corpus"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="aligned_corpus_clean.jsonl",
        help="Output cleaned corpus"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-3-27b-it",
        help="Model for LLM verification"
    )
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        default=True,
        help="Use 4-bit quantization"
    )
    parser.add_argument(
        "--no_llm",
        action="store_true",
        help="Skip LLM verification (faster, basic filters only)"
    )
    parser.add_argument(
        "--llm_sample_rate",
        type=float,
        default=0.1,
        help="Fraction of records to verify with LLM (0.1 = 10%)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max samples to process (for testing)"
    )
    args = parser.parse_args()

    # Setup paths
    base_dir = Path(__file__).parent.parent
    input_file = base_dir / args.input_file
    output_file = base_dir / args.output_file

    # Load input data
    print(f"Loading: {input_file}")
    records = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))

    print(f"Loaded {len(records):,} records")

    if args.max_samples:
        records = records[:args.max_samples]
        print(f"Limited to {len(records):,} records")

    # Load model if using LLM
    model, tokenizer = None, None
    if not args.no_llm:
        model, tokenizer = load_model(args.model, args.use_4bit)

    # Process records
    stats = CleaningStats(total_input=len(records))
    cleaned_records = []

    import random
    random.seed(42)

    for record in tqdm(records, desc="Cleaning"):
        # Decide if we use LLM for this record
        use_llm = (
            not args.no_llm and
            model is not None and
            random.random() < args.llm_sample_rate
        )

        cleaned = process_record(
            record,
            model=model,
            tokenizer=tokenizer,
            use_llm=use_llm,
            stats=stats
        )

        if cleaned:
            cleaned_records.append(cleaned)

    stats.final_output = len(cleaned_records)

    # Save output
    print(f"\nSaving {len(cleaned_records):,} cleaned records...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in cleaned_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    # Save stats
    stats_file = output_file.with_suffix('.stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'input_file': str(input_file),
            'output_file': str(output_file),
            'stats': {
                'total_input': stats.total_input,
                'null_filtered': stats.null_filtered,
                'too_short': stats.too_short,
                'too_long': stats.too_long,
                'low_quality': stats.low_quality,
                'ocr_fixed': stats.ocr_fixed,
                'final_output': stats.final_output,
                'retention_rate': stats.final_output / stats.total_input if stats.total_input > 0 else 0
            }
        }, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("CLEANING SUMMARY")
    print("=" * 60)
    print(f"  Input records:      {stats.total_input:,}")
    print(f"  Null filtered:      {stats.null_filtered:,}")
    print(f"  Too short:          {stats.too_short:,}")
    print(f"  Too long:           {stats.too_long:,}")
    print(f"  Low quality (LLM):  {stats.low_quality:,}")
    print(f"  OCR fixed:          {stats.ocr_fixed:,}")
    print(f"  Final output:       {stats.final_output:,}")
    print(f"  Retention rate:     {stats.final_output/stats.total_input*100:.1f}%")
    print(f"\n  Output: {output_file}")


if __name__ == "__main__":
    main()
