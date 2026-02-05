#!/usr/bin/env python3
"""
Phase 2: Sentence Chunking
==========================

Splits extracted markdown files into sentences.
Handles Kazakh, Russian, and English language specifics.

Usage:
    python scripts/02_chunk_sentences.py --input_dir extracted_markdown --output_dir chunked_sentences

Author: Dataset for Translator Project
Python: 3.11
"""

import argparse
import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
from tqdm import tqdm


# Common abbreviations that don't end sentences
ABBREVIATIONS = {
    # Russian/Kazakh
    "т.е", "т.б", "т.с.с", "т.д", "др", "пр", "г", "гг", "в",
    "млн", "млрд", "тыс", "руб", "тг", "долл", "евро",
    "ул", "пр-т", "д", "корп", "стр", "оф", "кв",
    "им", "проф", "акад", "доц", "канд", "д-р",
    "рис", "табл", "гл", "разд", "п", "пп", "ст",
    # English
    "Mr", "Mrs", "Ms", "Dr", "Prof", "Inc", "Ltd", "Corp",
    "vs", "etc", "i.e", "e.g", "fig", "no", "vol"
}

# Minimum sentence length (in characters)
MIN_SENTENCE_LENGTH = 20

# Maximum sentence length (likely extraction error if exceeded)
MAX_SENTENCE_LENGTH = 2000


def detect_language(filename: str) -> str:
    """
    Detect language from filename.
    Expected format: YYYY_Type_LANG.md
    """
    filename_upper = filename.upper()

    if "_KAZ" in filename_upper:
        return "kaz"
    elif "_RUS" in filename_upper:
        return "rus"
    elif "_ENG" in filename_upper:
        return "eng"
    else:
        return "unknown"


def is_abbreviation(word: str) -> bool:
    """
    Check if a word is a known abbreviation.
    """
    # Remove trailing period
    word_clean = word.rstrip('.')
    return word_clean.lower() in {abbr.lower() for abbr in ABBREVIATIONS}


def split_sentences_regex(text: str) -> List[str]:
    """
    Split text into sentences using regex.
    Handles abbreviations and special cases.
    """
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    # Protect abbreviations by replacing their periods
    protected_text = text
    for abbr in ABBREVIATIONS:
        # Match abbreviation followed by period and space/end
        pattern = rf'\b{re.escape(abbr)}\.(?=\s|$)'
        protected_text = re.sub(pattern, f'{abbr}<DOT>', protected_text, flags=re.IGNORECASE)

    # Protect numbers with periods (like "2.5" or "п.1")
    protected_text = re.sub(r'(\d)\.(\d)', r'\1<DOT>\2', protected_text)

    # Split on sentence-ending punctuation followed by space and capital letter
    # or followed by end of string
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-ZА-ЯӘҒҚҢӨҰҮІa-zа-яәғқңөұүі\d])|(?<=[.!?])$'

    # Split
    raw_sentences = re.split(sentence_pattern, protected_text)

    # Restore protected periods
    sentences = []
    for sent in raw_sentences:
        sent = sent.replace('<DOT>', '.')
        sent = sent.strip()

        # Skip empty or too short
        if len(sent) < MIN_SENTENCE_LENGTH:
            continue

        # Skip too long (likely extraction error)
        if len(sent) > MAX_SENTENCE_LENGTH:
            # Try to split further
            sub_sentences = sent.split('. ')
            for sub in sub_sentences:
                sub = sub.strip()
                if MIN_SENTENCE_LENGTH <= len(sub) <= MAX_SENTENCE_LENGTH:
                    sentences.append(sub)
            continue

        sentences.append(sent)

    return sentences


def clean_sentence(sentence: str) -> str:
    """
    Clean a sentence for further processing.
    """
    # Remove markdown formatting
    sentence = re.sub(r'\*\*|__|~~|`', '', sentence)

    # Remove markdown links but keep text
    sentence = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', sentence)

    # Remove markdown headers
    sentence = re.sub(r'^#+\s*', '', sentence)

    # Remove bullet points
    sentence = re.sub(r'^[\-\*\+]\s*', '', sentence)
    sentence = re.sub(r'^\d+\.\s*', '', sentence)

    # Normalize whitespace
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence.strip()


def is_valid_sentence(sentence: str, language: str) -> bool:
    """
    Check if a sentence is valid for inclusion in the dataset.
    """
    # Too short
    if len(sentence) < MIN_SENTENCE_LENGTH:
        return False

    # Too few words
    words = sentence.split()
    if len(words) < 3:
        return False

    # Contains too many numbers (likely a table row)
    num_count = len(re.findall(r'\d+', sentence))
    if num_count > 10:
        return False

    # Is mostly numbers/symbols
    alpha_chars = len(re.findall(r'[a-zA-Zа-яА-ЯәғқңөұүіӘҒҚҢӨҰҮІ]', sentence))
    if alpha_chars < len(sentence) * 0.5:
        return False

    # Check for common garbage patterns
    garbage_patterns = [
        r'^[\d\s\.\,\-]+$',  # Only numbers and punctuation
        r'^[A-Z\s]+$',  # All caps (likely header)
        r'©|®|™',  # Copyright symbols
        r'www\.|http',  # URLs
    ]
    for pattern in garbage_patterns:
        if re.search(pattern, sentence):
            return False

    return True


def extract_document_metadata(filename: str) -> dict:
    """
    Extract metadata from filename.
    Expected format: YYYY_Type_LANG.md
    """
    stem = Path(filename).stem
    parts = stem.split('_')

    metadata = {
        "year": None,
        "type": None,
        "language": None,
        "company": None
    }

    # Try to extract year
    for part in parts:
        if re.match(r'^\d{4}$', part):
            metadata["year"] = part
            break
        if re.match(r'^\d{4}-\d{4}$', part):
            metadata["year"] = part
            break

    # Detect language
    metadata["language"] = detect_language(filename)

    # Detect document type
    type_patterns = {
        "Annual_Report": ["annual", "report", "годовой", "отчет", "жылдық"],
        "Sustainability_Report": ["sustainab", "устойчив", "ESG"],
        "Research_Program": ["research", "исследов", "program", "программ"]
    }
    for doc_type, patterns in type_patterns.items():
        if any(p.lower() in filename.lower() for p in patterns):
            metadata["type"] = doc_type
            break

    # Detect company
    company_patterns = {
        "Halyk": ["halyk", "халык"],
        "Kazahtelekom": ["telecom", "телеком", "kaz_telecom"],
        "Baiterek": ["baiterek", "байтерек"],
        "KEGOC": ["kegoc", "кегок"],
        "National_Bank": ["national", "bank", "нацбанк", "национальн"]
    }
    for company, patterns in company_patterns.items():
        if any(p.lower() in filename.lower() for p in patterns):
            metadata["company"] = company
            break

    return metadata


def process_file(file_path: Path) -> Tuple[List[dict], dict]:
    """
    Process a single markdown file and extract sentences.
    Returns list of sentence dicts and statistics.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Extract metadata
    metadata = extract_document_metadata(file_path.name)
    language = metadata["language"]

    # Split into sentences
    raw_sentences = split_sentences_regex(text)

    # Clean and filter
    sentences = []
    stats = {
        "raw_count": len(raw_sentences),
        "valid_count": 0,
        "filtered_count": 0
    }

    for idx, sent in enumerate(raw_sentences):
        cleaned = clean_sentence(sent)

        if is_valid_sentence(cleaned, language):
            sentences.append({
                "text": cleaned,
                "source_file": file_path.name,
                "sentence_idx": idx,
                "language": language,
                "metadata": metadata
            })
            stats["valid_count"] += 1
        else:
            stats["filtered_count"] += 1

    return sentences, stats


def main():
    parser = argparse.ArgumentParser(description="Split extracted text into sentences")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="extracted_markdown",
        help="Input directory with markdown files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="chunked_sentences",
        help="Output directory for sentence files"
    )
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["jsonl", "json"],
        default="jsonl",
        help="Output format"
    )
    args = parser.parse_args()

    # Setup paths
    base_dir = Path(__file__).parent.parent
    input_dir = base_dir / args.input_dir
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(exist_ok=True)

    if not input_dir.exists():
        print(f"[ERROR] Input directory not found: {input_dir}")
        return

    # Collect markdown files
    md_files = list(input_dir.glob("*.md"))
    print(f"Found {len(md_files)} markdown files")
    print("=" * 60)

    # Process files by language
    all_sentences = {
        "kaz": [],
        "rus": [],
        "eng": [],
        "unknown": []
    }

    total_stats = {
        "total_files": len(md_files),
        "total_raw": 0,
        "total_valid": 0,
        "total_filtered": 0
    }

    for file_path in tqdm(md_files, desc="Processing files"):
        sentences, stats = process_file(file_path)

        # Update totals
        total_stats["total_raw"] += stats["raw_count"]
        total_stats["total_valid"] += stats["valid_count"]
        total_stats["total_filtered"] += stats["filtered_count"]

        # Group by language
        for sent in sentences:
            lang = sent["language"]
            all_sentences[lang].append(sent)

        print(f"  {file_path.name}: {stats['valid_count']} sentences")

    # Save output files
    print("\n" + "=" * 60)
    print("Saving output files...")

    for lang, sentences in all_sentences.items():
        if not sentences:
            continue

        output_file = output_dir / f"sentences_{lang}.jsonl"

        with open(output_file, 'w', encoding='utf-8') as f:
            for sent in sentences:
                f.write(json.dumps(sent, ensure_ascii=False) + '\n')

        print(f"  {lang.upper()}: {len(sentences):,} sentences -> {output_file.name}")

    # Save combined metadata
    metadata_file = output_dir / "chunking_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "stats": total_stats,
            "sentences_by_language": {
                lang: len(sents) for lang, sents in all_sentences.items()
            }
        }, f, ensure_ascii=False, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("CHUNKING SUMMARY")
    print("=" * 60)
    print(f"  Total files:      {total_stats['total_files']}")
    print(f"  Raw sentences:    {total_stats['total_raw']:,}")
    print(f"  Valid sentences:  {total_stats['total_valid']:,}")
    print(f"  Filtered out:     {total_stats['total_filtered']:,}")
    print()
    print("  By language:")
    for lang, sentences in all_sentences.items():
        if sentences:
            print(f"    {lang.upper()}: {len(sentences):,}")
    print(f"\n  Output dir: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
