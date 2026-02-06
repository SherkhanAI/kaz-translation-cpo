#!/usr/bin/env python3
"""
Phase 3b: Enhanced Post-processing (v2)
========================================

Comprehensive cleaning addressing all issues from Gemini analysis:
1. Aggressive text cleaning: dehyphenation, case fixes, spacing
2. TOC and noise removal: page numbers, headers, artifacts
3. Quality filtering: length, similarity, content
4. Deduplication: remove identical translations
5. Metadata enrichment: extract company names

Fixes from Gemini feedback:
- "денежно-кредитную полити - ку" -> "денежно-кредитную политику"
- "уст ОйчивОм" -> "устойчивом"
- "НарОДНЫй" -> "Народный"
- Remove TOC entries, page numbers, merged segments
- Extract company metadata

Usage:
    # Fast mode (no LLM):
    python scripts/03b_postprocess_v2.py \
        --input_file aligned_corpus.jsonl \
        --output_file aligned_corpus_clean.jsonl \
        --min_similarity 0.80

    # Strict mode:
    python scripts/03b_postprocess_v2.py \
        --input_file aligned_corpus.jsonl \
        --output_file aligned_corpus_clean.jsonl \
        --min_similarity 0.85 \
        --min_words 5 \
        --strict

Author: Dataset for Translator Project (Enhanced v2)
Python: 3.11
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter
from dataclasses import dataclass, asdict
from tqdm import tqdm


@dataclass
class CleaningConfig:
    """Configuration for cleaning process."""
    min_similarity: float = 0.80
    min_words: int = 3
    max_words: int = 300
    min_chars: int = 15
    max_length_ratio: float = 3.0
    remove_duplicates: bool = True
    fix_hyphenation: bool = True
    fix_spacing: bool = True
    fix_case: bool = True
    normalize_unicode: bool = True
    extract_company: bool = True
    remove_toc: bool = True
    remove_page_numbers: bool = True
    strict_mode: bool = False  # More aggressive filtering


# Known company names for extraction
COMPANY_PATTERNS = {
    "halyk": "Halyk Bank",
    "kegoc": "KEGOC",
    "казахтелеком": "Казахтелеком",
    "kazakhtelecom": "Казахтелеком",
    "air_astana": "Air Astana",
    "kaspi": "Kaspi",
    "kcell": "Kcell",
    "kmg": "KMG",
    "samruk": "Samruk-Kazyna",
}


class TextCleaner:
    """Comprehensive text cleaning engine."""

    def __init__(self, config: CleaningConfig):
        self.config = config

        # Pattern for broken hyphens from PDF line breaks
        # Matches: "денежно-кредитную полити - ку" -> "денежно-кредитную политику"
        self.hyphen_pattern = re.compile(
            r'(\w+)\s*-\s*\n?\s*(\w+)',
            re.UNICODE | re.MULTILINE
        )

        # Pattern for page numbers and navigation
        self.page_pattern = re.compile(
            r'^\s*\d+\s*$|'  # Just a number alone
            r'(?:стр\.|page|бет)\s*\d+|'  # Page N
            r'^\s*[A-ZА-ЯӘІҢҒҮҰҚӨҺ\s]{2,50}\s*\d+\s*$|'  # TITLE 123
            r'^\d+\s*/\s*\d+$'  # 1/100
        )

        # Pattern for table of contents entries
        self.toc_pattern = re.compile(
            r'(?:содержание|content|мазмұны)|'  # TOC keywords
            r'^\s*\d+\.\d+\s+|'  # 1.1 at start
            r'\.{3,}|'  # Dot leaders ...
            r'^\s*\d+\s+[А-ЯA-ZӘІҢҒҮҰҚӨҺا-ي]'  # Number followed by capital
        )

        # Pattern for weird spacing (broken words)
        # Matches: "у правление" -> "управление"
        self.broken_word_pattern = re.compile(
            r'\b([а-яА-ЯәіңғүұқөһӘІҢҒҮҰҚӨҺa-zA-Z])\s+([а-яәіңғүұқөһa-z]{2,})\b',
            re.UNICODE
        )

        # Pattern for mixed/broken case in single word
        # Matches: "уст ОйчивОм", "НарОДНЫй", "КазаХ стаНа"
        self.broken_case_pattern = re.compile(
            r'[а-яәіңғүұқөһa-z]+[А-ЯӘІҢҒҮҰҚӨҺA-Z]+[а-яәіңғүұқөһa-z]+|'
            r'[А-ЯӘІҢҒҮҰҚӨҺA-Z]+[а-яәіңғүұқөһa-z]+[А-ЯӘІҢҒҮҰҚӨҺA-Z]+',
            re.UNICODE
        )

        # Pattern for header artifacts
        self.header_pattern = re.compile(
            r'^\s*(?:Отчет|Report|Есеп|ОТЧЕТ|REPORT|ЕСЕП)\s+(?:об?|о|туралы)',
            re.IGNORECASE
        )

    def normalize_unicode(self, text: str) -> str:
        """Normalize Unicode using NFKC (canonical decomposition + composition)."""
        if not text or not self.config.normalize_unicode:
            return text
        return unicodedata.normalize('NFKC', text)

    def fix_hyphenation(self, text: str) -> str:
        """
        Fix broken hyphens from PDF extraction.
        "денежно-кредитную полити - ку" -> "денежно-кредитную политику"
        """
        if not text or not self.config.fix_hyphenation:
            return text

        # Remove space around hyphens within words
        # But keep hyphens in compounds like "денежно-кредитную"
        text = re.sub(r'(\w+)\s+-\s+(\w+)', r'\1\2', text)
        text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
        text = re.sub(r'(\w+)\s+-(\w+)', r'\1\2', text)

        return text

    def fix_spacing(self, text: str) -> str:
        """
        Fix broken word spacing.
        "у правление" -> "управление"
        "окружающая среда к орПоративное" -> "окружающая среда корпоративное"
        """
        if not text or not self.config.fix_spacing:
            return text

        # Multiple passes to catch all cases
        for _ in range(3):
            # Fix single letter followed by lowercase word
            text = re.sub(
                r'\b([а-яА-ЯәіңғүұқөһӘІҢҒҮҰҚӨҺ])\s+([а-яәіңғүұқөһ]{2,})\b',
                lambda m: m.group(1) + m.group(2) if m.group(1).isupper() else m.group(0),
                text,
                flags=re.UNICODE
            )

            # Fix two-letter prefix followed by lowercase word
            text = re.sub(
                r'\b([а-яА-ЯәіңғүұқөһӘІҢҒҮҰҚӨҺ]{2})\s+([а-яәіңғүұқөһ]{2,})\b',
                lambda m: m.group(1) + m.group(2) if m.group(1).isupper() else m.group(0),
                text,
                flags=re.UNICODE
            )

        # Normalize multiple spaces
        text = re.sub(r'\s+', ' ', text)

        return text

    def fix_case(self, text: str) -> str:
        """
        Fix broken case patterns.
        "уст ОйчивОм" -> "устойчивом"
        "НарОДНЫй" -> "Народный"
        "КазаХ стаНа" -> "Казахстана"
        """
        if not text or not self.config.fix_case:
            return text

        words = text.split()
        fixed_words = []

        for word in words:
            # Check if word has strange mixed case
            if self.broken_case_pattern.search(word):
                # Count lowercase vs uppercase
                lower_count = sum(1 for c in word if c.islower())
                upper_count = sum(1 for c in word if c.isupper())
                total_alpha = lower_count + upper_count

                if total_alpha == 0:
                    fixed_words.append(word)
                    continue

                # If mostly lowercase, convert to lowercase
                if lower_count > upper_count:
                    word = word.lower()
                # If mostly uppercase but long, likely a proper noun
                elif len(word) > 5 and upper_count / total_alpha > 0.5:
                    # Capitalize first letter, rest lowercase
                    word = word.capitalize()

            fixed_words.append(word)

        return ' '.join(fixed_words)

    def remove_noise(self, text: str) -> str:
        """Remove page numbers, TOC entries, and other noise."""
        if not text:
            return text

        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Remove page numbers
            if self.config.remove_page_numbers and self.page_pattern.match(line):
                continue

            # Remove TOC entries
            if self.config.remove_toc and self.toc_pattern.search(line):
                continue

            # Remove header artifacts
            if self.header_pattern.match(line):
                continue

            # Skip lines with only numbers and basic punctuation
            if re.match(r'^[\d\s\.\,\-\:\(\)]+$', line):
                continue

            cleaned_lines.append(line)

        text = ' '.join(cleaned_lines)

        # Remove excessive punctuation
        text = re.sub(r'\.{2,}', '', text)  # Remove dot leaders
        text = re.sub(r'\s+', ' ', text)  # Normalize spaces

        return text.strip()

    def clean_text(self, text: str) -> str:
        """Apply all cleaning steps in order."""
        if not text:
            return text

        # 1. Normalize Unicode
        text = self.normalize_unicode(text)

        # 2. Remove noise (page numbers, TOC)
        text = self.remove_noise(text)

        # 3. Fix hyphenation
        text = self.fix_hyphenation(text)

        # 4. Fix spacing
        text = self.fix_spacing(text)

        # 5. Fix case
        text = self.fix_case(text)

        # 6. Final cleanup
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        return text


class QualityFilter:
    """Quality filtering for aligned triplets."""

    def __init__(self, config: CleaningConfig):
        self.config = config

    def check_length(self, text: str) -> bool:
        """Check if text meets length requirements."""
        if not text:
            return False

        words = text.split()
        num_words = len(words)
        num_chars = len(text)

        if num_words < self.config.min_words:
            return False
        if num_words > self.config.max_words:
            return False
        if num_chars < self.config.min_chars:
            return False

        return True

    def check_content_quality(self, text: str) -> bool:
        """Check if content is meaningful."""
        if not text:
            return False

        # Must contain letters
        letter_count = sum(1 for c in text if c.isalpha())
        total_count = len(text)

        if total_count == 0:
            return False

        # At least 40% should be letters
        if letter_count / total_count < 0.4:
            return False

        # Check for minimum unique characters
        unique_chars = len(set(text.lower()))
        if unique_chars < 10:
            return False

        # Check for repetitive patterns
        words = text.split()
        if len(words) > 0:
            unique_words = len(set(words))
            # If too many repeated words, likely noise
            if unique_words / len(words) < 0.5:
                return False

        return True

    def check_length_ratio(self, text1: str, text2: str) -> bool:
        """Check if length ratio is reasonable."""
        if not text1 or not text2:
            return False

        len1 = len(text1.split())
        len2 = len(text2.split())

        if len2 == 0:
            return False

        ratio = max(len1, len2) / min(len1, len2)
        return ratio <= self.config.max_length_ratio

    def check_similarity(self, similarity: Optional[float]) -> bool:
        """Check if similarity meets threshold."""
        if similarity is None:
            return False
        return similarity >= self.config.min_similarity

    def is_toc_or_header(self, text: str) -> bool:
        """Detect if text is likely TOC or header content."""
        # Too many numbers relative to text
        digit_count = sum(1 for c in text if c.isdigit())
        if digit_count / len(text) > 0.3:
            return True

        # Contains TOC keywords
        toc_keywords = ['содержание', 'content', 'мазмұны', 'приложени', 'appendix']
        if any(kw in text.lower() for kw in toc_keywords):
            return True

        # Multiple lines with just numbers
        if re.search(r'\d+\s+\d+\s+\d+', text):
            return True

        return False

    def is_valid_triplet(self, triplet: Dict) -> Tuple[bool, str]:
        """
        Comprehensive quality check.
        Returns (is_valid, reason_if_invalid)
        """
        kaz_text = triplet.get("kaz", "")
        rus_text = triplet.get("rus")
        eng_text = triplet.get("eng")

        # Must have Kazakh
        if not kaz_text:
            return False, "missing_kaz"

        # Must have at least one translation
        if not rus_text and not eng_text:
            return False, "no_translations"

        # Check Kazakh quality
        if not self.check_length(kaz_text):
            return False, "kaz_length"
        if not self.check_content_quality(kaz_text):
            return False, "kaz_quality"
        if self.is_toc_or_header(kaz_text):
            return False, "kaz_toc"

        # Check Russian if present
        if rus_text:
            if not self.check_length(rus_text):
                return False, "rus_length"
            if not self.check_content_quality(rus_text):
                return False, "rus_quality"
            if self.is_toc_or_header(rus_text):
                return False, "rus_toc"
            if not self.check_length_ratio(kaz_text, rus_text):
                return False, "kaz_rus_ratio"

            # Similarity check
            sim = triplet.get("similarity_kaz_rus")
            if not self.check_similarity(sim):
                return False, f"kaz_rus_similarity_{sim}"

        # Check English if present
        if eng_text:
            if not self.check_length(eng_text):
                return False, "eng_length"
            if not self.check_content_quality(eng_text):
                return False, "eng_quality"
            if self.is_toc_or_header(eng_text):
                return False, "eng_toc"
            if not self.check_length_ratio(kaz_text, eng_text):
                return False, "kaz_eng_ratio"

            # Similarity check
            sim = triplet.get("similarity_kaz_eng")
            if not self.check_similarity(sim):
                return False, f"kaz_eng_similarity_{sim}"

        # In strict mode, require higher similarity for both
        if self.config.strict_mode:
            if rus_text:
                sim_rus = triplet.get("similarity_kaz_rus", 0)
                if sim_rus < 0.85:
                    return False, "strict_kaz_rus_similarity"

            if eng_text:
                sim_eng = triplet.get("similarity_kaz_eng", 0)
                if sim_eng < 0.85:
                    return False, "strict_kaz_eng_similarity"

        return True, ""


def extract_company_name(source_doc: str) -> Optional[str]:
    """Extract company name from document ID."""
    if not source_doc:
        return None

    source_lower = source_doc.lower()

    for pattern, company in COMPANY_PATTERNS.items():
        if pattern in source_lower:
            return company

    return None


def compute_text_hash(text: str) -> str:
    """Compute normalized hash for deduplication."""
    # Normalize: lowercase, remove extra spaces
    normalized = ' '.join(text.lower().split())
    return normalized


def deduplicate_triplets(triplets: List[Dict]) -> Tuple[List[Dict], int]:
    """
    Remove duplicates based on Kazakh text.
    Keeps the first occurrence.
    """
    seen = set()
    unique = []

    for triplet in triplets:
        kaz = triplet.get("kaz", "")
        text_hash = compute_text_hash(kaz)

        if text_hash not in seen:
            seen.add(text_hash)
            unique.append(triplet)

    return unique, len(triplets) - len(unique)


def process_corpus(
    input_file: Path,
    output_file: Path,
    config: CleaningConfig
) -> Dict:
    """Main processing pipeline."""

    print("\n" + "=" * 70)
    print("ENHANCED POST-PROCESSING (v2)")
    print("=" * 70)
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print("\nConfiguration:")
    print(f"  Min similarity:       {config.min_similarity}")
    print(f"  Min words:            {config.min_words}")
    print(f"  Max words:            {config.max_words}")
    print(f"  Max length ratio:     {config.max_length_ratio}")
    print(f"  Remove duplicates:    {config.remove_duplicates}")
    print(f"  Fix hyphenation:      {config.fix_hyphenation}")
    print(f"  Fix spacing:          {config.fix_spacing}")
    print(f"  Fix case:             {config.fix_case}")
    print(f"  Extract company:      {config.extract_company}")
    print(f"  Strict mode:          {config.strict_mode}")

    # Initialize processors
    cleaner = TextCleaner(config)
    quality_filter = QualityFilter(config)

    # Load data
    print("\nLoading triplets...")
    triplets = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            triplets.append(json.loads(line))

    print(f"  Loaded: {len(triplets):,} triplets")

    # Statistics
    stats = {
        "total_input": len(triplets),
        "text_cleaned": 0,
        "companies_extracted": 0,
        "passed_filters": 0,
        "filtered_out": 0,
        "duplicates_removed": 0,
        "total_output": 0,
        "filter_reasons": Counter(),
        "alignment_types_kaz_rus": Counter(),
        "alignment_types_kaz_eng": Counter(),
    }

    # Process
    print("\nCleaning texts...")
    cleaned_triplets = []

    for triplet in tqdm(triplets, desc="Processing"):
        # Track if any text was modified
        original_kaz = triplet.get("kaz", "")
        original_rus = triplet.get("rus", "")
        original_eng = triplet.get("eng", "")

        # Clean texts
        if triplet.get("kaz"):
            triplet["kaz"] = cleaner.clean_text(triplet["kaz"])
        if triplet.get("rus"):
            triplet["rus"] = cleaner.clean_text(triplet["rus"])
        if triplet.get("eng"):
            triplet["eng"] = cleaner.clean_text(triplet["eng"])

        # Track cleaning
        if (triplet.get("kaz") != original_kaz or
            triplet.get("rus") != original_rus or
            triplet.get("eng") != original_eng):
            stats["text_cleaned"] += 1

        # Extract company
        if config.extract_company:
            if "metadata" not in triplet:
                triplet["metadata"] = {}

            source_doc = triplet["metadata"].get("source_doc", "")
            if source_doc:
                company = extract_company_name(source_doc)
                if company:
                    triplet["metadata"]["company"] = company
                    stats["companies_extracted"] += 1

        # Quality filtering
        is_valid, reason = quality_filter.is_valid_triplet(triplet)

        if is_valid:
            cleaned_triplets.append(triplet)
            stats["passed_filters"] += 1
            # Track m:n alignment types if present
            at_rus = triplet.get("alignment_type_kaz_rus")
            if at_rus:
                stats["alignment_types_kaz_rus"][at_rus] += 1
            at_eng = triplet.get("alignment_type_kaz_eng")
            if at_eng:
                stats["alignment_types_kaz_eng"][at_eng] += 1
        else:
            stats["filtered_out"] += 1
            stats["filter_reasons"][reason] += 1

    # Deduplication
    if config.remove_duplicates:
        print("\nRemoving duplicates...")
        cleaned_triplets, num_dups = deduplicate_triplets(cleaned_triplets)
        stats["duplicates_removed"] = num_dups
        print(f"  Removed: {num_dups:,} duplicates")

    stats["total_output"] = len(cleaned_triplets)

    # Save output
    print(f"\nSaving {len(cleaned_triplets):,} cleaned triplets...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for triplet in cleaned_triplets:
            f.write(json.dumps(triplet, ensure_ascii=False) + '\n')

    # Save stats
    stats_file = output_file.with_suffix('.stats.json')
    stats_copy = dict(stats)
    stats_copy["filter_reasons"] = dict(stats["filter_reasons"])
    stats_copy["alignment_types_kaz_rus"] = dict(stats["alignment_types_kaz_rus"])
    stats_copy["alignment_types_kaz_eng"] = dict(stats["alignment_types_kaz_eng"])
    stats_copy["config"] = asdict(config)

    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats_copy, f, ensure_ascii=False, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("PROCESSING SUMMARY")
    print("=" * 70)
    print(f"  Input triplets:         {stats['total_input']:>10,}")
    print(f"  Texts cleaned:          {stats['text_cleaned']:>10,}")
    print(f"  Companies extracted:    {stats['companies_extracted']:>10,}")
    print(f"  Passed filters:         {stats['passed_filters']:>10,}")
    print(f"  Filtered out:           {stats['filtered_out']:>10,}")
    print(f"  Duplicates removed:     {stats['duplicates_removed']:>10,}")
    print(f"  Final output:           {stats['total_output']:>10,}")

    retention = stats['total_output'] / stats['total_input'] * 100
    print(f"\n  Retention rate:         {retention:.1f}%")

    if stats["alignment_types_kaz_rus"]:
        print("\n  Alignment types (KAZ-RUS):")
        for at, count in stats["alignment_types_kaz_rus"].most_common():
            print(f"    {at:<10} {count:>8,}")

    if stats["alignment_types_kaz_eng"]:
        print("\n  Alignment types (KAZ-ENG):")
        for at, count in stats["alignment_types_kaz_eng"].most_common():
            print(f"    {at:<10} {count:>8,}")

    if stats["filter_reasons"]:
        print("\n  Top filtering reasons:")
        for reason, count in stats["filter_reasons"].most_common(10):
            print(f"    {reason:<30} {count:>8,}")

    print(f"\n  Output:     {output_file.absolute()}")
    print(f"  Statistics: {stats_file.absolute()}")
    print("=" * 70)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced post-processing for aligned corpus (v2)"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Input aligned corpus JSONL"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output cleaned corpus JSONL"
    )
    parser.add_argument(
        "--min_similarity",
        type=float,
        default=0.80,
        help="Minimum similarity threshold (default: 0.80)"
    )
    parser.add_argument(
        "--min_words",
        type=int,
        default=3,
        help="Minimum words per sentence (default: 3)"
    )
    parser.add_argument(
        "--max_words",
        type=int,
        default=300,
        help="Maximum words per sentence (default: 300)"
    )
    parser.add_argument(
        "--max_length_ratio",
        type=float,
        default=3.0,
        help="Maximum length ratio between translations (default: 3.0)"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict mode (higher quality threshold)"
    )
    parser.add_argument(
        "--no_deduplicate",
        action="store_true",
        help="Disable deduplication"
    )
    parser.add_argument(
        "--no_fix_hyphenation",
        action="store_true",
        help="Disable hyphenation fixing"
    )
    parser.add_argument(
        "--no_fix_spacing",
        action="store_true",
        help="Disable spacing fixes"
    )
    parser.add_argument(
        "--no_fix_case",
        action="store_true",
        help="Disable case fixes"
    )

    args = parser.parse_args()

    # Setup
    base_dir = Path(__file__).parent.parent
    input_file = base_dir / args.input_file if not Path(args.input_file).is_absolute() else Path(args.input_file)
    output_file = base_dir / args.output_file

    if not input_file.exists():
        print(f"[ERROR] Input file not found: {input_file}")
        return 1

    # Config
    config = CleaningConfig(
        min_similarity=args.min_similarity,
        min_words=args.min_words,
        max_words=args.max_words,
        max_length_ratio=args.max_length_ratio,
        remove_duplicates=not args.no_deduplicate,
        fix_hyphenation=not args.no_fix_hyphenation,
        fix_spacing=not args.no_fix_spacing,
        fix_case=not args.no_fix_case,
        strict_mode=args.strict
    )

    # Process
    try:
        stats = process_corpus(input_file, output_file, config)
        return 0
    except Exception as e:
        print(f"\n[ERROR] Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
