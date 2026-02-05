#!/usr/bin/env python3
"""
Phase 1: PDF Extraction using Docling
=====================================

Extracts text from PDF files while preserving structure.
Outputs clean markdown files for each PDF.

Usage:
    python scripts/01_extract_pdfs.py --input_dirs Halyk_PDFs KEGOC_PDFs --output_dir extracted_markdown

Author: Dataset for Translator Project
Python: 3.11
"""

import argparse
import json
import re
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions


# Kazakh-specific characters for validation
KAZ_SPECIFIC_CHARS = set("ӘәҒғҚқҢңӨөҰұҮүІі")


def fix_unicode_escapes(text: str) -> str:
    """
    Fix malformed Unicode escape sequences from PDF extraction.

    Docling sometimes outputs sequences like /uni04AF instead of actual characters.
    This function converts them to proper Unicode characters.

    Pattern examples:
    - /uni04AF → ү (Kazakh letter)
    - /uni043F → п (Cyrillic letter)
    - /uni044F → я (Cyrillic letter)
    """
    def replace_uni(match):
        hex_code = match.group(1)
        try:
            return chr(int(hex_code, 16))
        except (ValueError, OverflowError):
            return match.group(0)  # Return original if invalid

    # Pattern for /uniXXXX (4 hex digits)
    text = re.sub(r'/uni([0-9A-Fa-f]{4})', replace_uni, text)

    # Pattern for /uniXXXXXX (6 hex digits for extended Unicode)
    text = re.sub(r'/uni([0-9A-Fa-f]{6})', replace_uni, text)

    # Also handle backslash variants: \uni04AF
    text = re.sub(r'\\uni([0-9A-Fa-f]{4})', replace_uni, text)
    text = re.sub(r'\\uni([0-9A-Fa-f]{6})', replace_uni, text)

    # Handle PDF glyph names: /afii10097 etc. (Cyrillic glyphs)
    # Common Cyrillic glyph mappings
    cyrillic_glyphs = {
        '/afii10017': 'А', '/afii10018': 'Б', '/afii10019': 'В', '/afii10020': 'Г',
        '/afii10021': 'Д', '/afii10022': 'Е', '/afii10023': 'Ё', '/afii10024': 'Ж',
        '/afii10025': 'З', '/afii10026': 'И', '/afii10027': 'Й', '/afii10028': 'К',
        '/afii10029': 'Л', '/afii10030': 'М', '/afii10031': 'Н', '/afii10032': 'О',
        '/afii10033': 'П', '/afii10034': 'Р', '/afii10035': 'С', '/afii10036': 'Т',
        '/afii10037': 'У', '/afii10038': 'Ф', '/afii10039': 'Х', '/afii10040': 'Ц',
        '/afii10041': 'Ч', '/afii10042': 'Ш', '/afii10043': 'Щ', '/afii10044': 'Ъ',
        '/afii10045': 'Ы', '/afii10046': 'Ь', '/afii10047': 'Э', '/afii10048': 'Ю',
        '/afii10049': 'Я',
        '/afii10065': 'а', '/afii10066': 'б', '/afii10067': 'в', '/afii10068': 'г',
        '/afii10069': 'д', '/afii10070': 'е', '/afii10071': 'ё', '/afii10072': 'ж',
        '/afii10073': 'з', '/afii10074': 'и', '/afii10075': 'й', '/afii10076': 'к',
        '/afii10077': 'л', '/afii10078': 'м', '/afii10079': 'н', '/afii10080': 'о',
        '/afii10081': 'п', '/afii10082': 'р', '/afii10083': 'с', '/afii10084': 'т',
        '/afii10085': 'у', '/afii10086': 'ф', '/afii10087': 'х', '/afii10088': 'ц',
        '/afii10089': 'ч', '/afii10090': 'ш', '/afii10091': 'щ', '/afii10092': 'ъ',
        '/afii10093': 'ы', '/afii10094': 'ь', '/afii10095': 'э', '/afii10096': 'ю',
        '/afii10097': 'я',
    }

    for glyph, char in cyrillic_glyphs.items():
        text = text.replace(glyph, char)

    return text


def validate_kazakh_encoding(text: str, filename: str) -> bool:
    """
    Validate that Kazakh text contains proper Kazakh-specific characters.
    Returns True if validation passes, False otherwise.
    """
    if "_KAZ" in filename or "_kaz" in filename.lower():
        has_kaz_chars = any(c in text for c in KAZ_SPECIFIC_CHARS)
        if not has_kaz_chars and len(text) > 100:
            print(f"  [WARNING] No Kazakh-specific characters found in {filename}")
            print(f"            This may indicate encoding issues.")
            return False
    return True


def clean_markdown(text: str) -> str:
    """
    Post-process extracted markdown:
    - Fix Unicode escape sequences
    - Remove excessive whitespace
    - Clean up headers/footers patterns
    - Remove page numbers
    """
    # Fix Unicode escape sequences FIRST
    text = fix_unicode_escapes(text)

    # Remove page numbers (common patterns)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    text = re.sub(r'\n\s*Page\s+\d+\s*(of\s+\d+)?\s*\n', '\n', text, flags=re.IGNORECASE)

    # Remove excessive blank lines
    text = re.sub(r'\n{4,}', '\n\n\n', text)

    # Remove common header/footer patterns
    text = re.sub(r'(?i)(confidential|internal use only|draft)\s*\n', '', text)

    # Clean up table artifacts (empty table rows)
    text = re.sub(r'\|\s*\|\s*\|\s*\|', '', text)

    return text.strip()


def extract_text_only(markdown_text: str) -> str:
    """
    Extract only paragraph text, skipping tables.
    Tables in markdown start with | character.
    """
    lines = markdown_text.split('\n')
    text_lines = []

    in_table = False
    for line in lines:
        stripped = line.strip()

        # Detect table start
        if stripped.startswith('|') or stripped.startswith('|-'):
            in_table = True
            continue

        # Detect table end (empty line after table)
        if in_table and not stripped:
            in_table = False
            continue

        # Skip table lines
        if in_table:
            continue

        # Keep non-table content
        text_lines.append(line)

    return '\n'.join(text_lines)


def setup_converter() -> DocumentConverter:
    """
    Setup Docling DocumentConverter with optimal settings for financial PDFs.
    """
    pdf_options = PdfPipelineOptions()
    pdf_options.do_ocr = True  # Enable OCR for scanned documents
    pdf_options.do_table_structure = True  # Preserve table structure

    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
    )

    return converter


def process_pdf(pdf_path: Path, converter: DocumentConverter) -> dict:
    """
    Process a single PDF and return extraction results.
    """
    result = {
        "filename": pdf_path.name,
        "success": False,
        "markdown": "",
        "text_only": "",
        "char_count": 0,
        "error": None
    }

    try:
        # Convert PDF to document
        conversion_result = converter.convert(str(pdf_path))

        # Export to markdown
        markdown_text = conversion_result.document.export_to_markdown()

        # Clean markdown
        markdown_text = clean_markdown(markdown_text)

        # Extract text only (no tables)
        text_only = extract_text_only(markdown_text)

        # Validate Kazakh encoding
        validate_kazakh_encoding(text_only, pdf_path.name)

        result["success"] = True
        result["markdown"] = markdown_text
        result["text_only"] = text_only
        result["char_count"] = len(text_only)

    except Exception as e:
        result["error"] = str(e)
        print(f"  [ERROR] Failed to process {pdf_path.name}: {e}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Extract text from PDFs using Docling")
    parser.add_argument(
        "--input_dirs",
        nargs="+",
        required=True,
        help="Input directories containing PDF files (e.g., Halyk_PDFs KEGOC_PDFs)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="extracted_markdown",
        help="Output directory for extracted markdown files"
    )
    parser.add_argument(
        "--text_only",
        action="store_true",
        help="Extract text only (skip tables)"
    )
    args = parser.parse_args()

    # Setup paths
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(exist_ok=True)

    # Initialize converter
    print("Initializing Docling DocumentConverter...")
    converter = setup_converter()

    # Collect all PDFs
    all_pdfs = []
    for input_dir_name in args.input_dirs:
        input_dir = base_dir / input_dir_name
        if not input_dir.exists():
            print(f"[WARNING] Directory not found: {input_dir}")
            continue
        pdfs = list(input_dir.glob("*.pdf")) + list(input_dir.glob("*.PDF"))
        all_pdfs.extend(pdfs)
        print(f"Found {len(pdfs)} PDFs in {input_dir_name}")

    print(f"\nTotal PDFs to process: {len(all_pdfs)}")
    print("=" * 60)

    # Process PDFs
    stats = {
        "total": len(all_pdfs),
        "success": 0,
        "failed": 0,
        "total_chars": 0
    }

    extraction_log = []

    for pdf_path in tqdm(all_pdfs, desc="Extracting PDFs"):
        print(f"\nProcessing: {pdf_path.name}")

        result = process_pdf(pdf_path, converter)
        extraction_log.append({
            "filename": result["filename"],
            "success": result["success"],
            "char_count": result["char_count"],
            "error": result["error"]
        })

        if result["success"]:
            stats["success"] += 1
            stats["total_chars"] += result["char_count"]

            # Determine output content
            content = result["text_only"] if args.text_only else result["markdown"]

            # Save markdown file
            output_filename = pdf_path.stem + ".md"
            output_path = output_dir / output_filename

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)

            print(f"  [OK] Saved: {output_filename} ({result['char_count']:,} chars)")
        else:
            stats["failed"] += 1

    # Save extraction log
    log_path = output_dir / "extraction_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "stats": stats,
            "files": extraction_log
        }, f, ensure_ascii=False, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"  Total PDFs:     {stats['total']}")
    print(f"  Successful:     {stats['success']}")
    print(f"  Failed:         {stats['failed']}")
    print(f"  Total chars:    {stats['total_chars']:,}")
    print(f"\n  Output dir:     {output_dir.absolute()}")
    print(f"  Log file:       {log_path.absolute()}")


if __name__ == "__main__":
    main()
