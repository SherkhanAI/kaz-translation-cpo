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
    - Remove excessive whitespace
    - Clean up headers/footers patterns
    - Remove page numbers
    """
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
