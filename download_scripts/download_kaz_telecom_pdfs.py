import requests
import re
import os
from pathlib import Path
from urllib.parse import unquote

def parse_links_file(file_path):
    """Parse the Kaz_Telecom file and extract PDF URLs with metadata."""

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    downloads = []
    current_year = None
    current_lang = None

    for i, line in enumerate(lines):
        line = line.strip()

        # Check if line contains both year and language (e.g., "2024 ГОД - Казахский (KAZ)")
        year_match = re.search(r'(\d{4})\s+ГОД', line)
        if year_match:
            current_year = year_match.group(1)

            # Check for language in the same line
            if 'Казахский (KAZ)' in line:
                current_lang = 'KAZ'
            elif 'Русский (RUS)' in line:
                current_lang = 'RUS'
            elif 'Английский (ENG)' in line:
                current_lang = 'ENG'
            else:
                current_lang = None

            continue

        # Extract URLs (both http and https)
        if line.startswith('http://') or line.startswith('https://'):
            # Get the previous line as document title
            prev_line = lines[i-1].strip() if i > 0 else ''

            # Determine document type
            if 'устойчивом развитии' in prev_line or 'Sustainability' in prev_line:
                doc_type = 'Sustainability_Report'
            elif 'Интегрирленген' in prev_line or 'Интегрированный' in prev_line or 'Integrated' in prev_line:
                doc_type = 'Integrated_Annual_Report'
            elif 'Годовой отчет' in prev_line or 'Annual Report' in prev_line:
                doc_type = 'Annual_Report'
            else:
                doc_type = 'Report'

            # Ensure we have both year and lang
            if current_year and current_lang:
                downloads.append({
                    'url': line,
                    'year': current_year,
                    'lang': current_lang,
                    'type': doc_type,
                    'title': prev_line
                })
            else:
                print(f"[WARNING] Skipping URL (year={current_year}, lang={current_lang}): {line[:60]}...")

    return downloads


def sanitize_filename(filename):
    """Remove invalid characters from filename."""
    return re.sub(r'[<>:"/\\|?*]', '_', filename)


def download_pdf(url, output_path):
    """Download a PDF file from URL."""
    try:
        print(f"Downloading: {url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=120, stream=True)
        response.raise_for_status()

        # Save the file
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"[OK] Saved: {output_path}")
        return True

    except Exception as e:
        print(f"[ERROR] Error downloading {url}: {str(e)}")
        return False


def main():
    # Set up paths
    script_dir = Path(__file__).parent
    links_file = script_dir / 'Kaz_Telecom_links.txt'
    output_dir = script_dir.parent / 'Kaz_Telecom_PDFs'

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    # Parse the links file
    print("Parsing links file...")
    downloads = parse_links_file(links_file)

    print(f"\nFound {len(downloads)} PDF files to download\n")
    print("="*60)

    # Download each file
    success_count = 0
    failed_count = 0

    for item in downloads:
        # Create filename: YYYY_Type_LANG.pdf
        filename = f"{item['year']}_{item['type']}_{item['lang']}.pdf"
        filename = sanitize_filename(filename)
        output_path = output_dir / filename

        # Skip if already exists
        if output_path.exists():
            print(f"[SKIP] Already exists: {filename}")
            success_count += 1
            continue

        # Download
        if download_pdf(item['url'], output_path):
            success_count += 1
        else:
            failed_count += 1

        print()

    # Summary
    print("="*60)
    print(f"\nDownload Summary:")
    print(f"  [OK] Successful: {success_count}")
    print(f"  [ERROR] Failed: {failed_count}")
    print(f"  Total: {len(downloads)}")
    print(f"\nAll PDFs saved to: {output_dir.absolute()}")


if __name__ == '__main__':
    main()
