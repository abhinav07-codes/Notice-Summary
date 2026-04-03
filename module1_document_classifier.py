"""
Module 1: Document Classifier
-------------------------------
Analyzes the input file and classifies it as:
  - "digital_pdf"  : Machine-readable PDF (text layer present)
  - "scanned_pdf"  : PDF with no text layer (image-based pages)
  - "image"        : Direct image input (JPEG, PNG, etc.)

Decision logic:
  - If the file extension is an image format → "image"
  - If it's a PDF, extract text from first N pages using PyMuPDF.
    If meaningful text is found  → "digital_pdf"
    Otherwise                    → "scanned_pdf"
"""

import fitz  # PyMuPDF
import os
from pathlib import Path

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# Minimum characters per page to consider it "digital"
TEXT_THRESHOLD = 50

# Number of pages to sample for classification
SAMPLE_PAGES = 3


def classify_document(file_path: str) -> str:
    """
    Classifies the input document.

    Args:
        file_path (str): Path to the input file.

    Returns:
        str: One of "digital_pdf", "scanned_pdf", or "image".

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is unsupported.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()

    # ── Step 1: Check if it's a direct image file ──────────────────────────
    if ext in IMAGE_EXTENSIONS:
        print(f"[Classifier] '{path.name}' → Detected as IMAGE")
        return "image"

    # ── Step 2: Check if it's a PDF ────────────────────────────────────────
    if ext == ".pdf":
        return _classify_pdf(file_path)

    raise ValueError(
        f"Unsupported file format: '{ext}'. "
        f"Supported formats: PDF, {', '.join(IMAGE_EXTENSIONS)}"
    )


def _classify_pdf(file_path: str) -> str:
    """
    Internal helper: Opens a PDF and checks if it contains a text layer.

    Samples up to SAMPLE_PAGES pages and counts total extracted characters.
    If average characters per sampled page >= TEXT_THRESHOLD → digital_pdf.
    Otherwise → scanned_pdf.
    """
    doc = fitz.open(file_path)
    total_pages = len(doc)
    pages_to_check = min(SAMPLE_PAGES, total_pages)

    total_chars = 0

    for page_num in range(pages_to_check):
        page = doc[page_num]
        text = page.get_text("text").strip()
        total_chars += len(text)

    doc.close()

    avg_chars = total_chars / pages_to_check if pages_to_check > 0 else 0

    print(
        f"[Classifier] '{Path(file_path).name}' → "
        f"Sampled {pages_to_check} page(s), avg {avg_chars:.1f} chars/page"
    )

    if avg_chars >= TEXT_THRESHOLD:
        print(f"[Classifier] → Classified as DIGITAL PDF")
        return "digital_pdf"
    else:
        print(f"[Classifier] → Classified as SCANNED PDF (OCR needed)")
        return "scanned_pdf"


def get_document_info(file_path: str) -> dict:
    """
    Returns basic metadata about the document alongside its classification.

    Args:
        file_path (str): Path to the input file.

    Returns:
        dict: {
            "file_name": str,
            "file_size_kb": float,
            "doc_type": str,        # "digital_pdf" | "scanned_pdf" | "image"
            "page_count": int | None
        }
    """
    path = Path(file_path)
    doc_type = classify_document(file_path)

    info = {
        "file_name": path.name,
        "file_size_kb": round(path.stat().st_size / 1024, 2),
        "doc_type": doc_type,
        "page_count": None,
    }

    if doc_type in ("digital_pdf", "scanned_pdf"):
        doc = fitz.open(file_path)
        info["page_count"] = len(doc)
        doc.close()

    return info


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python module1_document_classifier.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    info = get_document_info(file_path)

    print("\n── Document Info ──────────────────────────────")
    for key, value in info.items():
        print(f"  {key:<18}: {value}")
    print("────────────────────────────────────────────────")
