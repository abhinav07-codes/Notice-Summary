"""
Module 2: Text Extractor
--------------------------
Routes extraction based on document type from Module 1:

  ┌─────────────────┬────────────────────────────────────────────┐
  │ "digital_pdf"   │ PyMuPDF (fitz) for text + pdfplumber       │
  │                 │ for structured table detection              │
  ├─────────────────┼────────────────────────────────────────────┤
  │ "scanned_pdf"   │ Convert pages to images → PaddleOCR        │
  ├─────────────────┼────────────────────────────────────────────┤
  │ "image"         │ PaddleOCR directly on the image file       │
  └─────────────────┴────────────────────────────────────────────┘

Returns a unified ExtractedDocument dataclass containing:
  - raw_text      : Full plain text of the document
  - pages         : Per-page text list
  - tables        : List of tables (each table = list of row-lists)
  - metadata      : Extraction metadata (method used, page count, etc.)
"""

import fitz  # PyMuPDF
import pdfplumber
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import tempfile
import os

# ── Lazy import for PaddleOCR to avoid heavy startup time ──────────────────
# REPLACE WITH:
_easy_ocr = None

def _get_ocr():
    global _easy_ocr
    if _easy_ocr is None:
        import easyocr
        _easy_ocr = easyocr.Reader(['en'], gpu=False)
    return _easy_ocr


# ── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class ExtractedDocument:
    """Unified output structure from any extraction method."""
    raw_text: str = ""
    pages: list[str] = field(default_factory=list)      # per-page text
    tables: list[list[list]] = field(default_factory=list)  # [table[row[cell]]]
    metadata: dict = field(default_factory=dict)


# ── Main Entry Point ─────────────────────────────────────────────────────────

def extract_text(file_path: str, doc_type: str) -> ExtractedDocument:
    """
    Extracts text and tables from a document based on its type.

    Args:
        file_path (str): Path to the document.
        doc_type  (str): One of "digital_pdf", "scanned_pdf", "image".

    Returns:
        ExtractedDocument: Unified extraction result.
    """
    if doc_type == "digital_pdf":
        return _extract_digital_pdf(file_path)
    elif doc_type == "scanned_pdf":
        return _extract_scanned_pdf(file_path)
    elif doc_type == "image":
        return _extract_image(file_path)
    else:
        raise ValueError(f"Unknown doc_type: '{doc_type}'")


# ── Extractor: Digital PDF ───────────────────────────────────────────────────

def _extract_digital_pdf(file_path: str) -> ExtractedDocument:
    """
    Extracts text using PyMuPDF and tables using pdfplumber.
    PyMuPDF is faster and layout-aware; pdfplumber is better for tables.
    """
    print(f"[Extractor] Using PyMuPDF + pdfplumber for digital PDF...")

    pages_text = []
    all_tables = []

    # ── 1. Extract text page-by-page with PyMuPDF ──────────────────────────
    doc = fitz.open(file_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        # "dict" mode gives layout-aware text with block positions
        blocks = page.get_text("dict")["blocks"]
        page_text_parts = []

        for block in blocks:
            if block.get("type") == 0:  # type 0 = text block
                for line in block.get("lines", []):
                    line_text = " ".join(
                        span["text"] for span in line.get("spans", [])
                    ).strip()
                    if line_text:
                        page_text_parts.append(line_text)

        pages_text.append("\n".join(page_text_parts))

    doc.close()

    # ── 2. Extract tables using pdfplumber ─────────────────────────────────
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    # Filter out empty rows and None cells
                    clean_table = [
                        [cell if cell is not None else "" for cell in row]
                        for row in table
                        if any(cell for cell in row if cell)
                    ]
                    if clean_table:
                        all_tables.append(clean_table)
    except Exception as e:
        print(f"[Extractor] pdfplumber table extraction warning: {e}")

    raw_text = "\n\n--- PAGE BREAK ---\n\n".join(pages_text)

    return ExtractedDocument(
        raw_text=raw_text,
        pages=pages_text,
        tables=all_tables,
        metadata={
            "method": "PyMuPDF + pdfplumber",
            "page_count": len(pages_text),
            "table_count": len(all_tables),
        },
    )


# ── Extractor: Scanned PDF ───────────────────────────────────────────────────

def _extract_scanned_pdf(file_path: str) -> ExtractedDocument:
    """
    Converts each PDF page to a high-resolution image, then runs PaddleOCR.
    Uses a temporary directory to store intermediate page images.
    """
    print(f"[Extractor] Using PaddleOCR for scanned PDF...")

    ocr = _get_ocr()
    pages_text = []
    all_tables = []

    doc = fitz.open(file_path)

    with tempfile.TemporaryDirectory() as tmp_dir:
        for page_num in range(len(doc)):
            page = doc[page_num]

            # Render at 2x zoom for better OCR quality (150 DPI → 300 DPI)
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_path = os.path.join(tmp_dir, f"page_{page_num + 1}.png")
            pix.save(img_path)

            # Run OCR on the rendered image
            page_text = _run_paddleocr(ocr, img_path)
            pages_text.append(page_text)
            print(f"[Extractor]   Page {page_num + 1}/{len(doc)} extracted.")

    doc.close()
    raw_text = "\n\n--- PAGE BREAK ---\n\n".join(pages_text)

    return ExtractedDocument(
        raw_text=raw_text,
        pages=pages_text,
        tables=all_tables,  # Table extraction from scanned is handled by OCR grid analysis
        metadata={
            "method": "PaddleOCR (scanned PDF)",
            "page_count": len(pages_text),
            "table_count": 0,
        },
    )


# ── Extractor: Image ─────────────────────────────────────────────────────────

def _extract_image(file_path: str) -> ExtractedDocument:
    """
    Runs PaddleOCR directly on the image file.
    """
    print(f"[Extractor] Using PaddleOCR for image file...")

    ocr = _get_ocr()
    page_text = _run_paddleocr(ocr, file_path)

    return ExtractedDocument(
        raw_text=page_text,
        pages=[page_text],
        tables=[],
        metadata={
            "method": "PaddleOCR (image)",
            "page_count": 1,
            "table_count": 0,
        },
    )


# ── PaddleOCR Helper ─────────────────────────────────────────────────────────

# REPLACE entire _run_paddleocr with this:
def _run_paddleocr(ocr, image_path: str) -> str:
    result = ocr.readtext(image_path, detail=1)

    if not result:
        return ""

    # result items: (bbox, text, confidence)
    # Sort top-to-bottom by the y-coordinate of the top-left corner
    result.sort(key=lambda x: x[0][0][1])

    return "\n".join(text for _, text, conf in result if conf > 0.2)


# ── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from module1_document_classifier import classify_document, get_document_info

    if len(sys.argv) < 2:
        print("Usage: python module2_text_extractor.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    info = get_document_info(file_path)
    doc_type = info["doc_type"]

    result = extract_text(file_path, doc_type)

    print("\n── Extraction Result ──────────────────────────")
    print(f"  Method     : {result.metadata['method']}")
    print(f"  Pages      : {result.metadata['page_count']}")
    print(f"  Tables     : {result.metadata['table_count']}")
    print(f"\n── Extracted Text (first 500 chars) ──────────")
    print(result.raw_text[:500])
    print("────────────────────────────────────────────────")
