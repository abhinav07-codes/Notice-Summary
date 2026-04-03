"""
Module 3: Structure Parser
----------------------------
Takes the raw ExtractedDocument from Module 2 and applies
rule-based logic to identify and organize:

  - notice_title    : First prominent heading / all-caps line
  - issuing_body    : Department or institution name (near top)
  - notice_number   : Reference numbers like "No. XYZ/2024/01"
  - issue_date      : Date the notice was issued
  - body_content    : Main body paragraphs (cleaned)
  - important_dates : All dates mentioned with context labels
  - tables          : Structured tables (from Module 2 + raw text grids)

Date normalization:
  Ambiguous formats like "10/11/24" or "10th Nov" are resolved
  to ISO 8601 (YYYY-MM-DD) using deterministic dateutil parsing.
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from dateutil import parser as date_parser
from dateutil.parser import ParserError
from datetime import datetime


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class NoticeDate:
    """A date found in the document with its surrounding context."""
    raw_text: str           # Original text as found
    normalized: str         # ISO 8601: YYYY-MM-DD
    context: str            # Surrounding sentence for semantic label
    label: str = "date"     # e.g., "deadline", "exam_date", "issue_date"


@dataclass
class StructuredNotice:
    """Fully parsed notice with all key fields extracted."""
    notice_title: str = "Untitled Notice"
    issuing_body: str = ""
    notice_number: str = ""
    issue_date: str = ""
    body_content: str = ""
    important_dates: list[NoticeDate] = field(default_factory=list)
    tables: list[list[list]] = field(default_factory=list)
    raw_text: str = ""


# ── Regex Patterns ───────────────────────────────────────────────────────────

# Matches common date formats:
# DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY, DD Month YYYY, Month DD YYYY,
# DDth Month YYYY, YYYY-MM-DD, short forms like DD/MM/YY
DATE_PATTERNS = [
    # Full date: 15 January 2024, January 15, 2024
    r'\b(?:\d{1,2}(?:st|nd|rd|th)?\s+)?'
    r'(?:January|February|March|April|May|June|July|August|'
    r'September|October|November|December|'
    r'Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
    r'(?:\s+\d{1,2}(?:st|nd|rd|th)?)?\s*[,\s]\s*\d{2,4}\b',

    # Numeric: DD/MM/YYYY or DD-MM-YYYY or DD.MM.YYYY
    r'\b\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\b',

    # ISO: YYYY-MM-DD
    r'\b\d{4}-\d{2}-\d{2}\b',

    # Short month: 15 Jan 2024
    r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
    r'[\.,]?\s+\d{2,4}\b',
]

# Notice number patterns: No. 123/ABC/2024, Ref: XYZ/2024, Circular No.
NOTICE_NUMBER_PATTERN = re.compile(
    r'(?:No\.|Ref(?:erence)?[:\.]?|Circular\s+No\.?|Letter\s+No\.?)\s*'
    r'([A-Z0-9][A-Z0-9\/\-\.]+)',
    re.IGNORECASE,
)

# Date-related keywords to label context
DEADLINE_KEYWORDS = re.compile(
    r'\b(last\s+date|deadline|due\s+date|submit(?:ting|ted|ssion)?'
    r'|apply|application|registration|closing|final\s+date)\b',
    re.IGNORECASE,
)
EXAM_KEYWORDS = re.compile(
    r'\b(exam(?:ination)?|test|quiz|assessment|viva|practical|lab)\b',
    re.IGNORECASE,
)
EVENT_KEYWORDS = re.compile(
    r'\b(meeting|seminar|workshop|event|programme|ceremony|fest|competition)\b',
    re.IGNORECASE,
)


# ── Main Parser ──────────────────────────────────────────────────────────────

def parse_structure(extracted_doc) -> StructuredNotice:
    """
    Parses an ExtractedDocument into a StructuredNotice.

    Args:
        extracted_doc: ExtractedDocument from Module 2.

    Returns:
        StructuredNotice: All extracted fields.
    """
    text = extracted_doc.raw_text
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    notice = StructuredNotice(
        raw_text=text,
        tables=extracted_doc.tables,
    )

    notice.notice_title  = _extract_title(lines)
    notice.issuing_body  = _extract_issuing_body(lines)
    notice.notice_number = _extract_notice_number(text)
    notice.body_content  = _extract_body(lines)
    notice.important_dates = _extract_dates(text)

    # Tag issue date (first date near the top of the document)
    if notice.important_dates:
        # Heuristic: first date found is likely the issue date
        notice.issue_date = notice.important_dates[0].normalized
        notice.important_dates[0].label = "issue_date"

    print(f"[Parser] Title     : {notice.notice_title}")
    print(f"[Parser] Body Chars: {len(notice.body_content)}")
    print(f"[Parser] Dates Found: {len(notice.important_dates)}")
    print(f"[Parser] Tables Found: {len(notice.tables)}")

    return notice


# ── Field Extractors ─────────────────────────────────────────────────────────

def _extract_title(lines: list[str]) -> str:
    """
    Extracts the notice title using these heuristics (in priority order):
    1. Line explicitly labeled "NOTICE", "CIRCULAR", "ADVERTISEMENT"
    2. First ALL-CAPS line in the top 15 lines
    3. Longest line in the top 5 lines (fallback)
    """
    title_keywords = re.compile(
        r'^\s*(NOTICE|CIRCULAR|ADVERTISEMENT|NOTIFICATION|MEMO|ORDER)\s*$',
        re.IGNORECASE
    )

    # Priority 1: Explicit keyword line
    for line in lines[:20]:
        if title_keywords.match(line):
            # The actual subject line often follows immediately
            idx = lines.index(line)
            # Look for the next non-trivial line after keyword
            for candidate in lines[idx + 1: idx + 5]:
                if len(candidate) > 8:
                    return candidate
            return line

    # Priority 2: First all-caps line in top 15
    for line in lines[:15]:
        stripped = re.sub(r'[^A-Za-z]', '', line)
        if len(stripped) > 5 and stripped.isupper():
            return line

    # Priority 3: Longest line in top 5 (skip very short ones)
    candidates = [l for l in lines[:5] if len(l) > 10]
    if candidates:
        return max(candidates, key=len)

    return lines[0] if lines else "Untitled Notice"


def _extract_issuing_body(lines: list[str]) -> str:
    """
    Looks for institution/department name patterns in the first 8 lines.
    Common indicators: "University", "Department", "College", "Institute",
    "School", "Division", "Committee", "Board".
    """
    institution_keywords = re.compile(
        r'\b(university|college|institute|department|school|division|'
        r'committee|board|office|faculty|centre|center|academy)\b',
        re.IGNORECASE
    )

    for line in lines[:8]:
        if institution_keywords.search(line):
            return line.strip()

    return ""


def _extract_notice_number(text: str) -> str:
    """Extracts notice/reference number if present."""
    match = NOTICE_NUMBER_PATTERN.search(text)
    if match:
        return match.group(0).strip()
    return ""


def _extract_body(lines: list[str]) -> str:
    """
    Extracts the body content by removing header lines (first 3)
    and footer lines (last 2), then joining the rest.
    Also strips repeated whitespace and page-break markers.
    """
    if len(lines) <= 5:
        return "\n".join(lines)

    # Remove header (first 3 lines = title/institution/date area) and footer
    body_lines = lines[3:-2]

    # Remove page-break markers inserted during extraction
    body_lines = [l for l in body_lines if "--- PAGE BREAK ---" not in l]

    # Collapse multiple blank lines
    body_text = "\n".join(body_lines)
    body_text = re.sub(r'\n{3,}', '\n\n', body_text)

    return body_text.strip()


def _extract_dates(text: str) -> list[NoticeDate]:
    """
    Finds all dates in the text, normalizes them to ISO 8601,
    and labels each with semantic context.
    """
    found_dates = []
    seen_raw = set()

    combined_pattern = re.compile(
        "|".join(DATE_PATTERNS),
        re.IGNORECASE
    )

    for match in combined_pattern.finditer(text):
        raw = match.group(0).strip()

        if raw in seen_raw:
            continue
        seen_raw.add(raw)

        normalized = _normalize_date(raw)
        if not normalized:
            continue

        # Get surrounding context (50 chars before and after)
        start = max(0, match.start() - 60)
        end   = min(len(text), match.end() + 60)
        context = text[start:end].replace("\n", " ").strip()

        # Label based on surrounding keywords
        label = _label_date(context)

        found_dates.append(NoticeDate(
            raw_text=raw,
            normalized=normalized,
            context=context,
            label=label,
        ))

    # Sort by normalized date (chronological order)
    found_dates.sort(key=lambda d: d.normalized)
    return found_dates


def _normalize_date(raw: str) -> Optional[str]:
    """
    Converts a raw date string to ISO 8601 (YYYY-MM-DD).
    Uses dateutil for flexible parsing. Returns None on failure.

    Ambiguity rule: For DD/MM/YY formats, we set dayfirst=True
    (common in Indian academic notices).
    """
    try:
        # Clean up ordinal suffixes: 15th → 15, 1st → 1
        cleaned = re.sub(r'(\d+)(?:st|nd|rd|th)', r'\1', raw)
        dt = date_parser.parse(cleaned, dayfirst=True, fuzzy=True)

        # Sanity check: reject dates too far in the future or past
        if dt.year < 2000 or dt.year > 2035:
            return None

        return dt.strftime("%Y-%m-%d")
    except (ParserError, ValueError, OverflowError):
        return None


def _label_date(context: str) -> str:
    """Assigns a semantic label based on keywords near the date."""
    if DEADLINE_KEYWORDS.search(context):
        return "deadline"
    if EXAM_KEYWORDS.search(context):
        return "exam_date"
    if EVENT_KEYWORDS.search(context):
        return "event_date"
    return "mentioned_date"


# ── Pretty Print Helper ──────────────────────────────────────────────────────

def print_structured_notice(notice: StructuredNotice):
    """Utility to pretty-print a StructuredNotice for debugging."""
    divider = "─" * 52
    print(f"\n{divider}")
    print(f"  STRUCTURED NOTICE SUMMARY")
    print(divider)
    print(f"  Title       : {notice.notice_title}")
    print(f"  Issued By   : {notice.issuing_body or 'N/A'}")
    print(f"  Ref. Number : {notice.notice_number or 'N/A'}")
    print(f"  Issue Date  : {notice.issue_date or 'N/A'}")
    print(f"\n  Body Preview:")
    preview = notice.body_content[:300].replace('\n', ' ')
    print(f"  {preview}...")
    print(f"\n  Important Dates ({len(notice.important_dates)}):")
    for d in notice.important_dates:
        print(f"    [{d.label:<18}] {d.normalized}  ← \"{d.raw_text}\"")
    print(f"\n  Tables Found: {len(notice.tables)}")
    for i, table in enumerate(notice.tables[:2]):
        print(f"  Table {i+1}: {len(table)} rows × {len(table[0]) if table else 0} cols")
    print(divider)


# ── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from module1_document_classifier import get_document_info
    from module2_text_extractor import extract_text

    if len(sys.argv) < 2:
        print("Usage: python module3_structure_parser.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    info      = get_document_info(file_path)
    extracted = extract_text(file_path, info["doc_type"])
    notice    = parse_structure(extracted)

    print_structured_notice(notice)
