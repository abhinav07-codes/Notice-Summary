"""
Module 4: LangChain Summarizer
--------------------------------
Takes a StructuredNotice (Module 3) and passes it through a
LangChain pipeline to generate a concise, structured summary.

Pipeline stages:
  1. PromptTemplate   → Builds the LLM prompt from StructuredNotice fields
  2. ChatOpenAI (LLM) → Generates the summary (configurable: OpenAI / Ollama)
  3. PydanticOutputParser → Validates and structures the response

Pydantic output model:
  NoticeSummary {
    one_line_summary    : str
    key_points          : list[str]
    important_dates     : list[DateEntry]
    target_audience     : str
    action_required     : str
    urgency_level       : "Low" | "Medium" | "High"
    relevant_department : str
  }

Supported LLM backends (set via LLM_BACKEND env var):
  - "openai"  : Uses ChatOpenAI (requires OPENAI_API_KEY)
  - "ollama"  : Uses local Ollama (no API key required)
  - "groq"    : Uses Groq API (requires GROQ_API_KEY, fast + free tier)
"""

import os
from typing import Literal
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough


# ── Pydantic Output Models ────────────────────────────────────────────────────

class DateEntry(BaseModel):
    """Represents a single important date in the summary."""
    label: str  = Field(description="What this date is for, e.g. 'Submission Deadline'")
    date:  str  = Field(description="The date in YYYY-MM-DD or human-readable format")

# ADD this new class above NoticeSummary:
class ScheduleEntry(BaseModel):
    subject: str = Field(description="Subject or course name")
    date: str    = Field(description="Date of the exam/event")
    time: str    = Field(description="Time of the exam/event, e.g. '10:00 AM'")
    venue: str   = Field(description="Room, hall, or venue name if mentioned")


class NoticeSummary(BaseModel):
    """Structured summary of an academic notice."""

    one_line_summary: str = Field(
        description="A single sentence capturing the core purpose of the notice."
    )
    key_points: list[str] = Field(
        description="3 to 6 bullet-point key points from the notice body."
    )
    important_dates: list[DateEntry] = Field(
        description="All deadlines, exam dates, event dates, or issue dates mentioned."
    )
    target_audience: str = Field(
        description="Who this notice is for, e.g. 'All B.Tech Students', 'Faculty Members'."
    )
    action_required: str = Field(
        description="What the recipient needs to do, e.g. 'Register before the deadline'."
    )
    urgency_level: Literal["Low", "Medium", "High"] = Field(
        description="Urgency: High if deadline < 7 days or critical action needed, "
                    "Medium if 7-30 days, Low otherwise."
    )
    relevant_department: str = Field(
        description="The issuing or relevant department/office."
    )
    schedule: list[ScheduleEntry] = Field(
        default_factory=list,
        description="Subject-wise schedule table extracted from the notice. "
                    "Include subject, date, time, and venue for each entry."
    )


# ── LLM Factory ───────────────────────────────────────────────────────────────

def _build_llm(backend: str = None):
    """
    Builds and returns the LLM based on the selected backend.

    Priority: argument → LLM_BACKEND env var → "groq" (default)
    """
    backend = backend or os.getenv("LLM_BACKEND", "groq").lower()

    if backend == "openai":
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set for OpenAI backend.")
        print("[Summarizer] Using OpenAI (gpt-4o-mini)")
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=api_key)

    elif backend == "ollama":
        from langchain_ollama import ChatOllama
        model = os.getenv("OLLAMA_MODEL", "llama3")
        print(f"[Summarizer] Using Ollama ({model})")
        return ChatOllama(model=model, temperature=0.2)

    elif backend == "groq":
        from langchain_groq import ChatGroq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not set. Get a free key at https://console.groq.com"
            )
        # TO THIS:
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        print(f"[Summarizer] Using Groq ({model})")
        return ChatGroq(model=model, temperature=0.2, api_key=api_key)

    else:
        raise ValueError(
            f"Unknown LLM_BACKEND: '{backend}'. "
            f"Choose from: 'openai', 'ollama', 'groq'."
        )


# ── Prompt Template ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert academic notice summarizer. 
Your job is to extract structured information from academic notices, 
circulars, and announcements and return it in a precise JSON format.
Be concise, factual, and do not hallucinate information not present in the notice.
"""

HUMAN_PROMPT = """Below is an academic notice extracted from a document.
Analyze it carefully and produce a structured summary.

=== NOTICE DETAILS ===
Title         : {title}
Issuing Body  : {issuing_body}
Reference No  : {notice_number}
Issue Date    : {issue_date}

=== NOTICE BODY ===
{body_content}

=== IMPORTANT DATES DETECTED (by rule-based parser) ===
{dates_section}

=== TABLES FOUND ===
{tables_section}

=== OUTPUT FORMAT ===
{format_instructions}

Respond ONLY with the JSON object. Do not include any preamble or explanation.
"""


# ── Main Summarizer ───────────────────────────────────────────────────────────

def build_summarizer_chain(llm_backend: str = None):
    """
    Builds and returns a LangChain LCEL chain for notice summarization.

    Chain signature:
        Input  : dict with keys matching HUMAN_PROMPT placeholders
        Output : NoticeSummary (Pydantic model)

    Usage:
        chain = build_summarizer_chain()
        result = chain.invoke(prepare_input(structured_notice))
    """
    llm    = _build_llm(llm_backend)
    parser = PydanticOutputParser(pydantic_object=NoticeSummary)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ])

    # LCEL chain: prompt → llm → parser
    chain = prompt | llm | parser

    return chain


def prepare_input(notice) -> dict:
    """
    Converts a StructuredNotice into the dict expected by the LangChain chain.

    Args:
        notice: StructuredNotice from Module 3.

    Returns:
        dict: Input dict for chain.invoke()
    """
    parser = PydanticOutputParser(pydantic_object=NoticeSummary)

    # Format dates section
    if notice.important_dates:
        dates_lines = [
            f"  [{d.label:<18}] {d.normalized}  (raw: \"{d.raw_text}\")"
            for d in notice.important_dates
        ]
        dates_section = "\n".join(dates_lines)
    else:
        dates_section = "  No dates detected by parser."

    # Format tables section (first 2 tables, max 8 rows each)
    if notice.tables:
        tables_parts = []
        for i, table in enumerate(notice.tables[:2]):
            rows = table[:8]
            table_str = f"  Table {i+1}:\n"
            for row in rows:
                table_str += "    | " + " | ".join(str(c) for c in row) + " |\n"
            tables_parts.append(table_str)
        tables_section = "\n".join(tables_parts)
    else:
        tables_section = "  No tables found."

    # Truncate body to avoid exceeding context limits (keep first 2000 chars)
    body = notice.body_content[:2000]
    if len(notice.body_content) > 2000:
        body += "\n  ... [truncated for brevity] ..."

    return {
        "title":              notice.notice_title,
        "issuing_body":       notice.issuing_body or "Not specified",
        "notice_number":      notice.notice_number or "Not specified",
        "issue_date":         notice.issue_date or "Not specified",
        "body_content":       body,
        "dates_section":      dates_section,
        "tables_section":     tables_section,
        "format_instructions": parser.get_format_instructions(),
    }


def summarize_notice(notice, llm_backend: str = None) -> NoticeSummary:
    """
    Convenience function: builds chain, runs it, returns NoticeSummary.

    Args:
        notice      : StructuredNotice from Module 3.
        llm_backend : Override LLM backend (optional).

    Returns:
        NoticeSummary: Pydantic model with structured summary fields.
    """
    print("[Summarizer] Building LangChain pipeline...")
    chain = build_summarizer_chain(llm_backend)

    print("[Summarizer] Preparing input...")
    chain_input = prepare_input(notice)

    print("[Summarizer] Invoking LLM...")
    result = chain.invoke(chain_input)

    print(f"[Summarizer] Done. Urgency: {result.urgency_level}")
    return result


# ── Pretty Print Helper ───────────────────────────────────────────────────────

def print_summary(summary: NoticeSummary):
    """Pretty-prints a NoticeSummary for debugging or CLI output."""
    divider = "═" * 55
    print(f"\n{divider}")
    print(f"  📋  NOTICE SUMMARY")
    print(divider)
    print(f"  Summary    : {summary.one_line_summary}")
    print(f"  Audience   : {summary.target_audience}")
    print(f"  Department : {summary.relevant_department}")
    print(f"  Action     : {summary.action_required}")
    urgency_icon = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(summary.urgency_level, "⚪")
    print(f"  Urgency    : {urgency_icon} {summary.urgency_level}")

    print(f"\n  Key Points:")
    for i, point in enumerate(summary.key_points, 1):
        print(f"    {i}. {point}")

    print(f"\n  Important Dates:")
    if summary.important_dates:
        for d in summary.important_dates:
            print(f"    📅  {d.label:<25} → {d.date}")
    else:
        print("    (none)")

    print(divider)


# ── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from module1_document_classifier import get_document_info
    from module2_text_extractor import extract_text
    from module3_structure_parser import parse_structure

    if len(sys.argv) < 2:
        print("Usage: python module4_langchain_summarizer.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    info      = get_document_info(file_path)
    extracted = extract_text(file_path, info["doc_type"])
    notice    = parse_structure(extracted)
    summary   = summarize_notice(notice)

    print_summary(summary)
