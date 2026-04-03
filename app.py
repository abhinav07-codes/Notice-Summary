"""
Module 5: Streamlit Application
---------------------------------
The complete end-to-end UI that ties all four modules together.

Features:
  - Drag-and-drop file upload (PDF, JPEG, PNG)
  - Auto-classification display with icon
  - Expandable raw text viewer
  - Structured dates table
  - Extracted tables viewer
  - Final LLM summary with urgency badge
  - Download summary as JSON

Run:
    streamlit run app.py

Environment (set in .env or export):
    LLM_BACKEND = groq | openai | ollama
    GROQ_API_KEY = your_key_here      (for Groq)
    OPENAI_API_KEY = your_key_here    (for OpenAI)
"""

import tempfile

import streamlit as st
import json
import time
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Academic Notice Summarizer",
    page_icon="📋",
    layout="wide",
)

# ── Imports from our modules ─────────────────────────────────────────────────
from module1_document_classifier import get_document_info
from module2_text_extractor       import extract_text
from module3_structure_parser     import parse_structure
from module4_langchain_summarizer import summarize_notice, print_summary

# ── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .doc-type-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 13px;
        font-weight: bold;
        margin-bottom: 8px;
    }
    .badge-digital  { background:#d1fae5; color:#065f46; }
    .badge-scanned  { background:#fef9c3; color:#713f12; }
    .badge-image    { background:#dbeafe; color:#1e3a8a; }
    .urgency-high   { background:#fee2e2; color:#991b1b; padding:4px 12px; border-radius:8px; }
    .urgency-medium { background:#fef9c3; color:#92400e; padding:4px 12px; border-radius:8px; }
    .urgency-low    { background:#d1fae5; color:#064e3b; padding:4px 12px; border-radius:8px; }
    .section-header { font-size:18px; font-weight:700; margin-top:16px; margin-bottom:6px; }
    .summary-box {
        background:#f8fafc;
        border-left:4px solid #6366f1;
        padding:14px 18px;
        border-radius:6px;
        font-size:15px;
        margin-bottom:12px;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper: Badge HTML ────────────────────────────────────────────────────────
def doc_type_badge(doc_type: str) -> str:
    icons = {
        "digital_pdf": ("📄", "Digital PDF",  "badge-digital"),
        "scanned_pdf": ("🖨️", "Scanned PDF",  "badge-scanned"),
        "image":       ("🖼️", "Image File",   "badge-image"),
    }
    icon, label, css = icons.get(doc_type, ("❓", doc_type, "badge-digital"))
    return f'<span class="doc-type-badge {css}">{icon} {label}</span>'


def urgency_badge(level: str) -> str:
    icons = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
    css   = {"High": "urgency-high", "Medium": "urgency-medium", "Low": "urgency-low"}
    icon  = icons.get(level, "⚪")
    klass = css.get(level, "urgency-low")
    return f'<span class="{klass}">{icon} {level} Priority</span>'


# ── Main App ──────────────────────────────────────────────────────────────────

def main():
    # ── Header ───────────────────────────────────────────────────────────────
    st.title("📋 Academic Notice Summarizer")
    st.caption(
        "Upload any academic notice (digital PDF, scanned PDF, or image) "
        "to get an AI-powered structured summary."
    )

    # ── Sidebar: Configuration ────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")

        llm_backend = st.selectbox(
            "LLM Backend",
            options=["groq", "openai", "ollama"],
            index=0,
            help="Groq is free and fast. OpenAI needs API key. Ollama runs locally."
        )

        if llm_backend == "groq":
            groq_key = st.text_input(
                "Groq API Key",
                value=os.getenv("GROQ_API_KEY", ""),
                type="password",
                help="Get free key at https://console.groq.com"
            )
            if groq_key:
                os.environ["GROQ_API_KEY"] = groq_key

        elif llm_backend == "openai":
            oai_key = st.text_input(
                "OpenAI API Key",
                value=os.getenv("OPENAI_API_KEY", ""),
                type="password"
            )
            if oai_key:
                os.environ["OPENAI_API_KEY"] = oai_key

        os.environ["LLM_BACKEND"] = llm_backend

        st.divider()
        st.markdown("**Pipeline Modules:**")
        st.markdown("1. 🔍 Document Classifier")
        st.markdown("2. 📤 Text Extractor")
        st.markdown("3. 🔧 Structure Parser")
        st.markdown("4. 🤖 LangChain Summarizer")
        st.divider()
        st.markdown("Built with: PyMuPDF · pdfplumber · PaddleOCR · LangChain · Pydantic")

    # ── File Upload ───────────────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "Upload a Notice Document",
        type=["pdf", "jpg", "jpeg", "png", "tiff"],
        help="Supports: Digital PDFs, Scanned PDFs, JPEG/PNG images"
    )
    if uploaded_file is None:
        st.session_state["analyze_clicked"] = False
        st.session_state["do_summarize"] = False

    if uploaded_file is None:
        st.info("👆 Upload a notice document to get started.")

        # Demo notice for testing without a file
        st.divider()
        if st.button("🎯 Run with Demo Text (no file needed)"):
            _run_demo_mode(llm_backend)
        return

    # ── Processing Pipeline ───────────────────────────────────────────────────
    if st.button("🚀 Analyze & Summarize", type="primary", use_container_width=True):
        st.session_state["analyze_clicked"] = True
        st.session_state["do_summarize"] = False  # reset for new file

    if st.session_state.get("analyze_clicked", False):
        _run_pipeline(uploaded_file, llm_backend)


def _run_pipeline(uploaded_file, llm_backend: str):
    import tempfile
    import pandas as pd

    tmp_path = Path(tempfile.gettempdir()) / uploaded_file.name
    cache_key = f"extracted_{uploaded_file.name}_{uploaded_file.size}"

    # ── Step 1: Extract once, cache it ───────────────────────────────────────
    if cache_key not in st.session_state:
        tmp_path.write_bytes(uploaded_file.getvalue())
        progress_bar = st.progress(0, text="Starting pipeline...")
        status = st.empty()
        try:
            status.info("🔍 **Module 1:** Classifying document...")
            doc_info = get_document_info(str(tmp_path))
            progress_bar.progress(20, text="Classification complete")

            status.info("📤 **Module 2:** Extracting text from all pages...")
            extracted = extract_text(str(tmp_path), doc_info["doc_type"])
            progress_bar.progress(50, text="Extraction complete")

            st.session_state[cache_key] = (doc_info, extracted)
            st.session_state["selected_page_idx"] = 0
            st.session_state["do_summarize"] = False
            progress_bar.empty()
            status.empty()
        except Exception as e:
            st.error(f"❌ Extraction error: {str(e)}")
            with st.expander("🐛 Full traceback"):
                import traceback
                st.code(traceback.format_exc())
            return

    doc_info, extracted = st.session_state[cache_key]
    total_pages = len(extracted.pages)

    # ── Step 2: Page selector (only for multi-page) ───────────────────────────
    if total_pages > 1:

        def get_smart_label(page_text, page_num):
            lines = [l.strip() for l in page_text.splitlines() if l.strip()]
            skip_keywords = ["department of", "scanned with", "sikkim", "manipal"]
            meaningful = [
                l for l in lines
                if not any(kw in l.lower() for kw in skip_keywords)
            ]
            label = meaningful[0] if meaningful else (lines[0] if lines else f"Page {page_num + 1}")
            return label[:90] + "..." if len(label) > 90 else label

        page_labels = [
            f"Page {i + 1}: {get_smart_label(p, i)}"
            for i, p in enumerate(extracted.pages)
        ]

        st.info(f"📄 This document has **{total_pages} pages**. Identify and pick your page below.")

        # Overview table
        st.markdown("**📋 All pages in this document:**")
        overview_data = [
            {"Page": f"Page {i + 1}", "Content": get_smart_label(p, i)}
            for i, p in enumerate(extracted.pages)
        ]
        st.dataframe(pd.DataFrame(overview_data), use_container_width=True, hide_index=True)

        # Selectbox with "All Pages" as first option
        all_labels = ["All Pages (combine everything)"] + page_labels

        selected_option = st.selectbox(
            "Select the page to summarize:",
            options=list(range(len(all_labels))),
            format_func=lambda i: all_labels[i],
            index=st.session_state.get("selected_page_idx", 0),
            key="page_selector",
        )
        st.session_state["selected_page_idx"] = selected_option

        # Preview — always visible
        if selected_option == 0:
            st.caption("📄 Preview — All Pages combined:")
            preview_text = "\n--- PAGE BREAK ---\n".join(
                "\n".join(
                    l.strip() for l in p.splitlines() if l.strip()
                )[:300]
                for p in extracted.pages
            )
        else:
            actual_idx = selected_option - 1
            st.caption(f"📄 Preview — {page_labels[actual_idx]}:")
            preview_text = "\n".join(
                l.strip() for l in extracted.pages[actual_idx].splitlines()
                if l.strip()
            )

        st.text_area(
            "preview",
            value=preview_text[:1200],
            height=200,
            disabled=True,
            label_visibility="collapsed",
        )

        if st.button("✅ Summarize", type="primary", use_container_width=True):
            st.session_state["do_summarize"] = True

        if not st.session_state.get("do_summarize", False):
            return

        from module2_text_extractor import ExtractedDocument
        if st.session_state["selected_page_idx"] == 0:
            # All pages combined
            combined_text = "\n\n--- PAGE BREAK ---\n\n".join(extracted.pages)
            selected_extracted = ExtractedDocument(
                raw_text=combined_text,
                pages=extracted.pages,
                tables=extracted.tables,
                metadata=extracted.metadata,
            )
            selected_doc_info = doc_info
        else:
            actual_idx = st.session_state["selected_page_idx"] - 1
            selected_extracted = ExtractedDocument(
                raw_text=extracted.pages[actual_idx],
                pages=[extracted.pages[actual_idx]],
                tables=extracted.tables,
                metadata=extracted.metadata,
            )
            selected_doc_info = {**doc_info, "page_count": 1}

    else:
        selected_extracted = extracted
        selected_doc_info = doc_info
        st.session_state["do_summarize"] = True

    # ── Step 3: Summarize (cached per page selection) ─────────────────────────
    summary_key = f"summary_{cache_key}_{st.session_state.get('selected_page_idx', 0)}"

    if summary_key not in st.session_state:
        try:
            progress_bar = st.progress(65, text="Parsing structure...")
            status = st.empty()

            status.info("🔧 **Module 3:** Parsing structure...")
            notice = parse_structure(selected_extracted)
            progress_bar.progress(80, text="Almost done...")

            status.info("🤖 **Module 4:** Generating AI summary...")
            summary = summarize_notice(notice, llm_backend)
            progress_bar.progress(100, text="Done!")
            status.empty()
            progress_bar.empty()

            st.session_state[summary_key] = (notice, summary)

        except Exception as e:
            st.error(f"❌ Pipeline error: {str(e)}")
            with st.expander("🐛 Full traceback"):
                import traceback
                st.code(traceback.format_exc())
            return
    else:
        notice, summary = st.session_state[summary_key]
        st.success("✅ Showing cached summary — select a different page to re-summarize.")

    _render_results(selected_doc_info, selected_extracted, notice, summary)
def _render_results(doc_info, extracted, notice, summary):
    """Renders all pipeline results in a structured layout."""

    # ── Row 1: Document metadata ──────────────────────────────────────────────
    st.success("✅ Analysis complete!")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("File",        doc_info["file_name"])
    col2.metric("Size",        f"{doc_info['file_size_kb']} KB")
    col3.metric("Pages",       doc_info.get("page_count") or "N/A")
    col4.metric("Tables Found", len(notice.tables))

    st.markdown(doc_type_badge(doc_info["doc_type"]), unsafe_allow_html=True)

    st.divider()

    # ── Row 2: Two-column layout ──────────────────────────────────────────────
    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        # ── Extracted Structure ───────────────────────────────────────────────
        st.markdown('<div class="section-header">📐 Extracted Structure</div>',
                    unsafe_allow_html=True)

        st.markdown(f"**Title:** {notice.notice_title}")
        if notice.issuing_body:
            st.markdown(f"**Issued By:** {notice.issuing_body}")
        if notice.notice_number:
            st.markdown(f"**Reference No.:** `{notice.notice_number}`")
        if notice.issue_date:
            st.markdown(f"**Issue Date:** `{notice.issue_date}`")

        # Dates table
        # REPLACE the dates dataframe block with this:
        if notice.important_dates:
            st.markdown("**📅 Dates Detected:**")
            dates_data = [
                {
                    "Label": d.label.replace("_", " ").title(),
                    "Date":  d.normalized,
                    "Raw":   d.raw_text,
                }
                for d in notice.important_dates
                if d.label != "mentioned_date"   # hide generic unlabelled dates
            ]
            if dates_data:
                st.dataframe(dates_data, use_container_width=True, hide_index=True)

        # Tables
        if notice.tables:
            st.markdown(f"**📊 Tables ({len(notice.tables)} found):**")
            for i, table in enumerate(notice.tables):
                with st.expander(f"Table {i+1} ({len(table)} rows)"):
                    if table:
                        headers = table[0]
                        rows    = table[1:]
                        if rows:
                            import pandas as pd
                            try:
                                df = pd.DataFrame(rows, columns=headers)
                                st.dataframe(df, use_container_width=True)
                            except Exception:
                                st.write(table)
                        else:
                            st.write(table)

        # Raw text (collapsed by default)
        with st.expander("📄 View Raw Extracted Text"):
            st.text_area(
                "Extracted text",
                value=extracted.raw_text[:3000],
                height=250,
                disabled=True,
                label_visibility="collapsed",
            )

    with right_col:
        # ── AI Summary ────────────────────────────────────────────────────────
        st.markdown('<div class="section-header">🤖 AI-Generated Summary</div>',
                    unsafe_allow_html=True)

        st.markdown(
            f'<div class="summary-box">💬 {summary.one_line_summary}</div>',
            unsafe_allow_html=True
        )

        col_a, col_b = st.columns(2)
        col_a.markdown(f"**👥 Audience:** {summary.target_audience}")
        col_b.markdown(urgency_badge(summary.urgency_level), unsafe_allow_html=True)

        st.markdown(f"**🏢 Department:** {summary.relevant_department}")
        st.markdown(f"**✅ Action Required:** {summary.action_required}")

        st.markdown("**📌 Key Points:**")
        for point in summary.key_points:
            st.markdown(f"- {point}")

        # ADD this after the key_points loop:
        if summary.schedule:
            st.markdown("**🗓️ Subject-wise Schedule:**")
            import pandas as pd
            df = pd.DataFrame([
                {
                    "Subject": e.subject,
                    "Date":    e.date,
                    "Time":    e.time,
                    "Venue":   e.venue,
                }
                for e in summary.schedule
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)

        if summary.important_dates:
            st.markdown("**📅 Important Dates (from AI):**")
            for d in summary.important_dates:
                st.markdown(f"  - **{d.label}:** `{d.date}`")

        # ── Download JSON ─────────────────────────────────────────────────────
        st.divider()
        summary_json = json.dumps(summary.model_dump(), indent=2)
        st.download_button(
            label="⬇️ Download Summary as JSON",
            data=summary_json,
            file_name="notice_summary.json",
            mime="application/json",
            use_container_width=True,
        )


def _run_demo_mode(llm_backend: str):
    """Runs the pipeline on a built-in demo notice text (no file needed)."""
    from module3_structure_parser import StructuredNotice, NoticeDate
    from module4_langchain_summarizer import summarize_notice, print_summary
    from module2_text_extractor import ExtractedDocument

    st.info("Running demo mode with a sample notice...")

    # Create a fake extracted document
    demo_text = """
    DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING
    National Institute of Technology, Jamshedpur

    NOTICE
    Subject: Submission of Minor Project Reports - B.Tech Semester VI

    Ref. No.: CSE/NIT/2024/MP/042
    Date: 15 April 2024

    All students of B.Tech Semester VI are hereby informed that the submission
    of Minor Project Reports is scheduled as follows:

    1. Hard copy submission deadline: 30 April 2024
    2. Presentation/Viva date: 05 May 2024
    3. Final grade submission by faculty: 10 May 2024

    Students are required to submit two spiral-bound copies of their project
    report to the department office along with a soft copy (PDF) via the
    college portal by 30/04/2024.

    Reports not submitted by the deadline will be awarded zero marks.
    Students facing any issues should contact their project supervisor.

    HOD, Department of Computer Science and Engineering
    """

    extracted = ExtractedDocument(raw_text=demo_text, pages=[demo_text])
    notice    = parse_structure(extracted)

    with st.spinner("🤖 Generating AI summary..."):
        summary = summarize_notice(notice, llm_backend)

    _render_results(
        doc_info={"file_name": "demo_notice.txt", "file_size_kb": 1.2,
                  "page_count": 1, "doc_type": "digital_pdf"},
        extracted=extracted,
        notice=notice,
        summary=summary,
    )


if __name__ == "__main__":
    main()
