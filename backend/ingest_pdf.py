# :   — PDF Ingestion Pipeline
# What to change:
#   1. Replace sample data paths with real slide/paper PDFs.
#   2. Tune chunking strategy (chunk_size, overlap) for your content.
#
# Run-time steps:
#   1) pip install -r requirements.txt
#   2) python -m backend.ingest_pdf --input data/sample/slides.json --output data/processed/
#
# TODO[USER_ACTION]: Provide path to lecture slide PDFs in DATA_DIR or upload via UI.

"""
ingest_pdf.py — Extract text from PDF slides and past papers.

Supports:
  • Real PDF extraction via pdfplumber / PyMuPDF.
  • JSON-based sample data for demo mode (no PDFs needed).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PDF extraction helpers
# ---------------------------------------------------------------------------

def extract_text_from_pdf_pdfplumber(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract text page-by-page using pdfplumber.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of dicts: [{page: int, text: str}, ...]
    """
    try:
        import pdfplumber  # type: ignore
    except ImportError:
        logger.warning("pdfplumber not installed — returning empty extraction.")
        return []

    pages: List[Dict[str, Any]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append({"page": i + 1, "text": text.strip()})
    return pages


def extract_text_from_pdf_pymupdf(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract text page-by-page using PyMuPDF (fitz).

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of dicts: [{page: int, text: str}, ...]
    """
    try:
        import fitz  # PyMuPDF  # type: ignore
    except ImportError:
        logger.warning("PyMuPDF not installed — returning empty extraction.")
        return []

    pages: List[Dict[str, Any]] = []
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc):
        text = page.get_text()
        pages.append({"page": i + 1, "text": text.strip()})
    doc.close()
    return pages


# ---------------------------------------------------------------------------
# Slide ingestion (JSON sample data)
# ---------------------------------------------------------------------------

def load_slides_from_json(json_path: str) -> List[Dict[str, Any]]:
    """Load slide data from a JSON file (demo/sample mode).

    Args:
        json_path: Path to the slides JSON file.

    Returns:
        List of slide dicts with keys: slide_id, course, lecture, slide_number,
        title, body.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        slides = json.load(f)
    logger.info("Loaded %d slides from %s", len(slides), json_path)
    return slides


def slides_to_documents(slides: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert raw slide dicts into canonical document dicts for indexing.

    Each document has:
      - doc_id: unique identifier
      - source_type: "slide"
      - text: combined title + body
      - metadata: original slide metadata
    """
    docs: List[Dict[str, Any]] = []
    for s in slides:
        doc = {
            "doc_id": s["slide_id"],
            "source_type": "slide",
            "text": f"{s.get('title', '')}. {s.get('body', '')}",
            "metadata": {
                "course": s.get("course", ""),
                "lecture": s.get("lecture", 0),
                "slide_number": s.get("slide_number", 0),
                "filename": s.get("slide_id", ""),
            },
        }
        docs.append(doc)
    return docs


# ---------------------------------------------------------------------------
# Past paper ingestion
# ---------------------------------------------------------------------------

def load_past_papers_from_json(json_path: str) -> List[Dict[str, Any]]:
    """Load past paper questions from JSON (demo/sample mode).

    Args:
        json_path: Path to past_papers.json.

    Returns:
        List of paper dicts, each containing a list of questions.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        papers = json.load(f)
    logger.info("Loaded %d papers from %s", len(papers), json_path)
    return papers


def papers_to_documents(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert past-paper questions into canonical document dicts.

    Each question becomes its own document for retrieval.
    """
    docs: List[Dict[str, Any]] = []
    for paper in papers:
        for q in paper.get("questions", []):
            doc_id = f"{paper['paper_id']}_{q['q_id']}"
            docs.append({
                "doc_id": doc_id,
                "source_type": "past_paper",
                "text": q["text"],
                "metadata": {
                    "course": paper.get("course", ""),
                    "exam_type": paper.get("exam_type", ""),
                    "year": paper.get("year", 0),
                    "marks": q.get("marks", 0),
                    "topics": q.get("topics", []),
                    "filename": paper.get("paper_id", ""),
                },
            })
    return docs


# ---------------------------------------------------------------------------
# PDF ingestion (real PDFs)
# ---------------------------------------------------------------------------

def ingest_pdf_folder(
    folder_path: str,
    source_type: str = "slide",
    extractor: str = "pdfplumber",
) -> List[Dict[str, Any]]:
    """Ingest all PDFs in a folder and return canonical documents.

    Args:
        folder_path: Directory containing PDF files.
        source_type: "slide" or "past_paper".
        extractor: "pdfplumber" or "pymupdf".

    Returns:
        List of document dicts.
    """
    extract_fn = (
        extract_text_from_pdf_pdfplumber
        if extractor == "pdfplumber"
        else extract_text_from_pdf_pymupdf
    )
    docs: List[Dict[str, Any]] = []
    folder = Path(folder_path)
    if not folder.exists():
        logger.warning("Folder %s does not exist.", folder_path)
        return docs

    for pdf_file in sorted(folder.glob("*.pdf")):
        pages = extract_fn(str(pdf_file))
        for page in pages:
            doc_id = f"{pdf_file.stem}_p{page['page']}"
            docs.append({
                "doc_id": doc_id,
                "source_type": source_type,
                "text": page["text"],
                "metadata": {
                    "filename": pdf_file.name,
                    "page": page["page"],
                },
            })
    logger.info("Ingested %d pages from %s", len(docs), folder_path)
    return docs


# ---------------------------------------------------------------------------
# Convenience: build all documents from sample data
# ---------------------------------------------------------------------------

def build_all_sample_documents(data_dir: str = "data/sample") -> List[Dict[str, Any]]:
    """Build a complete document list from the sample dataset.

    Args:
        data_dir: Path to the sample data directory.

    Returns:
        Combined list of slide, transcript, and past-paper documents.
    """
    base = Path(data_dir)
    all_docs: List[Dict[str, Any]] = []

    # Slides
    slides_path = base / "slides.json"
    if slides_path.exists():
        slides = load_slides_from_json(str(slides_path))
        all_docs.extend(slides_to_documents(slides))

    # Past papers
    papers_path = base / "past_papers.json"
    if papers_path.exists():
        papers = load_past_papers_from_json(str(papers_path))
        all_docs.extend(papers_to_documents(papers))

    # Lecture transcript — delegate to ingest_audio for canonical form
    transcript_path = base / "lecture_transcript.json"
    if transcript_path.exists():
        from backend.ingest_audio import transcript_to_documents
        with open(str(transcript_path), "r", encoding="utf-8") as f:
            transcript = json.load(f)
        all_docs.extend(transcript_to_documents(transcript))

    logger.info("Total sample documents built: %d", len(all_docs))
    return all_docs


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Ingest PDFs / sample data")
    parser.add_argument("--input", default="data/sample/slides.json",
                        help="Path to slides JSON or PDF folder")
    parser.add_argument("--output", default="data/processed/",
                        help="Output directory for processed docs")
    parser.add_argument("--mock", action="store_true",
                        help="Use sample/mock data bundled in repo")
    args = parser.parse_args()

    if args.mock or args.input.endswith(".json"):
        docs = build_all_sample_documents(os.path.dirname(args.input))
    else:
        docs = ingest_pdf_folder(args.input)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "documents.json"
    with open(str(out_file), "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(docs)} documents to {out_file}")
