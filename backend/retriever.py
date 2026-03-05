# :   — Retriever + RAG Answer Generation
# Run-time steps:
#   1) Ensure FAISS index exists (upload docs via the UI or run scripts/build_faiss.py)
#   2) Import as: from backend.retriever import answer_query

"""
retriever.py — End-to-end Retrieval-Augmented Generation pipeline.

Given a student query:
  1. Embed the query.
  2. Retrieve top-k documents from FAISS.
  3. Generate a concise answer via Gemini (with template fallback).
  4. Attach provenance metadata.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

TOP_K: int = 5
SNIPPET_MAX_WORDS: int = 25


# ---------------------------------------------------------------------------
# Answer generation backends
# ---------------------------------------------------------------------------


def _generate_answer_gemini(query: str, passages: List[str]) -> str:
    """Generate answer using Google Gemini API."""
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set. Falling back to template.")
        return _generate_answer_template(passages)

    try:
        from google import genai  # type: ignore
        from google.genai import types as genai_types  # type: ignore

        client = genai.Client(api_key=api_key)
        # Truncate each passage to ~500 words and total context to ~3000 words
        trimmed = []
        total_words = 0
        for p in passages:
            words = p.split()
            if total_words + len(words) > 3000:
                remaining = 3000 - total_words
                if remaining > 50:
                    trimmed.append(" ".join(words[:remaining]))
                break
            trimmed.append(" ".join(words[:500]))
            total_words += min(len(words), 500)

        context = "\n\n".join(trimmed)
        prompt = (
            "You are CoursePilot, a knowledgeable university tutor. "
            "Answer the student's question using the provided course materials.\n\n"
            "Guidelines:\n"
            "- Provide a clear, well-structured answer (150-400 words)\n"
            "- Use **bold** for key terms and concepts\n"
            "- Break down complex ideas into numbered points when helpful\n"
            "- If the context is insufficient, acknowledge what is missing\n\n"
            f"Context from course materials:\n{context}\n\n"
            f"Student's Question: {query}"
        )
        GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                max_output_tokens=8192,
                temperature=0.3,
                http_options=genai_types.HttpOptions(timeout=60_000),
            ),
        )
        return response.text.strip()
    except Exception as exc:
        logger.warning("Gemini call failed (%s). Using template.", exc)
        return _generate_answer_template(passages)


def _generate_answer_template(passages: List[str]) -> str:
    """Deterministic templated answer from top-3 passages.

    Used when no LLM is available. Always returns something useful.
    """
    top = passages[:3]
    bullets = "\n".join(f"  • {p[:200]}" for p in top)
    return (
        "Based on the retrieved course materials:\n"
        f"{bullets}\n\n"
        "For a deeper explanation, consult the linked slides and "
        "lecture segments in the provenance section below."
    )


# ---------------------------------------------------------------------------
# Provenance formatting
# ---------------------------------------------------------------------------

def _make_snippet(text: str, max_words: int = SNIPPET_MAX_WORDS) -> str:
    """Truncate text to max_words."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " …"


def _build_provenance(result: Dict[str, Any]) -> Dict[str, Any]:
    """Build a provenance dict from a FAISS search result."""
    meta = result.get("metadata", result)
    return {
        "source_type": result.get("source_type", meta.get("source_type", "unknown")),
        "filename": meta.get("filename", ""),
        "slide_id": result.get("doc_id", meta.get("doc_id", "")),
        "start_time": meta.get("start_time", None),
        "snippet": _make_snippet(result.get("text", "")),
    }


# ---------------------------------------------------------------------------
# Gemini-powered generation helpers
# ---------------------------------------------------------------------------

def generate_follow_up_questions(query: str, answer: str, n: int = 5) -> List[str]:
    """Generate follow-up questions a student might ask next."""
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return []
    try:
        from google import genai  # type: ignore
        from google.genai import types as genai_types  # type: ignore

        client = genai.Client(api_key=api_key)
        prompt = (
            f"A student asked: \"{query}\"\n"
            f"They received this answer: \"{answer[:500]}\"\n\n"
            f"Generate exactly {n} natural follow-up questions the student "
            f"might ask next to deepen their understanding. Return ONLY the "
            f"questions, one per line, numbered 1 to {n}."
        )
        GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                max_output_tokens=8192,
                temperature=0.7,
                http_options=genai_types.HttpOptions(timeout=30_000),
            ),
        )
        questions = []
        for line in response.text.strip().split("\n"):
            line = re.sub(r"^\d+[\.)\-]\s*", "", line.strip()).strip()
            if line and len(line) > 10:
                questions.append(line)
        return questions[:n]
    except Exception as exc:
        logger.warning("Follow-up generation failed: %s", exc)
        return []


def generate_qa_from_documents(
    faiss_index_dir: str = "",
    n_questions: int = 15,
    question_type: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Generate Q&A study pairs from indexed documents using Gemini."""
    if not faiss_index_dir:
        faiss_index_dir = str(Path(__file__).resolve().parent.parent / "data" / "faiss_index")
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return []
    try:
        from backend.faiss_index import load_index

        _, metadata = load_index(faiss_index_dir)

        # Sample representative passages evenly
        passages: List[str] = []
        step = max(1, len(metadata) // 12)
        for i in range(0, len(metadata), step):
            text = metadata[i].get("text", "")
            if text.strip():
                passages.append(text[:400])
            if len(passages) >= 12:
                break
        if not passages:
            return []

        context = "\n\n---\n\n".join(passages)
        type_hint = (
            f" Focus on {question_type.lower()} questions."
            if question_type
            else ""
        )

        from google import genai  # type: ignore
        from google.genai import types as genai_types  # type: ignore

        client = genai.Client(api_key=api_key)
        prompt = (
            f"Based on the following course material, generate exactly "
            f"{n_questions} study questions with detailed answers.{type_hint}\n\n"
            f"Course Material:\n{context}\n\n"
            f"Format each Q&A pair exactly as:\n"
            f"Q: [question text]\n"
            f"A: [detailed answer text]\n\n"
            f"Generate exactly {n_questions} pairs."
        )
        GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                max_output_tokens=16384,
                temperature=0.5,
                http_options=genai_types.HttpOptions(timeout=120_000),
            ),
        )

        # Parse Q&A pairs
        text_out = response.text.strip()
        qa_pairs: List[Dict[str, str]] = []
        current_q: Optional[str] = None
        current_a: List[str] = []
        in_answer = False

        for raw_line in text_out.split("\n"):
            stripped = raw_line.strip()
            q_match = re.match(
                r"^(?:\*\*)?Q\d*[:\.\)]\s*(?:\*\*)?\s*(.*)",
                stripped, re.IGNORECASE,
            )
            a_match = re.match(
                r"^(?:\*\*)?A[:\.\)]\s*(?:\*\*)?\s*(.*)",
                stripped, re.IGNORECASE,
            )
            if q_match:
                if current_q and current_a:
                    qa_pairs.append({
                        "question": current_q,
                        "answer": " ".join(current_a).strip(),
                    })
                current_q = q_match.group(1).strip()
                current_a = []
                in_answer = False
            elif a_match:
                in_answer = True
                a_text = a_match.group(1).strip()
                if a_text:
                    current_a.append(a_text)
            elif in_answer and stripped:
                current_a.append(stripped)

        if current_q and current_a:
            qa_pairs.append({
                "question": current_q,
                "answer": " ".join(current_a).strip(),
            })

        return qa_pairs[:n_questions]
    except Exception as exc:
        logger.warning("Q&A generation failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Main retriever entry-point
# ---------------------------------------------------------------------------

def answer_query(
    query: str,
    k: int = TOP_K,
    data_dir: str = "",
    faiss_index_dir: str = "",
) -> Dict[str, Any]:
    """Answer a student query using RAG.

    Args:
        query: The student's natural-language question.
        k: Number of documents to retrieve.
        data_dir: Path to data directory.
        faiss_index_dir: Path to FAISS index directory.

    Returns:
        Dict with keys:
            answer: str — generated answer.
            provenance: List[Dict] — source metadata for each passage.
    """
    _root = Path(__file__).resolve().parent.parent
    if not faiss_index_dir:
        faiss_index_dir = str(_root / "data" / "faiss_index")
    if not data_dir:
        data_dir = str(_root / "data")

    from backend.embeddings import embed_query
    from backend.faiss_index import search, index_exists

    # 1. Check index
    if not index_exists(faiss_index_dir):
        return {
            "answer": (
                "⚠️ No documents indexed yet. Please upload course materials "
                "in the Upload & Ingest tab first."
            ),
            "provenance": [],
        }

    # 2. Embed query
    q_emb = embed_query(query)

    # 3. Retrieve
    results = search(q_emb, k=k, index_dir=faiss_index_dir)

    # 4. Build provenance
    provenance = [_build_provenance(r) for r in results]

    # 5. Extract passage texts
    passages = [r.get("text", "") for r in results]

    # 6. Generate answer (Gemini with template fallback)
    answer_text = _generate_answer_gemini(query, passages)

    return {
        "answer": answer_text,
        "provenance": provenance,
    }
