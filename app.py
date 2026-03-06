"""
app.py — CoursePilot: Campus Knowledge Graph & Just-In-Time Tutor.

Streamlit app with 7 tabs:
  0. How to Use      — guide for users and evaluation presentation
  1. Upload & Ingest — upload PDFs / audio / past papers
  2. Student Chat    — conversational RAG with follow-up suggestions
  3. Q&A Generator   — auto-generate study questions from documents
  4. NLP Analytics   — Word Cloud, FreqDist, TF-IDF similarity, VADER sentiment
  5. Faculty Dashboard — analytics, concept search, knowledge graph
  6. Admin Panel      — pipeline controls, data management, env status
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)          # ensure CWD = project root for all relative paths
load_dotenv()

# ─── Page config ──────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CoursePilot — CKG-JTT",
    page_icon="\U0001f393",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Logging ──────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

# ─── Session-state defaults ──────────────────────────────────────────────

_DEFAULTS = {
    "pipeline_logs": [],
    "messages": [],        # chat history [{role, content, provenance, follow_ups}]
    "generated_qa": [],    # Q&A pairs from generator
    "query_sentiments": [],  # sentiment scores from VADER for each user query
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def log_pipeline(msg: str) -> None:
    entry = f"[{time.strftime('%H:%M:%S')}] {msg}"
    st.session_state.pipeline_logs.append(entry)
    logger.info(msg)


# ─── Constants ────────────────────────────────────────────────────────────

FAISS_INDEX_DIR: str = os.getenv(
    "FAISS_INDEX_DIR", str(PROJECT_ROOT / "data" / "faiss_index")
)

# ─── Reusable pipeline helpers ────────────────────────────────────────────


def collect_all_documents(log_fn=None) -> list:
    """Scan all uploaded files and return canonical docs."""
    from backend.ingest_pdf import ingest_pdf_folder

    docs: list = []

    # Slide PDFs
    uploads_path = PROJECT_ROOT / "data" / "uploads"
    if uploads_path.exists() and list(uploads_path.glob("*.pdf")):
        pdf_docs = ingest_pdf_folder(str(uploads_path), source_type="slide")
        docs.extend(pdf_docs)
        if log_fn:
            log_fn(f"Slides: {len(pdf_docs)} pages from uploaded PDFs.")

    # Past paper PDFs
    papers_path = PROJECT_ROOT / "data" / "uploads" / "papers"
    if papers_path.exists() and list(papers_path.glob("*.pdf")):
        paper_docs = ingest_pdf_folder(str(papers_path), source_type="past_paper")
        docs.extend(paper_docs)
        if log_fn:
            log_fn(f"Past papers: {len(paper_docs)} pages.")

    # Audio transcripts
    audio_dir = PROJECT_ROOT / "data" / "uploads" / "audio"
    if audio_dir.exists():
        from backend.ingest_audio import transcribe_audio, transcript_to_documents

        for af in sorted(audio_dir.iterdir()):
            if af.suffix.lower() in (".mp3", ".wav", ".m4a"):
                if log_fn:
                    log_fn(f"Transcribing {af.name}\u2026")
                try:
                    transcript = transcribe_audio(str(af))
                except Exception as exc:
                    if log_fn:
                        log_fn(f"\u26a0\ufe0f {af.name}: transcription error — {exc}")
                    continue
                # Skip mock/error transcripts
                segs = transcript.get("segments", [])
                if segs and any(
                    "Mock transcript" in s.get("text", "")
                    or "Transcription failed" in s.get("text", "")
                    for s in segs
                ):
                    if log_fn:
                        reason = segs[0].get("text", "unknown error")
                        log_fn(f"\u26a0\ufe0f {af.name}: skipped — {reason}")
                    continue
                audio_docs = transcript_to_documents(transcript)
                docs.extend(audio_docs)
                if log_fn:
                    log_fn(f"Audio: {af.name} \u2192 {len(audio_docs)} segments.")

    return docs


def rebuild_pipeline(docs: list, log_fn=None) -> tuple:
    """Rebuild concepts, edges, FAISS index, and graph cache.

    Returns (n_concepts, n_edges, n_vectors).
    """
    from backend.embeddings import embed_texts
    from backend.extract_concepts import (
        extract_concepts_from_documents,
        extract_relations,
        save_concepts_csv,
        save_edges_csv,
    )
    from backend.faiss_index import build_index
    from backend.neo4j_import import reset_graph_cache

    # Concepts & edges
    concepts, concept_docs = extract_concepts_from_documents(docs)
    edges = extract_relations(docs, concepts, concept_docs)
    save_concepts_csv(concepts, str(PROJECT_ROOT / "data" / "concepts.csv"))
    save_edges_csv(edges, str(PROJECT_ROOT / "data" / "edges.csv"))
    if log_fn:
        log_fn(f"Concepts: {len(concepts)}, Edges: {len(edges)}")

    # FAISS index
    texts = [d["text"] for d in docs]
    if log_fn:
        log_fn("Computing embeddings\u2026")
    embeddings = embed_texts(texts)
    metadata = [
        {
            "doc_id": d["doc_id"],
            "source_type": d["source_type"],
            "text": d["text"],
            "metadata": d.get("metadata", {}),
        }
        for d in docs
    ]
    build_index(embeddings, metadata, FAISS_INDEX_DIR)
    if log_fn:
        log_fn(f"FAISS index: {len(docs)} vectors.")

    # Graph cache
    reset_graph_cache()

    return len(concepts), len(edges), len(docs)


# ─── Custom CSS ───────────────────────────────────────────────────────────

st.markdown(
    """<style>
    .hero-banner {
        padding: 1rem 1.25rem;
        border-radius: 12px;
        background: linear-gradient(135deg, rgba(99,102,241,.20), rgba(16,185,129,.15));
        border: 1px solid rgba(148,163,184,.18);
        margin-bottom: .75rem;
    }
    .hero-banner h3 { margin: 0 0 .2rem 0; }
    .hero-banner p  { margin: 0; opacity: .85; font-size: .88rem; }
</style>""",
    unsafe_allow_html=True,
)

# ─── Sidebar ─────────────────────────────────────────────────────────────

st.sidebar.markdown(
    '<div class="hero-banner">'
    "<h3>\U0001f393 CoursePilot</h3>"
    "<p>Campus Knowledge Graph &amp; Just-In-Time Tutor</p>"
    "</div>",
    unsafe_allow_html=True,
)

role = st.sidebar.selectbox(
    "\U0001f464 Select Role",
    ["student", "faculty", "admin"],
    index=0,
    help="Switch between student, faculty, and admin views.",
)

_USER_MAP = {
    "student": ("Alice Student", "alice@christuniversity.in"),
    "faculty": ("Dr. Ramesh Kumar", "ramesh.k@christuniversity.in"),
    "admin": ("Admin User", "admin@christuniversity.in"),
}
_uname, _uemail = _USER_MAP[role]
st.sidebar.markdown(f"**{_uname}**  \n`{_uemail}`")
st.sidebar.markdown("---")

# Sidebar quick stats
from backend.faiss_index import index_exists as _idx_exists  # noqa: E402

_CONCEPTS_CSV = PROJECT_ROOT / "data" / "concepts.csv"
_EDGES_CSV = PROJECT_ROOT / "data" / "edges.csv"

_n_docs = 0
_n_concepts = 0
_n_edges = 0
try:
    if _idx_exists(FAISS_INDEX_DIR):
        _mp = Path(FAISS_INDEX_DIR) / "metadata.json"
        if _mp.exists():
            _n_docs = len(json.loads(_mp.read_text(encoding="utf-8")))
    if _CONCEPTS_CSV.exists():
        _n_concepts = sum(1 for _ in open(_CONCEPTS_CSV, encoding="utf-8")) - 1
    if _EDGES_CSV.exists():
        _n_edges = sum(1 for _ in open(_EDGES_CSV, encoding="utf-8")) - 1
except Exception:
    pass

_sc1, _sc2, _sc3 = st.sidebar.columns(3)
_sc1.metric("\U0001f4c4 Pages", _n_docs)
_sc2.metric("\U0001f4a1 Concepts", _n_concepts)
_sc3.metric("\U0001f517 Relations", _n_edges)

st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit \u00b7 Gemini \u00b7 FAISS \u00b7 Neo4j")

# ─── Tab definitions ─────────────────────────────────────────────────────

tab_guide, tab_upload, tab_chat, tab_qa, tab_nlp, tab_faculty, tab_admin = st.tabs(
    [
        "\U0001f4d6 How to Use",
        "\U0001f4e4 Upload & Ingest",
        "\U0001f4ac Student Chat",
        "\U0001f4dd Q&A Generator",
        "\U0001f9e0 NLP Analytics",
        "\U0001f4ca Faculty Dashboard",
        "\u2699\ufe0f Admin",
    ]
)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 0 — How to Use Guide
# ═══════════════════════════════════════════════════════════════════════════

with tab_guide:
    st.header("\U0001f4d6 CoursePilot — How to Use")
    st.markdown(
        """
**CoursePilot** is a Streamlit-based educational tool that combines
**NLP**, **knowledge graphs**, **vector search**, and **generative AI**
to help students learn and help faculty analyze course materials.

---

### \U0001f680 Quick-Start (5 Steps)

| Step | Tab | What to Do |
|------|-----|-----------|
| **1** | **Upload & Ingest** | Upload your PDF lecture slides → click **Run Ingestion Pipeline** |
| **2** | **Student Chat** | Ask questions about your materials — AI answers using your documents |
| **3** | **Q&A Generator** | Auto-generate study questions for exam prep |
| **4** | **NLP Analytics** | Explore Word Clouds, concept frequencies, document similarity, and sentiment analysis |
| **5** | **Faculty Dashboard** | View knowledge graph, concept analytics, and course overview |

---

### \U0001f3af What Each Tab Does

**\U0001f4e4 Upload & Ingest**
- Upload PDF slides, audio recordings, or past question papers
- The pipeline extracts text (PDFPlumber), builds a **FAISS vector index**
  (sentence-transformers), extracts **concepts** (spaCy + NLTK lemmatization),
  and maps **relationships** between them — creating a knowledge graph

**\U0001f4ac Student Chat (RAG)**
- Type any question about your course → it searches the FAISS index for
  relevant document chunks → sends them to **Gemini AI** for a grounded answer
- Every answer includes source references and follow-up suggestions
- Your query sentiment is tracked (VADER) to show if questions are confused/neutral/confident

**\U0001f4dd Q&A Generator**
- Generates study questions and answers from your indexed documents
- Choose question type (Conceptual, Analytical, Short Answer) and quantity
- Download as a text file for offline revision

**\U0001f9e0 NLP Analytics** *(techniques from NLTK & sklearn labs)*
- **Word Cloud** — visual overview of the most prominent terms in your materials
- **Frequency Distribution** — bar chart of most frequent terms (NLTK `FreqDist`)
- **Document Similarity** — TF-IDF cosine similarity matrix between document pages (sklearn)
- **Sentiment Analysis** — VADER sentiment analyzer on student chat queries

**\U0001f4ca Faculty Dashboard**
- Concept frequency table and bar chart
- Interactive knowledge graph visualization (NetworkX + matplotlib)
- Course material overview metrics

**\u2699\ufe0f Admin Panel**
- Rebuild pipeline, import to Neo4j, view logs
- Clear all data for a new subject
- Environment status check

---

### \U0001f6e0\ufe0f Technology Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| UI Framework | **Streamlit** | Interactive web dashboard |
| PDF Extraction | **pdfplumber** + **PyMuPDF** | Extract text from lecture PDFs |
| Audio Transcription | **openai-whisper** | Transcribe lecture recordings |
| Text Preprocessing | **NLTK** + **spaCy** | Tokenization, lemmatization, stopword removal |
| Concept Extraction | **spaCy NER** + **NLTK** | Identify and normalize key concepts |
| Embeddings | **sentence-transformers** (all-MiniLM-L6-v2) | Convert text → 384-dim vectors |
| Vector Search | **FAISS** | Fast approximate nearest-neighbor retrieval |
| Knowledge Graph | **NetworkX** / **Neo4j** | Store concept relationships |
| Generative AI | **Google Gemini** | RAG answers, Q&A generation, follow-ups |
| NLP Analytics | **WordCloud**, **sklearn TF-IDF**, **NLTK VADER** | Visual analytics & sentiment |

---

### \U0001f3ac Evaluation Presentation Script

> **Step 1:** Open the app → *"This is CoursePilot, a knowledge-graph-powered
> tutor I built with Streamlit."*
>
> **Step 2:** Go to Upload & Ingest → upload a PDF → run the pipeline →
> *"The pipeline extracts text, builds concepts & relationships, and indexes
> everything in FAISS for semantic search."*
>
> **Step 3:** Open Student Chat → ask a question → *"It uses RAG — retrieves
> relevant chunks from FAISS, then sends them to Gemini for a grounded answer.
> Sentiment of each query is tracked using NLTK VADER."*
>
> **Step 4:** Open NLP Analytics → *"Here I applied techniques from our NLP labs —
> Word Cloud visualization, NLTK frequency distributions, TF-IDF cosine similarity
> between documents, and VADER sentiment analysis on student queries."*
>
> **Step 5:** Open Faculty Dashboard → *"Faculty can see the knowledge graph,
> search concepts, and review analytics."*
>
> **Step 6:** Show Admin → *"Admins can rebuild the pipeline, clear data for a
> new subject, and check environment health."*
"""
    )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Upload & Ingest
# ═══════════════════════════════════════════════════════════════════════════

with tab_upload:
    st.header("\U0001f4e4 Upload & Ingest Course Materials")
    st.markdown(
        "Upload lecture slides, audio recordings, or past question papers. "
        "CoursePilot extracts text, builds embeddings, and populates the "
        "knowledge graph automatically."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("\U0001f4d1 Lecture Slides")
        slide_files = st.file_uploader(
            "Upload PDF slides",
            type=["pdf"],
            accept_multiple_files=True,
            key="slide_upload",
        )
        if slide_files:
            st.success(f"{len(slide_files)} file(s) ready.")
    with col2:
        st.subheader("\U0001f399\ufe0f Lecture Audio")
        audio_files = st.file_uploader(
            "Upload audio files",
            type=["mp3", "wav", "m4a"],
            accept_multiple_files=True,
            key="audio_upload",
        )
        if audio_files:
            st.success(f"{len(audio_files)} file(s) ready.")
    with col3:
        st.subheader("\U0001f4dd Past Papers")
        paper_files = st.file_uploader(
            "Upload past question PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            key="paper_upload",
        )
        if paper_files:
            st.success(f"{len(paper_files)} file(s) ready.")

    st.markdown("---")

    if st.button("\u25b6\ufe0f Run Ingestion Pipeline", type="primary"):
        progress = st.progress(0, text="Starting ingestion\u2026")
        log_pipeline("Ingestion pipeline started.")

        upload_dir = PROJECT_ROOT / "data" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)

        all_files = (slide_files or []) + (audio_files or []) + (paper_files or [])
        if not all_files:
            # Check if there are existing files on disk to reprocess
            _existing = list((upload_dir).glob("*.pdf")) if upload_dir.exists() else []
            _existing += list((upload_dir / "papers").glob("*.pdf")) if (upload_dir / "papers").exists() else []
            _existing += [
                f for f in (upload_dir / "audio").iterdir()
                if f.suffix.lower() in (".mp3", ".wav", ".m4a")
            ] if (upload_dir / "audio").exists() else []
            if not _existing:
                st.warning("No files uploaded. Please upload at least one file.")
                log_pipeline("No uploads detected.")
            else:
                log_pipeline(f"Re-processing {len(_existing)} existing file(s) on disk.")

        # Save uploaded files to disk, tracking types separately
        _slide_names: list = []
        _paper_names: list = []
        _audio_names: list = []
        for f in (slide_files or []):
            (upload_dir / f.name).write_bytes(f.getbuffer())
            _slide_names.append(f.name)
            log_pipeline(f"Saved slide: {f.name}")
        for f in (paper_files or []):
            _papers_dir = PROJECT_ROOT / "data" / "uploads" / "papers"
            _papers_dir.mkdir(parents=True, exist_ok=True)
            (_papers_dir / f.name).write_bytes(f.getbuffer())
            _paper_names.append(f.name)
            log_pipeline(f"Saved past paper: {f.name}")
        for f in (audio_files or []):
            _audio_dir = PROJECT_ROOT / "data" / "uploads" / "audio"
            _audio_dir.mkdir(parents=True, exist_ok=True)
            (_audio_dir / f.name).write_bytes(f.getbuffer())
            _audio_names.append(f.name)
            log_pipeline(f"Saved audio: {f.name}")

        progress.progress(15, text="Files saved. Collecting documents\u2026")

        docs: list = []
        try:
            docs = collect_all_documents(log_fn=log_pipeline)
            if not docs:
                st.error(
                    "No processable documents found. "
                    "Upload PDF slides or install openai-whisper for audio support."
                )
                log_pipeline("ERROR: No documents to process.")
            else:
                log_pipeline(f"Total documents: {len(docs)}")
            progress.progress(40, text="Text extracted. Building concepts & index\u2026")
        except Exception as exc:
            st.error(f"Document extraction failed: {exc}")
            log_pipeline(f"ERROR: {exc}")

        _pipeline_ok = False
        if docs:
            try:
                n_c, n_e, n_v = rebuild_pipeline(docs, log_fn=log_pipeline)
                progress.progress(90, text="Pipeline complete. Finalizing\u2026")
                _pipeline_ok = True
            except Exception as exc:
                st.error(f"Pipeline failed: {exc}")
                log_pipeline(f"ERROR: {exc}")

        if _pipeline_ok:
            progress.progress(100, text="\u2705 Ingestion complete!")
            log_pipeline("Ingestion pipeline finished.")
            st.success(
                f"\u2705 Ingestion complete! "
                f"{n_v} document(s), {n_c} concepts, {n_e} relationships."
            )
            st.rerun()
        else:
            progress.progress(100, text="\u26a0\ufe0f Ingestion had issues.")
            if not docs:
                st.warning(
                    "No documents could be extracted. "
                    "**PDF slides** are recommended for best results. "
                    "Audio files require `openai-whisper` to be installed."
                )

    if st.session_state.pipeline_logs:
        with st.expander("\U0001f4cb Pipeline Logs", expanded=True):
            for entry in st.session_state.pipeline_logs[-20:]:
                st.text(entry)

    # ── Manage Uploaded Files ──────────────────────────────────────────────
    st.markdown("---")
    st.subheader("\U0001f4c1 Manage Uploaded Files")
    st.caption(
        "Remove individual files below. The pipeline will automatically "
        "rebuild concepts, edges, and the search index from the remaining files."
    )

    _up_root = PROJECT_ROOT / "data" / "uploads"
    _mgr_slides = sorted(_up_root.glob("*.pdf")) if _up_root.exists() else []
    _mgr_papers = sorted(((_up_root / "papers").glob("*.pdf"))) if (_up_root / "papers").exists() else []
    _mgr_audio = (
        sorted(
            f for f in (_up_root / "audio").iterdir()
            if f.suffix.lower() in (".mp3", ".wav", ".m4a")
        )
        if (_up_root / "audio").exists()
        else []
    )
    _mgr_total = len(_mgr_slides) + len(_mgr_papers) + len(_mgr_audio)

    if _mgr_total == 0:
        st.info("No uploaded files yet. Use the upload section above to add materials.")
    else:
        _files_to_delete: list = []

        if _mgr_slides:
            st.markdown("**\U0001f4d1 Lecture Slides**")
            for _f in _mgr_slides:
                _sz = _f.stat().st_size / 1024
                if st.checkbox(
                    f"\U0001f4c4 {_f.name} ({_sz:.0f} KB)",
                    key=f"mgr_slide_{_f.name}",
                ):
                    _files_to_delete.append(_f)

        if _mgr_papers:
            st.markdown("**\U0001f4dd Past Papers**")
            for _f in _mgr_papers:
                _sz = _f.stat().st_size / 1024
                if st.checkbox(
                    f"\U0001f4c4 {_f.name} ({_sz:.0f} KB)",
                    key=f"mgr_paper_{_f.name}",
                ):
                    _files_to_delete.append(_f)

        if _mgr_audio:
            st.markdown("**\U0001f399\ufe0f Audio Files**")
            for _f in _mgr_audio:
                _sz = _f.stat().st_size / 1024
                if st.checkbox(
                    f"\U0001f399\ufe0f {_f.name} ({_sz:.0f} KB)",
                    key=f"mgr_audio_{_f.name}",
                ):
                    _files_to_delete.append(_f)

        if _files_to_delete:
            st.warning(f"**{len(_files_to_delete)} file(s) selected for removal.**")
            if st.button(
                "\U0001f5d1\ufe0f Remove Selected & Rebuild Pipeline",
                type="primary",
                key="mgr_delete_rebuild",
            ):
                # Delete selected files
                for _f in _files_to_delete:
                    _f.unlink()
                    log_pipeline(f"Removed: {_f.name}")

                # Clean up empty subdirectories
                for _subdir in [_up_root / "papers", _up_root / "audio"]:
                    if _subdir.exists() and not any(_subdir.iterdir()):
                        _subdir.rmdir()

                # Rebuild pipeline from remaining data
                with st.spinner(
                    "Rebuilding concepts, edges & index from remaining files\u2026"
                ):
                    try:
                        remaining = collect_all_documents(log_fn=log_pipeline)
                        if remaining:
                            n_c, n_e, n_v = rebuild_pipeline(
                                remaining, log_fn=log_pipeline
                            )
                            log_pipeline(
                                f"Rebuild complete: {n_c} concepts, "
                                f"{n_e} edges, {n_v} vectors."
                            )
                            st.success(
                                f"\u2705 Removed {len(_files_to_delete)} file(s). "
                                f"Rebuilt with {n_v} documents, "
                                f"{n_c} concepts, {n_e} edges."
                            )
                        else:
                            # No documents left — clear derived data
                            import shutil

                            _fi = Path(FAISS_INDEX_DIR)
                            if _fi.exists():
                                shutil.rmtree(str(_fi))
                            for _csv in (_CONCEPTS_CSV, _EDGES_CSV):
                                _cp = Path(_csv)
                                if _cp.exists():
                                    _cp.unlink()
                            from backend.neo4j_import import reset_graph_cache

                            reset_graph_cache()
                            log_pipeline("All files removed. Derived data cleared.")
                            st.success(
                                "\u2705 All files removed. "
                                "Index and concepts cleared."
                            )
                    except Exception as _exc:
                        st.error(f"Rebuild failed: {_exc}")
                        log_pipeline(f"ERROR rebuild: {_exc}")
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — Student Chat (RAG)
# ═══════════════════════════════════════════════════════════════════════════

with tab_chat:
    st.header("\U0001f4ac Ask CoursePilot")

    if not _idx_exists(FAISS_INDEX_DIR):
        st.warning(
            "\u26a0\ufe0f No documents indexed yet. Go to the **Upload & Ingest** tab "
            "to upload your course materials first."
        )
    else:
        st.caption(
            "Ask any question about your course materials. CoursePilot retrieves "
            "relevant content and generates a detailed answer powered by Gemini AI."
        )

        # Consume any pending follow-up question
        _pending_q = st.session_state.pop("_pending_q", None)

        # ── Render chat history ──
        for _msg in st.session_state.messages:
            with st.chat_message(_msg["role"]):
                st.markdown(_msg["content"])
                if _msg["role"] == "assistant" and _msg.get("provenance"):
                    with st.expander("\U0001f4cc Sources & References", expanded=False):
                        for _j, _prov in enumerate(_msg["provenance"], 1):
                            _src = _prov.get("source_type", "unknown")
                            _snip = _prov.get("snippet", "")
                            _icon = {"slide": "\U0001f4d1", "lecture": "\U0001f399\ufe0f"}.get(
                                _src, "\U0001f4dd"
                            )
                            st.caption(f"{_icon} **[{_j}]** {_src.title()} \u2014 _{_snip}_")

        # ── Follow-up suggestions from last assistant message ──
        if (
            st.session_state.messages
            and st.session_state.messages[-1]["role"] == "assistant"
            and st.session_state.messages[-1].get("follow_ups")
        ):
            _fu_prefix = len(st.session_state.messages)
            st.markdown("**\U0001f4a1 Suggested follow-up questions:**")
            for _fi, _fq in enumerate(st.session_state.messages[-1]["follow_ups"]):
                if st.button(f"\u27a4 {_fq}", key=f"fu_{_fu_prefix}_{_fi}"):
                    st.session_state["_pending_q"] = _fq
                    st.rerun()

        # ── Starter questions (shown only when chat is empty) ──
        if not st.session_state.messages:
            st.markdown("---")
            st.markdown("**\U0001f3af Try asking:**")
            _starters = [
                "Summarize the key concepts from the uploaded documents",
                "What are the main topics covered in the course materials?",
                "Explain the most important concept from the lectures",
            ]
            _starter_cols = st.columns(len(_starters))
            for _si, _sq in enumerate(_starters):
                with _starter_cols[_si]:
                    if st.button(
                        _sq, key=f"starter_{_si}", use_container_width=True,
                    ):
                        st.session_state["_pending_q"] = _sq
                        st.rerun()

        # ── Chat input ──
        _user_input = st.chat_input("Ask about your course materials\u2026")
        _query = _pending_q or _user_input

        if _query:
            # Track query sentiment (VADER)
            try:
                from nltk.sentiment import SentimentIntensityAnalyzer as _SIA
                _sia = _SIA()
                _sent_scores = _sia.polarity_scores(_query)
                st.session_state.query_sentiments.append({
                    "query": _query[:80],
                    "compound": _sent_scores["compound"],
                    "pos": _sent_scores["pos"],
                    "neg": _sent_scores["neg"],
                    "neu": _sent_scores["neu"],
                })
            except Exception:
                pass

            # Show user message
            st.session_state.messages.append({"role": "user", "content": _query})
            with st.chat_message("user"):
                st.markdown(_query)

            # Generate answer
            with st.chat_message("assistant"):
                with st.spinner("Searching course materials\u2026"):
                    try:
                        from backend.retriever import (
                            answer_query,
                            generate_follow_up_questions,
                        )

                        _result = answer_query(
                            _query,
                            faiss_index_dir=FAISS_INDEX_DIR,
                        )
                        _answer = _result["answer"]
                        _provenance = _result.get("provenance", [])
                        _follow_ups = generate_follow_up_questions(_query, _answer)
                    except Exception as _exc:
                        _answer = f"Sorry, an error occurred: {_exc}"
                        _provenance = []
                        _follow_ups = []

                st.markdown(_answer)

            # Save to history and rerun to show follow-ups properly
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": _answer,
                    "provenance": _provenance,
                    "follow_ups": _follow_ups,
                }
            )
            st.rerun()

        # ── Clear chat ──
        if st.session_state.messages:
            if st.button("\U0001f5d1\ufe0f Clear Chat History"):
                st.session_state.messages = []
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — Q&A Generator
# ═══════════════════════════════════════════════════════════════════════════

with tab_qa:
    st.header("\U0001f4dd Study Q&A Generator")
    st.markdown(
        "Automatically generate study questions and answers from your "
        "uploaded course materials. Use these for self-assessment and "
        "exam preparation."
    )

    if not _idx_exists(FAISS_INDEX_DIR):
        st.warning("\u26a0\ufe0f Upload and ingest documents first.")
    else:
        _qa_c1, _qa_c2 = st.columns([2, 1])
        with _qa_c1:
            _n_qs = st.slider(
                "Number of questions to generate",
                min_value=5,
                max_value=25,
                value=10,
                step=5,
                help="More questions take longer to generate.",
            )
            _q_type = st.pills(
                "Question Type",
                ["All Types", "Conceptual", "Analytical", "Short Answer"],
                default="All Types",
            )
        with _qa_c2:
            st.info(
                f"\U0001f4c4 **{_n_docs}** document pages indexed\n\n"
                f"\U0001f4a1 **{_n_concepts}** concepts extracted\n\n"
                f"\U0001f517 **{_n_edges}** relationships mapped\n\n"
                "\U0001f916 Powered by Gemini AI"
            )

        if st.button("\U0001f680 Generate Questions", type="primary"):
            with st.spinner(
                "Generating study questions\u2026 this may take 30\u201360 seconds."
            ):
                try:
                    from backend.retriever import generate_qa_from_documents

                    _qa_pairs = generate_qa_from_documents(
                        faiss_index_dir=FAISS_INDEX_DIR,
                        n_questions=_n_qs,
                        question_type=_q_type if _q_type != "All Types" else None,
                    )
                    st.session_state.generated_qa = _qa_pairs
                except Exception as _exc:
                    st.error(f"Generation failed: {_exc}")

        # Display generated Q&A
        if st.session_state.generated_qa:
            st.markdown("---")
            st.subheader(
                f"\U0001f4cb Generated Questions ({len(st.session_state.generated_qa)})"
            )

            for _qi, _qa in enumerate(st.session_state.generated_qa, 1):
                with st.expander(
                    f"**Q{_qi}.** {_qa['question']}", expanded=(_qi <= 2)
                ):
                    st.markdown(_qa["answer"])
                    st.feedback("thumbs", key=f"qa_fb_{_qi}")

            # Download button
            _qa_text = "\n\n".join(
                f"Q{i}: {qa['question']}\nA: {qa['answer']}"
                for i, qa in enumerate(st.session_state.generated_qa, 1)
            )
            st.download_button(
                "\U0001f4e5 Download Q&A as Text",
                data=_qa_text,
                file_name="coursepilot_study_questions.txt",
                mime="text/plain",
            )


# ═══════════════════════════════════════════════════════════════════════════
# TAB — NLP Analytics (Word Cloud, FreqDist, Similarity, Sentiment)
# ═══════════════════════════════════════════════════════════════════════════

with tab_nlp:
    st.header("\U0001f9e0 NLP Analytics — Applied Techniques")
    st.caption(
        "Techniques from NLTK, sklearn, and WordCloud labs, applied to your "
        "actual course materials."
    )

    if not _idx_exists(FAISS_INDEX_DIR):
        st.warning(
            "\u26a0\ufe0f No documents indexed yet. Upload and ingest materials first."
        )
    else:
        # Load document texts from FAISS metadata
        _meta_path = Path(FAISS_INDEX_DIR) / "metadata.json"
        _all_texts: list[str] = []
        _doc_labels: list[str] = []
        if _meta_path.exists():
            _meta = json.loads(_meta_path.read_text(encoding="utf-8"))
            for _m in _meta:
                _t = _m.get("text", "")
                if _t.strip():
                    _all_texts.append(_t)
                    _doc_labels.append(
                        _m.get("metadata", {}).get("filename", "doc") + f" p{_m.get('doc_id', '?')}"
                    )

        if not _all_texts:
            st.info("No document text available.")
        else:
            import matplotlib.pyplot as plt
            import numpy as np
            import re as _re

            # Preprocessing helper
            from nltk.corpus import stopwords as _sw_corpus
            try:
                _stop_words = set(_sw_corpus.words("english"))
            except LookupError:
                import nltk as _nltk_dl
                _nltk_dl.download("stopwords", quiet=True)
                _stop_words = set(_sw_corpus.words("english"))

            _combined_text = " ".join(_all_texts)

            def _clean_text(text: str) -> str:
                text = text.lower()
                text = _re.sub(r"[^a-z\s]", "", text)
                text = _re.sub(r"\s+", " ", text).strip()
                return " ".join(w for w in text.split() if w not in _stop_words and len(w) > 2)

            _cleaned_combined = _clean_text(_combined_text)

            # ── Section 1: Word Cloud ──────────────────────────────────
            st.subheader("\u2601\ufe0f Word Cloud of Course Materials")
            st.caption("Visual representation of the most prominent terms (WordCloud library, from NLTK lab).")

            try:
                from wordcloud import WordCloud as _WC

                _wc = _WC(
                    width=1000,
                    height=400,
                    background_color="white",
                    max_words=150,
                    colormap="viridis",
                    collocations=False,
                ).generate(_cleaned_combined)

                _fig_wc, _ax_wc = plt.subplots(figsize=(12, 5))
                _ax_wc.imshow(_wc, interpolation="bilinear")
                _ax_wc.axis("off")
                _ax_wc.set_title("Word Cloud — Key Terms from Course Materials", fontsize=13)
                st.pyplot(_fig_wc)
                plt.close(_fig_wc)
            except ImportError:
                st.info("Install `wordcloud` package: `pip install wordcloud`")

            st.markdown("---")

            # ── Section 2: Frequency Distribution ──────────────────────
            st.subheader("\U0001f4ca Frequency Distribution (NLTK FreqDist)")
            st.caption("Top N most frequent terms after stopword removal and cleaning.")

            import nltk as _nltk_fd
            _words_list = _nltk_fd.tokenize.word_tokenize(_cleaned_combined)
            _fd = _nltk_fd.FreqDist(_words_list)

            _top_n_fd = st.slider("Number of terms", 10, 40, 20, key="nlp_fd_n")
            _top_words = _fd.most_common(_top_n_fd)
            _labels_fd, _counts_fd = zip(*_top_words) if _top_words else ([], [])

            _fig_fd, _ax_fd = plt.subplots(figsize=(12, 4))
            _ax_fd.bar(range(len(_labels_fd)), _counts_fd, color="#6366f1", edgecolor="white")
            _ax_fd.set_xticks(range(len(_labels_fd)))
            _ax_fd.set_xticklabels(_labels_fd, rotation=45, ha="right", fontsize=8)
            _ax_fd.set_ylabel("Frequency")
            _ax_fd.set_title(f"Top {_top_n_fd} Terms — Frequency Distribution")
            plt.tight_layout()
            st.pyplot(_fig_fd)
            plt.close(_fig_fd)

            st.markdown("---")

            # ── Section 3: TF-IDF Document Similarity ──────────────────
            st.subheader("\U0001f50d Document Similarity (TF-IDF + Cosine)")
            st.caption(
                "Compute TF-IDF vectors for each page, then measure cosine similarity "
                "between documents (sklearn, from NLTK/NLP lab)."
            )

            from sklearn.feature_extraction.text import TfidfVectorizer as _TfidfV
            from sklearn.metrics.pairwise import cosine_similarity as _cos_sim

            # Limit to first N docs for performance
            _max_sim_docs = min(30, len(_all_texts))
            _sim_texts = _all_texts[:_max_sim_docs]

            _tfidf = _TfidfV(max_features=500, stop_words="english")
            _tfidf_matrix = _tfidf.fit_transform(_sim_texts)
            _sim_matrix = _cos_sim(_tfidf_matrix)

            _fig_sim, _ax_sim = plt.subplots(figsize=(10, 8))
            _im = _ax_sim.imshow(_sim_matrix, cmap="YlOrRd", aspect="auto")
            _ax_sim.set_title(f"Cosine Similarity Matrix ({_max_sim_docs} documents)")
            _ax_sim.set_xlabel("Document Index")
            _ax_sim.set_ylabel("Document Index")
            _fig_sim.colorbar(_im, ax=_ax_sim, label="Similarity")
            st.pyplot(_fig_sim)
            plt.close(_fig_sim)

            # Show most similar pair
            _np = np.array(_sim_matrix)
            np.fill_diagonal(_np, 0)
            _max_idx = np.unravel_index(_np.argmax(), _np.shape)
            st.info(
                f"**Most similar pair:** Document {_max_idx[0]+1} & "
                f"Document {_max_idx[1]+1} — "
                f"similarity = {_np[_max_idx]:.3f}"
            )

            st.markdown("---")

            # ── Section 4: Sentiment Analysis of Student Queries ──────
            st.subheader("\U0001f4ac Sentiment Analysis (NLTK VADER)")
            st.caption(
                "VADER sentiment scores for student queries in the chat. "
                "Tracks whether questions are confident, neutral, or confused."
            )

            _sentiments = st.session_state.get("query_sentiments", [])
            if not _sentiments:
                st.info(
                    "No student queries yet. Go to the **Student Chat** tab, "
                    "ask some questions, and return here to see sentiment analysis."
                )
            else:
                import pandas as _pd_sent

                _df_sent = _pd_sent.DataFrame(_sentiments)

                # Sentiment distribution
                def _classify_sent(compound):
                    if compound >= 0.05:
                        return "Positive"
                    elif compound <= -0.05:
                        return "Negative"
                    return "Neutral"

                _df_sent["sentiment"] = _df_sent["compound"].apply(_classify_sent)

                _sc1, _sc2 = st.columns([1, 2])

                with _sc1:
                    _dist = _df_sent["sentiment"].value_counts()
                    _colors = {"Positive": "#22c55e", "Neutral": "#6366f1", "Negative": "#ef4444"}
                    _fig_sp, _ax_sp = plt.subplots(figsize=(4, 4))
                    _ax_sp.bar(
                        _dist.index,
                        _dist.values,
                        color=[_colors.get(s, "#999") for s in _dist.index],
                    )
                    _ax_sp.set_title("Query Sentiment Distribution")
                    _ax_sp.set_ylabel("Count")
                    plt.tight_layout()
                    st.pyplot(_fig_sp)
                    plt.close(_fig_sp)

                with _sc2:
                    st.markdown("**Query History with Sentiment Scores:**")
                    _display_sent = _df_sent[["query", "compound", "sentiment"]].copy()
                    _display_sent.columns = ["Query", "Score", "Sentiment"]
                    st.dataframe(_display_sent, hide_index=True, height=300)

                # Compound score over time
                if len(_sentiments) > 1:
                    _fig_ts, _ax_ts = plt.subplots(figsize=(10, 3))
                    _ax_ts.plot(
                        range(1, len(_sentiments) + 1),
                        _df_sent["compound"],
                        marker="o",
                        color="#6366f1",
                        linewidth=2,
                    )
                    _ax_ts.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
                    _ax_ts.fill_between(
                        range(1, len(_sentiments) + 1),
                        _df_sent["compound"],
                        alpha=0.15,
                        color="#6366f1",
                    )
                    _ax_ts.set_xlabel("Query Number")
                    _ax_ts.set_ylabel("Compound Score")
                    _ax_ts.set_title("Sentiment Trend Across Queries")
                    _ax_ts.set_ylim(-1.1, 1.1)
                    plt.tight_layout()
                    st.pyplot(_fig_ts)
                    plt.close(_fig_ts)


# ═══════════════════════════════════════════════════════════════════════════
# TAB — Faculty Dashboard
# ═══════════════════════════════════════════════════════════════════════════

with tab_faculty:
    st.header("\U0001f4ca Faculty Dashboard \u2014 Course Analytics")

    if role not in ("faculty", "admin"):
        st.warning(
            "\U0001f512 Faculty or Admin role required. Switch role in the sidebar."
        )
    else:
        try:
            from backend.neo4j_import import get_all_concepts, get_graph

            _concepts = get_all_concepts()
        except Exception:
            _concepts = []

        if not _concepts:
            st.info(
                "No concept data available yet. Run the ingestion pipeline first "
                "(Upload & Ingest tab)."
            )
        else:
            import pandas as pd

            _df = pd.DataFrame(_concepts)

            # ── Metrics Row ──
            st.subheader("\U0001f4c8 Course Material Overview")
            _m1, _m2, _m3, _m4 = st.columns(4)
            _m1.metric("\U0001f4c4 Document Pages", _n_docs)
            _m2.metric("\U0001f4a1 Unique Concepts", len(_concepts))
            _G_live = get_graph()
            _m3.metric("\U0001f517 Relationships", _G_live.number_of_edges())
            _avg_freq = (
                round(_df["frequency"].mean(), 1)
                if "frequency" in _df.columns
                else 0
            )
            _m4.metric("\U0001f4ca Avg Frequency", _avg_freq)

            st.markdown("---")

            # ── Concept Frequency Chart + Search Table ──
            _col_chart, _col_tbl = st.columns([3, 2])

            with _col_chart:
                st.subheader("\U0001f3f7\ufe0f Top Concepts by Frequency")
                if "label" in _df.columns and "frequency" in _df.columns:
                    _topn = st.slider(
                        "Number of concepts", 10, 50, 20, key="fac_topn"
                    )
                    _chart_data = _df.nlargest(_topn, "frequency").set_index("label")[
                        "frequency"
                    ]
                    st.bar_chart(_chart_data)

            with _col_tbl:
                st.subheader("\U0001f50d Concept Search")
                _search = st.text_input(
                    "Filter concepts",
                    placeholder="Type to search\u2026",
                    key="concept_search",
                )
                _disp = _df[["label", "frequency"]].copy()
                _disp.columns = ["Concept", "Frequency"]
                _disp = _disp.sort_values("Frequency", ascending=False)
                if _search:
                    _disp = _disp[
                        _disp["Concept"].str.contains(_search, case=False, na=False)
                    ]
                st.dataframe(_disp, height=400, hide_index=True)

            st.markdown("---")

            # ── Frequency Distribution Histogram ──
            st.subheader("\U0001f4ca Concept Frequency Distribution")
            if "frequency" in _df.columns:
                import matplotlib.pyplot as plt

                _fig_h, _ax_h = plt.subplots(figsize=(10, 3))
                _freq_clipped = _df["frequency"].clip(
                    upper=_df["frequency"].quantile(0.95)
                )
                _ax_h.hist(
                    _freq_clipped,
                    bins=30,
                    color="#6366f1",
                    alpha=0.7,
                    edgecolor="white",
                )
                _ax_h.set_xlabel("Frequency")
                _ax_h.set_ylabel("Number of Concepts")
                _ax_h.set_title("Distribution of Concept Frequencies")
                st.pyplot(_fig_h)
                plt.close(_fig_h)

            st.markdown("---")

            # ── Knowledge Graph Visualization ──
            st.subheader("\U0001f578\ufe0f Knowledge Graph Visualization")
            try:
                import matplotlib.pyplot as plt
                import networkx as nx

                _G = get_graph()
                if _G.number_of_nodes() > 0:
                    _max_nodes = st.slider(
                        "Concepts to visualize", 10, 60, 30, key="graph_n"
                    )
                    _sorted_n = sorted(
                        _G.nodes(data=True),
                        key=lambda x: x[1].get("frequency", 0),
                        reverse=True,
                    )
                    _top_ids = [n for n, _ in _sorted_n[:_max_nodes]]
                    _subG = _G.subgraph(_top_ids).copy()

                    _fig_g, _ax_g = plt.subplots(figsize=(12, 7))
                    _pos = nx.spring_layout(_subG, seed=42, k=1.5)
                    _nsizes = [
                        _subG.nodes[n].get("frequency", 1) * 80 for n in _subG.nodes
                    ]
                    _labels = {
                        n: _subG.nodes[n].get("label", n)[:18] for n in _subG.nodes
                    }
                    nx.draw_networkx(
                        _subG,
                        _pos,
                        ax=_ax_g,
                        labels=_labels,
                        node_size=_nsizes,
                        font_size=6,
                        node_color="#6366f1",
                        edge_color="#e2e8f0",
                        arrows=True,
                        alpha=0.9,
                    )
                    _ax_g.set_title(
                        f"Top {len(_subG.nodes)} Concepts "
                        f"(of {_G.number_of_nodes()} total)"
                    )
                    _ax_g.axis("off")
                    st.pyplot(_fig_g)
                    plt.close(_fig_g)

                    st.caption(
                        f"Showing {_subG.number_of_nodes()} nodes and "
                        f"{_subG.number_of_edges()} edges. "
                        f"Node size reflects concept frequency."
                    )
                else:
                    st.info("Graph is empty.")
            except Exception as _exc:
                st.warning(f"Graph visualization unavailable: {_exc}")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — Admin Panel
# ═══════════════════════════════════════════════════════════════════════════

with tab_admin:
    st.header("\u2699\ufe0f Admin Panel")

    if role != "admin":
        st.warning(
            "\U0001f512 Admin role required. Switch role in the sidebar."
        )
    else:
        # ── Pipeline Logs ──
        st.subheader("\U0001f4cb Pipeline Activity Logs")
        if st.session_state.pipeline_logs:
            st.text_area(
                "Logs",
                "\n".join(st.session_state.pipeline_logs),
                height=200,
            )
        else:
            st.info("No pipeline activity yet.")

        st.markdown("---")

        # ── Pipeline Controls ──
        st.subheader("\U0001f527 Pipeline Controls")
        _ca, _cb = st.columns(2)

        with _ca:
            if st.button("\U0001f504 Rebuild FAISS Index"):
                with st.spinner("Rebuilding index + concepts\u2026"):
                    try:
                        _docs = collect_all_documents(log_fn=log_pipeline)
                        if not _docs:
                            st.error("No documents found.")
                            st.stop()
                        n_c, n_e, n_v = rebuild_pipeline(
                            _docs, log_fn=log_pipeline
                        )
                        log_pipeline("Admin: Full rebuild complete.")
                        st.success(
                            f"\u2705 Rebuilt: {n_v} docs, "
                            f"{n_c} concepts, {n_e} edges."
                        )
                    except Exception as _exc:
                        st.error(f"Rebuild failed: {_exc}")
                        log_pipeline(f"ERROR: {_exc}")

        with _cb:
            if st.button("\U0001f504 Import to Neo4j"):
                with st.spinner("Importing to Neo4j\u2026"):
                    try:
                        from backend.neo4j_import import (
                            get_graph as _get_graph,
                            reset_graph_cache as _reset_cache,
                        )

                        _reset_cache()
                        _graph = _get_graph(csv_dir=str(PROJECT_ROOT / "data"))
                        log_pipeline(
                            f"Admin: Neo4j import \u2014 {_graph.number_of_nodes()} nodes."
                        )
                        st.success(
                            f"\u2705 Graph: {_graph.number_of_nodes()} nodes, "
                            f"{_graph.number_of_edges()} edges."
                        )
                    except Exception as _exc:
                        st.error(f"Import failed: {_exc}")
                        log_pipeline(f"ERROR: {_exc}")

        st.markdown("---")

        # ── Data Management ──
        st.subheader("\U0001f4c1 Data Management")
        _d1, _d2 = st.columns(2)

        with _d1:
            _up_path = PROJECT_ROOT / "data" / "uploads"
            _all_uploaded: list = []
            if _up_path.exists():
                _all_uploaded.extend(_up_path.glob("*.pdf"))
                _papers_dir = _up_path / "papers"
                if _papers_dir.exists():
                    _all_uploaded.extend(_papers_dir.glob("*.pdf"))
                _audio_dir = _up_path / "audio"
                if _audio_dir.exists():
                    _all_uploaded.extend(
                        f for f in _audio_dir.iterdir()
                        if f.suffix.lower() in (".mp3", ".wav", ".m4a")
                    )
            st.markdown(f"**Uploaded Files:** {len(_all_uploaded)}")
            for _f in _all_uploaded:
                _sz = _f.stat().st_size / 1024
                _icon = "\U0001f399\ufe0f" if _f.suffix.lower() in (".mp3", ".wav", ".m4a") else "\U0001f4c4"
                st.caption(f"{_icon} {_f.name} ({_sz:.0f} KB)")
            if not _all_uploaded:
                st.caption("No uploaded files.")

        with _d2:
            st.markdown("**Index Status:**")
            if _idx_exists(FAISS_INDEX_DIR):
                _ip = Path(FAISS_INDEX_DIR) / "index.faiss"
                _sz_mb = _ip.stat().st_size / (1024 * 1024)
                st.caption(f"\u2705 FAISS index: {_sz_mb:.2f} MB, {_n_docs} vectors")
            else:
                st.caption("\u274c No FAISS index built")
            if _CONCEPTS_CSV.exists():
                st.caption(f"\u2705 concepts.csv: {_n_concepts} concepts")
            if _EDGES_CSV.exists():
                st.caption(f"\u2705 edges.csv: {_n_edges} edges")

        st.markdown("---")

        # ── Clear All Data (New Subject) ──
        st.subheader("\U0001f5d1\ufe0f Clear Database for New Subject")
        st.warning(
            "This will **permanently delete** all uploaded files, FAISS index, "
            "concept/edge CSVs, graph cache, and chat history. "
            "Use this before ingesting materials for a new subject."
        )
        _confirm_clear = st.checkbox(
            "I confirm I want to delete all data and start fresh.",
            key="confirm_clear_db",
        )
        if st.button(
            "\U0001f5d1\ufe0f Clear All Data & Reset",
            type="primary",
            disabled=not _confirm_clear,
        ):
            import shutil

            _cleared: list = []

            # 1. Delete uploaded files
            _up = PROJECT_ROOT / "data" / "uploads"
            if _up.exists():
                shutil.rmtree(str(_up))
                _cleared.append("uploaded files")

            # 2. Delete FAISS index
            _fi = Path(FAISS_INDEX_DIR)
            if _fi.exists():
                shutil.rmtree(str(_fi))
                _cleared.append("FAISS index")

            # 3. Delete concepts.csv and edges.csv
            for _csv_name in (_CONCEPTS_CSV, _EDGES_CSV):
                _cp = Path(_csv_name)
                if _cp.exists():
                    _cp.unlink()
                    _cleared.append(_cp.name)

            # 4. Reset graph cache
            try:
                from backend.neo4j_import import reset_graph_cache
                reset_graph_cache()
                _cleared.append("graph cache")
            except Exception:
                pass

            # 5. Clear session state
            st.session_state.pipeline_logs = []
            st.session_state.messages = []
            st.session_state.generated_qa = []
            st.session_state.query_sentiments = []

            log_pipeline("Admin: All data cleared for new subject.")
            st.success(
                f"\u2705 Cleared: {', '.join(_cleared)}. "
                "You can now upload materials for a new subject."
            )
            st.rerun()

        st.markdown("---")

        # ── Faculty Consent ──
        st.subheader("\u2705 Faculty Consent Simulation")
        _consent = st.checkbox(
            "I, the faculty member, consent to having my lecture materials "
            "processed and stored for student learning purposes.",
            key="faculty_consent",
        )
        if _consent:
            st.success("\u2705 Consent recorded.")
            log_pipeline("Faculty consent granted (simulated).")
        else:
            st.info("\u2b1c Consent not yet given.")

        st.markdown("---")

        # ── Environment Status ──
        st.subheader("\U0001f4ca Environment Status")
        _checks = {
            "FAISS index exists": _idx_exists(FAISS_INDEX_DIR),
            "Uploaded data present": (
                (PROJECT_ROOT / "data" / "uploads").exists()
                and bool(list((PROJECT_ROOT / "data" / "uploads").glob("*.pdf")))
            ),
            "concepts.csv exists": _CONCEPTS_CSV.exists(),
            "edges.csv exists": _EDGES_CSV.exists(),
            "NEO4J_PASSWORD set": os.getenv("NEO4J_PASSWORD", "") != "changeme",
            "GEMINI_API_KEY set": bool(os.getenv("GEMINI_API_KEY", "")),
        }
        for _label, _ok in _checks.items():
            st.markdown(f"{'\u2705' if _ok else '\u274c'} {_label}")
        _unset = [k for k, v in _checks.items() if not v]
        if _unset:
            st.warning("**Missing:** " + ", ".join(_unset))


# ─── Startup checklist (console only) ────────────────────────────────────


def _print_demo_checklist() -> None:
    from backend.faiss_index import index_exists as _ie

    print("\n" + "=" * 60)
    print("  CoursePilot \u2014 CKG-JTT  |  Startup Checklist")
    print("=" * 60)
    _items = [
        ("FAISS index", _ie(FAISS_INDEX_DIR)),
        (
            "Uploaded data",
            (PROJECT_ROOT / "data" / "uploads").exists()
            and bool(list((PROJECT_ROOT / "data" / "uploads").glob("*.pdf"))),
        ),
        ("concepts.csv", (PROJECT_ROOT / "data" / "concepts.csv").exists()),
        ("NEO4J_PASSWORD", os.getenv("NEO4J_PASSWORD", "") != "changeme"),
        ("GEMINI_API_KEY", bool(os.getenv("GEMINI_API_KEY", ""))),
    ]
    for _name, _ok in _items:
        print(f"  [{'+'if _ok else 'X'}] {_name}: {'OK' if _ok else 'MISSING'}")
    print("=" * 60 + "\n")


if "_checklist_printed" not in st.session_state:
    _print_demo_checklist()
    st.session_state["_checklist_printed"] = True
