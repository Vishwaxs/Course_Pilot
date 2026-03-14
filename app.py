"""
app.py - CoursePilot: Course Material Analyzer & Visualizer

Streamlit app with 7 tabs:
  0. How to Use        - guide for users and evaluation presentation
  1. Upload & Process  - upload PDFs, extract text & images, run concept extraction
  2. Text Processing   - tokenization, POS tagging, NER, lemmatization (spaCy + NLTK)
  3. Word Cloud & Freq - word cloud visualization + NLTK FreqDist
  4. Document Analytics - TF-IDF similarity, VADER sentiment, N-gram analysis
  5. Image Processing  - extract images from PDFs, OpenCV transformations
  6. Knowledge Graph   - concept network visualization with NetworkX
"""

from __future__ import annotations

import io
import logging
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# --- Page config ---

st.set_page_config(
    page_title="CoursePilot - Course Material Analyzer",
    page_icon="\U0001f393",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Logging ---

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")


# --- NLTK data helper ---

def ensure_nltk_data():
    """Download required NLTK data packages if missing."""
    import nltk
    resources = {
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords",
        "sentiment/vader_lexicon": "vader_lexicon",
        "corpora/wordnet": "wordnet",
    }
    for path, name in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


# --- Session-state defaults ---

_DEFAULTS = {
    "page_texts": [],
    "pdf_images": [],
    "concepts": [],
    "edges": [],
    "processed": False,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# --- Helper: Extract text from PDFs ---

def extract_text_from_pdfs(files):
    """Extract text page-by-page from uploaded PDF files using pdfplumber."""
    import pdfplumber
    pages = []
    for f in files:
        f.seek(0)
        with pdfplumber.open(f) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append({"filename": f.name, "page": i + 1, "text": text.strip()})
    return pages


# --- Helper: Extract images from PDFs ---

def extract_images_from_pdfs(files):
    """Extract images from uploaded PDF files using PyMuPDF."""
    try:
        import fitz
    except ImportError:
        return []
    images = []
    for f in files:
        f.seek(0)
        doc = fitz.open(stream=f.read(), filetype="pdf")
        for page_num in range(len(doc)):
            page = doc[page_num]
            img_list = page.get_images(full=True)
            for img_idx, img in enumerate(img_list):
                xref = img[0]
                try:
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n >= 5:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    if pix.width < 50 or pix.height < 50:
                        continue
                    img_bytes = pix.tobytes("png")
                    images.append({
                        "filename": f.name, "page": page_num + 1,
                        "img_idx": img_idx, "image_bytes": img_bytes,
                        "width": pix.width, "height": pix.height,
                    })
                except Exception:
                    continue
        doc.close()
        f.seek(0)
    return images


# --- Helper: Text cleaning ---

def clean_text(text):
    """Clean text for NLP analysis."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_stopwords():
    """Get NLTK English stopwords."""
    try:
        from nltk.corpus import stopwords
        return set(stopwords.words("english"))
    except Exception:
        return set()


def remove_stopwords(text, stop_words):
    """Remove stopwords and short words from cleaned text."""
    return " ".join(w for w in text.split() if w not in stop_words and len(w) > 2)


def _filter_pages(page_texts):
    """Apply sidebar file-focus and page-range filters to a page list."""
    selected = st.session_state.get("sidebar_file_filter")
    if selected:
        page_texts = [p for p in page_texts if p["filename"] in selected]
    page_range = st.session_state.get("sidebar_page_range")
    if page_range and len(page_texts) > 0:
        lo, hi = page_range
        page_texts = page_texts[lo - 1:hi]
    return page_texts


# --- Custom CSS ---

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


# --- Sidebar ---

st.sidebar.markdown(
    '<div class="hero-banner">'
    '<h3>\U0001f393 CoursePilot</h3>'
    '<p>Course Material Analyzer &amp; Visualizer</p>'
    '</div>',
    unsafe_allow_html=True,
)

if st.session_state.processed:
    st.sidebar.markdown("### \U0001f4ca Dashboard Stats")
    _sc1, _sc2 = st.sidebar.columns(2)
    _sc1.metric("\U0001f4c4 Pages", len(st.session_state.page_texts))
    _sc2.metric("\U0001f4a1 Concepts", len(st.session_state.concepts))
    _sc3, _sc4 = st.sidebar.columns(2)
    _sc3.metric("\U0001f517 Relations", len(st.session_state.edges))
    _sc4.metric("\U0001f5bc\ufe0f Images", len(st.session_state.pdf_images))
    _n_files = len(set(p["filename"] for p in st.session_state.page_texts))
    st.sidebar.metric("\U0001f4c1 PDF Files", _n_files)
else:
    st.sidebar.info("\U0001f4e4 Upload PDFs in the **Upload & Process** tab to begin.")

st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit \u00b7 spaCy \u00b7 NLTK \u00b7 OpenCV \u00b7 NetworkX")
st.sidebar.markdown("---")
st.sidebar.date_input("Session Date", key="sidebar_date")
st.sidebar.text_area(
    "Session Notes",
    placeholder="Jot down observations\u2026",
    key="sidebar_notes",
    height=100,
)

if st.session_state.processed:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### \U0001f50d Filters")

    _all_filenames = sorted(set(p["filename"] for p in st.session_state.page_texts))
    st.sidebar.multiselect(
        "Focus on files",
        _all_filenames,
        default=_all_filenames,
        key="sidebar_file_filter",
    )

    _n_pgs = len(st.session_state.page_texts)
    if _n_pgs > 1:
        st.sidebar.slider(
            "Page range",
            1, _n_pgs, (1, _n_pgs),
            key="sidebar_page_range",
        )

    if st.session_state.concepts:
        _max_freq = max((int(c.get("frequency", 1)) for c in st.session_state.concepts), default=1)
        if _max_freq > 1:
            st.sidebar.slider(
                "Min concept frequency",
                1, _max_freq, 1,
                key="sidebar_min_freq",
            )

st.sidebar.markdown("---")
_PALETTE_MAP = {
    "Indigo": "#6366f1",
    "Teal": "#10b981",
    "Rose": "#f43f5e",
    "Amber": "#f59e0b",
}
_CHART_COLOR = _PALETTE_MAP[
    st.sidebar.selectbox("Chart palette", list(_PALETTE_MAP.keys()), key="sidebar_palette")
]


# --- Tab definitions ---

tab_guide, tab_upload, tab_text, tab_wc, tab_analytics, tab_img, tab_kg = st.tabs([
    "\U0001f4d6 How to Use",
    "\U0001f4e4 Upload & Process",
    "\U0001f4dd Text Processing",
    "\u2601\ufe0f Word Cloud & Frequency",
    "\U0001f4ca Document Analytics",
    "\U0001f5bc\ufe0f Image Processing",
    "\U0001f578\ufe0f Knowledge Graph",
])


# =====================================================================
# TAB 0 - How to Use Guide
# =====================================================================

with tab_guide:
    st.header("\U0001f4d6 CoursePilot - How to Use")
    st.markdown(
        """
**CoursePilot** is a Streamlit-based course material analyzer that applies
**NLP**, **text processing**, **image processing**, and **knowledge graphs**
to help explore and visualize educational content.

---

### \U0001f680 Quick-Start

| Step | Tab | What to Do |
|------|-----|-----------|
| **1** | **Upload & Process** | Upload PDF lecture slides \u2192 click **Process Materials** |
| **2** | **Text Processing** | Explore tokenization, POS tagging, Named Entity Recognition |
| **3** | **Word Cloud & Frequency** | Visualize prominent terms and frequency distributions |
| **4** | **Document Analytics** | TF-IDF similarity, VADER sentiment, N-gram analysis |
| **5** | **Image Processing** | Extract images from PDFs and apply OpenCV transformations |
| **6** | **Knowledge Graph** | Visualize concept relationships with NetworkX |

---

### \U0001f3af What Each Tab Does

**\U0001f4e4 Upload & Process**
- Upload PDF lecture slides or past question papers
- The pipeline extracts text page-by-page (pdfplumber), extracts embedded
  images (PyMuPDF), identifies **concepts** (spaCy noun-chunk extraction +
  lemmatization), and builds **co-occurrence relationships** between concepts

**\U0001f4dd Text Processing** *(NLTK + spaCy techniques)*
- **Tokenization** \u2014 word and sentence tokenization (NLTK)
- **POS Tagging** \u2014 Part-of-Speech tagging table (spaCy)
- **Named Entity Recognition** \u2014 identify people, organizations, dates, etc. (spaCy)
- **Lemmatization** \u2014 compare original tokens vs. lemmatized forms
- **POS Distribution** \u2014 bar chart of POS tag frequencies

**\u2601\ufe0f Word Cloud & Frequency** *(WordCloud + NLTK)*
- **Word Cloud** \u2014 visual overview of the most prominent terms (customizable)
- **Frequency Distribution** \u2014 bar chart of top-N terms (NLTK `FreqDist`)
- Vocabulary statistics (total words, unique words, lexical diversity)

**\U0001f4ca Document Analytics** *(sklearn + NLTK)*
- **TF-IDF Cosine Similarity** \u2014 heatmap showing how similar document pages are (sklearn)
- **VADER Sentiment Analysis** \u2014 compound sentiment score per page (NLTK)
- **N-gram Analysis** \u2014 most common bigrams and trigrams

**\U0001f5bc\ufe0f Image Processing** *(OpenCV + PyMuPDF)*
- Extract images embedded in PDF lecture slides
- Apply OpenCV transformations: **grayscale**, **Gaussian blur**, **Canny edge detection**, **binary thresholding**
- **Image histograms** (grayscale + RGB channels)
- **Histogram equalization** comparison
- Upload custom images for processing if PDFs have no images

**\U0001f578\ufe0f Knowledge Graph** *(spaCy + NetworkX)*
- Concept extraction using spaCy NLP (noun chunks + lemmatization)
- Co-occurrence relationship mapping between concepts
- Interactive NetworkX graph visualization with matplotlib
- Concept frequency analytics and search

---

### \U0001f6e0\ufe0f Technology Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| UI Framework | **Streamlit** | Interactive web dashboard |
| PDF Text Extraction | **pdfplumber** | Extract text from lecture PDFs |
| PDF Image Extraction | **PyMuPDF** | Extract embedded images |
| NLP Processing | **spaCy** (`en_core_web_sm`) + **NLTK** | Tokenization, POS, NER, lemmatization |
| Word Cloud | **WordCloud** | Visual term prominence |
| Text Analytics | **sklearn** (TF-IDF + cosine similarity) | Document similarity |
| Sentiment | **NLTK VADER** | Sentiment analysis |
| Image Processing | **OpenCV** | Image transformations & histograms |
| Knowledge Graph | **NetworkX** + **matplotlib** | Concept relationship visualization |
| Data Handling | **pandas** + **numpy** | DataFrames and numerical computation |

---

### \U0001f3ac Evaluation Presentation Script

> **Step 1:** Open Upload & Process tab \u2192 upload a PDF \u2192 click Process Materials \u2192
> *"The pipeline extracts text page-by-page using pdfplumber and identifies images using PyMuPDF."*
>
> **Step 2:** Open Text Processing tab \u2192
> *"Here I demonstrate tokenization, POS tagging, and Named Entity Recognition using spaCy and NLTK."*
>
> **Step 3:** Open Word Cloud & Frequency tab \u2192
> *"Word Cloud visualization shows the most prominent terms. The frequency distribution uses NLTK FreqDist."*
>
> **Step 4:** Open Document Analytics tab \u2192
> *"TF-IDF cosine similarity shows how similar document pages are. VADER sentiment analysis scores each page. N-gram analysis reveals common multi-word phrases."*
>
> **Step 5:** Open Image Processing tab \u2192
> *"Images extracted from PDFs are processed using OpenCV \u2014 grayscale conversion, Gaussian blur, Canny edge detection, and thresholding. Histograms show pixel intensity distributions."*
>
> **Step 6:** Open Knowledge Graph tab \u2192
> *"spaCy extracts key concepts through noun chunk analysis and lemmatization. NetworkX visualizes the concept co-occurrence relationships as a graph."*
"""
    )


# =====================================================================
# TAB 1 - Upload & Process
# =====================================================================

with tab_upload:
    st.header("\U0001f4e4 Upload & Process Course Materials")
    st.markdown(
        "Upload PDF lecture slides or past question papers. CoursePilot extracts "
        "text, images, and concepts automatically."
    )

    with st.form("upload_form", clear_on_submit=False):
        uploaded_pdfs = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_upload",
        )
        submitted = st.form_submit_button("\u25b6\ufe0f Process Materials", type="primary")

    if uploaded_pdfs:
        st.success(f"{len(uploaded_pdfs)} PDF(s) ready for processing.")

    if submitted and uploaded_pdfs:
        with st.status("Processing course materials\u2026", expanded=True) as status:
            st.write("\U0001f4c4 Extracting text from PDFs (pdfplumber)\u2026")
            page_texts = extract_text_from_pdfs(uploaded_pdfs)
            st.write(f"\u2705 Extracted text from {len(page_texts)} pages.")

            st.write("\U0001f5bc\ufe0f Extracting images from PDFs (PyMuPDF)\u2026")
            pdf_images = extract_images_from_pdfs(uploaded_pdfs)
            st.write(f"\u2705 Found {len(pdf_images)} image(s).")

            concepts = []
            edges = []
            if page_texts:
                st.write("\U0001f9e0 Extracting concepts with NLP (spaCy)\u2026")
                try:
                    from backend.extract_concepts import (
                        extract_concepts_from_documents,
                        extract_relations,
                    )
                    docs = [{
                        "doc_id": f"{p['filename']}_p{p['page']}",
                        "source_type": "slide",
                        "text": p["text"],
                        "metadata": {"filename": p["filename"], "page": p["page"]},
                    } for p in page_texts]
                    concepts, concept_docs = extract_concepts_from_documents(docs)
                    edges = extract_relations(docs, concepts, concept_docs)
                    st.write(f"\u2705 Extracted {len(concepts)} concepts and {len(edges)} relationships.")
                except Exception as e:
                    st.write(f"\u26a0\ufe0f Concept extraction: {e}")

            st.session_state.page_texts = page_texts
            st.session_state.pdf_images = pdf_images
            st.session_state.concepts = concepts
            st.session_state.edges = edges
            st.session_state.processed = True
            status.update(label="\u2705 Processing complete!", state="complete")
        st.rerun()

    if st.session_state.processed:
        st.markdown("---")
        page_texts = st.session_state.page_texts
        page_texts = _filter_pages(page_texts)
        m1, m2, m3, m4 = st.columns(4)
        n_files = len(set(p["filename"] for p in page_texts))
        m1.metric("\U0001f4c1 PDF Files", n_files)
        m2.metric("\U0001f4c4 Pages Extracted", len(page_texts))
        m3.metric("\U0001f4a1 Concepts Found", len(st.session_state.concepts))
        m4.metric("\U0001f5bc\ufe0f Images Found", len(st.session_state.pdf_images))

        st.markdown("---")
        st.subheader("\U0001f4c4 Extracted Text Preview")
        for filename in sorted(set(p["filename"] for p in page_texts)):
            file_pages = [p for p in page_texts if p["filename"] == filename]
            with st.expander(f"\U0001f4c1 {filename} ({len(file_pages)} pages)", expanded=False):
                for p in file_pages[:10]:
                    st.caption(f"**Page {p['page']}**")
                    st.text(p["text"][:500] + ("\u2026" if len(p["text"]) > 500 else ""))
                    st.markdown("---")
                if len(file_pages) > 10:
                    st.caption(f"\u2026 and {len(file_pages) - 10} more pages")

        st.markdown("---")
        confirm_clear = st.checkbox(
            "\u26a0\ufe0f Confirm: erase all extracted data", key="confirm_clear"
        )
        if st.button("\U0001f5d1\ufe0f Clear All Data & Start Over", disabled=not confirm_clear):
            for k, v in _DEFAULTS.items():
                st.session_state[k] = type(v)() if not isinstance(v, bool) else False
            st.rerun()


# =====================================================================
# TAB 2 - Text Processing
# =====================================================================

with tab_text:
    st.header("\U0001f4dd Text Processing \u2014 NLP Techniques")
    st.caption("Tokenization, POS tagging, NER, and lemmatization using spaCy and NLTK.")

    if not st.session_state.processed:
        st.warning("\u26a0\ufe0f Upload and process documents first (Upload & Process tab).")
    else:
        ensure_nltk_data()
        import nltk

        page_texts = st.session_state.page_texts
        page_texts = _filter_pages(page_texts)

        if not page_texts:
            st.info("No pages match the current sidebar filters. Adjust the **Focus on files** or **Page range** filter.")
        else:
            page_options = [f"{p['filename']} \u2014 Page {p['page']}" for p in page_texts]
            selected_idx = st.selectbox(
                "Select a page to analyze",
                range(len(page_options)),
                format_func=lambda i: page_options[i],
            )
            sample_text = page_texts[selected_idx]["text"]

            col_opts1, col_opts2 = st.columns(2)
            with col_opts1:
                show_analyses = st.multiselect(
                    "Show analyses",
                    ["POS Tagging & NER", "Lemmatization", "POS Distribution"],
                    default=["POS Tagging & NER", "Lemmatization", "POS Distribution"],
                    key="text_analyses",
                )
            with col_opts2:
                max_tokens_display = st.number_input(
                    "Max tokens to display", min_value=10, max_value=200, value=50, step=10, key="max_tok_disp"
                )

            st.subheader("\U0001f4c4 Raw Text")
            st.text_area("", sample_text, height=120, disabled=True, key="raw_text_view")
            st.markdown("---")

            # Tokenization
            col_wt, col_st = st.columns(2)
            with col_wt:
                st.subheader("\U0001f524 Word Tokenization (NLTK)")
                tokens = nltk.word_tokenize(sample_text)
                st.metric("Total Tokens", len(tokens))
                st.write(tokens[:int(max_tokens_display)])
                if len(tokens) > int(max_tokens_display):
                    st.caption(f"Showing first {int(max_tokens_display)} of {len(tokens)} tokens")

            with col_st:
                st.subheader("\U0001f4dd Sentence Tokenization (NLTK)")
                sentences = nltk.sent_tokenize(sample_text)
                st.metric("Total Sentences", len(sentences))
                for i, sent in enumerate(sentences[:10], 1):
                    st.caption(f"**{i}.** {sent}")
                if len(sentences) > 10:
                    st.caption(f"\u2026 and {len(sentences) - 10} more sentences")

            st.markdown("---")

            # POS Tagging & NER (spaCy)
            try:
                import spacy
                try:
                    nlp = spacy.load("en_core_web_sm")
                except OSError:
                    nlp = None
                    st.warning("spaCy model not found. Install: `python -m spacy download en_core_web_sm`")
            except ImportError:
                nlp = None
                st.warning("spaCy not installed.")

            if nlp:
                doc = nlp(sample_text[:5000])

                if "POS Tagging & NER" in show_analyses:
                    col_pos, col_ner = st.columns(2)
                    with col_pos:
                        st.subheader("\U0001f3f7\ufe0f POS Tagging (spaCy)")
                        pos_data = [
                            {"Token": token.text, "POS": token.pos_, "Tag": token.tag_,
                             "Explanation": spacy.explain(token.tag_) or ""}
                            for token in doc if not token.is_space
                        ][:50]
                        st.dataframe(pd.DataFrame(pos_data), height=400, hide_index=True)

                    with col_ner:
                        st.subheader("\U0001f50d Named Entity Recognition (spaCy)")
                        entities = [
                            {"Text": ent.text, "Label": ent.label_,
                             "Description": spacy.explain(ent.label_) or ""}
                            for ent in doc.ents
                        ]
                        if entities:
                            st.dataframe(pd.DataFrame(entities), height=400, hide_index=True)
                        else:
                            st.info("No named entities found in this page.")
                    st.markdown("---")

                if "Lemmatization" in show_analyses:
                    # Lemmatization
                    st.subheader("\U0001f4d6 Lemmatization (spaCy)")
                    lemma_data = [
                        {"Token": token.text, "Lemma": token.lemma_, "POS": token.pos_}
                        for token in doc
                        if not token.is_space and token.text.lower() != token.lemma_
                    ][:30]
                    if lemma_data:
                        st.dataframe(pd.DataFrame(lemma_data), hide_index=True)
                        st.caption("Showing tokens where the lemma differs from the original form.")
                    else:
                        st.info("All tokens match their lemmas in this page.")
                    st.markdown("---")

                if "POS Distribution" in show_analyses:
                    # POS Distribution
                    st.subheader("\U0001f4ca POS Tag Distribution")
                    pos_counts = pd.Series([token.pos_ for token in doc if not token.is_space]).value_counts()
                    fig_pos, ax_pos = plt.subplots(figsize=(10, 4))
                    pos_counts.plot(kind="bar", ax=ax_pos, color=_CHART_COLOR, edgecolor="white")
                    ax_pos.set_title("Part-of-Speech Tag Distribution")
                    ax_pos.set_ylabel("Count")
                    plt.xticks(rotation=45, ha="right")
                    plt.tight_layout()
                    st.pyplot(fig_pos)
                    plt.close(fig_pos)


# =====================================================================
# TAB 3 - Word Cloud & Frequency
# =====================================================================

with tab_wc:
    st.header("\u2601\ufe0f Word Cloud & Frequency Distribution")
    st.caption("Visual overview of prominent terms using WordCloud and NLTK FreqDist.")

    if not st.session_state.processed:
        st.warning("\u26a0\ufe0f Upload and process documents first (Upload & Process tab).")
    else:
        ensure_nltk_data()
        import nltk

        page_texts = st.session_state.page_texts
        page_texts = _filter_pages(page_texts)

        if not page_texts:
            st.info("No pages match the current sidebar filters. Adjust the **Focus on files** or **Page range** filter.")
        else:
            stop_words = get_stopwords()
            combined_text = " ".join(p["text"] for p in page_texts)
            cleaned = clean_text(combined_text)
            cleaned_words = remove_stopwords(cleaned, stop_words)

            # Word Cloud settings
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            with col_s1:
                max_words = st.slider("Max Words", 50, 300, 150, key="wc_max")
            with col_s2:
                colormap = st.selectbox("Color Map", ["viridis", "plasma", "inferno", "magma", "coolwarm", "Set2"])
            with col_s3:
                bg_color = st.color_picker("Background Color", "#ffffff", key="wc_bg")
            with col_s4:
                extra_stops_input = st.text_input(
                    "Extra stopwords (comma-separated)", placeholder="e.g. also, however", key="extra_stops"
                )
            extra_stops = {w.strip().lower() for w in extra_stops_input.split(",") if w.strip()}
            stop_words_wc = stop_words | extra_stops
            cleaned_words_filtered = remove_stopwords(cleaned, stop_words_wc)

            st.subheader("\u2601\ufe0f Word Cloud \u2014 Key Terms from Course Materials")
            try:
                from wordcloud import WordCloud
                wc = WordCloud(
                    width=1200, height=500, background_color=bg_color,
                    max_words=max_words, colormap=colormap, collocations=False,
                ).generate(cleaned_words_filtered)
                fig_wc, ax_wc = plt.subplots(figsize=(14, 6))
                ax_wc.imshow(wc, interpolation="bilinear")
                ax_wc.axis("off")
                st.pyplot(fig_wc)
                plt.close(fig_wc)
            except ImportError:
                st.error("Install `wordcloud`: `pip install wordcloud`")

            st.markdown("---")

            # Frequency Distribution
            st.subheader("\U0001f4ca Frequency Distribution (NLTK FreqDist)")
            word_tokens = nltk.word_tokenize(cleaned_words)
            fd = nltk.FreqDist(word_tokens)
            top_n = st.slider("Number of terms", 10, 50, 25, key="fd_n")
            top_words = fd.most_common(top_n)

            if top_words:
                labels_fd, counts_fd = zip(*top_words)
                fig_fd, ax_fd = plt.subplots(figsize=(14, 5))
                ax_fd.bar(range(len(labels_fd)), counts_fd, color=_CHART_COLOR, edgecolor="white")
                ax_fd.set_xticks(range(len(labels_fd)))
                ax_fd.set_xticklabels(labels_fd, rotation=45, ha="right", fontsize=8)
                ax_fd.set_ylabel("Frequency")
                ax_fd.set_title(f"Top {top_n} Terms \u2014 NLTK Frequency Distribution")
                plt.tight_layout()
                st.pyplot(fig_fd)
                plt.close(fig_fd)

            st.markdown("---")
            v1, v2, v3 = st.columns(3)
            v1.metric("Total Words", len(word_tokens))
            v2.metric("Unique Words", len(fd))
            v3.metric("Lexical Diversity", f"{len(fd) / max(len(word_tokens), 1):.2%}")


# =====================================================================
# TAB 4 - Document Analytics
# =====================================================================

with tab_analytics:
    st.header("\U0001f4ca Document Analytics")
    st.caption("TF-IDF similarity, VADER sentiment, and N-gram analysis.")

    if not st.session_state.processed:
        st.warning("\u26a0\ufe0f Upload and process documents first (Upload & Process tab).")
    else:
        ensure_nltk_data()
        import nltk

        page_texts = st.session_state.page_texts
        page_texts = _filter_pages(page_texts)
        all_texts = [p["text"] for p in page_texts]

        if not all_texts:
            st.info("No pages match the current sidebar filters. Adjust the **Focus on files** or **Page range** filter.")
        else:
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                show_sections = st.multiselect(
                    "Show analytics sections",
                    ["TF-IDF Similarity", "Sentiment Analysis", "N-gram Analysis"],
                    default=["TF-IDF Similarity", "Sentiment Analysis", "N-gram Analysis"],
                    key="analytics_sections",
                )
            with col_a2:
                tfidf_max_features = int(st.number_input(
                    "TF-IDF max features", min_value=100, max_value=5000, value=500, step=100, key="tfidf_feats"
                ))

            # TF-IDF Document Similarity
            if "TF-IDF Similarity" in show_sections:
                st.subheader("\U0001f50d Document Similarity (TF-IDF + Cosine)")
                st.caption("TF-IDF vectors for each page, cosine similarity between documents (sklearn).")

                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity

                max_docs = min(30, len(all_texts))
                sim_texts = all_texts[:max_docs]
                tfidf = TfidfVectorizer(max_features=tfidf_max_features, stop_words="english")
                tfidf_matrix = tfidf.fit_transform(sim_texts)
                sim_matrix = cosine_similarity(tfidf_matrix)

                fig_sim, ax_sim = plt.subplots(figsize=(10, 8))
                im = ax_sim.imshow(sim_matrix, cmap="YlOrRd", aspect="auto")
                ax_sim.set_title(f"Cosine Similarity Matrix ({max_docs} pages)")
                ax_sim.set_xlabel("Page Index")
                ax_sim.set_ylabel("Page Index")
                fig_sim.colorbar(im, ax=ax_sim, label="Similarity")
                st.pyplot(fig_sim)
                plt.close(fig_sim)

                sim_np = np.array(sim_matrix)
                np.fill_diagonal(sim_np, 0)
                if sim_np.max() > 0:
                    max_idx = np.unravel_index(sim_np.argmax(), sim_np.shape)
                    st.info(
                        f"**Most similar pair:** Page {max_idx[0] + 1} & "
                        f"Page {max_idx[1] + 1} \u2014 similarity = {sim_np[max_idx]:.3f}"
                    )
                st.markdown("---")

            if "Sentiment Analysis" in show_sections:
                # VADER Sentiment Analysis
                st.subheader("\U0001f4ac Sentiment Analysis (NLTK VADER)")
                st.caption("VADER sentiment scores for each page of the course materials.")

                try:
                    from nltk.sentiment import SentimentIntensityAnalyzer
                    sia = SentimentIntensityAnalyzer()
                    sentiments = []
                    for p in page_texts:
                        scores = sia.polarity_scores(p["text"])
                        sentiments.append({
                            "Page": f"{p['filename']} p{p['page']}",
                            "Positive": round(scores["pos"], 3),
                            "Negative": round(scores["neg"], 3),
                            "Neutral": round(scores["neu"], 3),
                            "Compound": round(scores["compound"], 3),
                        })
                    df_sent = pd.DataFrame(sentiments)

                    col_sc, col_sp = st.columns([2, 1])
                    with col_sc:
                        fig_sent, ax_sent = plt.subplots(figsize=(12, 4))
                        colors = [
                            "#22c55e" if c >= 0.05 else "#ef4444" if c <= -0.05 else "#6366f1"
                            for c in df_sent["Compound"]
                        ]
                        ax_sent.bar(range(len(df_sent)), df_sent["Compound"], color=colors, edgecolor="white")
                        ax_sent.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
                        ax_sent.set_xlabel("Page Index")
                        ax_sent.set_ylabel("Compound Score")
                        ax_sent.set_title("VADER Sentiment \u2014 Compound Score per Page")
                        ax_sent.set_ylim(-1.1, 1.1)
                        plt.tight_layout()
                        st.pyplot(fig_sent)
                        plt.close(fig_sent)

                    with col_sp:
                        def _classify_sent(c):
                            if c >= 0.05:
                                return "Positive"
                            elif c <= -0.05:
                                return "Negative"
                            return "Neutral"

                        dist = df_sent["Compound"].apply(_classify_sent).value_counts()
                        _pie_colors = {"Positive": "#22c55e", "Neutral": "#6366f1", "Negative": "#ef4444"}
                        fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
                        ax_pie.pie(
                            dist.values, labels=dist.index, autopct="%1.0f%%",
                            colors=[_pie_colors.get(s, "#999") for s in dist.index],
                        )
                        ax_pie.set_title("Sentiment Distribution")
                        st.pyplot(fig_pie)
                        plt.close(fig_pie)

                    st.dataframe(df_sent, hide_index=True, height=250)
                except Exception as exc:
                    st.warning(f"Sentiment analysis unavailable: {exc}")
                st.markdown("---")

            if "N-gram Analysis" in show_sections:
                # N-gram Analysis
                st.subheader("\U0001f4dd N-gram Analysis")
                st.caption("Most common bigrams and trigrams from course materials.")

                stop_words = get_stopwords()
                combined = clean_text(" ".join(all_texts))
                words = [w for w in combined.split() if w not in stop_words and len(w) > 2]
                from nltk import ngrams as _ngrams

                col_bi, col_tri = st.columns(2)
                with col_bi:
                    st.markdown("**Bigrams (2-word phrases)**")
                    bigrams = list(_ngrams(words, 2))
                    bigram_fd = nltk.FreqDist(bigrams)
                    top_bigrams = bigram_fd.most_common(15)
                    if top_bigrams:
                        bi_labels = [" ".join(b) for b, _ in top_bigrams]
                        bi_counts = [c for _, c in top_bigrams]
                        fig_bi, ax_bi = plt.subplots(figsize=(6, 5))
                        ax_bi.barh(bi_labels[::-1], bi_counts[::-1], color=_CHART_COLOR, edgecolor="white")
                        ax_bi.set_title("Top 15 Bigrams")
                        ax_bi.set_xlabel("Frequency")
                        plt.tight_layout()
                        st.pyplot(fig_bi)
                        plt.close(fig_bi)
                    else:
                        st.info("Not enough text for bigram analysis.")

                with col_tri:
                    st.markdown("**Trigrams (3-word phrases)**")
                    trigrams = list(_ngrams(words, 3))
                    trigram_fd = nltk.FreqDist(trigrams)
                    top_trigrams = trigram_fd.most_common(15)
                    if top_trigrams:
                        tri_labels = [" ".join(t) for t, _ in top_trigrams]
                        tri_counts = [c for _, c in top_trigrams]
                        fig_tri, ax_tri = plt.subplots(figsize=(6, 5))
                        ax_tri.barh(tri_labels[::-1], tri_counts[::-1], color="#10b981", edgecolor="white")
                        ax_tri.set_title("Top 15 Trigrams")
                        ax_tri.set_xlabel("Frequency")
                        plt.tight_layout()
                        st.pyplot(fig_tri)
                        plt.close(fig_tri)
                    else:
                        st.info("Not enough text for trigram analysis.")


# =====================================================================
# TAB 5 - Image Processing (OpenCV)
# =====================================================================

with tab_img:
    st.header("\U0001f5bc\ufe0f Image Processing \u2014 OpenCV Techniques")
    st.caption("Extract images from PDFs and apply computer vision transformations.")

    if not st.session_state.processed:
        st.warning("\u26a0\ufe0f Upload and process documents first (Upload & Process tab).")
    else:
        pdf_images = st.session_state.pdf_images

        if not pdf_images:
            st.info("No images found in uploaded PDFs. Upload an image directly for processing.")

        uploaded_img = st.file_uploader(
            "Or upload an image directly",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
            key="direct_img_upload",
        )

        # Determine image source
        img_bytes = None
        img_label = ""

        if pdf_images:
            img_options = [
                f"{img['filename']} \u2014 Page {img['page']} \u2014 "
                f"Image {img['img_idx'] + 1} ({img['width']}\u00d7{img['height']})"
                for img in pdf_images
            ]
            source_choices = ["From PDF"]
            if uploaded_img:
                source_choices.append("Uploaded image")
            source_choice = st.radio("Image source", source_choices, horizontal=True)

            if source_choice == "From PDF":
                selected_img_idx = st.selectbox(
                    "Select an image", range(len(img_options)),
                    format_func=lambda i: img_options[i],
                )
                img_bytes = pdf_images[selected_img_idx]["image_bytes"]
                img_label = img_options[selected_img_idx]
            elif uploaded_img:
                img_bytes = uploaded_img.read()
                img_label = uploaded_img.name
                uploaded_img.seek(0)
        elif uploaded_img:
            img_bytes = uploaded_img.read()
            img_label = uploaded_img.name
            uploaded_img.seek(0)

        if img_bytes:
            try:
                import cv2
                from PIL import Image

                img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                img_np = np.array(img_pil)

                st.subheader(f"\U0001f4f7 Original Image")
                st.image(img_pil, caption=img_label, width=600)
                st.markdown("---")

                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

                st.subheader("\U0001f527 OpenCV Transformations")

                # Row 1: Grayscale + Blur
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Grayscale Conversion**")
                    fig_g, ax_g = plt.subplots(figsize=(6, 4))
                    ax_g.imshow(gray, cmap="gray")
                    ax_g.set_title("Grayscale")
                    ax_g.axis("off")
                    st.pyplot(fig_g)
                    plt.close(fig_g)

                with col2:
                    blur_k = st.slider("Blur kernel size", 1, 15, 5, step=2, key="blur_k")
                    st.markdown("**Gaussian Blur**")
                    blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
                    fig_b, ax_b = plt.subplots(figsize=(6, 4))
                    ax_b.imshow(blurred, cmap="gray")
                    ax_b.set_title(f"Gaussian Blur (kernel={blur_k})")
                    ax_b.axis("off")
                    st.pyplot(fig_b)
                    plt.close(fig_b)

                # Row 2: Edge Detection + Thresholding
                col3, col4 = st.columns(2)
                with col3:
                    t_low = st.slider("Canny low threshold", 10, 200, 50, key="canny_low")
                    t_high = st.slider("Canny high threshold", 50, 300, 150, key="canny_high")
                    st.markdown("**Canny Edge Detection**")
                    edges_img = cv2.Canny(gray, t_low, t_high)
                    fig_e, ax_e = plt.subplots(figsize=(6, 4))
                    ax_e.imshow(edges_img, cmap="gray")
                    ax_e.set_title(f"Canny Edges (low={t_low}, high={t_high})")
                    ax_e.axis("off")
                    st.pyplot(fig_e)
                    plt.close(fig_e)

                with col4:
                    thresh_val = st.slider("Threshold value", 0, 255, 127, key="thresh_v")
                    st.markdown("**Binary Thresholding**")
                    _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
                    fig_t, ax_t = plt.subplots(figsize=(6, 4))
                    ax_t.imshow(thresh, cmap="gray")
                    ax_t.set_title(f"Binary Threshold ({thresh_val})")
                    ax_t.axis("off")
                    st.pyplot(fig_t)
                    plt.close(fig_t)

                st.markdown("---")

                # Image Histograms
                st.subheader("\U0001f4ca Image Histograms")
                fig_h, axes_h = plt.subplots(1, 2, figsize=(14, 4))
                axes_h[0].hist(gray.ravel(), bins=256, range=(0, 256), color="gray", alpha=0.7)
                axes_h[0].set_title("Grayscale Histogram")
                axes_h[0].set_xlabel("Pixel Value")
                axes_h[0].set_ylabel("Frequency")

                color_names = ("red", "green", "blue")
                for i, color in enumerate(color_names):
                    axes_h[1].hist(
                        img_np[:, :, i].ravel(), bins=256, range=(0, 256),
                        color=color, alpha=0.5, label=color.upper(),
                    )
                axes_h[1].set_title("RGB Channel Histograms")
                axes_h[1].set_xlabel("Pixel Value")
                axes_h[1].set_ylabel("Frequency")
                axes_h[1].legend()
                plt.tight_layout()
                st.pyplot(fig_h)
                plt.close(fig_h)

                st.markdown("---")

                # Histogram Equalization
                st.subheader("\U0001f504 Histogram Equalization")
                equalized = cv2.equalizeHist(gray)

                col_eq1, col_eq2 = st.columns(2)
                with col_eq1:
                    fig_eq1, ax_eq1 = plt.subplots(figsize=(6, 4))
                    ax_eq1.imshow(gray, cmap="gray")
                    ax_eq1.set_title("Before Equalization")
                    ax_eq1.axis("off")
                    st.pyplot(fig_eq1)
                    plt.close(fig_eq1)
                with col_eq2:
                    fig_eq2, ax_eq2 = plt.subplots(figsize=(6, 4))
                    ax_eq2.imshow(equalized, cmap="gray")
                    ax_eq2.set_title("After Histogram Equalization")
                    ax_eq2.axis("off")
                    st.pyplot(fig_eq2)
                    plt.close(fig_eq2)

                fig_eqh, axes_eqh = plt.subplots(1, 2, figsize=(14, 3))
                axes_eqh[0].hist(gray.ravel(), bins=256, range=(0, 256), color=_CHART_COLOR, alpha=0.7)
                axes_eqh[0].set_title("Histogram Before")
                axes_eqh[0].set_xlabel("Pixel Value")
                axes_eqh[1].hist(equalized.ravel(), bins=256, range=(0, 256), color="#10b981", alpha=0.7)
                axes_eqh[1].set_title("Histogram After Equalization")
                axes_eqh[1].set_xlabel("Pixel Value")
                plt.tight_layout()
                st.pyplot(fig_eqh)
                plt.close(fig_eqh)

            except ImportError as e:
                st.error(f"Required library not available: {e}. Install: `pip install opencv-python-headless Pillow`")
        elif not pdf_images and not uploaded_img:
            st.info("\U0001f4e4 Upload an image above to start processing.")


# =====================================================================
# TAB 6 - Knowledge Graph
# =====================================================================

with tab_kg:
    st.header("\U0001f578\ufe0f Knowledge Graph \u2014 Concept Visualization")
    st.caption("Concepts extracted with spaCy NLP, relationships visualized with NetworkX.")

    if not st.session_state.processed:
        st.warning("\u26a0\ufe0f Upload and process documents first (Upload & Process tab).")
    else:
        concepts = st.session_state.concepts
        edges = st.session_state.edges

        _min_freq = st.session_state.get("sidebar_min_freq", 1)
        if _min_freq > 1:
            _allowed = {c["concept_id"] for c in concepts if int(c.get("frequency", 0)) >= _min_freq}
            concepts = [c for c in concepts if c["concept_id"] in _allowed]
            edges = [e for e in edges if e["source"] in _allowed and e["target"] in _allowed]

        if not concepts:
            st.info(
                "No concepts extracted. Try uploading PDFs with more text content, "
                "or ensure spaCy is installed (`python -m spacy download en_core_web_sm`)."
            )
        else:
            import networkx as nx

            G = nx.Graph()
            for c in concepts:
                G.add_node(c["concept_id"], label=c["label"], frequency=c["frequency"])
            for e in edges:
                G.add_edge(
                    e["source"], e["target"],
                    relation=e.get("relation", "co_occurs"),
                    weight=e.get("weight", 1),
                )

            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("\U0001f4a1 Concepts", G.number_of_nodes())
            m2.metric("\U0001f517 Relations", G.number_of_edges())
            avg_freq = round(
                sum(c.get("frequency", 0) for c in concepts) / max(len(concepts), 1), 1
            )
            m3.metric("\U0001f4ca Avg Frequency", avg_freq)
            m4.metric("\U0001f504 Graph Density", f"{nx.density(G):.3f}")

            st.markdown("---")

            # Concept Frequency Chart + Search
            col_chart, col_tbl = st.columns([3, 2])
            df_concepts = pd.DataFrame(concepts)

            with col_chart:
                st.subheader("\U0001f3f7\ufe0f Top Concepts by Frequency")
                if "label" in df_concepts.columns and "frequency" in df_concepts.columns:
                    topn = st.slider("Number of concepts", 10, 50, 20, key="kg_topn")
                    chart_data = df_concepts.nlargest(topn, "frequency")
                    fig_cf, ax_cf = plt.subplots(figsize=(12, 5))
                    ax_cf.barh(
                        chart_data["label"].values[::-1],
                        chart_data["frequency"].values[::-1],
                        color=_CHART_COLOR, edgecolor="white",
                    )
                    ax_cf.set_title(f"Top {topn} Concepts by Frequency")
                    ax_cf.set_xlabel("Frequency")
                    plt.tight_layout()
                    st.pyplot(fig_cf)
                    plt.close(fig_cf)

            with col_tbl:
                st.subheader("\U0001f50d Concept Search")
                search = st.text_input("Filter concepts", placeholder="Type to search\u2026", key="concept_search")
                disp = df_concepts[["label", "frequency"]].copy()
                disp.columns = ["Concept", "Frequency"]
                disp = disp.sort_values("Frequency", ascending=False)
                if search:
                    disp = disp[disp["Concept"].str.contains(search, case=False, na=False)]
                st.dataframe(disp, height=400, hide_index=True)

            st.markdown("---")

            # Knowledge Graph Visualization
            st.subheader("\U0001f578\ufe0f Concept Relationship Network")
            max_nodes = st.slider("Concepts to visualize", 10, 60, 30, key="graph_n")

            sorted_nodes = sorted(
                G.nodes(data=True), key=lambda x: x[1].get("frequency", 0), reverse=True
            )
            top_ids = [n for n, _ in sorted_nodes[:max_nodes]]
            subG = G.subgraph(top_ids).copy()

            if subG.number_of_nodes() > 0:
                fig_g, ax_g = plt.subplots(figsize=(14, 8))
                pos = nx.spring_layout(subG, seed=42, k=1.5)
                node_sizes = [subG.nodes[n].get("frequency", 1) * 80 for n in subG.nodes]
                labels = {n: subG.nodes[n].get("label", n)[:20] for n in subG.nodes}

                nx.draw_networkx(
                    subG, pos, ax=ax_g, labels=labels,
                    node_size=node_sizes, font_size=7,
                    node_color=_CHART_COLOR, edge_color="#e2e8f0",
                    alpha=0.9, width=0.5,
                )
                ax_g.set_title(
                    f"Knowledge Graph \u2014 Top {len(subG.nodes)} Concepts "
                    f"(of {G.number_of_nodes()} total)"
                )
                ax_g.axis("off")
                st.pyplot(fig_g)
                plt.close(fig_g)

                st.caption(
                    f"Showing {subG.number_of_nodes()} nodes and "
                    f"{subG.number_of_edges()} edges. "
                    f"Node size reflects concept frequency."
                )

            st.markdown("---")

            # Concept Frequency Distribution
            st.subheader("\U0001f4ca Concept Frequency Distribution")
            if "frequency" in df_concepts.columns:
                fig_hist, ax_hist = plt.subplots(figsize=(10, 3))
                freq_clipped = df_concepts["frequency"].clip(
                    upper=df_concepts["frequency"].quantile(0.95)
                )
                ax_hist.hist(freq_clipped, bins=30, color=_CHART_COLOR, alpha=0.7, edgecolor="white")
                ax_hist.set_xlabel("Frequency")
                ax_hist.set_ylabel("Number of Concepts")
                ax_hist.set_title("Distribution of Concept Frequencies")
                plt.tight_layout()
                st.pyplot(fig_hist)
                plt.close(fig_hist)

            st.markdown("---")
            st.subheader("\U0001f4dd Your Notes")
            st.text_area(
                "Add notes about this knowledge graph",
                placeholder="e.g. Key concepts discovered, observations, follow-up questions\u2026",
                key="kg_notes",
                height=120,
            )
            st.markdown("---")
            st.subheader("\U0001f31f Rate This Analysis")
            try:
                st.feedback("stars", key="kg_feedback")
            except AttributeError:
                st.radio("Rate this analysis", ["\u2605", "\u2605\u2605", "\u2605\u2605\u2605", "\u2605\u2605\u2605\u2605", "\u2605\u2605\u2605\u2605\u2605"], horizontal=True, key="kg_feedback_fallback")
