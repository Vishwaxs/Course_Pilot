# :   — Concept & Relation Extraction (NLP Pipeline)
# What to change:
#   1. Tune MIN_PHRASE_FREQ and scoring weights for your domain.
#   2. Optionally replace spaCy model (en_core_web_sm → en_core_web_lg).
#
# Run-time steps:
#   1) python -m spacy download en_core_web_sm
#   2) python -m backend.extract_concepts --data-dir data/sample
#
# FIXME[REVIEW]: Noun-chunk heuristics may miss domain-specific phrases.
#   Consider adding a domain dictionary or using a transformer NER model.

"""
extract_concepts.py — Extract concepts (noun chunks) from documents and
infer simple prerequisite / co-occurrence relations.

Outputs:
  • data/concepts.csv   — concept_id, label, frequency, source_doc_ids
  • data/edges.csv      — source_concept, target_concept, relation, weight
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MIN_PHRASE_FREQ: int = 2          # Minimum occurrences to keep a concept
MIN_PHRASE_LEN: int = 1           # Minimum word-count in a phrase
MAX_PHRASE_LEN: int = 5           # Maximum word-count
MIN_WORD_LEN: int = 3            # Minimum characters for a single-word concept
COOCCURRENCE_WINDOW: int = 1     # Same document = 1; could be paragraph-level

# Domain-agnostic junk words that should never be concepts
STOPWORDS: Set[str] = {
    # presentation / formatting artefacts
    "example", "figure", "slide", "page", "e.g", "i.e", "etc", "note",
    "today", "lecture", "class", "question", "answer", "case", "chapter",
    "section", "table", "image", "diagram", "reference", "appendix",
    "overview", "introduction", "conclusion", "summary", "content",
    # generic verbs / adjectives that leak through noun chunks
    "use", "using", "used", "make", "making", "made", "get", "getting",
    "set", "setting", "run", "running", "thing", "way", "kind", "type",
    "lot", "bit", "part", "number", "point", "end", "start", "time",
    "day", "year", "place", "work", "working", "result", "need",
    "following", "given", "based", "called", "related", "shown",
    "different", "various", "many", "several", "new", "old", "good",
    "high", "low", "large", "small", "first", "last", "next",
    "general", "specific", "particular", "certain", "important",
    "total", "main", "basic", "simple", "common", "current",
}

# Extended stop words (determiners, pronouns, prepositions, etc.)
_COMMON_STOPWORDS: Set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "of", "in", "to",
    "for", "with", "on", "at", "from", "by", "about", "as", "into",
    "through", "during", "before", "after", "above", "below", "between",
    "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
    "each", "every", "all", "any", "few", "more", "most", "other",
    "some", "such", "no", "only", "own", "same", "than", "too", "very",
    "it", "its", "this", "that", "these", "those", "i", "we", "you",
    "he", "she", "they", "me", "him", "her", "us", "them", "my", "our",
    "your", "his", "their", "what", "which", "who", "whom", "how",
    "when", "where", "why", "if", "then", "also", "just", "like",
}

# POS tags that should be kept when building noun-chunk lemmas (spaCy path)
_KEEP_POS: Set[str] = {"NOUN", "PROPN", "ADJ", "NUM"}


# ---------------------------------------------------------------------------
# Lemmatizer setup (spaCy primary, NLTK WordNet fallback, regex last resort)
# ---------------------------------------------------------------------------

_SPACY_AVAILABLE: bool = False
_nlp_instance = None
_lemmatizer_fn = None  # callable(word) -> lemma


def _get_nlp():
    """Load a spaCy language model (lazy singleton)."""
    global _SPACY_AVAILABLE, _nlp_instance
    if _nlp_instance is not None:
        return _nlp_instance
    try:
        import spacy  # type: ignore
    except ImportError:
        logger.warning("spaCy not installed — using fallback extraction.")
        _SPACY_AVAILABLE = False
        return None
    try:
        _nlp_instance = spacy.load("en_core_web_sm")
        _SPACY_AVAILABLE = True
        return _nlp_instance
    except OSError:
        logger.warning("spaCy model 'en_core_web_sm' not found — using fallback. "
                        "Install with: python -m spacy download en_core_web_sm")
        _SPACY_AVAILABLE = False
        return None


def _get_lemmatizer():
    """Return a word→lemma function.  Tries NLTK WordNet, then regex rules."""
    global _lemmatizer_fn
    if _lemmatizer_fn is not None:
        return _lemmatizer_fn

    # Try NLTK WordNetLemmatizer
    try:
        import nltk  # type: ignore
        from nltk.stem import WordNetLemmatizer  # type: ignore

        # Ensure wordnet data is present
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)
        try:
            nltk.data.find("corpora/omw-1.4")
        except LookupError:
            nltk.download("omw-1.4", quiet=True)

        _wnl = WordNetLemmatizer()
        _lemmatizer_fn = lambda w: _wnl.lemmatize(w.lower(), pos="n")
        logger.info("Using NLTK WordNet lemmatizer.")
        return _lemmatizer_fn
    except ImportError:
        pass

    # Regex-based portable lemmatizer (good enough for plural/gerund)
    def _regex_lemma(word: str) -> str:
        w = word.lower()
        # Common plural rules (order matters)
        if w.endswith("ies") and len(w) > 4:
            return w[:-3] + "y"
        if w.endswith("sses"):
            return w[:-2]
        if w.endswith("ses") and len(w) > 4:
            return w[:-1]
        if w.endswith("ves") and len(w) > 4:
            return w[:-3] + "f"
        if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
            return w[:-1]
        # Gerund / -ing
        if w.endswith("ing") and len(w) > 5:
            base = w[:-3]
            if base and base[-1] == base[-2]:  # e.g. running → run
                return base[:-1]
            return base
        return w

    _lemmatizer_fn = _regex_lemma
    logger.info("Using regex-based lemmatizer (NLTK not available).")
    return _lemmatizer_fn


def _lemmatize_phrase(phrase: str) -> str:
    """Lemmatize each word in a multi-word phrase; drop empty tokens."""
    lemmatize = _get_lemmatizer()
    parts = [lemmatize(w) for w in phrase.split()]
    parts = [p for p in parts if p and len(p) >= 2]
    return " ".join(parts)


def _is_junk(phrase: str) -> bool:
    """Return True if `phrase` is too short, numeric-only, or a stopword."""
    if not phrase:
        return True
    words = phrase.split()
    # Single word checks
    if len(words) == 1:
        w = words[0]
        if len(w) < MIN_WORD_LEN:
            return True
        if w.isdigit():
            return True
        if w in STOPWORDS or w in _COMMON_STOPWORDS:
            return True
    # If *all* words are stopwords → junk
    if all(w in STOPWORDS or w in _COMMON_STOPWORDS for w in words):
        return True
    # If phrase starts or ends with a stopword → junk (for multi-word)
    if len(words) > 1:
        if words[0] in _COMMON_STOPWORDS or words[-1] in _COMMON_STOPWORDS:
            return True
    return False


# ---------------------------------------------------------------------------
# Concept extraction — spaCy path (with lemmatization)
# ---------------------------------------------------------------------------

def _extract_spacy(text: str, nlp) -> List[str]:
    """Extract lemmatized noun chunks using spaCy."""
    doc = nlp(text)
    chunks: List[str] = []

    for chunk in doc.noun_chunks:
        # Build the lemmatized phrase from meaningful tokens only
        lemma_parts: List[str] = []
        for token in chunk:
            # Skip determiners, pronouns, punctuation, and auxiliary verbs
            if token.pos_ in ("DET", "PRON", "PUNCT", "ADP", "CCONJ",
                              "SCONJ", "AUX", "PART", "INTJ"):
                continue
            if token.is_stop and token.pos_ not in ("NOUN", "PROPN"):
                continue
            lemma = token.lemma_.lower().strip()
            lemma = re.sub(r"[^a-z0-9\-]", "", lemma)
            if lemma and len(lemma) >= 2:
                lemma_parts.append(lemma)

        if not lemma_parts:
            continue
        phrase = " ".join(lemma_parts)

        # Length filter
        words = phrase.split()
        if len(words) > MAX_PHRASE_LEN:
            continue

        if _is_junk(phrase):
            continue

        chunks.append(phrase)

    return chunks


# ---------------------------------------------------------------------------
# Concept extraction — regex fallback (with lemmatization)
# ---------------------------------------------------------------------------

def _extract_regex(text: str) -> List[str]:
    """Regex-based extraction with lemmatization when spaCy is unavailable."""
    words = re.findall(r"[a-z][a-z\-]+", text.lower())
    lemmatize = _get_lemmatizer()
    lemmatized = [lemmatize(w) for w in words]
    chunks: List[str] = []

    # Multi-word phrases
    for window_size in range(MAX_PHRASE_LEN, 1, -1):
        for i in range(len(lemmatized) - window_size + 1):
            phrase_words = lemmatized[i : i + window_size]
            phrase = " ".join(phrase_words)
            if not _is_junk(phrase):
                chunks.append(phrase)

    # Single meaningful words
    for w in lemmatized:
        if not _is_junk(w):
            chunks.append(w)

    return chunks


# ---------------------------------------------------------------------------
# Public extraction API
# ---------------------------------------------------------------------------

def extract_noun_chunks(text: str, nlp=None) -> List[str]:
    """Extract cleaned, lemmatized concept phrases from text.

    Uses spaCy if available, otherwise falls back to regex + lemmatizer.
    """
    if nlp is None:
        nlp = _get_nlp()

    if nlp is not None:
        return _extract_spacy(text, nlp)
    else:
        return _extract_regex(text)


def extract_concepts_from_documents(
    documents: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    """Extract concepts across all documents with lemma-based deduplication.

    Phrases like "users" and "user", or "running" and "run", are merged
    into a single concept via lemmatization.

    Args:
        documents: List of canonical document dicts (must have 'doc_id', 'text').

    Returns:
        (concepts_list, concept_to_docs)
        concepts_list: [{concept_id, label, frequency}]
        concept_to_docs: {label: [doc_id, ...]}
    """
    nlp = _get_nlp()
    concept_counter: Counter = Counter()
    concept_docs: Dict[str, List[str]] = defaultdict(list)
    # Track the most-common surface form for each lemmatized key
    surface_counter: Dict[str, Counter] = defaultdict(Counter)

    for doc in documents:
        chunks = extract_noun_chunks(doc["text"], nlp=nlp)
        seen_in_doc: Set[str] = set()
        for raw_chunk in chunks:
            # Lemmatize the chunk to get a canonical key
            lemma_key = _lemmatize_phrase(raw_chunk)
            if _is_junk(lemma_key):
                continue
            concept_counter[lemma_key] += 1
            surface_counter[lemma_key][raw_chunk] += 1
            if lemma_key not in seen_in_doc:
                concept_docs[lemma_key].append(doc["doc_id"])
                seen_in_doc.add(lemma_key)

    # Filter by minimum frequency
    concepts: List[Dict[str, Any]] = []
    for lemma_key, freq in concept_counter.most_common():
        if freq < MIN_PHRASE_FREQ:
            continue
        # Pick the most frequent surface form as the display label
        best_label = surface_counter[lemma_key].most_common(1)[0][0]
        concepts.append({
            "concept_id": lemma_key.replace(" ", "_"),
            "label": best_label,
            "frequency": freq,
            "source_doc_ids": concept_docs[lemma_key],
        })

    logger.info("Extracted %d concepts from %d documents.", len(concepts), len(documents))
    return concepts, dict(concept_docs)


# ---------------------------------------------------------------------------
# Relation extraction (co-occurrence heuristic)
# ---------------------------------------------------------------------------

def extract_relations(
    documents: List[Dict[str, Any]],
    concepts: List[Dict[str, Any]],
    concept_docs: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    """Infer prerequisite / co-occurrence relations between concepts.

    Heuristic: two concepts that appear in the same document are related.
    Weight = number of shared documents.

    FIXME[REVIEW]: This is a naive co-occurrence heuristic. For better
    prerequisite inference, consider:
      - OrderedPair: if concept A always appears before concept B in lecture
        sequence, infer A → B (prerequisite).
      - TF-IDF weighting.
      - Manual curation via faculty dashboard.

    Args:
        documents: List of document dicts.
        concepts: Output of extract_concepts_from_documents.
        concept_docs: Mapping label → [doc_ids].

    Returns:
        List of edge dicts: [{source, target, relation, weight}]
    """
    # Build inverted index: doc_id → set of concept_ids
    doc_concepts: Dict[str, Set[str]] = defaultdict(set)
    for c in concepts:
        cid = c["concept_id"]
        for did in c.get("source_doc_ids", []):
            doc_concepts[did].add(cid)

    # Count co-occurrences
    pair_count: Counter = Counter()
    for doc_id, cids in doc_concepts.items():
        for a, b in combinations(sorted(cids), 2):
            pair_count[(a, b)] += 1

    edges: List[Dict[str, Any]] = []
    for (a, b), weight in pair_count.most_common():
        if weight < 1:
            continue
        edges.append({
            "source": a,
            "target": b,
            "relation": "co_occurs_with",
            "weight": weight,
        })

    # FIXME[REVIEW]: Add prerequisite inference based on lecture ordering.
    # Simple heuristic: if concept A only appears in earlier lectures than B,
    # mark A as prerequisite of B.

    logger.info("Extracted %d concept relations.", len(edges))
    return edges


# ---------------------------------------------------------------------------
# CSV export (for Neo4j import)
# ---------------------------------------------------------------------------

def save_concepts_csv(
    concepts: List[Dict[str, Any]],
    output_path: str = "data/concepts.csv",
) -> None:
    """Write concepts to CSV for Neo4j LOAD CSV.

    Args:
        concepts: List of concept dicts.
        output_path: Destination CSV file.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["concept_id", "label", "frequency"])
        writer.writeheader()
        for c in concepts:
            writer.writerow({
                "concept_id": c["concept_id"],
                "label": c["label"],
                "frequency": c["frequency"],
            })
    logger.info("Saved %d concepts to %s", len(concepts), output_path)


def save_edges_csv(
    edges: List[Dict[str, Any]],
    output_path: str = "data/edges.csv",
) -> None:
    """Write edges to CSV for Neo4j LOAD CSV.

    Args:
        edges: List of edge dicts.
        output_path: Destination CSV file.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "target", "relation", "weight"])
        writer.writeheader()
        for e in edges:
            writer.writerow(e)
    logger.info("Saved %d edges to %s", len(edges), output_path)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Extract concepts and relations")
    parser.add_argument("--data-dir", default="data/sample",
                        help="Directory with sample JSON data")
    parser.add_argument("--output-dir", default="data/",
                        help="Output directory for CSVs")
    args = parser.parse_args()

    from backend.ingest_pdf import build_all_sample_documents
    docs = build_all_sample_documents(args.data_dir)

    concepts, concept_docs = extract_concepts_from_documents(docs)
    edges = extract_relations(docs, concepts, concept_docs)

    save_concepts_csv(concepts, os.path.join(args.output_dir, "concepts.csv"))
    save_edges_csv(edges, os.path.join(args.output_dir, "edges.csv"))

    print(f"Concepts: {len(concepts)}, Edges: {len(edges)}")
