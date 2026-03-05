# :   — Unit Tests for Retriever
# Run: pytest tests/test_retriever.py -v

"""
test_retriever.py — Tests for backend.retriever and backend.faiss_index.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Test: FAISS index build and search
# ---------------------------------------------------------------------------

class TestFAISSIndex:
    """Tests for FAISS index operations."""

    def test_build_and_search(self, tmp_path):
        """Build an index, then search should return nearest docs."""
        try:
            from backend.faiss_index import build_index, search, index_exists
        except ImportError:
            pytest.skip("faiss-cpu not installed.")

        dim = 16
        n_docs = 10
        embeddings = np.random.randn(n_docs, dim).astype(np.float32)
        metadata = [
            {"doc_id": f"doc_{i}", "text": f"Document {i}", "source_type": "slide"}
            for i in range(n_docs)
        ]

        idx_dir = str(tmp_path / "faiss_test")
        build_index(embeddings, metadata, idx_dir)
        assert index_exists(idx_dir)

        # Search with the first document's embedding
        results = search(embeddings[0], k=3, index_dir=idx_dir)
        assert len(results) == 3
        # Nearest should be doc_0 itself
        assert results[0]["doc_id"] == "doc_0"
        assert results[0]["score"] == pytest.approx(0.0, abs=1e-5)

    def test_index_exists_false(self, tmp_path):
        """index_exists should return False for an empty dir."""
        from backend.faiss_index import index_exists

        assert not index_exists(str(tmp_path / "nonexistent"))

    def test_search_metadata_preserved(self, tmp_path):
        """Metadata fields should be preserved in search results."""
        try:
            from backend.faiss_index import build_index, search
        except ImportError:
            pytest.skip("faiss-cpu not installed.")

        dim = 8
        embeddings = np.random.randn(3, dim).astype(np.float32)
        metadata = [
            {"doc_id": "s1", "text": "Hello world", "source_type": "slide",
             "metadata": {"course": "CS101"}},
            {"doc_id": "s2", "text": "Goodbye world", "source_type": "lecture",
             "metadata": {"course": "CS102"}},
            {"doc_id": "s3", "text": "Test document", "source_type": "past_paper",
             "metadata": {"course": "CS103"}},
        ]

        idx_dir = str(tmp_path / "meta_test")
        build_index(embeddings, metadata, idx_dir)

        results = search(embeddings[1], k=1, index_dir=idx_dir)
        assert results[0]["doc_id"] == "s2"
        assert results[0]["source_type"] == "lecture"


# ---------------------------------------------------------------------------
# Test: Retriever answer generation
# ---------------------------------------------------------------------------

class TestRetriever:
    """Tests for the retriever module."""

    def test_canned_response_structure(self):
        """answer_query should return required keys."""
        # Test the template answer function directly
        from backend.retriever import _generate_answer_template

        passages = ["Test passage"]
        answer = _generate_answer_template(passages)
        assert isinstance(answer, str)
        assert len(answer) > 10

    def test_template_answer_generation(self):
        """Templated answer should work without any LLM."""
        from backend.retriever import _generate_answer_template

        passages = [
            "A stack uses LIFO ordering.",
            "A queue uses FIFO ordering.",
            "Both are abstract data types.",
        ]
        answer = _generate_answer_template(passages)
        assert isinstance(answer, str)
        assert len(answer) > 20
        assert "LIFO" in answer or "stack" in answer.lower() or "retrieved" in answer.lower()

    def test_snippet_truncation(self):
        """Snippets should be truncated to SNIPPET_MAX_WORDS."""
        from backend.retriever import _make_snippet

        long_text = " ".join(["word"] * 50)
        snippet = _make_snippet(long_text, max_words=10)
        assert snippet.endswith("…")
        assert len(snippet.split()) <= 12  # 10 words + "…"

        short_text = "Short text here."
        assert _make_snippet(short_text) == short_text


# ---------------------------------------------------------------------------
# Test: Provenance building
# ---------------------------------------------------------------------------

class TestProvenance:
    """Tests for provenance formatting."""

    def test_build_provenance(self):
        """Provenance dict should have all required keys."""
        from backend.retriever import _build_provenance

        result = {
            "doc_id": "CS501_L1_S3",
            "source_type": "slide",
            "text": "A stack is a LIFO structure with push and pop operations.",
            "metadata": {
                "filename": "CS501_L1_S3",
                "start_time": None,
            },
        }
        prov = _build_provenance(result)
        assert prov["source_type"] == "slide"
        assert prov["filename"] == "CS501_L1_S3"
        assert prov["slide_id"] == "CS501_L1_S3"
        assert "snippet" in prov
        assert len(prov["snippet"]) > 0
