# :   — Unit Tests for Concept Extraction
# Run: pytest tests/test_extract.py -v

"""
test_extract.py — Tests for backend.extract_concepts.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Test: concept extraction
# ---------------------------------------------------------------------------

class TestExtractConcepts:
    """Tests for concept and relation extraction."""

    def test_extract_noun_chunks_basic(self):
        """Noun chunks should be extracted from a simple sentence."""
        try:
            from backend.extract_concepts import extract_noun_chunks
        except (ImportError, OSError):
            pytest.skip("spaCy model not available.")

        text = "A binary search tree maintains sorted order for efficient searching."
        chunks = extract_noun_chunks(text)
        assert isinstance(chunks, list)
        # Should find at least some concepts
        assert len(chunks) >= 1

    def test_extract_concepts_from_documents(self):
        """Extraction should find concepts across multiple documents."""
        try:
            from backend.extract_concepts import extract_concepts_from_documents
        except (ImportError, OSError):
            pytest.skip("spaCy model not available.")

        docs = [
            {"doc_id": "d1", "text": "Arrays store elements in contiguous memory locations."},
            {"doc_id": "d2", "text": "Linked lists use pointers to connect nodes in memory."},
            {"doc_id": "d3", "text": "Arrays and linked lists are fundamental data structures."},
        ]
        concepts, concept_docs = extract_concepts_from_documents(docs)
        assert isinstance(concepts, list)
        assert len(concepts) >= 1
        # Each concept should have expected keys
        for c in concepts:
            assert "concept_id" in c
            assert "label" in c
            assert "frequency" in c

    def test_extract_relations(self):
        """Relations should be inferred from co-occurrences."""
        try:
            from backend.extract_concepts import (
                extract_concepts_from_documents,
                extract_relations,
            )
        except (ImportError, OSError):
            pytest.skip("spaCy model not available.")

        docs = [
            {"doc_id": "d1", "text": "Binary search trees enable efficient search and insert operations."},
            {"doc_id": "d2", "text": "AVL trees are balanced binary search trees with rotations."},
        ]
        concepts, concept_docs = extract_concepts_from_documents(docs)
        edges = extract_relations(docs, concepts, concept_docs)
        assert isinstance(edges, list)
        # Edges may or may not exist depending on extraction
        for e in edges:
            assert "source" in e
            assert "target" in e
            assert "relation" in e

    def test_csv_roundtrip(self, tmp_path):
        """Concepts and edges should survive CSV save/load."""
        from backend.extract_concepts import save_concepts_csv, save_edges_csv

        concepts = [
            {"concept_id": "binary_tree", "label": "binary tree", "frequency": 5, "source_doc_ids": ["d1"]},
        ]
        edges = [
            {"source": "binary_tree", "target": "avl_tree", "relation": "co_occurs_with", "weight": 2},
        ]

        c_path = str(tmp_path / "concepts.csv")
        e_path = str(tmp_path / "edges.csv")
        save_concepts_csv(concepts, c_path)
        save_edges_csv(edges, e_path)

        assert Path(c_path).exists()
        assert Path(e_path).exists()

        # Verify CSV content by reading back
        import csv
        with open(c_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["concept_id"] == "binary_tree"

        with open(e_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["source"] == "binary_tree"
