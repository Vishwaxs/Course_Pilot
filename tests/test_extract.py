# :   — Unit Tests for Concept Extraction
# Run: pytest tests/test_extract.py -v

"""
test_extract.py — Tests for backend.extract_concepts and backend.ingest_pdf.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Test: slide ingestion from JSON
# ---------------------------------------------------------------------------

class TestIngestPDF:
    """Tests for the PDF / JSON ingestion module."""

    def test_load_slides_from_json(self):
        """Sample slides.json should load correctly."""
        from backend.ingest_pdf import load_slides_from_json

        path = "data/sample/slides.json"
        if not Path(path).exists():
            pytest.skip("Sample data not found.")

        slides = load_slides_from_json(path)
        assert isinstance(slides, list)
        assert len(slides) >= 1
        # Each slide should have required keys
        for s in slides:
            assert "slide_id" in s
            assert "body" in s

    def test_slides_to_documents(self):
        """Slides should convert to canonical documents."""
        from backend.ingest_pdf import slides_to_documents

        slides = [
            {
                "slide_id": "TEST_S1",
                "course": "TEST",
                "lecture": 1,
                "slide_number": 1,
                "title": "Test Title",
                "body": "Test body content about algorithms.",
            }
        ]
        docs = slides_to_documents(slides)
        assert len(docs) == 1
        assert docs[0]["doc_id"] == "TEST_S1"
        assert docs[0]["source_type"] == "slide"
        assert "Test Title" in docs[0]["text"]

    def test_papers_to_documents(self):
        """Past papers should convert to per-question documents."""
        from backend.ingest_pdf import papers_to_documents

        papers = [
            {
                "paper_id": "TEST_MID",
                "course": "TEST",
                "exam_type": "Mid",
                "year": 2024,
                "questions": [
                    {"q_id": "Q1", "text": "Explain stacks.", "marks": 10, "topics": ["stacks"]},
                    {"q_id": "Q2", "text": "What is BFS?", "marks": 15, "topics": ["BFS"]},
                ],
            }
        ]
        docs = papers_to_documents(papers)
        assert len(docs) == 2
        assert docs[0]["source_type"] == "past_paper"
        assert "stacks" in docs[0]["metadata"]["topics"]

    def test_build_all_sample_documents(self):
        """build_all_sample_documents should return non-empty list."""
        from backend.ingest_pdf import build_all_sample_documents

        if not Path("data/sample/slides.json").exists():
            pytest.skip("Sample data not found.")

        docs = build_all_sample_documents("data/sample")
        assert isinstance(docs, list)
        assert len(docs) >= 5  # At least slides + transcript segments + papers


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

        from backend.neo4j_import import load_concepts_csv, load_edges_csv

        loaded_c = load_concepts_csv(c_path)
        loaded_e = load_edges_csv(e_path)
        assert len(loaded_c) == 1
        assert loaded_c[0]["concept_id"] == "binary_tree"
        assert len(loaded_e) == 1
        assert loaded_e[0]["source"] == "binary_tree"
