# :   — Generate Neo4j Import CSVs CLI
# What to change:
#   1. Point --data-dir to your real data if not using samples.
#
# Usage:
#   python scripts/generate_neo4j_csvs.py --data-dir data/sample
#   python scripts/generate_neo4j_csvs.py --mock

"""
generate_neo4j_csvs.py — Extract concepts and relations from documents
and export as CSV files suitable for Neo4j LOAD CSV import.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate concepts.csv and edges.csv for Neo4j import."
    )
    parser.add_argument(
        "--data-dir",
        default=os.getenv("DATA_DIR", "data/sample"),
        help="Directory containing sample JSON data.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/neo4j_import",
        help="Output directory for CSV files.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use sample data bundled in the repo.",
    )
    args = parser.parse_args()

    # ── Step 1: Build documents ──
    logger.info("Building documents from %s …", args.data_dir)
    from backend.ingest_pdf import build_all_sample_documents

    docs = build_all_sample_documents(args.data_dir)
    if not docs:
        logger.error("No documents found.")
        sys.exit(1)

    # ── Step 2: Extract concepts & relations ──
    from backend.extract_concepts import (
        extract_concepts_from_documents,
        extract_relations,
        save_concepts_csv,
        save_edges_csv,
    )

    concepts, concept_docs = extract_concepts_from_documents(docs)
    edges = extract_relations(docs, concepts, concept_docs)

    # ── Step 3: Save CSVs ──
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    save_concepts_csv(concepts, str(out / "concepts.csv"))
    save_edges_csv(edges, str(out / "edges.csv"))

    # Also save to data/ root for the app
    save_concepts_csv(concepts, "data/concepts.csv")
    save_edges_csv(edges, "data/edges.csv")

    logger.info("✅ CSVs saved to %s", args.output_dir)
    logger.info("   Concepts: %d, Edges: %d", len(concepts), len(edges))


if __name__ == "__main__":
    main()
