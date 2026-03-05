# :   — Build FAISS Index CLI
# What to change:
#   1. Point --data-dir to your real data if not using samples.
#
# Usage:
#   python scripts/build_faiss.py --data-dir data/sample
#   python scripts/build_faiss.py --mock

"""
build_faiss.py — CLI script to build the FAISS vector index from
course documents (slides, transcripts, past papers).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build FAISS index from course documents."
    )
    parser.add_argument(
        "--data-dir",
        default=os.getenv("DATA_DIR", "data/sample"),
        help="Directory containing sample JSON data.",
    )
    parser.add_argument(
        "--index-dir",
        default=os.getenv("FAISS_INDEX_DIR", "data/faiss_index"),
        help="Output directory for the FAISS index.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Generate and use mock/sample data bundled in the repo.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    index_dir = args.index_dir

    # ── Step 1: Build documents ──
    logger.info("Building documents from %s …", data_dir)
    from backend.ingest_pdf import build_all_sample_documents

    docs = build_all_sample_documents(data_dir)
    if not docs:
        logger.error("No documents found. Check your --data-dir path.")
        sys.exit(1)
    logger.info("Total documents: %d", len(docs))

    # ── Step 2: Compute embeddings ──
    logger.info("Computing sentence embeddings …")
    from backend.embeddings import embed_texts

    texts = [d["text"] for d in docs]
    embeddings = embed_texts(texts)
    logger.info("Embedding shape: %s", embeddings.shape)

    # ── Step 3: Prepare metadata ──
    metadata = []
    for d in docs:
        metadata.append({
            "doc_id": d["doc_id"],
            "source_type": d["source_type"],
            "text": d["text"],
            "metadata": d.get("metadata", {}),
        })

    # ── Step 4: Build and save FAISS index ──
    from backend.faiss_index import build_index

    build_index(embeddings, metadata, index_dir)
    logger.info("✅ FAISS index saved to %s", index_dir)

    # ── Step 5: Also save processed docs JSON ──
    proc_dir = Path("data/processed")
    proc_dir.mkdir(parents=True, exist_ok=True)
    with open(str(proc_dir / "documents.json"), "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2, ensure_ascii=False)
    logger.info("Processed documents saved to data/processed/documents.json")


if __name__ == "__main__":
    main()
