# :   — Retrieval Evaluation Script
# What to change:
#   1. Add more query→relevant_doc pairs in data/eval/eval_queries.json.
#
# Usage:
#   python scripts/eval_retrieval.py

"""
eval_retrieval.py — Evaluate retrieval quality by computing recall@k
for a small set of query → relevant_doc_id pairs.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    eval_path = Path("data/eval/eval_queries.json")
    if not eval_path.exists():
        print(f"Eval file not found: {eval_path}")
        sys.exit(1)

    with open(str(eval_path), "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    from backend.embeddings import embed_query
    from backend.faiss_index import search, index_exists

    faiss_dir = os.getenv("FAISS_INDEX_DIR", "data/faiss_index")
    if not index_exists(faiss_dir):
        print("FAISS index not found. Run: python scripts/build_faiss.py --data-dir data/sample")
        sys.exit(1)

    k_values = [1, 3, 5]
    recall_at_k = {k: 0.0 for k in k_values}
    total = len(eval_set)

    print(f"\nEvaluating {total} queries …\n")

    for item in eval_set:
        query = item["query"]
        relevant = set(item["relevant_doc_ids"])

        q_emb = embed_query(query)
        results = search(q_emb, k=max(k_values), index_dir=faiss_dir)
        retrieved_ids = [r.get("doc_id", "") for r in results]

        for k in k_values:
            hits = set(retrieved_ids[:k]) & relevant
            recall_at_k[k] += len(hits) / len(relevant)

        print(f"  Q: {query[:60]}…")
        print(f"    Relevant: {relevant}")
        print(f"    Retrieved@5: {retrieved_ids}")
        print()

    print("=" * 50)
    print("Recall@k Results:")
    for k in k_values:
        avg = recall_at_k[k] / total if total > 0 else 0
        print(f"  Recall@{k}: {avg:.2%}")
    print("=" * 50)


if __name__ == "__main__":
    main()
