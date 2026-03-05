# :   — FAISS Vector Index Wrapper
# What to change:
#   1. Adjust INDEX_TYPE if you want IVF or HNSW instead of Flat.
#   2. Change FAISS_INDEX_DIR in .env if needed.
#
# Run-time steps:
#   1) python scripts/build_faiss.py --data-dir data/sample
#   2) Index is saved to data/faiss_index/

"""
faiss_index.py — Build, save, load, and search a FAISS index over
document embeddings.  Stores metadata alongside the index so that search
results can be mapped back to documents with provenance.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

FAISS_INDEX_DIR: str = os.getenv(
    "FAISS_INDEX_DIR",
    str(Path(__file__).resolve().parent.parent / "data" / "faiss_index"),
)
INDEX_FILE: str = "index.faiss"
META_FILE: str = "metadata.json"


# ---------------------------------------------------------------------------
# Build index
# ---------------------------------------------------------------------------

def build_index(
    embeddings: np.ndarray,
    metadata: List[Dict[str, Any]],
    index_dir: str = FAISS_INDEX_DIR,
) -> None:
    """Build a FAISS Flat L2 index and persist to disk.

    Args:
        embeddings: (N, D) float32 array of document embeddings.
        metadata: List of N metadata dicts (must include 'doc_id').
        index_dir: Directory to write index + metadata files.
    """
    try:
        import faiss  # type: ignore
    except ImportError:
        raise ImportError("faiss-cpu not installed. Run: pip install faiss-cpu")

    assert len(embeddings) == len(metadata), (
        f"Embedding count ({len(embeddings)}) != metadata count ({len(metadata)})"
    )

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))

    out = Path(index_dir)
    out.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(out / INDEX_FILE))
    with open(str(out / META_FILE), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(
        "FAISS index built: %d vectors, dim=%d → %s",
        index.ntotal, dim, index_dir,
    )


# ---------------------------------------------------------------------------
# Load index
# ---------------------------------------------------------------------------

def load_index(
    index_dir: str = FAISS_INDEX_DIR,
) -> Tuple[Any, List[Dict[str, Any]]]:
    """Load a persisted FAISS index and its metadata.

    Args:
        index_dir: Directory containing index.faiss and metadata.json.

    Returns:
        (faiss_index, metadata_list)
    """
    try:
        import faiss  # type: ignore
    except ImportError:
        raise ImportError("faiss-cpu not installed. Run: pip install faiss-cpu")

    idx_path = Path(index_dir) / INDEX_FILE
    meta_path = Path(index_dir) / META_FILE

    if not idx_path.exists():
        raise FileNotFoundError(f"FAISS index not found at {idx_path}. Run build_faiss.py first.")

    index = faiss.read_index(str(idx_path))
    with open(str(meta_path), "r", encoding="utf-8") as f:
        metadata = json.load(f)

    logger.info("Loaded FAISS index: %d vectors from %s", index.ntotal, index_dir)
    return index, metadata


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search(
    query_embedding: np.ndarray,
    k: int = 5,
    index_dir: str = FAISS_INDEX_DIR,
    index=None,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Search the FAISS index for the k nearest neighbours.

    Args:
        query_embedding: 1-D float32 array of shape (dim,).
        k: Number of results.
        index_dir: Path to persisted index (used if index/metadata not passed).
        index: Pre-loaded FAISS index (optional).
        metadata: Pre-loaded metadata (optional).

    Returns:
        List of result dicts: [{score, doc_id, ...metadata}]
    """
    if index is None or metadata is None:
        index, metadata = load_index(index_dir)

    query = query_embedding.astype(np.float32).reshape(1, -1)
    distances, indices = index.search(query, k)

    results: List[Dict[str, Any]] = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        entry = dict(metadata[idx])
        entry["score"] = float(dist)
        results.append(entry)
    return results


# ---------------------------------------------------------------------------
# Utility: check if index exists
# ---------------------------------------------------------------------------

def index_exists(index_dir: str = FAISS_INDEX_DIR) -> bool:
    """Return True if a FAISS index has been built in index_dir."""
    return (Path(index_dir) / INDEX_FILE).exists()
