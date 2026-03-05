# :   — Sentence Embedding Wrapper
# What to change:
#   1. Optionally replace EMBEDDING_MODEL with a different model in .env.
#   2. Set HF_API_KEY if you want to use HuggingFace Inference API instead.
#
# Run-time steps:
#   1) pip install sentence-transformers
#   2) First call will download the model (~420 MB for all-mpnet-base-v2).
#
# TODO[USER_ACTION]: OPTIONAL_REPLACE_WITH_HF_API_KEY if you want remote embeddings.

"""
embeddings.py — Wrap sentence-transformers to produce dense vector
embeddings for text passages and queries.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import numpy as np
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Singleton model cache
_model = None


def _load_model():
    """Lazy-load the sentence-transformers model."""
    global _model
    if _model is not None:
        return _model
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError:
        raise ImportError(
            "sentence-transformers not installed. "
            "Run: pip install sentence-transformers"
        )
    logger.info("Loading embedding model '%s' …", EMBEDDING_MODEL)
    _model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info("Embedding model loaded (dim=%d).", _model.get_sentence_embedding_dimension())
    return _model


def embed_texts(texts: List[str], batch_size: int = 64) -> np.ndarray:
    """Encode a list of texts into dense embeddings.

    Args:
        texts: List of plain-text strings.
        batch_size: Encoding batch size.

    Returns:
        numpy array of shape (len(texts), embedding_dim).
    """
    model = _load_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 100,
        convert_to_numpy=True,
    )
    return embeddings  # type: ignore[return-value]


def embed_query(query: str) -> np.ndarray:
    """Encode a single query string.

    Args:
        query: The search query.

    Returns:
        1-D numpy array of shape (embedding_dim,).
    """
    return embed_texts([query])[0]


def get_embedding_dim() -> int:
    """Return the dimensionality of the loaded model."""
    model = _load_model()
    return model.get_sentence_embedding_dimension()
