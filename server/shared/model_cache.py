"""
Shared model cache — loads heavy models once and reuses them across pipeline stages.

Usage:
    from server.shared.model_cache import get_sentence_transformer
    model = get_sentence_transformer()
"""
import os
import logging
from sentence_transformers import SentenceTransformer

from server.shared.config import (
    SENTENCE_TRANSFORMER_MODEL,
    SENTENCE_TRANSFORMERS_CACHE,
)

logger = logging.getLogger(__name__)

# ── Module-level singleton ───────────────────────────────────────
_sentence_transformer: SentenceTransformer | None = None


def get_sentence_transformer() -> SentenceTransformer:
    """
    Return a cached SentenceTransformer instance.

    First call loads the model (~1-3 s); subsequent calls return the
    same object in < 1 ms.  Also forces ``HF_HUB_OFFLINE=1`` so that
    the locally-cached weights are used without any HTTP round-trips
    to huggingface.co.
    """
    global _sentence_transformer

    if _sentence_transformer is not None:
        logger.debug("SentenceTransformer cache hit — reusing model")
        return _sentence_transformer

    # Skip network checks — model is already cached locally
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    logger.info(
        "Loading SentenceTransformer: %s (first load, will be cached)",
        SENTENCE_TRANSFORMER_MODEL,
    )

    _sentence_transformer = SentenceTransformer(
        SENTENCE_TRANSFORMER_MODEL,
        cache_folder=str(SENTENCE_TRANSFORMERS_CACHE),
    )

    logger.info("SentenceTransformer loaded and cached ✓")
    return _sentence_transformer


def clear_cache() -> None:
    """Release the cached model (useful for tests / memory pressure)."""
    global _sentence_transformer
    _sentence_transformer = None
