"""ModernBERT embedding cache."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from routers.core.modernbert_encoder import ModernBertEncoder
from routers.core.splits import project_root

CACHE_ROOT = project_root() / "cache"
ENCODER_KEY = "modernbert_base_v1"


def embedding_path(encoder: str) -> Path:
    p = CACHE_ROOT / "embeddings"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{encoder}_full.npz"


def load_embedding_cache(encoder: str) -> dict[str, Any] | None:
    path = embedding_path(encoder)
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return {"ids": data["ids"].tolist(), "vectors": data["vectors"]}


def save_embedding_cache(encoder: str, ids: list[Any], vectors: np.ndarray) -> Path:
    path = embedding_path(encoder)
    np.savez_compressed(path, ids=np.array(ids, dtype=object), vectors=vectors)
    logger.info("Saved {} embeddings to {}", len(ids), path)
    return path


def vectors_for_ids(cache: dict[str, Any], ids: list[Any]) -> np.ndarray:
    id_to_row = {cache["ids"][i]: i for i in range(len(cache["ids"]))}
    rows = [id_to_row[rid] for rid in ids]
    return cache["vectors"][rows]


def embed_modernbert(texts: list[str], *, batch_size: int = 8) -> np.ndarray:
    return ModernBertEncoder.get().encode_batch(texts, batch_size=batch_size)


def get_or_build_modernbert_cache(all_ids: list[Any], texts: list[str], *, key: str = ENCODER_KEY) -> dict[str, Any]:
    cached = load_embedding_cache(key)
    if cached is not None and len(cached["ids"]) == len(all_ids):
        return cached
    logger.info("Building ModernBERT cache for {} texts", len(texts))
    vectors = embed_modernbert(texts, batch_size=8)
    save_embedding_cache(key, all_ids, vectors)
    return {"ids": all_ids, "vectors": vectors}
