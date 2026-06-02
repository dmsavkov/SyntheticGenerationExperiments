"""Frozen ModernBERT embeddings + logistic regression Domain probe."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from routers.core.cache import embed_modernbert, get_or_build_modernbert_cache, vectors_for_ids
from routers.core.constants import CLASSIFIER, MODERNBERT_MODEL_ID, TARGET_COL


@dataclass
class ModernBertProbe:
    clf: LogisticRegression
    classes_: list[str]
    id_to_cache_row: dict[Any, int]
    cache: dict[str, Any]

    def predict(self, row: dict) -> str:
        vec = self._vector_for_row(row)
        return str(self.clf.predict(vec.reshape(1, -1))[0])

    def predict_proba_dict(self, row: dict) -> dict[str, float]:
        vec = self._vector_for_row(row)
        probs = self.clf.predict_proba(vec.reshape(1, -1))[0]
        return {c: round(float(p), 6) for c, p in zip(self.classes_, probs)}

    def predict_extended(self, row: dict) -> dict[str, Any]:
        pred = self.predict(row)
        return {"pred": pred, "pred_probs": self.predict_proba_dict(row)}

    def _vector_for_row(self, row: dict) -> np.ndarray:
        rid = row["id"]
        if rid in self.id_to_cache_row:
            return self.cache["vectors"][self.id_to_cache_row[rid]]
        return embed_modernbert([str(row["prompt"])], batch_size=1)[0]


def train_probe(
    train_rows: list[dict],
    *,
    cache_ids: list[Any] | None = None,
    cache_texts: list[str] | None = None,
) -> tuple[ModernBertProbe, float]:
    t0 = time.perf_counter()
    ids = cache_ids or [r["id"] for r in train_rows]
    texts = cache_texts or [str(r["prompt"]) for r in train_rows]
    cache = get_or_build_modernbert_cache(ids, texts)
    X = vectors_for_ids(cache, [r["id"] for r in train_rows])
    y = [str(r["gold"]) for r in train_rows]
    clf = LogisticRegression(max_iter=500)
    clf.fit(X, y)
    id_to_row = {cache["ids"][i]: i for i in range(len(cache["ids"]))}
    train_seconds = time.perf_counter() - t0
    probe = ModernBertProbe(clf=clf, classes_=list(clf.classes_), id_to_cache_row=id_to_row, cache=cache)
    return probe, train_seconds


def probe_config_extra() -> dict[str, Any]:
    return {
        "classifier": CLASSIFIER,
        "modernbert_model_id": MODERNBERT_MODEL_ID,
        "target_col": TARGET_COL,
    }
