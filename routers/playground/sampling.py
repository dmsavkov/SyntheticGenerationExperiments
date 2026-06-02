"""Stratified sampling with per-source ratio preservation."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any

import pandas as pd
from loguru import logger


def _source_key(row: pd.Series, id_col: str) -> str:
    rid = str(row[id_col]) if id_col in row.index else ""
    dname = str(row.get("Dataset name", "") or "")
    if dname:
        return dname.split("_")[0] if "_" in dname else dname
    if "_" in rid:
        return rid.split("_", 1)[0]
    return rid or "unknown"


def stratified_by_dataset(
    df: pd.DataFrame,
    pool_ids: list[Any],
    n: int,
    *,
    label_col: str = "Domain",
    id_col: str = "Global Index",
    id_to_idx: dict[Any, int],
    seed: int = 42,
) -> list[Any]:
    """Sample n ids stratified by label, preserving global source ratios within each class."""
    if n <= 0 or not pool_ids:
        return []

    by_label_source: dict[str, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
    global_source_counts: Counter[str] = Counter()

    for rid in pool_ids:
        idx = id_to_idx.get(rid)
        if idx is None:
            continue
        row = df.iloc[idx]
        label = str(row[label_col])
        src = _source_key(row, id_col)
        by_label_source[label][src].append(rid)
        global_source_counts[src] += 1

    total_global = sum(global_source_counts.values()) or 1
    source_weights = {s: c / total_global for s, c in global_source_counts.items()}

    labels = sorted(by_label_source.keys())
    per_class = max(1, n // max(1, len(labels)))
    rng = random.Random(seed)
    chosen: list[Any] = []
    used: set[Any] = set()

    for label in labels:
        src_pools = by_label_source[label]
        class_target = per_class
        label_chosen: list[Any] = []

        for src, weight in sorted(source_weights.items()):
            ids = src_pools.get(src, [])
            if not ids:
                continue
            want = max(0, int(round(class_target * weight)))
            rng.shuffle(ids)
            take = [i for i in ids if i not in used][:want]
            if len(take) < want:
                logger.warning(
                    "Short cell label={} source={}: wanted {} got {}",
                    label,
                    src,
                    want,
                    len(take),
                )
            label_chosen.extend(take)
            used.update(take)

        if len(label_chosen) < class_target:
            rest = [i for pools in src_pools.values() for i in pools if i not in used]
            rng.shuffle(rest)
            extra = rest[: class_target - len(label_chosen)]
            label_chosen.extend(extra)
            used.update(extra)

        chosen.extend(label_chosen[:class_target])

    if len(chosen) > n:
        rng.shuffle(chosen)
        chosen = chosen[:n]
    elif len(chosen) < n:
        rest = [i for i in pool_ids if i not in used]
        rng.shuffle(rest)
        chosen.extend(rest[: n - len(chosen)])

    return chosen[:n]
