"""Frozen playground split JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from routers.core.data import load_arena
from routers.core.splits import SHUFFLE_SEED, get_splits, project_root
from routers.playground.sampling import stratified_by_dataset
from routers.playground.source_filter import PRESETS, filter_ids, row_matches_source

PLAYGROUND_TRAIN_PER_CLASS = 50
PLAYGROUND_EVAL_PER_CLASS = 100


def playground_splits_path(preset: str = "arc_med_mmlu") -> Path:
    return project_root() / "data" / "splits" / f"playground_{preset}_seed42.json"


@dataclass
class PlaygroundSplits:
    preset: str
    train_ids: list[Any]
    eval_ids: list[Any]
    pool_ids: list[Any]


def build_playground_splits(
    df: pd.DataFrame,
    bundle: Any,
    *,
    preset: str = "arc_med_mmlu",
    train_per_class: int = PLAYGROUND_TRAIN_PER_CLASS,
    eval_per_class: int = PLAYGROUND_EVAL_PER_CLASS,
    seed: int = SHUFFLE_SEED,
) -> dict[str, Any]:
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}")

    id_col = "Global Index"
    id_to_idx = bundle.id_to_idx
    pool_union = list(
        dict.fromkeys(
            filter_ids(bundle.train_pool_ids, preset)
            + filter_ids(bundle.eval_1000_ids, preset)
        )
    )
    n_classes = len({str(df.iloc[id_to_idx[i]]["Domain"]) for i in pool_union if i in id_to_idx})
    train_n = train_per_class * max(1, n_classes)
    eval_n = eval_per_class * max(1, n_classes)

    train_ids = stratified_by_dataset(
        df, pool_union, train_n, id_to_idx=id_to_idx, seed=seed, id_col=id_col
    )
    used = set(train_ids)
    eval_pool = [i for i in pool_union if i not in used]
    eval_ids = stratified_by_dataset(
        df, eval_pool, eval_n, id_to_idx=id_to_idx, seed=seed + 1, id_col=id_col
    )

    logger.info(
        "Playground {} splits: train={} eval={} (pool={})",
        preset,
        len(train_ids),
        len(eval_ids),
        len(pool_union),
    )
    return {
        "seed": seed,
        "preset": preset,
        "train_per_class": train_per_class,
        "eval_per_class": eval_per_class,
        "pool_ids": pool_union,
        "train_ids": train_ids,
        "eval_ids": eval_ids,
    }


def get_playground_splits(
    df: pd.DataFrame | None = None,
    bundle: Any | None = None,
    *,
    preset: str = "arc_med_mmlu",
    rebuild: bool = False,
) -> PlaygroundSplits:
    path = playground_splits_path(preset)
    if df is None:
        df = load_arena()
    if bundle is None:
        bundle = get_splits(df)

    if path.exists() and not rebuild:
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = build_playground_splits(df, bundle, preset=preset)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Wrote playground splits to {}", path)

    return PlaygroundSplits(
        preset=payload["preset"],
        train_ids=payload["train_ids"],
        eval_ids=payload["eval_ids"],
        pool_ids=payload["pool_ids"],
    )


def filter_rows_by_preset(rows: list[dict], preset: str) -> list[dict]:
    return [r for r in rows if row_matches_source(r, preset)]
