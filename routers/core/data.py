"""RouterArena load, text formats, row assembly."""

from __future__ import annotations

import json
import re
from typing import Any, Literal

import pandas as pd
from loguru import logger

from routers.core.constants import CONTEXT_MAX_CHARS

DATASET_ID = "RouteWorks/RouterArena"
SPLITS = {
    "sub_10": "data/sub_10-00000-of-00001.parquet",
    "full": "data/full-00000-of-00001.parquet",
    "robustness": "data/robustness-00000-of-00001.parquet",
}
DEFAULT_SPLIT = "full"
SHUFFLE_SEED = 42

TARGET_COLUMNS = ("Domain", "Difficulty")
FEATURE_COLUMNS = ("Context", "Question", "Options")

TextFormat = Literal["simple_ml", "embedder", "bert"]


def load_arena(split: str = DEFAULT_SPLIT, seed: int = SHUFFLE_SEED) -> pd.DataFrame:
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; choose from {list(SPLITS)}")
    path = f"hf://datasets/{DATASET_ID}/{SPLITS[split]}"
    logger.info("Loading RouterArena split={} from {}", split, path)
    df = pd.read_parquet(path)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    logger.info("Loaded {} rows, {} columns", len(df), len(df.columns))
    return df


def _serialize_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def _field(row: pd.Series, col: str, context_max_chars: int = CONTEXT_MAX_CHARS) -> str:
    if col not in row.index:
        return ""
    raw = _serialize_value(row[col])
    if col == "Context" and len(raw) > context_max_chars:
        logger.debug("Truncating Context from {} to {} chars", len(raw), context_max_chars)
        raw = raw[:context_max_chars] + "…"
    return raw


def normalize_simple_ml(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def build_text(
    row: pd.Series,
    target_col: str,
    text_format: TextFormat = "embedder",
    context_max_chars: int = CONTEXT_MAX_CHARS,
) -> str:
    if target_col not in TARGET_COLUMNS:
        raise ValueError(f"target_col must be one of {TARGET_COLUMNS}, got {target_col!r}")

    ctx = _field(row, "Context", context_max_chars)
    question = _field(row, "Question", context_max_chars)
    options = _field(row, "Options", context_max_chars)

    if text_format == "simple_ml":
        return normalize_simple_ml(f"{ctx} {question} {options}")
    if text_format == "embedder":
        return f"Context: {ctx} | Question: {question} | Options: {options}"
    if text_format == "bert":
        return f"[CLS] {ctx} {question} {options} [SEP]"
    raise ValueError(f"Unknown text_format {text_format!r}")


def build_text_from_parts(context: str, question: str, options: str) -> str:
    return f"Context: {context} | Question: {question} | Options: {options}"


def row_dict_from_series(row: pd.Series, rid: Any, target_col: str) -> dict[str, Any]:
    return {
        "id": rid,
        "prompt": build_text(row, target_col, "embedder"),
        "gold": _serialize_value(row[target_col]),
        "dataset_name": _serialize_value(row.get("Dataset name", "")),
        "context": _field(row, "Context"),
        "question": _field(row, "Question"),
        "options": _field(row, "Options"),
    }


def rows_from_ids(
    df: pd.DataFrame,
    ids: list[Any],
    target_col: str,
    *,
    id_to_idx: dict[Any, int] | None = None,
    text_format: TextFormat = "embedder",
    with_features: bool = False,
) -> list[dict[str, Any]]:
    if id_to_idx is None:
        id_col = "Global Index" if "Global Index" in df.columns else None
        if id_col:
            id_to_idx = {df.iloc[i][id_col]: i for i in range(len(df))}
        else:
            id_to_idx = {i: i for i in range(len(df))}

    rows: list[dict[str, Any]] = []
    for rid in ids:
        idx = id_to_idx.get(rid)
        if idx is None:
            logger.warning("Missing id {} in dataframe", rid)
            continue
        row = df.iloc[idx]
        if with_features:
            rows.append(row_dict_from_series(row, rid, target_col))
        else:
            rows.append(
                {
                    "id": rid,
                    "prompt": build_text(row, target_col, text_format),
                    "gold": _serialize_value(row[target_col]),
                    "dataset_name": _serialize_value(row.get("Dataset name", "")),
                }
            )
    return rows


def label_vocab(df: pd.DataFrame, target_col: str) -> list[str]:
    return sorted(df[target_col].dropna().astype(str).unique().tolist())


def dataset_metadata(split: str = DEFAULT_SPLIT) -> dict[str, str]:
    return {"dataset_id": DATASET_ID, "dataset_split": split}
