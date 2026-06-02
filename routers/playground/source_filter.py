"""Source-dataset presets for playground filtering."""

from __future__ import annotations

from typing import Any

SourcePreset = str

PRESETS: dict[str, tuple[str, ...]] = {
    "arc_med_mmlu": ("ArcMMLU", "MedMCQA", "MMLU_management"),
    "opentdb_only": ("OpenTDB",),
    "all": (),
}


def row_matches_source(row_or_id: Any, preset: SourcePreset) -> bool:
    """Match Global Index prefix or Dataset name against preset."""
    prefixes = PRESETS.get(preset)
    if prefixes is None:
        raise ValueError(f"Unknown source_preset {preset!r}; choose from {list(PRESETS)}")
    if preset == "all":
        return True

    if isinstance(row_or_id, dict):
        rid = str(row_or_id.get("id", ""))
        dname = str(row_or_id.get("dataset_name", ""))
    else:
        rid = str(row_or_id)
        dname = ""

    for p in prefixes:
        if rid.startswith(p) or dname.startswith(p):
            return True
    return False


def filter_ids(ids: list[Any], preset: SourcePreset) -> list[Any]:
    return [i for i in ids if row_matches_source(i, preset)]
