"""Holdout reference selection and cycling for SetFit synthetic generation."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Literal

from loguru import logger

from routers.core.constants import SETFIT_SYNTHETIC_CAP, SETFIT_UNCERTAINTY_HIGH, SETFIT_UNCERTAINTY_LOW

SelectionMode = Literal[
    "uncertainty",
    "high_conf_correct",
    "low_conf_correct",
    "rewrite_split",
]


def score_holdout_rows(holdout_rows: list[dict], probe: Any) -> list[dict]:
    scored: list[dict] = []
    for row in holdout_rows:
        ext = probe.predict_extended(row)
        gold = str(row["gold"])
        p_gold = float(ext["pred_probs"].get(gold, 0.0))
        enriched = dict(row)
        enriched["pred"] = ext["pred"]
        enriched["pred_probs"] = ext["pred_probs"]
        enriched["p_gold"] = p_gold
        enriched["correct"] = ext["pred"] == gold
        scored.append(enriched)
    return scored


def expand_references(exemplars: list[dict], n_slots: int) -> tuple[list[dict], dict[str, Any]]:
    if not exemplars or n_slots <= 0:
        return [], {"n_unique_refs": 0, "n_ref_slots": n_slots, "cycling_used": False}
    unique = len(exemplars)
    cycling_used = unique < n_slots
    if cycling_used:
        logger.warning("Only {} unique refs; cycling to {} slots", unique, n_slots)
    pool = [dict(exemplars[i % unique]) for i in range(n_slots)]
    stats = {
        "n_unique_refs": unique,
        "n_ref_slots": n_slots,
        "cycling_used": cycling_used,
        "cycle_multiplier": round(n_slots / unique, 4) if unique else 0.0,
    }
    return pool, stats


def _stratified_pick(candidates: list[dict], cap: int, seed: int) -> list[dict]:
    if not candidates or cap <= 0:
        return []
    by_gold: dict[str, list[dict]] = defaultdict(list)
    for r in candidates:
        by_gold[str(r["gold"])].append(r)
    labels = sorted(by_gold.keys())
    per = max(1, cap // len(labels))
    rng = random.Random(seed)
    picked: list[dict] = []
    for label in labels:
        rows = by_gold[label]
        rng.shuffle(rows)
        picked.extend(rows[: min(per, len(rows))])
    if len(picked) < cap:
        rest = [r for r in candidates if r not in picked]
        rng.shuffle(rest)
        picked.extend(rest[: cap - len(picked)])
    return picked[:cap]


def _top_bottom_fraction(rows: list[dict], frac: float, *, top: bool) -> list[dict]:
    if not rows:
        return []
    k = max(1, int(len(rows) * frac))
    ordered = sorted(rows, key=lambda r: r["p_gold"], reverse=top)
    return ordered[:k]


def select_holdout_references(
    scored_rows: list[dict],
    *,
    mode: SelectionMode,
    cap: int = SETFIT_SYNTHETIC_CAP,
    seed: int = 42,
    uncertainty_low: float = SETFIT_UNCERTAINTY_LOW,
    uncertainty_high: float = SETFIT_UNCERTAINTY_HIGH,
) -> tuple[list[dict], dict[str, Any]]:
    if mode == "uncertainty":
        candidates = [
            r
            for r in scored_rows
            if uncertainty_low <= r["p_gold"] <= uncertainty_high
        ]
        unique = _stratified_pick(candidates, cap, seed)
    elif mode == "high_conf_correct":
        correct = [r for r in scored_rows if r["correct"]]
        pool = _top_bottom_fraction(correct, 0.10, top=True)
        unique = _stratified_pick(pool, cap, seed)
    elif mode == "low_conf_correct":
        correct = [r for r in scored_rows if r["correct"]]
        pool = _top_bottom_fraction(correct, 0.10, top=False)
        unique = _stratified_pick(pool, cap, seed)
    elif mode == "rewrite_split":
        high = [r for r in scored_rows if r["correct"] and r["p_gold"] > 0.9]
        mid = [
            r
            for r in scored_rows
            if uncertainty_low <= r["p_gold"] <= uncertainty_high
        ]
        half = cap // 2
        unique = _stratified_pick(high, half, seed) + _stratified_pick(mid, cap - half, seed + 1)
    else:
        raise ValueError(f"Unknown selection mode: {mode}")

    pool, cycle_stats = expand_references(unique, cap)
    stats: dict[str, Any] = {
        "selection_mode": mode,
        "n_holdout_scored": len(scored_rows),
        "n_candidates": len(unique),
        "uncertainty_band": [uncertainty_low, uncertainty_high],
        **cycle_stats,
    }
    logger.info(
        "Selection {}: {} unique → {} slots (cycling={})",
        mode,
        stats["n_unique_refs"],
        stats["n_ref_slots"],
        stats["cycling_used"],
    )
    return pool, stats
