"""Percentile-based reference selection and CV failure collection for smart experiments."""

from __future__ import annotations

import random
from typing import Any

from loguru import logger
from sklearn.model_selection import StratifiedKFold

from routers.baselines.setfit_probe import train_setfit_probe
from routers.core.constants import SMART_CV_FOLDS, SMART_REF_PERCENTILE
from routers.experiments._common import train_rows_from_ids
from routers.synthetic.holdout_selection import _stratified_pick, _top_bottom_fraction


def score_rows(probe: Any, rows: list[dict]) -> list[dict]:
    scored: list[dict] = []
    for row in rows:
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


def mislabeled_rows(scored: list[dict]) -> list[dict]:
    return [r for r in scored if not r["correct"]]


def bottom_correct_rows(
    scored: list[dict],
    *,
    frac: float = SMART_REF_PERCENTILE,
) -> list[dict]:
    correct = [r for r in scored if r["correct"]]
    return _top_bottom_fraction(correct, frac, top=False)


def top_correct_rows(
    scored: list[dict],
    *,
    frac: float = SMART_REF_PERCENTILE,
) -> list[dict]:
    correct = [r for r in scored if r["correct"]]
    return _top_bottom_fraction(correct, frac, top=True)


def pick_random(rows: list[dict], n: int, seed: int = 42) -> list[dict]:
    if not rows or n <= 0:
        return []
    rng = random.Random(seed)
    if len(rows) <= n:
        return list(rows)
    return rng.sample(rows, n)


def collect_cv_mislabels(
    df: Any,
    train_ids: list[Any],
    bundle: Any,
    *,
    n_folds: int = SMART_CV_FOLDS,
    seed: int = 42,
) -> tuple[list[dict], dict[str, Any]]:
    """5-fold CV on train_100: aggregate unique val mislabels for synth refs."""
    rows = train_rows_from_ids(df, train_ids, bundle)
    labels = [str(r["gold"]) for r in rows]
    ids = [r["id"] for r in rows]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    seen: set[Any] = set()
    failures: list[dict] = []
    fold_stats: list[dict] = []

    for fold_i, (tr_idx, val_idx) in enumerate(skf.split(ids, labels)):
        tr_rows = [rows[i] for i in tr_idx]
        val_rows = [rows[i] for i in val_idx]
        probe, _ = train_setfit_probe(tr_rows)
        scored_val = score_rows(probe, val_rows)
        fold_fail = mislabeled_rows(scored_val)
        fold_stats.append(
            {"fold": fold_i + 1, "n_train": len(tr_rows), "n_val": len(val_rows), "n_mislabeled": len(fold_fail)}
        )
        for r in fold_fail:
            rid = r.get("id")
            if rid not in seen:
                seen.add(rid)
                failures.append(r)
        logger.info("CV fold {}: {} val mislabels ({} unique total)", fold_i + 1, len(fold_fail), len(failures))

    stats = {"n_folds": n_folds, "n_unique_failures": len(failures), "folds": fold_stats}
    return failures, stats


def partition_expansion_refs(
    scored: list[dict],
    *,
    frac: float = SMART_REF_PERCENTILE,
    cap_per_bucket: int | None = None,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Route refs into top / bottom / mislabeled buckets for dataset expansion."""
    mis = mislabeled_rows(scored)
    top = top_correct_rows(scored, frac=frac)
    bottom = bottom_correct_rows(scored, frac=frac)
    if cap_per_bucket:
        mis = _stratified_pick(mis, cap_per_bucket, seed)
        top = _stratified_pick(top, cap_per_bucket, seed + 1)
        bottom = _stratified_pick(bottom, cap_per_bucket, seed + 2)
    return {"mislabeled": mis, "top_correct": top, "bottom_correct": bottom}


def neighbor_class_for_row(row: dict) -> str:
    """Confusing class: pred if wrong, else second-highest prob label."""
    if not row.get("correct") and row.get("pred"):
        return str(row["pred"])
    probs = row.get("pred_probs") or {}
    gold = str(row.get("gold", ""))
    ranked = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    for lab, _ in ranked:
        if lab != gold:
            return lab
    return gold
