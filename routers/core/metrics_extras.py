"""Metrics helpers for ensemble OVR routing with confused samples."""

from __future__ import annotations

from typing import Any

from sklearn.metrics import accuracy_score, f1_score

from routers.core.harness import _per_class_pr


def compute_subset_metrics(
    golds: list[str],
    preds: list[str],
    *,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    if not golds:
        return {"n": 0, "accuracy": 0.0, "macro_f1": 0.0}
    label_list = labels or sorted(set(golds) | set(preds))
    acc = accuracy_score(golds, preds)
    macro_f1 = f1_score(golds, preds, labels=label_list, average="macro", zero_division=0)
    per_prec, per_rec = _per_class_pr(golds, preds, label_list)
    return {
        "n": len(golds),
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "per_class_precision": per_prec,
        "per_class_recall": per_rec,
    }


def metrics_with_confused_breakdown(
    golds: list[str],
    preds: list[str],
    confused_flags: list[bool],
    *,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """Global metrics use resolved preds; also report metrics excluding confused rows."""
    label_list = labels or sorted(set(golds) | set(preds))
    all_metrics = compute_subset_metrics(golds, preds, labels=label_list)
    n_confused = sum(1 for c in confused_flags if c)
    all_metrics["n_confused"] = n_confused
    all_metrics["confused_rate"] = round(n_confused / len(golds), 4) if golds else 0.0

    clear_g = [g for g, c in zip(golds, confused_flags) if not c]
    clear_p = [p for g, p, c in zip(golds, preds, confused_flags) if not c]
    all_metrics["metrics_excluding_confused"] = compute_subset_metrics(
        clear_g, clear_p, labels=label_list
    )
    return all_metrics
