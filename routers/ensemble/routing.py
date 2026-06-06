"""Expert + general routing and OVR panel routing with scream-based confusion."""

from __future__ import annotations

from typing import Any

from routers.baselines.setfit_ovr import SetFitOvrProbe
from routers.baselines.setfit_probe import SetFitProbe
from routers.core.constants import ENSEMBLE_EXPERT_SCREAM_THRESHOLD, ENSEMBLE_OVR_POS_THRESHOLD


def _screaming_labels_from_probs(
    label_probs: dict[str, float],
    *,
    threshold: float,
) -> list[str]:
    return [lab for lab, p in label_probs.items() if p >= threshold]


def resolve_expert_general_prediction(
    general: SetFitProbe,
    expert_probes: dict[str, SetFitOvrProbe],
    row: dict,
    *,
    scream_threshold: float = ENSEMBLE_EXPERT_SCREAM_THRESHOLD,
) -> dict[str, Any]:
    """
    Case A: max expert P >= scream_threshold → expert class wins.
    Case B: else → general multiclass pred.
    Confused when >=2 distinct labels scream (general and/or experts) at >= threshold.
    """
    expert_probs = {label: probe.p_positive(row) for label, probe in expert_probes.items()}
    max_exp_class = max(expert_probs, key=expert_probs.get) if expert_probs else ""
    max_exp_prob = expert_probs.get(max_exp_class, 0.0)

    gen_ext = general.predict_extended(row)
    general_pred = str(gen_ext["pred"])
    general_probs = {str(k): float(v) for k, v in gen_ext["pred_probs"].items()}
    general_max_prob = max(general_probs.values()) if general_probs else 0.0
    general_scream_label = max(general_probs, key=general_probs.get) if general_probs else general_pred

    scream_map: dict[str, float] = {}
    if general_max_prob >= scream_threshold:
        scream_map[general_scream_label] = general_max_prob
    for lab, p in expert_probs.items():
        if p >= scream_threshold:
            scream_map[lab] = max(scream_map.get(lab, 0.0), p)

    confused = len(scream_map) > 1
    pred = max_exp_class if max_exp_prob >= scream_threshold else general_pred

    return {
        "pred": pred,
        "confused": confused,
        "general_pred": general_pred,
        "general_max_prob": round(general_max_prob, 6),
        "max_expert_class": max_exp_class,
        "max_expert_prob": round(max_exp_prob, 6),
        "routing_case": "expert_override" if max_exp_prob >= scream_threshold else "general_fallback",
        "screaming_labels": sorted(scream_map.keys()),
        "scream_probs": {k: round(v, 6) for k, v in scream_map.items()},
        "ovr_probs": {k: round(v, 6) for k, v in expert_probs.items()},
    }


def resolve_ovr_panel_prediction(
    probes: dict[str, SetFitOvrProbe],
    row: dict,
    *,
    scream_threshold: float = ENSEMBLE_EXPERT_SCREAM_THRESHOLD,
) -> dict[str, Any]:
    """Full OVR panel: pred = argmax P(positive); confused if >1 class screams."""
    probs = {label: probe.p_positive(row) for label, probe in probes.items()}
    screamers = _screaming_labels_from_probs(probs, threshold=scream_threshold)
    confused = len(set(screamers)) > 1
    pred = max(probs, key=probs.get) if probs else ""
    return {
        "pred": pred,
        "confused": confused,
        "n_expert_positive": len([p for p in probs.values() if p >= ENSEMBLE_OVR_POS_THRESHOLD]),
        "ovr_probs": {k: round(v, 6) for k, v in probs.items()},
        "positive_labels": screamers,
        "screaming_labels": sorted(set(screamers)),
    }


def resolve_ovr_prediction(
    probes: dict[str, SetFitOvrProbe],
    row: dict,
    *,
    threshold: float = ENSEMBLE_OVR_POS_THRESHOLD,
) -> dict[str, Any]:
    """Legacy alias — delegates to panel prediction with scream-based confusion."""
    out = resolve_ovr_panel_prediction(probes, row, scream_threshold=ENSEMBLE_EXPERT_SCREAM_THRESHOLD)
    out["positive_labels"] = [
        lab for lab, p in out["ovr_probs"].items() if p >= threshold
    ]
    return out


def per_class_f1_from_records(
    records: list[dict],
    labels: list[str],
) -> dict[str, float]:
    from sklearn.metrics import classification_report

    golds = [str(r["gold"]) for r in records]
    preds = [str(r["pred"]) for r in records]
    report = classification_report(
        golds, preds, labels=labels, output_dict=True, zero_division=0
    )
    return {
        lab: round(float(report[lab]["f1-score"]), 4)
        for lab in labels
        if lab in report and isinstance(report[lab], dict)
    }


def select_weak_expert_classes(
    holdout_records: list[dict],
    label_universe: list[str],
    *,
    max_experts: int = 2,
    min_support: int = 8,
) -> list[str]:
    """Pick up to max_experts classes with lowest per-class F1 and gold support >= min_support."""
    from collections import Counter

    gold_counts = Counter(str(r["gold"]) for r in holdout_records)
    f1_by_class = per_class_f1_from_records(holdout_records, label_universe)
    eligible = [
        lab
        for lab in label_universe
        if gold_counts.get(lab, 0) >= min_support and lab in f1_by_class
    ]
    ranked = sorted(eligible, key=lambda lab: f1_by_class[lab])
    return ranked[:max_experts]
