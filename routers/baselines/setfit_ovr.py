"""Binary SetFit OVR probes for a single positive Domain label."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from routers.baselines.setfit_probe import default_training_args, train_setfit_probe
from routers.core.constants import SETFIT_MODEL_ID

OVR_POS_LABEL = "1"
OVR_NEG_LABEL = "0"


@dataclass
class SetFitOvrProbe:
    """One-vs-rest probe: predict P(positive) for positive_domain."""

    probe: Any
    positive_domain: str

    def p_positive(self, row: dict) -> float:
        probs = self.probe.predict_proba_dict(row)
        return float(probs.get(OVR_POS_LABEL, 0.0))

    def is_positive(self, row: dict, *, threshold: float = 0.5) -> bool:
        return self.p_positive(row) >= threshold


def rows_to_binary(train_rows: list[dict], positive_domain: str) -> list[dict]:
    out: list[dict] = []
    for r in train_rows:
        row = dict(r)
        row["gold"] = OVR_POS_LABEL if str(r["gold"]) == positive_domain else OVR_NEG_LABEL
        out.append(row)
    return out


def cap_class_imbalance(
    rows: list[dict],
    *,
    max_spread: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], dict[str, Any]]:
    """Downsample majority gold labels until (max_count - min_count) / n <= max_spread."""
    import random
    from collections import Counter

    if not rows:
        return [], {"imbalance_before": 0.0, "imbalance_after": 0.0, "n_before": 0, "n_after": 0}

    def spread(rs: list[dict]) -> float:
        c = Counter(str(r["gold"]) for r in rs)
        if len(c) < 2:
            return 0.0
        counts = list(c.values())
        return (max(counts) - min(counts)) / len(rs)

    before = spread(rows)
    if before <= max_spread:
        return list(rows), {
            "imbalance_before": round(before, 4),
            "imbalance_after": round(before, 4),
            "n_before": len(rows),
            "n_after": len(rows),
            "resampled": False,
        }

    by_gold: dict[str, list[dict]] = {}
    for r in rows:
        by_gold.setdefault(str(r["gold"]), []).append(r)
    rng = random.Random(seed)
    target_max = max(1, int(min(len(by_gold[g]) for g in by_gold) + max_spread * len(rows)))
    trimmed: list[dict] = []
    for gold, group in by_gold.items():
        rng.shuffle(group)
        trimmed.extend(group[: min(len(group), target_max)])
    after = spread(trimmed)
    return trimmed, {
        "imbalance_before": round(before, 4),
        "imbalance_after": round(after, 4),
        "n_before": len(rows),
        "n_after": len(trimmed),
        "resampled": True,
    }


def train_ovr_probe(
    train_rows: list[dict],
    positive_domain: str,
    *,
    balance: bool = True,
    max_spread: float = 0.2,
    model_id: str = SETFIT_MODEL_ID,
) -> tuple[SetFitOvrProbe, float, dict[str, Any]]:
    binary = rows_to_binary(train_rows, positive_domain)
    balance_stats: dict[str, Any] = {"resampled": False}
    if balance:
        binary, balance_stats = cap_class_imbalance(binary, max_spread=max_spread)
    probe, train_s = train_setfit_probe(binary, model_id=model_id)
    return SetFitOvrProbe(probe=probe, positive_domain=positive_domain), train_s, balance_stats
