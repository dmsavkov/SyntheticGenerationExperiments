"""Smart reference selection and CV collector."""

from __future__ import annotations

from unittest.mock import MagicMock

from routers.synthetic.smart_selection import (
    bottom_correct_rows,
    mislabeled_rows,
    top_correct_rows,
)


def _row(gold: str, pred: str, p: float) -> dict:
    return {
        "gold": gold,
        "pred": pred,
        "p_gold": p,
        "correct": pred == gold,
        "pred_probs": {gold: p, "other": 1.0 - p},
    }


def test_mislabeled_filter():
    rows = [_row("A", "A", 0.9), _row("B", "A", 0.8)]
    assert len(mislabeled_rows(rows)) == 1


def test_bottom_top_percentile():
    correct = [_row("A", "A", p) for p in [0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05]]
    bottom = bottom_correct_rows(correct, frac=0.10)
    top = top_correct_rows(correct, frac=0.10)
    assert len(bottom) == 1
    assert bottom[0]["p_gold"] == 0.05
    assert len(top) == 1
    assert top[0]["p_gold"] == 0.95
