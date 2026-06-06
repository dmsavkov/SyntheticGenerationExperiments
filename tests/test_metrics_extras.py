"""Confused metrics: global pred always set; subset excludes confused."""

from __future__ import annotations

from routers.core.metrics_extras import metrics_with_confused_breakdown


def test_confused_excluded_subset():
    golds = ["A", "B", "A"]
    preds = ["A", "A", "B"]
    confused = [False, True, False]
    m = metrics_with_confused_breakdown(golds, preds, confused, labels=["A", "B"])
    assert m["n"] == 3
    assert m["n_confused"] == 1
    assert m["metrics_excluding_confused"]["n"] == 2
