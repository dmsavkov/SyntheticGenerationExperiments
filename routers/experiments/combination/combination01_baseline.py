"""Combination 01: same baseline as ensemble01."""

from __future__ import annotations

from routers.core.constants import COMBINATION_TOPIC
from routers.experiments.opentdb_ensemble._common import HYPOTHESES_BASE, run_baseline_multiclass


def run(*, save: bool = True, rebuild_splits: bool = False) -> dict:
    return run_baseline_multiclass(
        experiment="combination01_baseline",
        topic=COMBINATION_TOPIC,
        hypotheses=f"{HYPOTHESES_BASE} Combination01: multiclass SetFit baseline.",
        save=save,
        rebuild_splits=rebuild_splits,
    )
