"""Ensemble 01: OpenTDB multiclass baseline (100 train, 100 holdout, rest test)."""

from __future__ import annotations

from routers.core.constants import ENSEMBLE_TOPIC
from routers.experiments.opentdb_ensemble._common import HYPOTHESES_BASE, run_baseline_multiclass


def run(*, save: bool = True, rebuild_splits: bool = False) -> dict:
    return run_baseline_multiclass(
        experiment="ensemble01_baseline",
        topic=ENSEMBLE_TOPIC,
        hypotheses=f"{HYPOTHESES_BASE} Ensemble01: multiclass SetFit baseline, Gemini protocol, no synth.",
        save=save,
        rebuild_splits=rebuild_splits,
    )
