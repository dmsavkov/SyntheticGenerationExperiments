"""SetFit exp01: train_100 real only, no synthetic generation."""

from __future__ import annotations

from routers.experiments.setfit._common import HYPOTHESES_BASE, run_setfit_experiment


def run(*, save: bool = True, rebuild_splits: bool = False) -> dict:
    return run_setfit_experiment(
        experiment="setfit01_baseline",
        hypotheses=f"{HYPOTHESES_BASE} SetFit01: train_100 floor, no synthetic generation.",
        generate_synthetics=False,
        save=save,
        rebuild_splits=rebuild_splits,
    )
