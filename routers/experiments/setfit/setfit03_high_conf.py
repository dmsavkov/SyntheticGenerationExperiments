"""SetFit exp03: synth from top 10% confidence correct holdout refs."""

from __future__ import annotations

from routers.experiments.setfit._common import HYPOTHESES_BASE, run_setfit_experiment


def run(*, save: bool = True, rebuild_splits: bool = False) -> dict:
    return run_setfit_experiment(
        experiment="setfit03_high_conf",
        hypotheses=f"{HYPOTHESES_BASE} SetFit03: synth from top 10% confidence correct holdout refs.",
        generate_synthetics=True,
        selection_mode="high_conf_correct",
        generation_mode="label_failure",
        save=save,
        rebuild_splits=rebuild_splits,
    )
