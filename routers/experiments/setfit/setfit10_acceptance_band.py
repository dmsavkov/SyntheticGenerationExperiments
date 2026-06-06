"""SetFit exp10: setfit05 verification pipeline with narrow prob band [0.4, 0.6]."""

from __future__ import annotations

from routers.core.constants import SETFIT_ACCEPTANCE_HIGH, SETFIT_ACCEPTANCE_LOW
from routers.experiments.setfit._common import HYPOTHESES_BASE, run_setfit_experiment


def run(*, save: bool = True, rebuild_splits: bool = False) -> dict:
    return run_setfit_experiment(
        experiment="setfit10_acceptance_band",
        hypotheses=(
            f"{HYPOTHESES_BASE} SetFit10: uncertainty refs + verification v2 (diversity + SetFit prob) "
            f"with band [{SETFIT_ACCEPTANCE_LOW}, {SETFIT_ACCEPTANCE_HIGH}] (vs setfit05 [0.2, 0.8])."
        ),
        generate_synthetics=True,
        selection_mode="uncertainty",
        generation_mode="label_failure",
        verification_v2=True,
        uncertainty_band=(SETFIT_ACCEPTANCE_LOW, SETFIT_ACCEPTANCE_HIGH),
        save=save,
        rebuild_splits=rebuild_splits,
    )
