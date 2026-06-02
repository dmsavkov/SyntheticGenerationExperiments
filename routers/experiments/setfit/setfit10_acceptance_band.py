"""SetFit exp10: narrow acceptance zone [0.4, 0.6] for uncertainty selection."""

from __future__ import annotations

from routers.core.constants import SETFIT_ACCEPTANCE_HIGH, SETFIT_ACCEPTANCE_LOW
from routers.experiments.setfit._common import HYPOTHESES_BASE, run_setfit_experiment


def run(*, save: bool = True, rebuild_splits: bool = False) -> dict:
    return run_setfit_experiment(
        experiment="setfit10_acceptance_band",
        hypotheses=(
            f"{HYPOTHESES_BASE} SetFit10: uncertainty synth with acceptance band "
            f"[{SETFIT_ACCEPTANCE_LOW}, {SETFIT_ACCEPTANCE_HIGH}]."
        ),
        generate_synthetics=True,
        selection_mode="uncertainty",
        generation_mode="label_failure",
        uncertainty_band=(SETFIT_ACCEPTANCE_LOW, SETFIT_ACCEPTANCE_HIGH),
        save=save,
        rebuild_splits=rebuild_splits,
    )
