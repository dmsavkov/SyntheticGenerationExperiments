"""SetFit exp06: diversity triplet generation (3 refs / 3 strategies per call)."""

from __future__ import annotations

from routers.experiments.setfit._common import HYPOTHESES_BASE, run_setfit_experiment


def run(*, save: bool = True, rebuild_splits: bool = False) -> dict:
    return run_setfit_experiment(
        experiment="setfit06_diversity_mutation",
        hypotheses=f"{HYPOTHESES_BASE} SetFit06: diversity triplet mutation prompts (3 items/call).",
        generate_synthetics=True,
        selection_mode="uncertainty",
        generation_mode="diversity_triplet",
        save=save,
        rebuild_splits=rebuild_splits,
    )
