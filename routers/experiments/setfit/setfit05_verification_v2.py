"""SetFit exp05: uncertainty synth + verification v2 filter."""

from __future__ import annotations

from routers.experiments.setfit._common import HYPOTHESES_BASE, run_setfit_experiment


def run(*, save: bool = True, rebuild_splits: bool = False) -> dict:
    return run_setfit_experiment(
        experiment="setfit05_verification_v2",
        hypotheses=(
            f"{HYPOTHESES_BASE} SetFit05: uncertainty synth + fastembed diversity + SetFit prob filter."
        ),
        generate_synthetics=True,
        selection_mode="uncertainty",
        generation_mode="label_failure",
        verification_v2=True,
        save=save,
        rebuild_splits=rebuild_splits,
    )
