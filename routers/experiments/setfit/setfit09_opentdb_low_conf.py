"""SetFit exp09: OpenTDB-only data, low-confidence correct holdout refs."""

from __future__ import annotations

from routers.experiments.setfit._common import HYPOTHESES_BASE, run_setfit_experiment


def run(*, save: bool = True, rebuild_splits: bool = False) -> dict:
    return run_setfit_experiment(
        experiment="setfit09_opentdb_low_conf",
        hypotheses=f"{HYPOTHESES_BASE} SetFit09: OpenTDB-only pool, bottom 10% conf correct synth refs.",
        generate_synthetics=True,
        selection_mode="low_conf_correct",
        generation_mode="label_failure",
        split_variant="opentdb",
        save=save,
        rebuild_splits=rebuild_splits,
    )
