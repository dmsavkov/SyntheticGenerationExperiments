"""SetFit exp07: hard + diverse rewrite from high-conf and uncertainty refs."""

from __future__ import annotations

from routers.experiments.setfit._common import HYPOTHESES_BASE, run_setfit_experiment


def run(*, save: bool = True, rebuild_splits: bool = False) -> dict:
    return run_setfit_experiment(
        experiment="setfit07_variation_rewrite",
        hypotheses=(
            f"{HYPOTHESES_BASE} SetFit07: 15 hard rewrites (conf>0.9) + 15 diverse rewrites "
            "(p in [0.2,0.8]), balanced per class."
        ),
        generate_synthetics=True,
        selection_mode="rewrite_split",
        generation_mode="mixed_rewrite",
        save=save,
        rebuild_splits=rebuild_splits,
    )
