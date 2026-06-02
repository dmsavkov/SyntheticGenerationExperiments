"""SetFit exp02: train_100 + 30 synth from uncertainty-band holdout refs (Ollama)."""

from __future__ import annotations

from routers.experiments.setfit._common import HYPOTHESES_BASE, run_setfit_experiment


def run(*, save: bool = True, rebuild_splits: bool = False) -> dict:
    return run_setfit_experiment(
        experiment="setfit02_uncertainty_synth",
        hypotheses=(
            f"{HYPOTHESES_BASE} SetFit02: append 30 synth from holdout uncertainty band "
            "p(true) in [0.2, 0.8] (Ollama)."
        ),
        generate_synthetics=True,
        selection_mode="uncertainty",
        generation_mode="label_failure",
        llm_backend="ollama",
        save=save,
        rebuild_splits=rebuild_splits,
    )
