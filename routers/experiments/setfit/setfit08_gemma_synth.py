"""SetFit exp08: uncertainty synth via Google GenAI gemini-3.1-flash-lite."""

from __future__ import annotations

from routers.core.constants import GOOGLE_FLASH_MODEL_DEFAULT
from routers.experiments.setfit._common import HYPOTHESES_BASE, run_setfit_experiment


def run(*, save: bool = True, rebuild_splits: bool = False) -> dict:
    return run_setfit_experiment(
        experiment="setfit08_gemma_synth",
        hypotheses=(
            f"{HYPOTHESES_BASE} SetFit08: uncertainty synth via Google GenAI {GOOGLE_FLASH_MODEL_DEFAULT}."
        ),
        generate_synthetics=True,
        selection_mode="uncertainty",
        generation_mode="label_failure",
        llm_backend="google",
        generation_model=GOOGLE_FLASH_MODEL_DEFAULT,
        save=save,
        rebuild_splits=rebuild_splits,
    )
