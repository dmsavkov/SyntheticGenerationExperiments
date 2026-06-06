"""Shared smart experiment runner."""

from __future__ import annotations

from typing import Any

from routers.baselines.setfit_probe import probe_config_extra
from routers.core.constants import (
    COMBINATION_GOOGLE_THINKING_LEVEL,
    ENSEMBLE_MAX_ITEMS_PER_REQUEST,
    GENERATION_TEMPERATURE,
    GOOGLE_FLASH_MODEL_DEFAULT,
    SMART_MAX_IN_FLIGHT,
    SMART_PARSE_MAX_RETRIES,
    SMART_TOPIC,
    TARGET_COL,
    TEXT_FORMAT,
)
from routers.core.data import dataset_metadata

HYPOTHESES_BASE = "OpenTDB smart SetFit experiments (Gemini, percentile refs, async LLM)."


def generate_mislabel_lowconf_synthetics(
    scored: list[dict],
    labels: list[str],
    *,
    stats: dict[str, Any] | None = None,
    cap: int | None = None,
) -> tuple[list[dict], dict[str, int]]:
    """Smart03-style: mislabeled failures + bottom-percentile correct → hard negatives."""
    from routers.core.constants import SMART_SYNTHETIC_CAP
    from routers.synthetic.generator_smart import generate_hard_negative_pairs_smart
    from routers.synthetic.smart_selection import bottom_correct_rows, mislabeled_rows

    limit = cap if cap is not None else SMART_SYNTHETIC_CAP
    failures = mislabeled_rows(scored)
    low_conf = bottom_correct_rows(scored)
    sel = {"n_mislabeled": len(failures), "n_bottom_correct": len(low_conf)}
    synthetics = generate_hard_negative_pairs_smart(failures, labels, stats=stats)
    if low_conf:
        synthetics.extend(
            generate_hard_negative_pairs_smart(
                low_conf[: max(1, limit // 2)], labels, stats=stats
            )
        )
    return synthetics[:limit], sel


def smart_base_config(
    *,
    experiment: str,
    hypotheses: str,
    extra: dict[str, Any] | None = None,
    use_reasoning: bool = True,
) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        **dataset_metadata(),
        "topic": SMART_TOPIC,
        "target_col": TARGET_COL,
        "train_split": "train_100",
        "holdout_split": "holdout_100",
        "eval_split": "test_ids",
        "text_format": TEXT_FORMAT,
        "generation_temperature": GENERATION_TEMPERATURE,
        "max_items_per_llm_request": ENSEMBLE_MAX_ITEMS_PER_REQUEST,
        "llm_backend": "google",
        "generation_model": GOOGLE_FLASH_MODEL_DEFAULT,
        "parse_max_retries": SMART_PARSE_MAX_RETRIES,
        "max_llm_in_flight": SMART_MAX_IN_FLIGHT,
        "hypotheses": hypotheses,
        "experiment": experiment,
        **probe_config_extra(),
    }
    if use_reasoning:
        cfg["google_thinking_level"] = COMBINATION_GOOGLE_THINKING_LEVEL
    else:
        cfg["google_thinking_level"] = None
        cfg["use_reasoning"] = False
    if extra:
        cfg.update(extra)
    return cfg
