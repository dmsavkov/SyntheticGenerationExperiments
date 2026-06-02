"""Shared SetFit experiment runner."""

from __future__ import annotations

from typing import Any, Literal

from loguru import logger

from routers.baselines.setfit_probe import probe_config_extra, train_setfit_probe
from routers.core.constants import (
    GENERATION_TEMPERATURE,
    GOOGLE_FLASH_MODEL_DEFAULT,
    GOOGLE_GEMMA_MODEL_DEFAULT,
    SETFIT_SYNTHETIC_CAP,
    SETFIT_TRAIN_N,
    SETFIT_UNCERTAINTY_HIGH,
    SETFIT_UNCERTAINTY_LOW,
    TARGET_COL,
    TEXT_FORMAT,
)
from routers.core.data import dataset_metadata, label_vocab
from routers.experiments._common import (
    eval_holdout,
    load_context,
    save_primary_eval,
    train_rows_from_ids,
)
from routers.experiments.session import ExperimentSession
from routers.synthetic.generator_setfit import GenerationMode, generate_from_references
from routers.synthetic.holdout_selection import SelectionMode, score_holdout_rows, select_holdout_references
from routers.synthetic.llm_client import LlmBackend
from routers.synthetic.ollama_client import ollama_model
from routers.synthetic.verification_v2 import filter_verification_v2

SETFIT_TOPIC = "setfit"
HYPOTHESES_BASE = "SetFit Domain synthetic collision experiments on RouterArena."


def load_setfit_context(
    *,
    rebuild_splits: bool = False,
    split_variant: Literal["default", "opentdb"] = "default",
) -> tuple[Any, Any, Any]:
    """Load arena + splits; opentdb variant uses filtered split JSON."""
    from routers.core.splits import get_opentdb_setfit_splits

    if split_variant == "default":
        return load_context(rebuild_splits=rebuild_splits)
    df, bundle, ds = load_context(rebuild_splits=rebuild_splits)
    otd = get_opentdb_setfit_splits(df, bundle, rebuild=rebuild_splits)
    eval_ids = otd.eval_ids if otd.eval_ids else bundle.eval_1000_ids

    class _EvalBundle:
        def __init__(self, base: Any, eval_ids: list[Any]) -> None:
            self._base = base
            self.eval_1000_ids = eval_ids

        def __getattr__(self, name: str) -> Any:
            return getattr(self._base, name)

    return df, _EvalBundle(bundle, eval_ids), otd


def setfit_base_config(
    *,
    experiment: str,
    hypotheses: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        **dataset_metadata(),
        "topic": SETFIT_TOPIC,
        "target_col": TARGET_COL,
        "train_split": "train_100",
        "eval_split": "eval_1000",
        "text_format": TEXT_FORMAT,
        "ollama_model": ollama_model(),
        "generation_temperature": GENERATION_TEMPERATURE,
        "max_items_per_llm_request": 3,
        "synthetic_cap": SETFIT_SYNTHETIC_CAP,
        "hypotheses": hypotheses,
        "experiment": experiment,
        **probe_config_extra(),
    }
    if extra:
        cfg.update(extra)
    return cfg


def run_setfit_experiment(
    *,
    experiment: str,
    hypotheses: str,
    generate_synthetics: bool = True,
    selection_mode: SelectionMode | None = "uncertainty",
    generation_mode: GenerationMode = "label_failure",
    verification_v2: bool = False,
    llm_backend: LlmBackend = "ollama",
    generation_model: str | None = None,
    synthetic_cap: int = SETFIT_SYNTHETIC_CAP,
    uncertainty_band: tuple[float, float] | None = None,
    split_variant: Literal["default", "opentdb"] = "default",
    save: bool = True,
    rebuild_splits: bool = False,
) -> dict:
    u_low, u_high = uncertainty_band or (SETFIT_UNCERTAINTY_LOW, SETFIT_UNCERTAINTY_HIGH)
    df, bundle, ds = load_setfit_context(rebuild_splits=rebuild_splits, split_variant=split_variant)
    labels = label_vocab(df, "Domain")
    cfg = setfit_base_config(
        experiment=experiment,
        hypotheses=hypotheses,
        extra={
            "training_mode": "real_plus_synthetic" if generate_synthetics else "real_only",
            "selection_mode": selection_mode,
            "generation_mode": generation_mode if generate_synthetics else None,
            "verification_v2": verification_v2,
            "llm_backend": llm_backend if generate_synthetics else None,
            "generation_model": generation_model
            or (GOOGLE_FLASH_MODEL_DEFAULT if llm_backend == "google" else ollama_model()),
            "uncertainty_band": [u_low, u_high],
            "split_variant": split_variant,
        },
    )
    session = ExperimentSession(experiment, topic=SETFIT_TOPIC)

    train_real = train_rows_from_ids(df, ds.train_100_ids, bundle)
    cfg["n_train_real"] = len(train_real)
    if len(train_real) != SETFIT_TRAIN_N:
        logger.warning("train_100 has {} rows (expected {})", len(train_real), SETFIT_TRAIN_N)

    holdout_rows = train_rows_from_ids(df, ds.holdout_500_ids, bundle)
    probe0, train_s0 = train_setfit_probe(train_real)

    if generate_synthetics:
        eval_holdout(
            session, probe0, df, ds.holdout_500_ids, bundle,
            phase="pre", hypotheses=hypotheses, config=cfg, train_seconds=train_s0,
        )
        scored = score_holdout_rows(holdout_rows, probe0)
        assert selection_mode is not None
        ref_pool, sel_stats = select_holdout_references(
            scored,
            mode=selection_mode,
            cap=synthetic_cap,
            uncertainty_low=u_low,
            uncertainty_high=u_high,
        )
        session.selection_stats = sel_stats
        cfg["selection_stats"] = sel_stats

        synthetics = generate_from_references(
            ref_pool,
            labels,
            mode=generation_mode,
            cap=synthetic_cap,
            stats=session.generation_stats,
            backend=llm_backend,
            generation_model=generation_model,
        )

        if verification_v2:
            synthetics, _, vstats = filter_verification_v2(
                synthetics,
                train_real,
                probe0,
                session=session,
                uncertainty_low=u_low,
                uncertainty_high=u_high,
            )
            cfg["verification_stats"] = vstats

        session.save_synthetics(synthetics, stats=session.generation_stats)
        cfg["n_synthetic"] = len(synthetics)
        train_final = train_real + synthetics
        probe1, train_s1 = train_setfit_probe(train_final)
        eval_holdout(
            session, probe1, df, ds.holdout_500_ids, bundle,
            phase="post", hypotheses=hypotheses, config=cfg, train_seconds=train_s1,
        )
        cfg["n_train_total"] = len(train_final)
        final_probe = probe1
        final_train_s = train_s1
    else:
        eval_holdout(
            session, probe0, df, ds.holdout_500_ids, bundle,
            phase="baseline", hypotheses=hypotheses, config=cfg, train_seconds=train_s0,
        )
        cfg["n_synthetic"] = 0
        cfg["n_train_total"] = len(train_real)
        final_probe = probe0
        final_train_s = train_s0

    if not save:
        return {}
    return save_primary_eval(
        session, final_probe, df, bundle.eval_1000_ids, bundle,
        hypotheses=hypotheses, config=cfg, train_seconds=final_train_s,
    )
