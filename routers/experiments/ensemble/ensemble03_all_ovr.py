"""Ensemble 03: full OVR panel (one per label), one synth iter, scream-based routing."""

from __future__ import annotations

from loguru import logger

from routers.baselines.setfit_ovr import SetFitOvrProbe, train_ovr_probe
from routers.core.constants import ENSEMBLE_TOPIC, GOOGLE_FLASH_MODEL_DEFAULT
from routers.experiments._common import train_rows_from_ids
from routers.experiments.opentdb_ensemble._common import (
    HYPOTHESES_BASE,
    ensemble_base_config,
    eval_ovr_panel_holdout,
    eval_ovr_panel_test,
    load_ensemble_context,
)
from routers.experiments.session import ExperimentSession
from routers.synthetic.generator_opentdb import generate_ovr_mutation


def run(*, save: bool = True, rebuild_splits: bool = False) -> dict:
    df, bundle, splits, labels = load_ensemble_context(rebuild_splits=rebuild_splits)
    hyp = (
        f"{HYPOTHESES_BASE} Ensemble03: {len(labels)} OVR models, 1 synth iter, "
        "scream-based confused routing (>=0.85)."
    )
    cfg = ensemble_base_config(
        experiment="ensemble03_all_ovr",
        topic=ENSEMBLE_TOPIC,
        hypotheses=hyp,
        extra={"training_mode": "full_ovr_panel", "label_universe": labels},
    )
    session = ExperimentSession("ensemble03_all_ovr", topic=ENSEMBLE_TOPIC)
    train_rows = train_rows_from_ids(df, splits.train_100_ids, bundle)
    panel: dict[str, SetFitOvrProbe] = {}
    all_synth: list[dict] = []
    total_train_s = 0.0

    for lab in labels:
        ovr, ts, _ = train_ovr_probe(train_rows, lab)
        panel[lab] = ovr
        total_train_s += ts

    eval_ovr_panel_holdout(
        session,
        panel,
        df,
        splits.holdout_100_ids,
        bundle,
        phase="pre",
        hypotheses=hyp,
        config=cfg,
        train_seconds=total_train_s,
    )

    holdout_rows = train_rows_from_ids(df, splits.holdout_100_ids, bundle)
    for lab in labels:
        fps = [
            dict(row, pred=lab)
            for row in holdout_rows
            if panel[lab].is_positive(row) and str(row["gold"]) != lab
        ]
        if not fps:
            continue
        batch = generate_ovr_mutation(
            fps[:8],
            labels,
            lab,
            cap=8,
            stats=session.generation_stats,
            generation_model=GOOGLE_FLASH_MODEL_DEFAULT,
        )
        for row in batch:
            row["ovr_class"] = lab
        all_synth.extend(batch)
        ovr, ts, _ = train_ovr_probe(train_rows + batch, lab)
        panel[lab] = ovr
        total_train_s += ts

    session.save_synthetics(all_synth, stats=session.generation_stats)
    cfg["n_synthetic"] = len(all_synth)
    eval_ovr_panel_holdout(
        session,
        panel,
        df,
        splits.holdout_100_ids,
        bundle,
        phase="post",
        hypotheses=hyp,
        config=cfg,
        train_seconds=total_train_s,
    )
    if not save:
        return {}
    logger.info("Ensemble03: {} synthetics, evaluating test set", len(all_synth))
    return eval_ovr_panel_test(
        session,
        panel,
        df,
        splits.test_ids,
        bundle,
        hypotheses=hyp,
        config=cfg,
        train_seconds=total_train_s,
        label_universe=labels,
    )
