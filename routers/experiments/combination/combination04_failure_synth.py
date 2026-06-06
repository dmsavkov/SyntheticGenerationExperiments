"""Combination 04: failure-driven label_failure synthesis (Gemini, 8/request)."""

from __future__ import annotations

from routers.baselines.setfit_probe import train_setfit_probe
from routers.core.constants import COMBINATION_TOPIC, GOOGLE_FLASH_MODEL_DEFAULT
from routers.experiments._common import eval_holdout, failures_from_holdout, train_rows_from_ids
from routers.experiments.opentdb_ensemble._common import (
    HYPOTHESES_BASE,
    ensemble_base_config,
    eval_multiclass_test,
    load_ensemble_context,
)
from routers.experiments.session import ExperimentSession
from routers.synthetic.generator_opentdb import generate_label_failure_opentdb


def run(*, save: bool = True, rebuild_splits: bool = False) -> dict:
    df, bundle, splits, labels = load_ensemble_context(rebuild_splits=rebuild_splits)
    hyp = f"{HYPOTHESES_BASE} Combination04: holdout failure synth (label_failure), Gemini."
    cfg = ensemble_base_config(
        experiment="combination04_failure_synth",
        topic=COMBINATION_TOPIC,
        hypotheses=hyp,
        extra={"generation_mode": "label_failure"},
    )
    session = ExperimentSession("combination04_failure_synth", topic=COMBINATION_TOPIC)
    train_rows = train_rows_from_ids(df, splits.train_100_ids, bundle)
    probe, train_s = train_setfit_probe(train_rows)
    holdout_rows = train_rows_from_ids(df, splits.holdout_100_ids, bundle)
    eval_holdout(session, probe, df, splits.holdout_100_ids, bundle, phase="pre", hypotheses=hyp, config=cfg, train_seconds=train_s)
    failures = failures_from_holdout(holdout_rows, probe)
    synthetics = generate_label_failure_opentdb(
        failures,
        labels,
        cap=len(failures) * 2,
        stats=session.generation_stats,
        generation_model=GOOGLE_FLASH_MODEL_DEFAULT,
    )
    session.save_synthetics(synthetics, stats=session.generation_stats)
    cfg["n_synthetic"] = len(synthetics)
    probe2, train_s2 = train_setfit_probe(train_rows + synthetics)
    eval_holdout(session, probe2, df, splits.holdout_100_ids, bundle, phase="post", hypotheses=hyp, config=cfg)
    if not save:
        return {}
    return eval_multiclass_test(
        session, probe2, df, splits.test_ids, bundle,
        hypotheses=hyp, config=cfg, train_seconds=train_s2,
    )
