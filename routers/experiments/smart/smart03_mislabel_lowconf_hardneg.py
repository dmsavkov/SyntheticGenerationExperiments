"""Smart03: mislabeled + bottom-percentile correct → hard negatives."""

from __future__ import annotations

from routers.baselines.setfit_probe import train_setfit_probe
from routers.core.constants import SMART_REF_PERCENTILE
from routers.experiments._common import eval_holdout, train_rows_from_ids
from routers.experiments.opentdb_ensemble._common import eval_multiclass_test, load_ensemble_context
from routers.experiments.smart._common import (
    HYPOTHESES_BASE,
    generate_mislabel_lowconf_synthetics,
    smart_base_config,
)
from routers.experiments.session import ExperimentSession
from routers.synthetic.smart_selection import score_rows


def run(*, save: bool = True, rebuild_splits: bool = False) -> dict:
    df, bundle, splits, labels = load_ensemble_context(rebuild_splits=rebuild_splits)
    hyp = (
        f"{HYPOTHESES_BASE} Smart03: mislabeled + bottom {SMART_REF_PERCENTILE:.0%} "
        "correct holdout refs → hard negatives."
    )
    cfg = smart_base_config(
        experiment="smart03_mislabel_lowconf_hardneg",
        hypotheses=hyp,
        extra={"generation_mode": "mislabel_lowconf_hardneg", "label_universe": labels},
    )
    session = ExperimentSession("smart03_mislabel_lowconf_hardneg", topic=cfg["topic"])
    train_rows = train_rows_from_ids(df, splits.train_100_ids, bundle)
    probe, train_s = train_setfit_probe(train_rows)
    holdout_rows = train_rows_from_ids(df, splits.holdout_100_ids, bundle)
    eval_holdout(
        session, probe, df, splits.holdout_100_ids, bundle,
        phase="pre", hypotheses=hyp, config=cfg, train_seconds=train_s,
    )
    scored = score_rows(probe, holdout_rows)
    synthetics, sel = generate_mislabel_lowconf_synthetics(
        scored, labels, stats=session.generation_stats
    )
    cfg.update(sel)
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
