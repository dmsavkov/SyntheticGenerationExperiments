"""Smart08: percentile buckets → hard-negative pair per reference (2 items each)."""

from __future__ import annotations

from routers.baselines.setfit_probe import train_setfit_probe
from routers.core.constants import SMART_REF_PERCENTILE, SMART_SYNTHETIC_CAP
from routers.experiments._common import eval_holdout, train_rows_from_ids
from routers.experiments.opentdb_ensemble._common import eval_multiclass_test, load_ensemble_context
from routers.experiments.smart._common import HYPOTHESES_BASE, smart_base_config
from routers.experiments.session import ExperimentSession
from routers.synthetic.generator_smart import generate_expansion_hard_negatives
from routers.synthetic.smart_selection import partition_expansion_refs, score_rows


def run(*, save: bool = True, rebuild_splits: bool = False) -> dict:
    df, bundle, splits, labels = load_ensemble_context(rebuild_splits=rebuild_splits)
    hyp = (
        f"{HYPOTHESES_BASE} Smart08: top/bottom/mislabeled buckets → "
        f"hard-negative pair (2 items) per reference ({SMART_REF_PERCENTILE:.0%} percentiles)."
    )
    cfg = smart_base_config(
        experiment="smart08_dataset_expansion",
        hypotheses=hyp,
        extra={"generation_mode": "expansion_hard_negative", "label_universe": labels},
    )
    session = ExperimentSession("smart08_dataset_expansion", topic=cfg["topic"])
    train_rows = train_rows_from_ids(df, splits.train_100_ids, bundle)
    probe, train_s = train_setfit_probe(train_rows)
    holdout_rows = train_rows_from_ids(df, splits.holdout_100_ids, bundle)
    eval_holdout(
        session, probe, df, splits.holdout_100_ids, bundle,
        phase="pre", hypotheses=hyp, config=cfg, train_seconds=train_s,
    )
    scored = score_rows(probe, holdout_rows)
    buckets = partition_expansion_refs(scored, cap_per_bucket=SMART_SYNTHETIC_CAP // 3)
    cfg["bucket_sizes"] = {k: len(v) for k, v in buckets.items()}
    synthetics = generate_expansion_hard_negatives(buckets, labels, stats=session.generation_stats)
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
