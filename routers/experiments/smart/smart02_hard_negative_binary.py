"""Smart02: binary CS vs Technology hard-negative pairs (combination02 protocol)."""

from __future__ import annotations

from routers.baselines.setfit_probe import train_setfit_probe
from routers.core.constants import COMBINATION_BINARY_HOLDOUT_N, COMBINATION_BINARY_TRAIN_PER_CLASS
from routers.experiments._common import eval_holdout, failures_from_holdout, train_rows_from_ids
from routers.experiments.opentdb_ensemble._common import eval_multiclass_test, load_combination_binary_context
from routers.experiments.smart._common import HYPOTHESES_BASE, smart_base_config
from routers.experiments.session import ExperimentSession
from routers.synthetic.generator_smart import generate_hard_negative_pairs_smart


def run(*, save: bool = True, rebuild_splits: bool = False) -> dict:
    df, bundle, splits, labels = load_combination_binary_context(rebuild_splits=rebuild_splits)
    hyp = f"{HYPOTHESES_BASE} Smart02: binary hard-negative pairs with few-shot."
    cfg = smart_base_config(
        experiment="smart02_hard_negative_binary",
        hypotheses=hyp,
        extra={
            "generation_mode": "hard_negative_pair",
            "split_profile": "binary_cs_tech",
            "train_split": "train_20",
            "holdout_split": "holdout_80",
            "n_train": COMBINATION_BINARY_TRAIN_PER_CLASS * len(labels),
            "n_holdout": COMBINATION_BINARY_HOLDOUT_N,
            "label_universe": labels,
        },
    )
    session = ExperimentSession("smart02_hard_negative_binary", topic=cfg["topic"])
    train_rows = train_rows_from_ids(df, splits.train_20_ids, bundle)
    probe, train_s = train_setfit_probe(train_rows)
    holdout_rows = train_rows_from_ids(df, splits.holdout_80_ids, bundle)
    eval_holdout(session, probe, df, splits.holdout_80_ids, bundle, phase="pre", hypotheses=hyp, config=cfg, train_seconds=train_s)
    failures = failures_from_holdout(holdout_rows, probe)
    synthetics = generate_hard_negative_pairs_smart(failures, labels, stats=session.generation_stats)
    session.save_synthetics(synthetics, stats=session.generation_stats)
    cfg["n_synthetic"] = len(synthetics)
    probe2, train_s2 = train_setfit_probe(train_rows + synthetics)
    eval_holdout(session, probe2, df, splits.holdout_80_ids, bundle, phase="post", hypotheses=hyp, config=cfg)
    if not save:
        return {}
    return eval_multiclass_test(
        session, probe2, df, splits.test_ids, bundle,
        hypotheses=hyp, config=cfg, train_seconds=train_s2,
    )
