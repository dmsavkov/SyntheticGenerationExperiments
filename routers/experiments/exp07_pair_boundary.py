"""Exp 7: confusion-pair boundary synthesis."""

from __future__ import annotations

from routers.core.constants import SYNTHETIC_CAP_DEFAULT
from routers.core.data import label_vocab
from routers.experiments._common import (
    HYPOTHESES_BASE,
    base_config,
    eval_holdout,
    failures_from_holdout,
    load_context,
    save_primary_eval,
    train_rows_from_ids,
)
from routers.experiments.session import ExperimentSession
from routers.baselines.modernbert_probe import train_probe
from routers.synthetic.generator import generate_confusion_pair_boundary


def run(*, save: bool = True, rebuild_splits: bool = False, synthetic_cap: int = SYNTHETIC_CAP_DEFAULT) -> dict:
    df, bundle, ds = load_context(rebuild_splits=rebuild_splits)
    labels = label_vocab(df, "Domain")
    exp = "exp07_pair_boundary"
    hyp = f"{HYPOTHESES_BASE} Exp7: confusion_pair_boundary synthesis."
    cfg = base_config(
        experiment=exp,
        train_split="train_2k",
        eval_split="eval_1000",
        hypotheses=hyp,
        extra={"generation_mode": "confusion_pair_boundary"},
    )
    session = ExperimentSession(exp)
    train_rows = train_rows_from_ids(df, ds.train_2k_ids, bundle)
    holdout_rows = train_rows_from_ids(df, ds.holdout_500_ids, bundle)
    probe, train_s = train_probe(train_rows)
    eval_holdout(session, probe, df, ds.holdout_500_ids, bundle, phase="pre", hypotheses=hyp, config=cfg, train_seconds=train_s)
    failures = failures_from_holdout(holdout_rows, probe)
    synthetics = generate_confusion_pair_boundary(
        failures, train_rows, labels, cap=synthetic_cap, stats=session.generation_stats
    )
    session.save_synthetics(synthetics, stats=session.generation_stats)
    probe2, train_s2 = train_probe(train_rows + synthetics)
    eval_holdout(session, probe2, df, ds.holdout_500_ids, bundle, phase="post", hypotheses=hyp, config=cfg)
    if not save:
        return {}
    cfg["n_synthetic"] = len(synthetics)
    return save_primary_eval(session, probe2, df, bundle.eval_1000_ids, bundle, hypotheses=hyp, config=cfg, train_seconds=train_s2)
