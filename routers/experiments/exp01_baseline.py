"""Exp 1: ModernBERT baseline on train_2k."""

from __future__ import annotations

from routers.experiments._common import (
    HYPOTHESES_BASE,
    base_config,
    eval_holdout,
    load_context,
    save_primary_eval,
    train_rows_from_ids,
)
from routers.experiments.session import ExperimentSession
from routers.baselines.modernbert_probe import train_probe


def run(*, save: bool = True, rebuild_splits: bool = False) -> dict:
    df, bundle, ds = load_context(rebuild_splits=rebuild_splits)
    exp = "exp01_baseline"
    hyp = f"{HYPOTHESES_BASE} Exp1: 2k ModernBERT floor."
    cfg = base_config(
        experiment=exp,
        train_split="train_2k",
        eval_split="eval_1000",
        hypotheses=hyp,
    )
    session = ExperimentSession(exp)
    train_rows = train_rows_from_ids(df, ds.train_2k_ids, bundle)
    probe, train_s = train_probe(train_rows)
    eval_holdout(session, probe, df, ds.holdout_500_ids, bundle, phase="baseline", hypotheses=hyp, config=cfg, train_seconds=train_s)
    if not save:
        return {}
    return save_primary_eval(session, probe, df, bundle.eval_1000_ids, bundle, hypotheses=hyp, config=cfg, train_seconds=0.0)
