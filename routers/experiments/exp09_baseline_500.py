"""Exp 9: ModernBERT baseline on train_500 (exp06 real-data floor)."""

from __future__ import annotations

from routers.experiments._common import (
    HYPOTHESES_BASE,
    base_config,
    load_context,
    save_primary_eval,
    train_rows_from_ids,
)
from routers.experiments.session import ExperimentSession
from routers.baselines.modernbert_probe import train_probe


def run(*, save: bool = True, rebuild_splits: bool = False) -> dict:
    df, bundle, ds = load_context(rebuild_splits=rebuild_splits)
    exp = "exp09_baseline_500"
    hyp = f"{HYPOTHESES_BASE} Exp9: ModernBERT baseline on train_500."
    cfg = base_config(
        experiment=exp,
        train_split="train_500",
        eval_split="eval_1000",
        hypotheses=hyp,
    )
    session = ExperimentSession(exp)
    train_rows = train_rows_from_ids(df, ds.train_500_ids, bundle)
    cfg["n_train_real"] = len(train_rows)
    probe, train_s = train_probe(train_rows)
    if not save:
        return {}
    return save_primary_eval(
        session, probe, df, bundle.eval_1000_ids, bundle, hypotheses=hyp, config=cfg, train_seconds=train_s
    )
