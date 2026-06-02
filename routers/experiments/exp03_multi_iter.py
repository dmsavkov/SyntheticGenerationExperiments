"""Exp 3: three synthetic iterations on 2k."""

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
from routers.synthetic.generator import generate_label_failure


def run(
    *,
    save: bool = True,
    rebuild_splits: bool = False,
    n_iterations: int = 3,
    synthetic_cap: int = SYNTHETIC_CAP_DEFAULT,
) -> dict:
    df, bundle, ds = load_context(rebuild_splits=rebuild_splits)
    labels = label_vocab(df, "Domain")
    exp = "exp03_multi_iter"
    hyp = f"{HYPOTHESES_BASE} Exp3: {n_iterations} synthetic iterations."
    cfg = base_config(
        experiment=exp,
        train_split="train_2k",
        eval_split="eval_1000",
        hypotheses=hyp,
        extra={"generation_mode": "label_failure", "n_iterations": n_iterations},
    )
    session = ExperimentSession(exp)
    train_rows = train_rows_from_ids(df, ds.train_2k_ids, bundle)
    holdout_rows = train_rows_from_ids(df, ds.holdout_500_ids, bundle)
    probe, train_s = train_probe(train_rows)
    all_synth: list[dict] = []
    for it in range(1, n_iterations + 1):
        eval_holdout(
            session, probe, df, ds.holdout_500_ids, bundle,
            phase=f"pre_iter{it:02d}", hypotheses=hyp, config={**cfg, "iteration": it},
            train_seconds=train_s if it == 1 else 0.0,
        )
        failures = failures_from_holdout(holdout_rows, probe)
        batch = generate_label_failure(failures, labels, cap=synthetic_cap, stats=session.generation_stats)
        for row in batch:
            row["iteration"] = it
        all_synth.extend(batch)
        train_rows = train_rows + batch
        probe, train_s = train_probe(train_rows)
    session.save_synthetics(all_synth, stats=session.generation_stats)
    eval_holdout(session, probe, df, ds.holdout_500_ids, bundle, phase="post", hypotheses=hyp, config=cfg)
    if not save:
        return {}
    cfg["n_synthetic"] = len(all_synth)
    return save_primary_eval(session, probe, df, bundle.eval_1000_ids, bundle, hypotheses=hyp, config=cfg, train_seconds=0.0)
