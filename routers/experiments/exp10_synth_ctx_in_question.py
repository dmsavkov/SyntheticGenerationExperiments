"""Exp 10: train_2k + frozen exp02 synthetics with context merged into question."""

from __future__ import annotations

from pathlib import Path

from routers.experiments._common import (
    HYPOTHESES_BASE,
    base_config,
    load_context,
    save_primary_eval,
    train_rows_from_ids,
)
from routers.experiments.session import ExperimentSession
from routers.baselines.modernbert_probe import train_probe
from routers.synthetic.frozen_loader import DEFAULT_EXP02_SYNTHETICS, load_formatted_synthetics


def run(
    *,
    save: bool = True,
    rebuild_splits: bool = False,
    synthetic_path: Path | str | None = None,
) -> dict:
    df, bundle, ds = load_context(rebuild_splits=rebuild_splits)
    exp = "exp10_synth_ctx_in_question"
    hyp = (
        f"{HYPOTHESES_BASE} Exp10: train_2k + frozen exp02 synthetics; "
        "non-empty context pasted into question (embedder prompt unchanged)."
    )
    frozen_path = synthetic_path or DEFAULT_EXP02_SYNTHETICS
    cfg = base_config(
        experiment=exp,
        train_split="train_2k",
        eval_split="eval_1000",
        hypotheses=hyp,
        extra={
            "synthetic_source": str(frozen_path),
            "synth_format": "context_in_question",
        },
    )
    session = ExperimentSession(exp)
    train_rows = train_rows_from_ids(df, ds.train_2k_ids, bundle)
    synthetics = load_formatted_synthetics("context_in_question", frozen_path)
    session.save_synthetics(synthetics)
    cfg["n_synthetic"] = len(synthetics)
    cfg["n_train_real"] = len(train_rows)
    probe, train_s = train_probe(train_rows + synthetics)
    if not save:
        return {}
    return save_primary_eval(
        session, probe, df, bundle.eval_1000_ids, bundle, hypotheses=hyp, config=cfg, train_seconds=train_s
    )
