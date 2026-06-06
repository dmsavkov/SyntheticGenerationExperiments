"""Smart05: SKIP gate for ambiguous gold, else hard-negative pair."""

from __future__ import annotations

from routers.baselines.setfit_probe import train_setfit_probe
from routers.core.harness import save_json_artifact
from routers.experiments._common import eval_holdout, failures_from_holdout, train_rows_from_ids
from routers.experiments.opentdb_ensemble._common import eval_multiclass_test, load_ensemble_context
from routers.experiments.smart._common import HYPOTHESES_BASE, smart_base_config
from routers.experiments.session import ExperimentSession
from routers.synthetic.generator_smart import generate_hard_negative_skip


def run(*, save: bool = True, rebuild_splits: bool = False) -> dict:
    df, bundle, splits, labels = load_ensemble_context(rebuild_splits=rebuild_splits)
    hyp = (
        f"{HYPOTHESES_BASE} Smart05: SKIP ambiguous gold else hard-negative pair "
        "(H0: noisy labels hurt performance)."
    )
    cfg = smart_base_config(
        experiment="smart05_hard_negative_skip",
        hypotheses=hyp,
        extra={"generation_mode": "hard_negative_skip", "label_universe": labels},
    )
    session = ExperimentSession("smart05_hard_negative_skip", topic=cfg["topic"])
    train_rows = train_rows_from_ids(df, splits.train_100_ids, bundle)
    probe, train_s = train_setfit_probe(train_rows)
    holdout_rows = train_rows_from_ids(df, splits.holdout_100_ids, bundle)
    eval_holdout(session, probe, df, splits.holdout_100_ids, bundle, phase="pre", hypotheses=hyp, config=cfg, train_seconds=train_s)
    failures = failures_from_holdout(holdout_rows, probe)
    synthetics, skip_log = generate_hard_negative_skip(failures, labels, stats=session.generation_stats)
    save_json_artifact(session.out_dir, "skip_log.json", skip_log)
    cfg["n_skip"] = sum(1 for x in skip_log if x.get("action") == "SKIP")
    cfg["n_generated"] = len(synthetics)
    session.save_synthetics(synthetics, stats=session.generation_stats)
    probe2, train_s2 = train_setfit_probe(train_rows + synthetics)
    eval_holdout(session, probe2, df, splits.holdout_100_ids, bundle, phase="post", hypotheses=hyp, config=cfg)
    if not save:
        return {}
    return eval_multiclass_test(
        session, probe2, df, splits.test_ids, bundle,
        hypotheses=hyp, config=cfg, train_seconds=train_s2,
    )
