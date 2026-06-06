"""Smart01: 3-step contrastive pipeline (draft → critique → refine)."""

from __future__ import annotations

from routers.core.harness import save_json_artifact
from routers.baselines.setfit_probe import train_setfit_probe
from routers.experiments._common import eval_holdout, failures_from_holdout, train_rows_from_ids
from routers.experiments.opentdb_ensemble._common import eval_multiclass_test, load_ensemble_context
from routers.experiments.smart._common import HYPOTHESES_BASE, smart_base_config
from routers.experiments.session import ExperimentSession
from routers.synthetic.generator_smart import generate_contrastive_pipeline


def run(*, save: bool = True, rebuild_splits: bool = False) -> dict:
    df, bundle, splits, labels = load_ensemble_context(rebuild_splits=rebuild_splits)
    hyp = f"{HYPOTHESES_BASE} Smart01: contrastive 3-step hard-negative pipeline."
    cfg = smart_base_config(
        experiment="smart01_contrastive_pipeline",
        hypotheses=hyp,
        extra={"generation_mode": "contrastive_pipeline", "label_universe": labels},
    )
    session = ExperimentSession("smart01_contrastive_pipeline", topic=cfg["topic"])
    train_rows = train_rows_from_ids(df, splits.train_100_ids, bundle)
    probe, train_s = train_setfit_probe(train_rows)
    holdout_rows = train_rows_from_ids(df, splits.holdout_100_ids, bundle)
    eval_holdout(session, probe, df, splits.holdout_100_ids, bundle, phase="pre", hypotheses=hyp, config=cfg, train_seconds=train_s)
    failures = failures_from_holdout(holdout_rows, probe)
    cfg["n_failures"] = len(failures)
    synthetics, traces = generate_contrastive_pipeline(
        failures, labels, stats=session.generation_stats
    )
    save_json_artifact(session.out_dir, "contrastive_traces.json", traces)
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
