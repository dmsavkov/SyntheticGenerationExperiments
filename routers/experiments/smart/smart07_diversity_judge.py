"""Smart07: diversity paths A/B/C + judge per reference."""

from __future__ import annotations

from routers.baselines.setfit_probe import train_setfit_probe
from routers.core.constants import SMART_CREATIVE_TEMPERATURE, SMART_CREATIVE_TOP_P, SMART_REF_PERCENTILE
from routers.core.harness import save_json_artifact
from routers.experiments._common import eval_holdout, train_rows_from_ids
from routers.experiments.opentdb_ensemble._common import eval_multiclass_test, load_ensemble_context
from routers.experiments.smart._common import HYPOTHESES_BASE, smart_base_config
from routers.experiments.session import ExperimentSession
from routers.synthetic.generator_smart import generate_diversity_paths_judge_pool
from routers.synthetic.smart_selection import bottom_correct_rows, mislabeled_rows, score_rows


def run(*, save: bool = True, rebuild_splits: bool = False) -> dict:
    df, bundle, splits, labels = load_ensemble_context(rebuild_splits=rebuild_splits)
    hyp = (
        f"{HYPOTHESES_BASE} Smart07: per-ref diversity paths + judge "
        f"(mislabeled + bottom {SMART_REF_PERCENTILE:.0%} correct pool)."
    )
    cfg = smart_base_config(
        experiment="smart07_diversity_judge",
        hypotheses=hyp,
        extra={"generation_mode": "diversity_judge", "label_universe": labels},
        use_reasoning=False,
    )
    cfg["creative_temperature"] = SMART_CREATIVE_TEMPERATURE
    cfg["creative_top_p"] = SMART_CREATIVE_TOP_P
    session = ExperimentSession("smart07_diversity_judge", topic=cfg["topic"])
    train_rows = train_rows_from_ids(df, splits.train_100_ids, bundle)
    probe, train_s = train_setfit_probe(train_rows)
    holdout_rows = train_rows_from_ids(df, splits.holdout_100_ids, bundle)
    eval_holdout(
        session, probe, df, splits.holdout_100_ids, bundle,
        phase="pre", hypotheses=hyp, config=cfg, train_seconds=train_s,
    )
    scored = score_rows(probe, holdout_rows)
    pool = mislabeled_rows(scored) + bottom_correct_rows(scored)
    cfg["n_pool"] = len(pool)
    judge_log: list[dict] = []
    synthetics = generate_diversity_paths_judge_pool(
        pool, labels, stats=session.generation_stats, judge_log=judge_log
    )
    save_json_artifact(session.out_dir, "judge_log.json", judge_log)
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
