"""Smart06: parallel hard-negative gen + pair judge per mislabeled ref."""

from __future__ import annotations

from routers.baselines.setfit_probe import train_setfit_probe
from routers.core.constants import SMART_CREATIVE_TEMPERATURE, SMART_CREATIVE_TOP_P
from routers.core.harness import save_json_artifact
from routers.experiments._common import eval_holdout, failures_from_holdout, train_rows_from_ids
from routers.experiments.opentdb_ensemble._common import eval_multiclass_test, load_ensemble_context
from routers.experiments.smart._common import HYPOTHESES_BASE, smart_base_config
from routers.experiments.session import ExperimentSession
from routers.synthetic.generator_smart import generate_parallel_hard_negative_judge_pool


def run(*, save: bool = True, rebuild_splits: bool = False) -> dict:
    df, bundle, splits, labels = load_ensemble_context(rebuild_splits=rebuild_splits)
    hyp = (
        f"{HYPOTHESES_BASE} Smart06: per-failure parallel hard-negative (3×2 pairs) "
        f"+ single pair judge → 2 items (temp={SMART_CREATIVE_TEMPERATURE}, top_p={SMART_CREATIVE_TOP_P})."
    )
    cfg = smart_base_config(
        experiment="smart06_parallel_judge",
        hypotheses=hyp,
        extra={
            "generation_mode": "parallel_judge",
            "label_universe": labels,
            "creative_temperature": SMART_CREATIVE_TEMPERATURE,
            "creative_top_p": SMART_CREATIVE_TOP_P,
        },
        use_reasoning=False,
    )
    session = ExperimentSession("smart06_parallel_judge", topic=cfg["topic"])
    train_rows = train_rows_from_ids(df, splits.train_100_ids, bundle)
    probe, train_s = train_setfit_probe(train_rows)
    holdout_rows = train_rows_from_ids(df, splits.holdout_100_ids, bundle)
    eval_holdout(
        session, probe, df, splits.holdout_100_ids, bundle,
        phase="pre", hypotheses=hyp, config=cfg, train_seconds=train_s,
    )
    failures = failures_from_holdout(holdout_rows, probe)
    cfg["n_failures"] = len(failures)
    judge_log: list[dict] = []
    synthetics = generate_parallel_hard_negative_judge_pool(
        failures, labels, stats=session.generation_stats, judge_log=judge_log
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
