"""Smart09: smart03 hard-negative loop for 3 iterations with per-iter metrics."""

from __future__ import annotations

from loguru import logger

from routers.baselines.setfit_probe import train_setfit_probe
from routers.core.constants import SMART_ITERATIVE_ITERS, SMART_REF_PERCENTILE, SMART_SYNTHETIC_CAP
from routers.core.harness import save_json_artifact
from routers.experiments._common import eval_holdout, train_rows_from_ids
from routers.experiments.opentdb_ensemble._common import (
    eval_multiclass_metrics_only,
    eval_multiclass_test,
    load_ensemble_context,
)
from routers.experiments.session import ExperimentSession
from routers.experiments.smart._common import (
    HYPOTHESES_BASE,
    generate_mislabel_lowconf_synthetics,
    smart_base_config,
)
from routers.synthetic.smart_selection import score_rows


def _metric_summary(metrics: dict) -> dict:
    return {
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "micro_f1": metrics.get("micro_f1"),
    }


def run(*, save: bool = True, rebuild_splits: bool = False) -> dict:
    df, bundle, splits, labels = load_ensemble_context(rebuild_splits=rebuild_splits)
    hyp = (
        f"{HYPOTHESES_BASE} Smart09: {SMART_ITERATIVE_ITERS}-iter smart03 cycle "
        f"(mislabeled + bottom {SMART_REF_PERCENTILE:.0%} correct → synth → retrain)."
    )
    cfg = smart_base_config(
        experiment="smart09_iterative_hardneg",
        hypotheses=hyp,
        extra={
            "generation_mode": "iterative_mislabel_lowconf_hardneg",
            "iterative_iters": SMART_ITERATIVE_ITERS,
            "label_universe": labels,
        },
    )
    session = ExperimentSession("smart09_iterative_hardneg", topic=cfg["topic"])
    train_real = train_rows_from_ids(df, splits.train_100_ids, bundle)
    holdout_rows = train_rows_from_ids(df, splits.holdout_100_ids, bundle)
    all_synth: list[dict] = []
    synth_by_iter: dict[int, list[dict]] = {}
    iter_metrics: list[dict] = []

    probe, train_s = train_setfit_probe(train_real)
    eval_holdout(
        session, probe, df, splits.holdout_100_ids, bundle,
        phase="pre", hypotheses=hyp, config=cfg, train_seconds=train_s,
    )

    for it in range(1, SMART_ITERATIVE_ITERS + 1):
        logger.info("Smart09 iteration {}/{}", it, SMART_ITERATIVE_ITERS)
        scored = score_rows(probe, holdout_rows)
        batch, sel = generate_mislabel_lowconf_synthetics(
            scored, labels, stats=session.generation_stats, cap=SMART_SYNTHETIC_CAP
        )
        for row in batch:
            row["smart_iter"] = it
        synth_by_iter[it] = batch
        all_synth.extend(batch)
        save_json_artifact(session.out_dir, f"synthetic_samples_iter{it}.json", batch)

        probe, train_s = train_setfit_probe(train_real + all_synth)
        holdout_m = eval_holdout(
            session, probe, df, splits.holdout_100_ids, bundle,
            phase=f"iter{it}", hypotheses=hyp, config=cfg, train_seconds=train_s,
        )
        test_m = eval_multiclass_metrics_only(
            probe, df, splits.test_ids, bundle, label_universe=labels
        )
        iter_metrics.append(
            {
                "iter": it,
                "selection": sel,
                "n_synth_iter": len(batch),
                "n_synth_cumulative": len(all_synth),
                "holdout": _metric_summary(holdout_m),
                "test": _metric_summary(test_m),
            }
        )
        logger.info(
            "Smart09 iter{} holdout macro_f1={:.4f} test macro_f1={:.4f} (+{} synth)",
            it,
            holdout_m["macro_f1"],
            test_m["macro_f1"],
            len(batch),
        )

    session.save_synthetics(all_synth, stats=session.generation_stats)
    cfg["n_synthetic"] = len(all_synth)
    cfg["synth_by_iter"] = {str(k): len(v) for k, v in synth_by_iter.items()}
    cfg["iter_metrics"] = iter_metrics
    save_json_artifact(session.out_dir, "iter_metrics.json", iter_metrics)

    if not save:
        return {}
    final = eval_multiclass_test(
        session, probe, df, splits.test_ids, bundle,
        hypotheses=hyp, config=cfg, train_seconds=train_s,
    )
    session.patch_metrics({"iter_metrics": iter_metrics})
    return final
