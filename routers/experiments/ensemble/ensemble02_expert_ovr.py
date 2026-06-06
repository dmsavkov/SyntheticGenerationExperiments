"""Ensemble 02: general + up to 2 OVR experts with mutation synth (2 iters)."""

from __future__ import annotations

from loguru import logger

from routers.baselines.setfit_ovr import SetFitOvrProbe, train_ovr_probe
from routers.baselines.setfit_probe import train_setfit_probe
from routers.core.constants import (
    ENSEMBLE_EXPERT_MAX,
    ENSEMBLE_EXPERT_MIN_SUPPORT,
    ENSEMBLE_OVR_MAX_ITERS,
    ENSEMBLE_TOPIC,
    GOOGLE_FLASH_MODEL_DEFAULT,
)
from routers.core.harness import save_json_artifact
from routers.ensemble.routing import select_weak_expert_classes
from routers.experiments._common import eval_holdout, train_rows_from_ids
from routers.experiments.opentdb_ensemble._common import (
    HYPOTHESES_BASE,
    ensemble_base_config,
    eval_expert_general_test,
    eval_multiclass_metrics_only,
    holdout_records_from_probe,
    load_ensemble_context,
    score_rows_multiclass,
)
from routers.experiments.session import ExperimentSession
from routers.synthetic.generator_opentdb import generate_ovr_mutation


def run(*, save: bool = True, rebuild_splits: bool = False) -> dict:
    df, bundle, splits, labels = load_ensemble_context(rebuild_splits=rebuild_splits)
    hyp = (
        f"{HYPOTHESES_BASE} Ensemble02: general + <=2 OVR experts "
        f"(min support {ENSEMBLE_EXPERT_MIN_SUPPORT}), 2 mutation iters, Gemini."
    )
    cfg = ensemble_base_config(
        experiment="ensemble02_expert_ovr",
        topic=ENSEMBLE_TOPIC,
        hypotheses=hyp,
        extra={
            "training_mode": "general_plus_ovr_experts",
            "expert_max": ENSEMBLE_EXPERT_MAX,
            "expert_min_support": ENSEMBLE_EXPERT_MIN_SUPPORT,
            "ovr_iters": ENSEMBLE_OVR_MAX_ITERS,
            "label_universe": labels,
        },
    )
    session = ExperimentSession("ensemble02_expert_ovr", topic=ENSEMBLE_TOPIC)
    train_rows = train_rows_from_ids(df, splits.train_100_ids, bundle)
    general, train_s = train_setfit_probe(train_rows)
    holdout_scored = holdout_records_from_probe(general, df, splits.holdout_100_ids, bundle)
    expert_classes = select_weak_expert_classes(
        holdout_scored,
        labels,
        max_experts=ENSEMBLE_EXPERT_MAX,
        min_support=ENSEMBLE_EXPERT_MIN_SUPPORT,
    )
    cfg["expert_classes"] = expert_classes
    save_json_artifact(
        session.out_dir,
        "expert_selection.json",
        {"expert_classes": expert_classes, "n_holdout": len(holdout_scored)},
    )
    logger.info("Selected expert classes: {}", expert_classes)

    pre_synth = eval_multiclass_metrics_only(
        general,
        df,
        splits.test_ids,
        bundle,
        label_universe=labels,
    )
    cfg["pre_synth_test_metrics"] = pre_synth
    logger.info(
        "Pre-synth test metrics: accuracy={:.4f} macro_f1={:.4f}",
        pre_synth.get("accuracy", 0.0),
        pre_synth.get("macro_f1", 0.0),
    )

    expert_probes: dict[str, SetFitOvrProbe] = {}
    expert_train_stats: dict[str, list] = {c: [] for c in expert_classes}
    all_synth: list[dict] = []

    for expert_label in expert_classes:
        expert_train = list(train_rows)
        ovr_probe: SetFitOvrProbe | None = None
        for it in range(1, ENSEMBLE_OVR_MAX_ITERS + 1):
            holdout_rows = train_rows_from_ids(df, splits.holdout_100_ids, bundle)
            if ovr_probe is not None:
                fps = [
                    dict(row, pred=expert_label)
                    for row in holdout_rows
                    if ovr_probe.is_positive(row) and str(row["gold"]) != expert_label
                ]
            else:
                scored = score_rows_multiclass(holdout_rows, general)
                fps = [
                    r
                    for r in scored
                    if str(r.get("pred")) == expert_label and str(r.get("gold")) != expert_label
                ]
            if not fps:
                logger.warning("No FPs for expert {} iter {}", expert_label, it)
                break
            batch = generate_ovr_mutation(
                fps[:8],
                labels,
                expert_label,
                cap=8,
                stats=session.generation_stats,
                generation_model=GOOGLE_FLASH_MODEL_DEFAULT,
            )
            for row in batch:
                row["expert_class"] = expert_label
                row["ovr_iter"] = it
            all_synth.extend(batch)
            expert_train = expert_train + batch
            ovr, ts, bal = train_ovr_probe(expert_train, expert_label)
            expert_train_stats[expert_label].append({"iter": it, "balance": bal, "train_seconds": ts})
            expert_probes[expert_label] = ovr
            ovr_probe = ovr

    cfg["expert_train_stats"] = expert_train_stats
    session.save_synthetics(all_synth, stats=session.generation_stats)
    cfg["n_synthetic"] = len(all_synth)

    eval_holdout(
        session,
        general,
        df,
        splits.holdout_100_ids,
        bundle,
        phase="general_only",
        hypotheses=hyp,
        config=cfg,
        train_seconds=train_s,
    )
    if not save:
        return {}
    metrics = eval_expert_general_test(
        session,
        general,
        expert_probes,
        df,
        splits.test_ids,
        bundle,
        hypotheses=hyp,
        config=cfg,
        train_seconds=train_s,
        label_universe=labels,
    )
    session.patch_metrics({"pre_synth_test_metrics": pre_synth})
    return metrics
