"""Smart04: 5-fold CV on train_100 for failure refs; no holdout; train full 100 + synth."""

from __future__ import annotations

from routers.baselines.setfit_probe import train_setfit_probe
from routers.core.harness import save_json_artifact
from routers.experiments._common import train_rows_from_ids
from routers.experiments.opentdb_ensemble._common import eval_multiclass_test, load_ensemble_context
from routers.experiments.smart._common import HYPOTHESES_BASE, smart_base_config
from routers.experiments.session import ExperimentSession
from routers.synthetic.generator_smart import generate_hard_negative_pairs_smart
from routers.synthetic.smart_selection import collect_cv_mislabels


def run(*, save: bool = True, rebuild_splits: bool = False) -> dict:
    df, bundle, splits, labels = load_ensemble_context(rebuild_splits=rebuild_splits)
    hyp = f"{HYPOTHESES_BASE} Smart04: 5-fold CV failure mining → hard negatives (no holdout)."
    cfg = smart_base_config(
        experiment="smart04_cv_hard_negative",
        hypotheses=hyp,
        extra={"generation_mode": "cv_hard_negative", "use_holdout": False, "label_universe": labels},
    )
    session = ExperimentSession("smart04_cv_hard_negative", topic=cfg["topic"])
    train_rows = train_rows_from_ids(df, splits.train_100_ids, bundle)
    failures, cv_stats = collect_cv_mislabels(df, splits.train_100_ids, bundle)
    cfg["cv_stats"] = cv_stats
    save_json_artifact(session.out_dir, "cv_stats.json", cv_stats)
    synthetics = generate_hard_negative_pairs_smart(failures, labels, stats=session.generation_stats)
    session.save_synthetics(synthetics, stats=session.generation_stats)
    cfg["n_synthetic"] = len(synthetics)
    probe, train_s = train_setfit_probe(train_rows + synthetics)
    if not save:
        return {}
    return eval_multiclass_test(
        session, probe, df, splits.test_ids, bundle,
        hypotheses=hyp, config=cfg, train_seconds=train_s,
    )
