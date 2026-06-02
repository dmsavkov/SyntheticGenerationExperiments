"""Exp 5: donor-pool real copies instead of synthetic."""

from __future__ import annotations

from collections import Counter

from routers.core.constants import SYNTHETIC_CAP_DEFAULT
from routers.experiments._common import (
    HYPOTHESES_BASE,
    base_config,
    eval_holdout,
    failures_from_holdout,
    load_context,
    sample_donor_rows,
    save_primary_eval,
    train_rows_from_ids,
)
from routers.experiments.session import ExperimentSession, serialize_synthetic_rows
from routers.core.harness import save_json_artifact
from routers.baselines.modernbert_probe import train_probe


def run(*, save: bool = True, rebuild_splits: bool = False, synthetic_cap: int = SYNTHETIC_CAP_DEFAULT) -> dict:
    df, bundle, ds = load_context(rebuild_splits=rebuild_splits)
    exp = "exp05_donor_ablation"
    hyp = f"{HYPOTHESES_BASE} Exp5: donor-pool real-copy ablation."
    cfg = base_config(
        experiment=exp,
        train_split="train_2k",
        eval_split="eval_1000",
        hypotheses=hyp,
        extra={"augmentation_source": "donor_pool"},
    )
    session = ExperimentSession(exp)
    train_rows = train_rows_from_ids(df, ds.train_2k_ids, bundle)
    holdout_rows = train_rows_from_ids(df, ds.holdout_500_ids, bundle)
    probe, train_s = train_probe(train_rows)
    eval_holdout(session, probe, df, ds.holdout_500_ids, bundle, phase="pre", hypotheses=hyp, config=cfg, train_seconds=train_s)
    failures = failures_from_holdout(holdout_rows, probe)[:synthetic_cap]
    by_gold = Counter(str(r["gold"]) for r in failures)
    augmented: list[dict] = []
    seed = 50
    for gold_label, count in by_gold.items():
        copies = sample_donor_rows(df, ds.donor_pool_ids, bundle, gold_label, count, seed)
        seed += 1
        for c in copies:
            c = dict(c)
            c["source"] = "real_copy"
            c["donor_id"] = c["id"]
            c["id"] = f"donor_copy_{c['id']}"
            augmented.append(c)
    save_json_artifact(session.out_dir, "augmentation_samples.json", serialize_synthetic_rows(augmented))
    session.generation_stats["n_synthetic"] = len(augmented)
    probe2, train_s2 = train_probe(train_rows + augmented)
    eval_holdout(session, probe2, df, ds.holdout_500_ids, bundle, phase="post", hypotheses=hyp, config=cfg)
    if not save:
        return {}
    cfg["n_augmented"] = len(augmented)
    return save_primary_eval(session, probe2, df, bundle.eval_1000_ids, bundle, hypotheses=hyp, config=cfg, train_seconds=train_s2)
