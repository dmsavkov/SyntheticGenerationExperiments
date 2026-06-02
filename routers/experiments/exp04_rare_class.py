"""Exp 4: within train_2k, downsample one Domain to 10%, synth-restore the other 90%."""

from __future__ import annotations

import os
import random
from collections import Counter

from routers.core.data import label_vocab
from routers.experiments._common import (
    HYPOTHESES_BASE,
    base_config,
    load_context,
    save_primary_eval,
    train_rows_from_ids,
)
from routers.experiments.session import ExperimentSession
from routers.baselines.modernbert_probe import train_probe
from routers.synthetic.generator import generate_label_failure


def _expand_exemplars(exemplars: list[dict], n: int) -> list[dict]:
    """Cycle kept references so generation can fill n_restore slots."""
    if not exemplars or n <= 0:
        return []
    return [dict(exemplars[i % len(exemplars)]) for i in range(n)]


def run(
    *,
    save: bool = True,
    rebuild_splits: bool = False,
    target_domain: str | None = None,
    rare_domain: str | None = None,
    downsample_frac: float = 0.1,
) -> dict:
    df, bundle, ds = load_context(rebuild_splits=rebuild_splits)
    labels = label_vocab(df, "Domain")
    train_rows = train_rows_from_ids(df, ds.train_2k_ids, bundle)

    target = target_domain or rare_domain or os.environ.get("RARE_DOMAIN") or os.environ.get("TARGET_DOMAIN")
    if not target:
        target = Counter(str(r["gold"]) for r in train_rows).most_common(1)[0][0]

    exp = "exp04_rare_class"
    hyp = (
        f"{HYPOTHESES_BASE} Exp4: within train_2k, downsample {target!r} to "
        f"{downsample_frac:.0%}, synth-restore removed rows from kept references."
    )
    cfg = base_config(
        experiment=exp,
        train_split="train_2k",
        eval_split="eval_1000",
        hypotheses=hyp,
        extra={
            "target_domain": target,
            "downsample_frac": downsample_frac,
            "train_base": "train_2k",
        },
    )
    session = ExperimentSession(exp)

    target_rows = [r for r in train_rows if str(r["gold"]) == target]
    other_rows = [r for r in train_rows if str(r["gold"]) != target]
    n_keep = max(1, int(len(target_rows) * downsample_frac))
    rng = random.Random(42)
    kept = rng.sample(target_rows, min(n_keep, len(target_rows)))
    n_restore = len(target_rows) - len(kept)

    reference_pool = _expand_exemplars(kept, n_restore)
    synthetics = generate_label_failure(
        reference_pool, labels, cap=n_restore, stats=session.generation_stats
    )
    synthetics = synthetics[:n_restore]
    for row in synthetics:
        row["seed_domain"] = target
    session.save_synthetics(synthetics, stats=session.generation_stats)

    train_final = other_rows + kept + synthetics
    cfg["n_train_real"] = len(other_rows) + len(kept)
    cfg["n_target_in_2k"] = len(target_rows)
    cfg["n_target_kept"] = len(kept)
    cfg["n_synthetic"] = len(synthetics)

    probe, train_s = train_probe(train_final)
    if not save:
        return {}
    return save_primary_eval(
        session, probe, df, bundle.eval_1000_ids, bundle, hypotheses=hyp, config=cfg, train_seconds=train_s
    )
