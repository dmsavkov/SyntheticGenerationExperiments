"""Combination 05: CS vs Technology binary cascade (20 train, 80 holdout)."""

from __future__ import annotations

from loguru import logger

from routers.baselines.setfit_probe import train_setfit_probe
from routers.core.constants import (
    COMBINATION_BINARY_HOLDOUT_N,
    COMBINATION_BINARY_TRAIN_PER_CLASS,
    COMBINATION_CASCADE_ITERS,
    COMBINATION_TOPIC,
    ENSEMBLE_MAX_ITEMS_PER_REQUEST,
    GOOGLE_FLASH_MODEL_DEFAULT,
)
from routers.core.harness import save_json_artifact
from routers.experiments._common import eval_holdout, failures_from_holdout, train_rows_from_ids
from routers.experiments.opentdb_ensemble._common import (
    HYPOTHESES_BASE,
    ensemble_base_config,
    eval_multiclass_test,
    load_combination_binary_context,
)
from routers.experiments.session import ExperimentSession
from routers.synthetic.generator_opentdb import generate_from_cascade_session
from routers.synthetic.google_chat_session import GoogleChatSession
from routers.synthetic.prompts_combination import (
    build_cascade_analysis_prompt,
    build_cascade_contrastive_prompt,
    build_cascade_generation_prompt,
)


def _eval_variant(
    session: ExperimentSession,
    train_rows: list[dict],
    df,
    splits,
    bundle,
    *,
    hyp: str,
    cfg: dict,
    variant_key: str,
) -> dict:
    probe, train_s = train_setfit_probe(train_rows)
    return eval_multiclass_test(
        session,
        probe,
        df,
        splits.test_ids,
        bundle,
        hypotheses=hyp,
        config={**cfg, "eval_variant": variant_key},
        train_seconds=train_s,
    )


def run(*, save: bool = True, rebuild_splits: bool = False) -> dict:
    df, bundle, splits, labels = load_combination_binary_context(rebuild_splits=rebuild_splits)
    hyp = (
        f"{HYPOTHESES_BASE} Combination05: binary CS vs Technology cascade "
        f"({COMBINATION_CASCADE_ITERS} iters, {COMBINATION_BINARY_TRAIN_PER_CLASS}/class train)."
    )
    cfg = ensemble_base_config(
        experiment="combination05_binary_cascade",
        topic=COMBINATION_TOPIC,
        hypotheses=hyp,
        extra={
            "generation_mode": "cascade",
            "cascade_iters": COMBINATION_CASCADE_ITERS,
            "split_profile": "binary_cs_tech",
            "label_universe": labels,
            "train_split": "train_20",
            "holdout_split": "holdout_80",
            "n_train": COMBINATION_BINARY_TRAIN_PER_CLASS * len(labels),
            "n_holdout": COMBINATION_BINARY_HOLDOUT_N,
            "n_test": len(splits.test_ids),
        },
    )
    session = ExperimentSession("combination05_binary_cascade", topic=COMBINATION_TOPIC)
    train_rows = train_rows_from_ids(df, splits.train_20_ids, bundle)
    probe, train_s = train_setfit_probe(train_rows)
    holdout_rows = train_rows_from_ids(df, splits.holdout_80_ids, bundle)
    eval_holdout(
        session,
        probe,
        df,
        splits.holdout_80_ids,
        bundle,
        phase="pre",
        hypotheses=hyp,
        config=cfg,
        train_seconds=train_s,
    )

    failures = failures_from_holdout(holdout_rows, probe)
    synth_by_iter: dict[int, list[dict]] = {}
    system = (
        "You help improve a small SetFit domain router via synthetic training data.\n"
        "Stay in this conversation across turns."
    )
    chat = GoogleChatSession(system=system, model=GOOGLE_FLASH_MODEL_DEFAULT)

    for it in range(1, COMBINATION_CASCADE_ITERS + 1):
        if it == 1:
            chat.complete_json(
                build_cascade_analysis_prompt(failures),
                parse_fn=lambda t: {"ok": True},
            )
        gen_prompt = build_cascade_generation_prompt(
            labels, n_items=ENSEMBLE_MAX_ITEMS_PER_REQUEST
        )
        batch = generate_from_cascade_session(
            chat,
            gen_prompt,
            labels,
            n_expected=ENSEMBLE_MAX_ITEMS_PER_REQUEST,
            stats=session.generation_stats,
            mode=f"cascade_iter{it}",
        )
        for row in batch:
            row["cascade_iter"] = it
        synth_by_iter[it] = batch
        save_json_artifact(session.out_dir, f"synthetic_samples_iter{it}.json", batch)

        probe_it, _ = train_setfit_probe(train_rows + batch)
        remaining = failures_from_holdout(holdout_rows, probe_it)
        if remaining and it < COMBINATION_CASCADE_ITERS:
            chat.complete_json(
                build_cascade_contrastive_prompt(remaining),
                parse_fn=lambda t: {"ok": True},
            )

    save_json_artifact(session.out_dir, "cascade_chat_log.json", chat.transcript())
    session.save_synthetics(
        [r for i in sorted(synth_by_iter) for r in synth_by_iter[i]],
        stats=session.generation_stats,
    )

    cumulative: dict[int, list[dict]] = {}
    for it in sorted(synth_by_iter):
        cumulative[it] = cumulative.get(it - 1, []) + synth_by_iter[it]

    eval_variants: dict[str, dict] = {}
    if save:
        all_synth = cumulative[max(synth_by_iter)]
        eval_variants["iter123"] = _eval_variant(
            session,
            train_rows + all_synth,
            df,
            splits,
            bundle,
            hyp=hyp,
            cfg=cfg,
            variant_key="iter123",
        )
        if len(synth_by_iter) >= 2:
            only_23 = synth_by_iter.get(2, []) + synth_by_iter.get(3, [])
            sub_session = ExperimentSession(
                "combination05_binary_cascade_iter23",
                topic=COMBINATION_TOPIC,
            )
            eval_variants["iter23"] = _eval_variant(
                sub_session,
                train_rows + only_23,
                df,
                splits,
                bundle,
                hyp=hyp,
                cfg=cfg,
                variant_key="iter23",
            )
        cfg["eval_variants"] = {
            k: {
                "accuracy": v.get("accuracy"),
                "macro_f1": v.get("macro_f1"),
                "results_path": v.get("results_path"),
            }
            for k, v in eval_variants.items()
        }
        session.patch_metrics({"eval_variants": cfg["eval_variants"]})

    logger.info("Combination05 complete; variants={}", list(eval_variants.keys()))
    return eval_variants.get("iter123", {})
