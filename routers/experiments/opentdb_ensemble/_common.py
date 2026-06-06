"""Shared OpenTDB ensemble/combination experiment runner."""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from routers.baselines.setfit_probe import SetFitProbe, probe_config_extra, train_setfit_probe
from routers.baselines.setfit_ovr import SetFitOvrProbe
from routers.core.constants import (
    COMBINATION_BINARY_LABELS,
    COMBINATION_GOOGLE_THINKING_LEVEL,
    COMBINATION_TOPIC,
    ENSEMBLE_EXPERT_SCREAM_THRESHOLD,
    ENSEMBLE_MAX_ITEMS_PER_REQUEST,
    ENSEMBLE_TOPIC,
    GENERATION_TEMPERATURE,
    GOOGLE_FLASH_MODEL_DEFAULT,
    OPENTDB_ENSEMBLE_HOLDOUT_N,
    OPENTDB_ENSEMBLE_TRAIN_N,
    TARGET_COL,
    TEXT_FORMAT,
)
from routers.core.data import dataset_metadata, label_vocab, load_arena
from routers.core.harness import run_experiment, save_json_artifact, _build_metrics, _save_confusion_plot
from routers.core.metrics_extras import compute_subset_metrics, metrics_with_confused_breakdown
from routers.core.splits import (
    SplitBundle,
    get_opentdb_combination_binary_splits,
    get_opentdb_ensemble_splits,
    get_splits,
    set_global_seed,
)
from routers.ensemble.routing import (
    resolve_expert_general_prediction,
    resolve_ovr_panel_prediction,
)
from routers.experiments._common import eval_holdout, train_rows_from_ids
from routers.experiments.session import ExperimentSession
HYPOTHESES_BASE = "OpenTDB ensemble/combination SetFit experiments on RouterArena."


def load_ensemble_context(*, rebuild_splits: bool = False):
    set_global_seed(42)
    df = load_arena()
    bundle = get_splits(df, rebuild=rebuild_splits)
    splits = get_opentdb_ensemble_splits(df, bundle, rebuild=rebuild_splits)
    labels = splits.label_universe or label_vocab(df, TARGET_COL)
    return df, bundle, splits, labels


def ensemble_base_config(
    *,
    experiment: str,
    topic: str,
    hypotheses: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        **dataset_metadata(),
        "topic": topic,
        "target_col": TARGET_COL,
        "train_split": "train_100",
        "holdout_split": "holdout_100",
        "eval_split": "test_ids",
        "text_format": TEXT_FORMAT,
        "generation_temperature": GENERATION_TEMPERATURE,
        "max_items_per_llm_request": ENSEMBLE_MAX_ITEMS_PER_REQUEST,
        "llm_backend": "google",
        "generation_model": GOOGLE_FLASH_MODEL_DEFAULT,
        "hypotheses": hypotheses,
        "experiment": experiment,
        **probe_config_extra(),
    }
    if extra:
        cfg.update(extra)
    if topic == COMBINATION_TOPIC:
        cfg["google_thinking_level"] = COMBINATION_GOOGLE_THINKING_LEVEL
        cfg["expert_scream_threshold"] = ENSEMBLE_EXPERT_SCREAM_THRESHOLD
    return cfg


def load_combination_binary_context(*, rebuild_splits: bool = False):
    set_global_seed(42)
    df = load_arena()
    bundle = get_splits(df, rebuild=rebuild_splits)
    splits = get_opentdb_combination_binary_splits(df, bundle, rebuild=rebuild_splits)
    return df, bundle, splits, list(COMBINATION_BINARY_LABELS)


def score_rows_multiclass(rows: list[dict], probe: SetFitProbe) -> list[dict]:
    out: list[dict] = []
    for row in rows:
        ext = probe.predict_extended(row)
        enriched = dict(row)
        enriched["pred"] = ext["pred"]
        enriched["pred_probs"] = ext["pred_probs"]
        enriched["correct"] = ext["pred"] == str(row["gold"])
        out.append(enriched)
    return out


def run_baseline_multiclass(
    *,
    experiment: str,
    topic: str,
    hypotheses: str,
    save: bool = True,
    rebuild_splits: bool = False,
) -> dict:
    df, bundle, splits, labels = load_ensemble_context(rebuild_splits=rebuild_splits)
    cfg = ensemble_base_config(
        experiment=experiment,
        topic=topic,
        hypotheses=hypotheses,
        extra={
            "training_mode": "real_only",
            "label_universe": labels,
            "n_train": OPENTDB_ENSEMBLE_TRAIN_N,
            "n_holdout": OPENTDB_ENSEMBLE_HOLDOUT_N,
            "n_test": len(splits.test_ids),
        },
    )
    session = ExperimentSession(experiment, topic=topic)
    train_rows = train_rows_from_ids(df, splits.train_100_ids, bundle)
    probe, train_s = train_setfit_probe(train_rows)
    eval_holdout(
        session,
        probe,
        df,
        splits.holdout_100_ids,
        bundle,
        phase="baseline",
        hypotheses=hypotheses,
        config=cfg,
        train_seconds=train_s,
    )
    if not save:
        return {}
    return eval_multiclass_test(
        session,
        probe,
        df,
        splits.test_ids,
        bundle,
        hypotheses=hypotheses,
        config=cfg,
        train_seconds=train_s,
    )


def holdout_records_from_probe(
    probe: SetFitProbe,
    df: Any,
    holdout_ids: list[Any],
    bundle: SplitBundle,
) -> list[dict]:
    rows = train_rows_from_ids(df, holdout_ids, bundle)
    return score_rows_multiclass(rows, probe)


def eval_multiclass_test(
    session: ExperimentSession,
    probe: SetFitProbe,
    df: Any,
    test_ids: list[Any],
    bundle: SplitBundle,
    *,
    hypotheses: str,
    config: dict[str, Any],
    train_seconds: float = 0.0,
    persist_artifacts: bool = True,
) -> dict[str, Any]:
    if persist_artifacts:
        session.save_config(config)
    supplemental = session.supplemental_metrics() if persist_artifacts else {}
    rows = train_rows_from_ids(df, test_ids, bundle)
    return run_experiment(
        session.topic,
        session.experiment,
        rows,
        hypotheses=hypotheses,
        config=config,
        predict_row=probe.predict,
        save=persist_artifacts,
        out_dir=session.out_dir if persist_artifacts else None,
        supplemental_metrics=supplemental if persist_artifacts else None,
        train_seconds=train_seconds,
    )


def _routed_test_metrics(
    session: ExperimentSession,
    rows: list[dict],
    *,
    hypotheses: str,
    config: dict[str, Any],
    train_seconds: float,
    label_universe: list[str],
    route_fn,
) -> tuple[dict[str, Any], list[dict]]:
    golds: list[str] = []
    preds: list[str] = []
    confused_flags: list[bool] = []
    records: list[dict] = []

    for row in rows:
        gold = str(row["gold"])
        routed = route_fn(row)
        golds.append(gold)
        preds.append(routed["pred"])
        confused_flags.append(routed["confused"])
        rec = {
            "id": row["id"],
            "gold": gold,
            "pred": routed["pred"],
            "correct": gold == routed["pred"],
            "confused": routed["confused"],
            "prompt": str(row.get("prompt", ""))[:2000],
        }
        rec.update({k: v for k, v in routed.items() if k not in rec})
        records.append(rec)

    metrics = _build_metrics(
        session.topic,
        session.experiment,
        golds,
        preds,
        hypotheses,
        config,
        train_seconds,
        0.0,
    )
    metrics.update(metrics_with_confused_breakdown(golds, preds, confused_flags, labels=label_universe))
    metrics.update(session.supplemental_metrics())
    return metrics, records


def eval_expert_general_test(
    session: ExperimentSession,
    general: SetFitProbe,
    expert_probes: dict[str, SetFitOvrProbe],
    df: Any,
    test_ids: list[Any],
    bundle: SplitBundle,
    *,
    hypotheses: str,
    config: dict[str, Any],
    train_seconds: float = 0.0,
    label_universe: list[str],
    persist_artifacts: bool = True,
) -> dict[str, Any]:
    session.save_config(config)
    rows = train_rows_from_ids(df, test_ids, bundle)
    metrics, records = _routed_test_metrics(
        session,
        rows,
        hypotheses=hypotheses,
        config=config,
        train_seconds=train_seconds,
        label_universe=label_universe,
        route_fn=lambda row: resolve_expert_general_prediction(general, expert_probes, row),
    )
    if not persist_artifacts:
        return metrics
    out_dir = session.out_dir
    metrics["results_path"] = str(out_dir)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    correct = [r for r in records if r["correct"]]
    incorrect = [r for r in records if not r["correct"]]
    with open(out_dir / "predictions_correct.json", "w", encoding="utf-8") as f:
        json.dump(correct, f, indent=2, ensure_ascii=False)
    with open(out_dir / "predictions_incorrect.json", "w", encoding="utf-8") as f:
        json.dump(incorrect, f, indent=2, ensure_ascii=False)
    save_json_artifact(out_dir, "predictions_confused.json", [r for r in records if r["confused"]])
    labels = sorted(set(r["gold"] for r in records) | set(r["pred"] for r in records))
    _save_confusion_plot(out_dir / "confusion.png", [r["gold"] for r in records], [r["pred"] for r in records], labels)
    logger.info(
        "{} test accuracy={:.4f} macro_f1={:.4f} confused={}",
        session.experiment,
        metrics["accuracy"],
        metrics["macro_f1"],
        metrics.get("n_confused", 0),
    )
    return metrics


def eval_multiclass_metrics_only(
    probe: SetFitProbe,
    df: Any,
    test_ids: list[Any],
    bundle: SplitBundle,
    *,
    label_universe: list[str],
) -> dict[str, Any]:
    rows = train_rows_from_ids(df, test_ids, bundle)
    golds = [str(r["gold"]) for r in rows]
    preds = [probe.predict(r) for r in rows]
    return compute_subset_metrics(golds, preds, labels=label_universe)


def eval_ovr_panel_test(
    session: ExperimentSession,
    probes: dict[str, SetFitOvrProbe],
    df: Any,
    test_ids: list[Any],
    bundle: SplitBundle,
    *,
    hypotheses: str,
    config: dict[str, Any],
    train_seconds: float = 0.0,
    label_universe: list[str],
) -> dict[str, Any]:
    session.save_config(config)
    rows = train_rows_from_ids(df, test_ids, bundle)
    metrics, records = _routed_test_metrics(
        session,
        rows,
        hypotheses=hypotheses,
        config=config,
        train_seconds=train_seconds,
        label_universe=label_universe,
        route_fn=lambda row: resolve_ovr_panel_prediction(probes, row),
    )

    out_dir = session.out_dir
    metrics["results_path"] = str(out_dir)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    correct = [r for r in records if r["correct"]]
    incorrect = [r for r in records if not r["correct"]]
    with open(out_dir / "predictions_correct.json", "w", encoding="utf-8") as f:
        json.dump(correct, f, indent=2, ensure_ascii=False)
    with open(out_dir / "predictions_incorrect.json", "w", encoding="utf-8") as f:
        json.dump(incorrect, f, indent=2, ensure_ascii=False)
    save_json_artifact(out_dir, "predictions_confused.json", [r for r in records if r["confused"]])
    labels = sorted(set(r["gold"] for r in records) | set(r["pred"] for r in records))
    _save_confusion_plot(out_dir / "confusion.png", [r["gold"] for r in records], [r["pred"] for r in records], labels)
    logger.info(
        "{} test accuracy={:.4f} macro_f1={:.4f} confused={}",
        session.experiment,
        metrics["accuracy"],
        metrics["macro_f1"],
        metrics.get("n_confused", 0),
    )
    return metrics


class OvrPanelRouter:
    """Adapter so eval_holdout can score full OVR panel routing."""

    def __init__(self, probes: dict[str, SetFitOvrProbe]) -> None:
        self.probes = probes

    def predict(self, row: dict) -> str:
        return resolve_ovr_panel_prediction(self.probes, row)["pred"]


def eval_ovr_panel_holdout(
    session: ExperimentSession,
    probes: dict[str, SetFitOvrProbe],
    df: Any,
    holdout_ids: list[Any],
    bundle: SplitBundle,
    *,
    phase: str,
    hypotheses: str,
    config: dict[str, Any],
    train_seconds: float = 0.0,
) -> None:
    eval_holdout(
        session,
        OvrPanelRouter(probes),
        df,
        holdout_ids,
        bundle,
        phase=phase,
        hypotheses=hypotheses,
        config=config,
        train_seconds=train_seconds,
    )
