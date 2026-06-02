"""Eval harness with per-class precision/recall and 2k prompt saves."""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tqdm import tqdm

from routers.core.constants import PROMPT_SAVE_MAX_CHARS
from routers.core.splits import project_root

PredictFn = Callable[[str], str]
BatchPredictFn = Callable[[list[str]], list[str]]


def _results_dir(topic: str, variation: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return project_root() / "results" / topic / variation / ts


def run_experiment(
    topic: str,
    variation: str,
    rows: list[dict],
    *,
    hypotheses: str,
    config: dict[str, Any],
    predict: PredictFn | None = None,
    predict_batch: BatchPredictFn | None = None,
    predict_row: Callable[[dict], str] | None = None,
    predict_row_extended: Callable[[dict], dict] | None = None,
    batch_size: int = 5,
    save: bool = True,
    out_dir: Path | None = None,
    supplemental_metrics: dict[str, Any] | None = None,
    train_seconds: float = 0.0,
    tqdm_desc: str | None = None,
) -> dict[str, Any]:
    if predict_row_extended is not None:
        predict_row = None
    if predict is None and predict_batch is None and predict_row is None and predict_row_extended is None:
        raise ValueError("Provide a predict function")

    infer_start = time.perf_counter()
    golds: list[str] = []
    preds: list[str] = []
    records: list[dict] = []

    desc = tqdm_desc or variation

    if predict_batch is not None:
        prompts = [str(r["prompt"]) for r in rows]
        n = len(rows)
        for start in tqdm(range(0, n, batch_size), desc=desc):
            batch_rows = rows[start : start + batch_size]
            batch_preds = predict_batch(prompts[start : start + batch_size])
            if len(batch_preds) != len(batch_rows):
                if len(batch_preds) < len(batch_rows):
                    batch_preds = list(batch_preds) + ["unavailable"] * (
                        len(batch_rows) - len(batch_preds)
                    )
                else:
                    batch_preds = batch_preds[: len(batch_rows)]
            for row, pred in zip(batch_rows, batch_preds):
                gold, pred_s = str(row["gold"]), str(pred)
                golds.append(gold)
                preds.append(pred_s)
                records.append(_record(row, gold, pred_s))
    else:
        for row in tqdm(rows, desc=desc):
            gold = str(row["gold"])
            if predict_row_extended is not None:
                out = predict_row_extended(row)
                pred = str(out["pred"])
                extra = {k: v for k, v in out.items() if k != "pred"}
                golds.append(gold)
                preds.append(pred)
                records.append(_record(row, gold, pred, extra=extra))
            elif predict_row is not None:
                pred = predict_row(row)
                golds.append(gold)
                preds.append(str(pred))
                records.append(_record(row, gold, str(pred)))
            else:
                assert predict is not None
                pred = predict(str(row["prompt"]))
                golds.append(gold)
                preds.append(str(pred))
                records.append(_record(row, gold, str(pred)))

    infer_seconds = time.perf_counter() - infer_start
    metrics = _build_metrics(
        topic, variation, golds, preds, hypotheses, config, train_seconds, infer_seconds
    )
    if supplemental_metrics:
        metrics.update(supplemental_metrics)

    if save:
        target = out_dir or _results_dir(topic, variation)
        target.mkdir(parents=True, exist_ok=True)
        metrics["results_path"] = str(target)
        with open(target / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        with open(target / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        correct = [r for r in records if r["correct"]]
        incorrect = [r for r in records if not r["correct"]]
        with open(target / "predictions_correct.json", "w", encoding="utf-8") as f:
            json.dump(correct, f, indent=2, ensure_ascii=False)
        with open(target / "predictions_incorrect.json", "w", encoding="utf-8") as f:
            json.dump(incorrect, f, indent=2, ensure_ascii=False)
        labels = sorted(set(golds) | set(preds))
        _save_confusion_plot(target / "confusion.png", golds, preds, labels)
        logger.info("Saved results to {}", target)

    logger.info(
        "{} / {} accuracy={:.4f} macro_f1={:.4f}",
        topic,
        variation,
        metrics["accuracy"],
        metrics["macro_f1"],
    )
    return metrics


def _record(row: dict, gold: str, pred: str, extra: dict | None = None) -> dict:
    rec = {
        "id": row["id"],
        "gold": gold,
        "pred": pred,
        "correct": gold == pred,
        "score": row.get("score"),
        "dataset_name": row.get("dataset_name"),
        "prompt": str(row.get("prompt", ""))[:PROMPT_SAVE_MAX_CHARS],
    }
    if extra:
        rec.update(extra)
    for key, val in row.items():
        if key.startswith("_") and key not in rec:
            rec[key.lstrip("_")] = val
    return rec


def _per_class_pr(golds: list[str], preds: list[str], labels: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    report = classification_report(
        golds, preds, labels=labels, output_dict=True, zero_division=0
    )
    precision: dict[str, float] = {}
    recall: dict[str, float] = {}
    for lab in labels:
        if lab in report and isinstance(report[lab], dict):
            precision[lab] = round(float(report[lab]["precision"]), 4)
            recall[lab] = round(float(report[lab]["recall"]), 4)
    return precision, recall


def _build_metrics(
    topic: str,
    variation: str,
    golds: list[str],
    preds: list[str],
    hypotheses: str,
    config: dict[str, Any],
    train_seconds: float,
    infer_seconds: float,
) -> dict[str, Any]:
    labels = sorted(set(golds) | set(preds))
    acc = accuracy_score(golds, preds)
    macro_f1 = f1_score(golds, preds, labels=labels, average="macro", zero_division=0)
    per_prec, per_rec = _per_class_pr(golds, preds, labels)
    return {
        "topic": topic,
        "variation": variation,
        "hypotheses": hypotheses,
        "n": len(golds),
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "per_class_precision": per_prec,
        "per_class_recall": per_rec,
        "per_class_gold": dict(Counter(golds)),
        "per_class_pred": dict(Counter(preds)),
        "train_seconds": round(train_seconds, 4),
        "infer_seconds": round(infer_seconds, 4),
        "total_seconds": round(train_seconds + infer_seconds, 4),
        **{k: v for k, v in config.items() if k != "hypotheses"},
    }


def _save_confusion_plot(path: Path, golds: list[str], preds: list[str], labels: list[str]) -> None:
    if not golds:
        return
    cm = confusion_matrix(golds, preds, labels=labels)
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.6), max(5, len(labels) * 0.5)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="Gold",
        xlabel="Predicted",
        title="Confusion matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_json_artifact(out_dir: Path, name: str, payload: Any) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path
