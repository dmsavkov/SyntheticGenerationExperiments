"""Shared experiment utilities."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
from loguru import logger

from routers.baselines.modernbert_probe import ModernBertProbe, probe_config_extra, train_probe
from routers.core.constants import (
    CONTEXT_MAX_CHARS,
    GENERATION_TEMPERATURE,
    TARGET_COL,
    TEXT_FORMAT,
)
from routers.core.data import dataset_metadata, label_vocab, load_arena, rows_from_ids
from routers.core.harness import run_experiment, save_json_artifact
from routers.experiments.session import RESULTS_TOPIC, ExperimentSession, serialize_synthetic_rows
from routers.core.splits import (
    DomainSynthSplits,
    SplitBundle,
    get_domain_synth_splits,
    get_splits,
    set_global_seed,
)
from routers.synthetic.generator import validate_synthetics
from routers.synthetic.ollama_client import ollama_model

HYPOTHESES_BASE = "Domain synthetic collision experiments on RouterArena."


class ClassifierProbe(Protocol):
    def predict(self, row: dict) -> str: ...

    def predict_extended(self, row: dict) -> dict[str, Any]: ...


def load_context(*, rebuild_splits: bool = False) -> tuple[pd.DataFrame, SplitBundle, DomainSynthSplits]:
    set_global_seed(42)
    df = load_arena()
    bundle = get_splits(df, rebuild=rebuild_splits)
    dsplits = get_domain_synth_splits(df, bundle, rebuild=rebuild_splits)
    return df, bundle, dsplits


def base_config(
    *,
    experiment: str,
    train_split: str,
    eval_split: str,
    hypotheses: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        **dataset_metadata(),
        "target_col": TARGET_COL,
        "train_split": train_split,
        "eval_split": eval_split,
        "text_format": TEXT_FORMAT,
        "context_max_chars": CONTEXT_MAX_CHARS,
        "ollama_model": ollama_model(),
        "generation_temperature": GENERATION_TEMPERATURE,
        "length_policy": {"abs_max_chars": 600, "match_exemplar": True},
        "batching_policy": {"max_exemplars": 10, "max_exemplar_chars": 2000},
        "hypotheses": hypotheses,
        "experiment": experiment,
        **probe_config_extra(),
    }
    if extra:
        cfg.update(extra)
    return cfg


def train_rows_from_ids(
    df: pd.DataFrame,
    ids: list[Any],
    bundle: SplitBundle,
    *,
    extra_rows: list[dict] | None = None,
) -> list[dict]:
    rows = rows_from_ids(
        df, ids, TARGET_COL, id_to_idx=bundle.id_to_idx, text_format="embedder", with_features=True
    )
    if extra_rows:
        rows = rows + extra_rows
    return rows


def eval_probe(
    probe: ClassifierProbe,
    df: pd.DataFrame,
    ids: list[Any],
    bundle: SplitBundle,
    *,
    slug: str,
    hypotheses: str,
    config: dict[str, Any],
    topic: str = RESULTS_TOPIC,
    save: bool = True,
    out_dir: Path | None = None,
    supplemental_metrics: dict[str, Any] | None = None,
    train_seconds: float = 0.0,
) -> dict[str, Any]:
    rows = rows_from_ids(df, ids, TARGET_COL, id_to_idx=bundle.id_to_idx, text_format="embedder")
    return run_experiment(
        topic,
        slug,
        rows,
        hypotheses=hypotheses,
        config=config,
        predict_row=probe.predict,
        save=save,
        out_dir=out_dir,
        supplemental_metrics=supplemental_metrics,
        train_seconds=train_seconds,
    )


def eval_holdout(
    session: ExperimentSession,
    probe: ClassifierProbe,
    df: pd.DataFrame,
    holdout_ids: list[Any],
    bundle: SplitBundle,
    *,
    phase: str,
    hypotheses: str,
    config: dict[str, Any],
    train_seconds: float = 0.0,
) -> dict[str, Any]:
    metrics = eval_probe(
        probe,
        df,
        holdout_ids,
        bundle,
        slug=session.experiment,
        hypotheses=hypotheses,
        config=config,
        topic=session.topic,
        save=False,
        train_seconds=train_seconds,
    )
    session.record_holdout(phase, metrics)
    logger.info("Holdout {} macro_f1={:.4f}", phase, metrics["macro_f1"])
    return metrics


def save_primary_eval(
    session: ExperimentSession,
    probe: ClassifierProbe,
    df: pd.DataFrame,
    eval_ids: list[Any],
    bundle: SplitBundle,
    *,
    hypotheses: str,
    config: dict[str, Any],
    train_seconds: float = 0.0,
) -> dict[str, Any]:
    session.save_config(config)
    supplemental = session.supplemental_metrics()
    metrics = eval_probe(
        probe,
        df,
        eval_ids,
        bundle,
        slug=session.experiment,
        hypotheses=hypotheses,
        config=config,
        save=True,
        out_dir=session.out_dir,
        topic=session.topic,
        supplemental_metrics=supplemental,
        train_seconds=train_seconds,
    )
    return metrics


def failures_from_holdout(holdout_rows: list[dict], probe: ModernBertProbe) -> list[dict]:
    incorrect: list[dict] = []
    for row in holdout_rows:
        pred = probe.predict(row)
        if pred != str(row["gold"]):
            enriched = dict(row)
            enriched["pred"] = pred
            incorrect.append(enriched)
    return incorrect


def sample_donor_rows(
    df: pd.DataFrame,
    donor_ids: list[Any],
    bundle: SplitBundle,
    gold_label: str,
    n: int,
    seed: int,
) -> list[dict]:
    candidates = rows_from_ids(
        df,
        donor_ids,
        TARGET_COL,
        id_to_idx=bundle.id_to_idx,
        text_format="embedder",
        with_features=True,
    )
    matched = [r for r in candidates if str(r["gold"]) == gold_label]
    if not matched:
        return []
    rng = random.Random(seed)
    if len(matched) >= n:
        return rng.sample(matched, n)
    out: list[dict] = []
    while len(out) < n:
        out.extend(rng.sample(matched, min(len(matched), n - len(out))))
    return out


def run_validation_pipeline(
    probe: ModernBertProbe,
    synthetics: list[dict],
    domain_labels: list[str],
    session: ExperimentSession,
) -> tuple[list[dict], list[dict]]:
    val_meta = validate_synthetics(synthetics, domain_labels, stats=session.generation_stats)
    val_by_id = {v["synthetic_id"]: v for v in val_meta}
    records: list[dict] = []
    yes_rows: list[dict] = []
    for syn in synthetics:
        ext = probe.predict_extended(syn)
        v = val_by_id.get(syn["id"], {})
        rec = {
            "synthetic_id": syn["id"],
            "gold_domain": syn.get("gold"),
            "llm_generated": syn.get("llm_generated"),
            "context": syn.get("context", ""),
            "question": syn.get("question", ""),
            "options": syn.get("options", ""),
            "prompt": syn.get("prompt", ""),
            "validator_verdict": v.get("validator_verdict", "NO"),
            "validator_reason": v.get("validator_reason", ""),
            "validator_raw_response": v.get("validator_raw_response"),
            "modernbert_pred": ext["pred"],
            "modernbert_probs": ext["pred_probs"],
            "used_in_training": False,
        }
        if rec["validator_verdict"] == "YES":
            rec["used_in_training"] = True
            yes_rows.append(syn)
        records.append(rec)
    save_json_artifact(session.out_dir, "synthetic_validation.json", records)
    session.save_generation_log()
    return yes_rows, records
