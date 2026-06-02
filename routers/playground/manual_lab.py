"""Manual SetFit laboratory: load splits, train, eval, generate."""

from __future__ import annotations

import json
import random
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import classification_report, f1_score

from routers.core.data import load_arena, row_dict_from_series, rows_from_ids
from routers.core.splits import get_splits
from routers.playground.ids import new_row_id
from routers.playground.splits import get_playground_splits
from routers.synthetic.parse import parse_synthetic_batch
from routers.synthetic.prompts import synthetic_item_to_train_row

DEFAULT_CFG: dict[str, Any] = {
    "source_preset": "arc_med_mmlu",
    "domains": None,
    "train_per_class": 50,
    "eval_per_class": 100,
    "shuffle_seed": 42,
    "setfit_model": "sentence-transformers/all-MiniLM-L6-v2",
    "setfit_epochs": 1,
    "setfit_batch_size": 16,
    "setfit_num_iterations": 20,
    "ollama_model": "llama3:8b-instruct-q4_K_M",
    "temperature": 0.8,
    "synth_per_class": 20,
    "parse_retries": 3,
    "show_gold_in_exemplars": True,
    "system_prompt": (
        "You generate RouterArena-style training examples for domain classification.\n"
        "Allowed Domain labels: {domain_labels}\n"
        "Return ONLY valid JSON:\n"
        '{{"items": [{{"context": "...", "question": "...", "options": "...", "domain": "<label>"}}]}}\n'
        "No markdown fences or commentary."
    ),
    "user_prompt_template": (
        "Generate exactly {n} new examples for domain **{label}**.\n"
        "Match the logic and style of the references below.\n\n"
        "References:\n\n{examples}"
    ),
}


def resolve_domains(df: pd.DataFrame, requested: list[str] | None) -> list[str]:
    available = df["Domain"].value_counts()
    if requested:
        missing = [d for d in requested if d not in available.index]
        if not missing:
            return list(requested)
        logger.warning("Domains not found {}. Top-5: {}", missing, available.head(5).to_dict())
    top2 = available.head(2).index.tolist()
    logger.info("Using top-2 domains: {}", top2)
    return top2


def load_playground_data(cfg: dict[str, Any] | None = None) -> tuple[pd.DataFrame, Any, list[str]]:
    """Load arena, frozen playground splits, restrict to 2 domains."""
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    df = load_arena(seed=cfg["shuffle_seed"])
    bundle = get_splits(df)
    preset = cfg["source_preset"]
    pg = get_playground_splits(df, bundle, preset=preset)

    all_rows = rows_from_ids(df, pg.train_ids + pg.eval_ids, "Domain", id_to_idx=bundle.id_to_idx, with_features=True)
    domains = resolve_domains(
        pd.DataFrame([{"Domain": r["gold"]} for r in all_rows]),
        cfg.get("domains"),
    )
    domain_set = set(domains)
    train_rows = [r for r in rows_from_ids(df, pg.train_ids, "Domain", id_to_idx=bundle.id_to_idx, with_features=True) if r["gold"] in domain_set]
    eval_rows = [r for r in rows_from_ids(df, pg.eval_ids, "Domain", id_to_idx=bundle.id_to_idx, with_features=True) if r["gold"] in domain_set]

    per_train = cfg["train_per_class"]
    per_eval = cfg["eval_per_class"]
    train_final = _cap_per_class(train_rows, per_train, cfg["shuffle_seed"])
    eval_final = _cap_per_class(eval_rows, per_eval, cfg["shuffle_seed"] + 1)

    for r in train_final + eval_final:
        r["source"] = "real"

    logger.info("train: {}", Counter(r["gold"] for r in train_final))
    logger.info("eval: {}", Counter(r["gold"] for r in eval_final))
    return df, {"train": train_final, "eval": eval_final, "domains": domains, "cfg": cfg}, domains


def _cap_per_class(rows: list[dict], per_class: int, seed: int) -> list[dict]:
    by_label: dict[str, list[dict]] = {}
    for r in rows:
        by_label.setdefault(str(r["gold"]), []).append(r)
    rng = random.Random(seed)
    out: list[dict] = []
    for lab in sorted(by_label):
        pool = by_label[lab]
        rng.shuffle(pool)
        out.extend(pool[: min(per_class, len(pool))])
    return out


def build_train_eval(cfg: dict[str, Any] | None = None) -> tuple[list[dict], list[dict], list[str]]:
    _, state, domains = load_playground_data(cfg)
    return state["train"], state["eval"], domains


def train_setfit(rows: list[dict], cfg: dict[str, Any]) -> tuple[Any, dict[int, str]]:
    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments

    labels = sorted({r["gold"] for r in rows})
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    ds = Dataset.from_dict({
        "text": [r["prompt"] for r in rows],
        "label": [label2id[r["gold"]] for r in rows],
    })
    model = SetFitModel.from_pretrained(cfg["setfit_model"], labels=list(range(len(labels))))
    args = TrainingArguments(
        batch_size=cfg["setfit_batch_size"],
        num_epochs=cfg["setfit_epochs"],
        sampling_strategy="num_iterations",
        num_iterations=cfg["setfit_num_iterations"],
        show_progress_bar=True,
    )
    Trainer(model=model, args=args, train_dataset=ds).train()
    return model, id2label


def eval_with_report(model: Any, id2label: dict[int, str], rows: list[dict]) -> dict[str, Any]:
    texts = [r["prompt"] for r in rows]
    golds = [str(r["gold"]) for r in rows]
    raw = model.predict(texts)
    arr = np.asarray(raw)
    preds = [id2label[int(x)] for x in arr.flat]
    labels = sorted(set(golds) | set(preds))
    report = classification_report(golds, preds, labels=labels, output_dict=True, zero_division=0)
    return {
        "n": len(rows),
        "accuracy": round(float(report["accuracy"]), 4),
        "macro_f1": round(float(f1_score(golds, preds, average="macro", zero_division=0)), 4),
        "per_class_precision": {k: round(report[k]["precision"], 4) for k in labels if k in report},
        "per_class_recall": {k: round(report[k]["recall"], 4) for k in labels if k in report},
        "golds": golds,
        "preds": preds,
        "rows": rows,
    }


def format_example(row: dict, idx: int, show_gold: bool) -> str:
    lines = [f"### Example {idx}"]
    if show_gold:
        lines.append(f"**Domain (gold):** {row['gold']}")
    lines += [
        f"\n**Context**\n{row.get('context', '')}",
        f"\n**Question**\n{row.get('question', '')}",
        f"\n**Options**\n{row.get('options', '')}",
    ]
    return "\n".join(lines)


def generate_for_class(
    seed_rows: list[dict],
    label: str,
    cfg: dict[str, Any],
    domain_labels: list[str],
) -> list[dict]:
    import ollama

    n = cfg["synth_per_class"]
    examples = "\n\n".join(
        format_example(r, i + 1, cfg["show_gold_in_exemplars"]) for i, r in enumerate(seed_rows)
    )
    system = cfg["system_prompt"].format(domain_labels=json.dumps(domain_labels))
    user = cfg["user_prompt_template"].format(n=n, label=label, examples=examples)
    last_err: Exception | None = None
    for attempt in range(1, cfg["parse_retries"] + 1):
        try:
            user_msg = user
            if attempt > 1:
                user_msg += "\n\nReturn ONLY valid JSON matching the schema."
            resp = ollama.chat(
                model=cfg["ollama_model"],
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
                format="json",
                options={"temperature": cfg["temperature"]},
            )
            text = (resp.get("message") or {}).get("content") or ""
            batch = parse_synthetic_batch(text, n_expected=n, domain_labels=domain_labels)
            out: list[dict] = []
            for item in batch.items:
                sid = new_row_id()
                row = synthetic_item_to_train_row(item, sid)
                row["source"] = "synthetic"
                row["seed_domain"] = label
                if row["gold"] != label:
                    row["gold"] = label
                    row["prompt"] = (
                        f"Context: {row.get('context', '')} | Question: {row.get('question', '')} | "
                        f"Options: {row.get('options', '')}"
                    )
                out.append(row)
            return out
        except Exception as e:
            last_err = e
            logger.warning("parse attempt {}/{} failed: {}", attempt, cfg["parse_retries"], e)
    raise RuntimeError(f"Generation failed for {label}") from last_err


def row_from_series(row: pd.Series, rid: Any) -> dict[str, Any]:
    """Build playground row dict from dataframe series."""
    d = row_dict_from_series(row, rid, "Domain")
    d["source"] = "real"
    return d
