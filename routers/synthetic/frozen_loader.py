"""Load pre-generated synthetic rows and apply prompt-format variants."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from loguru import logger

from routers.core.data import build_text_from_parts
from routers.core.splits import project_root

SynthFormat = Literal["as_is", "context_in_question", "no_context"]

DEFAULT_EXP02_SYNTHETICS = project_root() / "data" / "synthetic" / "baseline_exp02.json"


def _strip(s: str | None) -> str:
    return (s or "").strip()


def _has_context(ctx: str) -> bool:
    return bool(_strip(ctx))


def _item_to_fields(item: dict[str, Any]) -> tuple[str, str, str, str]:
    ctx = _strip(item.get("context"))
    question = _strip(item.get("question"))
    options = _strip(item.get("options"))
    domain = _strip(item.get("domain") or item.get("gold"))
    return ctx, question, options, domain


def _rows_from_generation_log(payload: dict[str, Any], *, source_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    idx = 0
    for batch in payload.get("batches", []):
        for item in batch.get("parsed_items") or []:
            ctx, question, options, domain = _item_to_fields(item)
            if not domain:
                continue
            idx += 1
            sid = f"synth_frozen_{idx:04d}"
            llm = {"context": ctx, "question": question, "options": options, "domain": domain}
            rows.append(
                {
                    "id": sid,
                    "context": ctx,
                    "question": question,
                    "options": options,
                    "gold": domain,
                    "dataset_name": "synthetic_frozen",
                    "prompt": build_text_from_parts(ctx, question, options),
                    "source": "synthetic_frozen",
                    "llm_generated": llm,
                    "frozen_source": str(source_path.name),
                }
            )
    return rows


def _rows_from_samples_list(samples: list[dict[str, Any]], *, source_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        llm = sample.get("llm_generated") or sample.get("normalized_for_training") or sample
        norm = sample.get("normalized_for_training") or {}
        ctx, question, options, domain = _item_to_fields({**norm, **llm, "domain": llm.get("domain") or norm.get("domain")})
        if not domain:
            domain = _strip(sample.get("gold") or norm.get("domain"))
        if not domain:
            continue
        sid = _strip(sample.get("id")) or f"synth_frozen_{len(rows) + 1:04d}"
        rows.append(
            {
                "id": sid,
                "context": ctx,
                "question": question,
                "options": options,
                "gold": domain,
                "dataset_name": "synthetic_frozen",
                "prompt": build_text_from_parts(ctx, question, options),
                "source": "synthetic_frozen",
                "llm_generated": {"context": ctx, "question": question, "options": options, "domain": domain},
                "frozen_source": str(source_path.name),
            }
        )
    return rows


def load_frozen_synthetics(path: Path | str | None = None) -> list[dict[str, Any]]:
    source = Path(path) if path is not None else DEFAULT_EXP02_SYNTHETICS
    if not source.is_file():
        raise FileNotFoundError(f"Frozen synthetic file not found: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = _rows_from_samples_list(payload, source_path=source)
    elif isinstance(payload, dict) and "batches" in payload:
        rows = _rows_from_generation_log(payload, source_path=source)
    else:
        raise ValueError(f"Unsupported frozen synthetic format in {source}")
    logger.info("Loaded {} frozen synthetic rows from {}", len(rows), source)
    return rows


def apply_synth_format(rows: list[dict[str, Any]], fmt: SynthFormat) -> list[dict[str, Any]]:
    if fmt == "as_is":
        return [dict(r) for r in rows]

    out: list[dict[str, Any]] = []
    for row in rows:
        r = dict(row)
        ctx = _strip(r.get("context"))
        question = _strip(r.get("question"))
        options = _strip(r.get("options"))

        if fmt == "context_in_question":
            if _has_context(ctx):
                question = f"{ctx} {question}".strip() if question else ctx
            ctx = ""
        elif fmt == "no_context":
            ctx = ""

        r["context"] = ctx
        r["question"] = question
        r["options"] = options
        r["prompt"] = build_text_from_parts(ctx, question, options)
        r["synth_format"] = fmt
        out.append(r)
    return out


def load_formatted_synthetics(
    fmt: SynthFormat,
    path: Path | str | None = None,
) -> list[dict[str, Any]]:
    return apply_synth_format(load_frozen_synthetics(path), fmt)
