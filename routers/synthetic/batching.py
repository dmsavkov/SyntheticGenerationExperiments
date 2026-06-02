"""Length-aware truncation and dynamic exemplar batching."""

from __future__ import annotations

from typing import Any

from routers.core.constants import (
    BATCH_MAX_EXEMPLAR_CHARS,
    BATCH_MAX_EXEMPLARS,
    GENERATION_ABS_MAX_CHARS,
)


def exemplar_char_count(row: dict) -> int:
    return len(str(row.get("context", ""))) + len(str(row.get("question", ""))) + len(str(row.get("options", "")))


def truncate_fields_proportional(
    context: str, question: str, options: str, max_chars: int = GENERATION_ABS_MAX_CHARS
) -> tuple[str, str, str]:
    total = len(context) + len(question) + len(options)
    if total <= max_chars:
        return context, question, options
    if total == 0:
        return "", "", ""
    ratio = max_chars / total
    return (
        context[: max(1, int(len(context) * ratio))],
        question[: max(1, int(len(question) * ratio))],
        options[: max(1, int(len(options) * ratio))],
    )


def prepare_exemplar(row: dict) -> dict[str, Any]:
    ctx = str(row.get("context", ""))
    q = str(row.get("question", ""))
    opt = str(row.get("options", ""))
    ctx, q, opt = truncate_fields_proportional(ctx, q, opt)
    out = dict(row)
    out["context"] = ctx
    out["question"] = q
    out["options"] = opt
    out["exemplar_chars"] = len(ctx) + len(q) + len(opt)
    out["target_length"] = out["exemplar_chars"]
    return out


def sort_exemplars(rows: list[dict]) -> list[dict]:
    prepared = [prepare_exemplar(r) for r in rows]
    return sorted(prepared, key=lambda r: r["exemplar_chars"])


def fixed_size_batches(rows: list[dict], batch_size: int) -> list[list[dict]]:
    if batch_size <= 0:
        return [rows] if rows else []
    return [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]


def dynamic_batches(rows: list[dict]) -> list[list[dict]]:
    sorted_rows = sort_exemplars(rows)
    batches: list[list[dict]] = []
    current: list[dict] = []
    char_sum = 0
    for row in sorted_rows:
        c = row["exemplar_chars"]
        would_exceed = len(current) >= BATCH_MAX_EXEMPLARS or (
            current and char_sum + c >= BATCH_MAX_EXEMPLAR_CHARS
        )
        if would_exceed and current:
            batches.append(current)
            current = []
            char_sum = 0
        current.append(row)
        char_sum += c
    if current:
        batches.append(current)
    return batches


def truncate_synthetic_item(context: str, question: str, options: str) -> tuple[str, str, str]:
    return truncate_fields_proportional(context, question, options, GENERATION_ABS_MAX_CHARS)
