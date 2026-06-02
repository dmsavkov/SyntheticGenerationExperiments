"""LLM prompt templates with full label vocabulary."""

from __future__ import annotations

import json
from typing import Any

from routers.core.data import build_text_from_parts
from routers.synthetic.parse import SyntheticItem, ValidationItem


def format_row_for_llm(row: dict, index: int | None = None) -> str:
    prefix = f"### Example {index}\n" if index is not None else "### Example\n"
    return (
        f"{prefix}"
        f"**Domain (gold):** {row.get('gold', '')}\n\n"
        f"**Context**\n{row.get('context', '')}\n\n"
        f"**Question**\n{row.get('question', '')}\n\n"
        f"**Options**\n{row.get('options', '')}\n"
    )


def _labels_block(domain_labels: list[str]) -> str:
    return "Allowed Domain labels (use exactly one per item):\n" + json.dumps(domain_labels, ensure_ascii=False)


def _length_instruction(n: int) -> str:
    return (
        f"Generate exactly {n} new items (one per reference example, same order).\n"
        "Each new item's total length should approximate its reference example.\n"
        "Absolute maximum ~150 words (~600 characters) combined across context, question, and options.\n"
    )


def build_label_failure_prompt(
    exemplars: list[dict],
    domain_labels: list[str],
    target_domain: str,
) -> tuple[str, str]:
    n = len(exemplars)
    system = (
        "You generate RouterArena-style training examples for domain classification.\n"
        f"{_labels_block(domain_labels)}\n"
        "Return ONLY valid JSON matching this schema:\n"
        '{"items": [{"context": "...", "question": "...", "options": "...", "domain": "<label>"}]}\n'
        "No markdown fences or commentary outside JSON."
    )
    blocks = [format_row_for_llm(ex, i + 1) for i, ex in enumerate(exemplars)]
    user = (
        f"Domain **{target_domain}** was misclassified on similar items.\n"
        "Generate new examples that match the logic and style of the references and are clearly "
        f"**{target_domain}**.\n\n"
        f"{_length_instruction(n)}\n"
        "References:\n\n"
        + "\n".join(blocks)
    )
    return system, user


def build_confusion_pair_prompt(
    pair: tuple[str, str],
    exemplars_failures: list[dict],
    fewshot_a: list[dict],
    fewshot_b: list[dict],
    domain_labels: list[str],
    n_generate: int,
) -> tuple[str, str]:
    gold_a, gold_b = pair
    system = (
        "You generate RouterArena-style examples that clarify the boundary between two domains.\n"
        f"{_labels_block(domain_labels)}\n"
        "Return ONLY valid JSON:\n"
        '{"items": [{"context": "...", "question": "...", "options": "...", "domain": "<label>"}]}\n'
    )
    fs = []
    for i, ex in enumerate(fewshot_a[:5], 1):
        fs.append(format_row_for_llm({**ex, "gold": gold_a}, i))
    for i, ex in enumerate(fewshot_b[:5], 1):
        fs.append(format_row_for_llm({**ex, "gold": gold_b}, i))
    fail_blocks = [format_row_for_llm(ex, i + 1) for i, ex in enumerate(exemplars_failures)]
    user = (
        f"Distinguish **{gold_a}** vs **{gold_b}**. Misclassified holdout examples:\n\n"
        + "\n".join(fail_blocks)
        + "\n\nFew-shot per class:\n\n"
        + "\n".join(fs)
        + f"\n\n{_length_instruction(n_generate)}\n"
        f"Split domains across items between {gold_a} and {gold_b} where helpful."
    )
    return system, user


def build_validator_prompt(
    synthetics: list[dict],
    domain_labels: list[str],
) -> tuple[str, str]:
    n = len(synthetics)
    system = (
        "You verify whether a domain label correctly describes a generated sample.\n"
        f"{_labels_block(domain_labels)}\n"
        "Return ONLY JSON:\n"
        '{"items": [{"id": "<synthetic_id>", "verdict": "YES" or "NO", "reason": "..."}]}\n'
    )
    blocks = []
    for ex in synthetics:
        blocks.append(
            f"### id={ex['id']}\n"
            f"**Claimed Domain:** {ex.get('gold', '')}\n\n"
            f"**Context**\n{ex.get('context', '')}\n\n"
            f"**Question**\n{ex.get('question', '')}\n\n"
            f"**Options**\n{ex.get('options', '')}\n"
        )
    user = (
        f"For each sample, answer: Does the claimed Domain label correctly describe this sample?\n"
        f"Return exactly {n} items with matching ids.\n\n"
        + "\n".join(blocks)
    )
    return system, user


def synthetic_item_to_train_row(item: SyntheticItem, synth_id: str) -> dict[str, Any]:
    ctx, q, opt = item.context, item.question, item.options
    from routers.synthetic.batching import truncate_synthetic_item

    ctx, q, opt = truncate_synthetic_item(ctx, q, opt)
    return {
        "id": synth_id,
        "context": ctx,
        "question": q,
        "options": opt,
        "gold": item.domain,
        "dataset_name": "synthetic",
        "prompt": build_text_from_parts(ctx, q, opt),
        "source": "synthetic",
    }


def validation_items_to_map(batch_items: list[ValidationItem]) -> dict[str, ValidationItem]:
    return {it.id: it for it in batch_items}
