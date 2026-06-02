"""SetFit-specific LLM prompt templates (max 3 items per call)."""

from __future__ import annotations

import json
from typing import Any

from routers.synthetic.prompts import _labels_block, _length_instruction, format_row_for_llm


def build_diversity_triplet_prompt(
    exemplars: list[dict],
    domain_labels: list[str],
) -> tuple[str, str]:
    if len(exemplars) != 3:
        raise ValueError("diversity_triplet requires exactly 3 exemplars")
    system = (
        "You generate RouterArena-style training examples for domain classification.\n"
        f"{_labels_block(domain_labels)}\n"
        "Ensure extreme structural diversity across the 3 items.\n"
        "Return ONLY valid JSON:\n"
        '{"items": [{"context": "...", "question": "...", "options": "...", "domain": "<label>"}]}\n'
    )
    blocks = [format_row_for_llm(ex, i + 1) for i, ex in enumerate(exemplars)]
    user = (
        "Generate exactly 3 new items (one per reference, same order).\n"
        "Item 1 strategy: factual question starting with 'What is the' or similar direct query.\n"
        "Item 2 strategy: complex real-world scenario paragraph with embedded question.\n"
        "Item 3 strategy: short fill-in-the-blank style query.\n"
        "Rule: No two questions may start with the same word.\n"
        "Use advanced, varied vocabulary.\n\n"
        f"{_length_instruction(3)}\n"
        "References:\n\n"
        + "\n".join(blocks)
    )
    return system, user


def build_hard_rewrite_prompt(exemplar: dict, domain_labels: list[str]) -> tuple[str, str]:
    gold = str(exemplar.get("gold", ""))
    system = (
        "You rewrite RouterArena-style training examples.\n"
        f"{_labels_block(domain_labels)}\n"
        "Return ONLY valid JSON:\n"
        '{"items": [{"context": "...", "question": "...", "options": "...", "domain": "<label>"}]}\n'
    )
    user = (
        f"Write a new example that tests the EXACT SAME concept as the reference, but rewrite it to be "
        "2 times longer, use completely different vocabulary, and frame it as a real-world scenario "
        "rather than a direct question.\n"
        f"The domain must remain **{gold}**.\n\n"
        f"{_length_instruction(1)}\n"
        "Reference:\n\n"
        + format_row_for_llm(exemplar, 1)
    )
    return system, user


def build_diverse_rewrite_prompt(exemplar: dict, domain_labels: list[str]) -> tuple[str, str]:
    gold = str(exemplar.get("gold", ""))
    system = (
        "You rewrite RouterArena-style training examples.\n"
        f"{_labels_block(domain_labels)}\n"
        "Return ONLY valid JSON:\n"
        '{"items": [{"context": "...", "question": "...", "options": "...", "domain": "<label>"}]}\n'
    )
    user = (
        "Rewrite this question using an entirely different sentence structure, vocabulary and synonyms, "
        "but keep the logic, correct option and the domain identical.\n"
        f"The domain must remain **{gold}**.\n\n"
        f"{_length_instruction(1)}\n"
        "Reference:\n\n"
        + format_row_for_llm(exemplar, 1)
    )
    return system, user


def prompt_for_mode(
    mode: str,
    batch: list[dict],
    domain_labels: list[str],
    target_domain: str | None = None,
) -> tuple[str, str]:
    from routers.synthetic.prompts import build_label_failure_prompt

    if mode == "label_failure":
        td = target_domain or str(batch[0].get("gold", ""))
        return build_label_failure_prompt(batch, domain_labels, td)
    if mode == "diversity_triplet":
        return build_diversity_triplet_prompt(batch, domain_labels)
    if mode == "hard_rewrite":
        return build_hard_rewrite_prompt(batch[0], domain_labels)
    if mode == "diverse_rewrite":
        return build_diverse_rewrite_prompt(batch[0], domain_labels)
    raise ValueError(f"Unknown generation mode: {mode}")
