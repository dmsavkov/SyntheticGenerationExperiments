"""Prompts for combination experiment group."""

from __future__ import annotations

import json

from routers.synthetic.prompts import _labels_block, _length_instruction, format_row_for_llm


HARD_NEGATIVE_FEWSHOT = """
Few-shot: similar general wording, contrasting specific vocabulary:

Domain A (Technology): "The resolution of IOPA is considered optimal at..."
Domain B (Computer Science): "The resolution of a standard VGA display interface is..."

Domain A (Technology): "Impedance based apex locator..."
Domain B (Computer Science): "The characteristic impedance of a standard coaxial network cable..."

Domain A (Library Science): "Which Indian University first started M. Lib. SC..."
Domain B (Medical): "Which Indian University first started MBBS and BDS..."
""".strip()


def build_hard_negative_pair_prompt(
    failure_row: dict,
    domain_labels: list[str],
    *,
    n_items: int = 2,
) -> tuple[str, str]:
    gold = str(failure_row.get("gold", ""))
    pred = str(failure_row.get("pred", ""))
    system = (
        "You generate formal, high-quality RouterArena training examples for domain classification.\n"
        f"{_labels_block(domain_labels)}\n"
        "Return ONLY valid JSON:\n"
        '{"items": [{"context": "...", "question": "...", "options": "...", "domain": "<label>"}]}\n'
    )
    user = (
        f"A classifier confused gold **{gold}** with prediction **{pred}**.\n"
        f"Generate exactly {n_items} new items: one clearly **{gold}**, one clearly **{pred}**.\n"
        "Use similar general topic words but contrasting specific technical vocabulary.\n"
        "Show what discriminates the domains.\n\n"
        f"{HARD_NEGATIVE_FEWSHOT}\n\n"
        f"{_length_instruction(n_items)}\n"
        "Misclassified example:\n\n"
        + format_row_for_llm(failure_row)
    )
    return system, user


def build_cascade_analysis_prompt(failures: list[dict]) -> str:
    blocks = [format_row_for_llm(r, i + 1) for i, r in enumerate(failures[:20])]
    return (
        "SetFit failed on these holdout samples. For each, describe technical properties "
        "of the item and explain why a small embedding classifier might confuse domains.\n\n"
        + "\n".join(blocks)
    )


def build_cascade_generation_prompt(
    domain_labels: list[str],
    *,
    n_items: int = 8,
    focus_note: str = "",
) -> str:
    return (
        "Using your prior analysis in this conversation, generate new training examples "
        "that teach the model the mechanics it missed.\n"
        f"{focus_note}"
        f"{_labels_block(domain_labels)}\n"
        f"Generate exactly {n_items} items.\n"
        "Return ONLY valid JSON:\n"
        '{"items": [{"context": "...", "question": "...", "options": "...", "domain": "<label>"}]}\n'
    )


def build_cascade_contrastive_prompt(remaining_failures: list[dict]) -> str:
    blocks = [format_row_for_llm(r, i + 1) for i, r in enumerate(remaining_failures[:10])]
    return (
        "Your previous generation helped SetFit learn some concepts, but it STILL FAILED on these:\n\n"
        + "\n".join(blocks)
        + "\n\nGenerate new samples for these errors using a completely different structural "
        "strategy than your first attempt (e.g. True/False, fill-in-blank, long scenario, short keyword).\n"
        "Return ONLY valid JSON with an items array."
    )
