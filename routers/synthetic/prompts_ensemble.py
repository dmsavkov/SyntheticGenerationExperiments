"""Prompts for ensemble OVR mutation generation."""

from __future__ import annotations

from routers.synthetic.prompts import _labels_block, _length_instruction, format_row_for_llm


def build_ovr_mutation_prompt(
    exemplars: list[dict],
    domain_labels: list[str],
    target_domain: str,
    *,
    n_items: int,
) -> tuple[str, str]:
    system = (
        "You generate RouterArena training examples for one-vs-rest domain classification.\n"
        f"{_labels_block(domain_labels)}\n"
        "Items were predicted as the target domain but gold differs — teach the boundary.\n"
        "Return ONLY valid JSON:\n"
        '{"items": [{"context": "...", "question": "...", "options": "...", "domain": "<label>"}]}\n'
    )
    blocks = [format_row_for_llm(ex, i + 1) for i, ex in enumerate(exemplars)]
    user = (
        f"Generate exactly {n_items} items clearly labeled **{target_domain}**.\n"
        "Rewrite with different vocabulary and real-world framing while preserving domain logic.\n\n"
        f"{_length_instruction(n_items)}\n"
        "References (predicted as target, gold differs):\n\n"
        + "\n".join(blocks)
    )
    return system, user
