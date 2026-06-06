"""Prompts for smart experiment group."""

from __future__ import annotations

import json
from typing import Any

from routers.synthetic.prompts import _labels_block, _length_instruction, format_row_for_llm
from routers.synthetic.prompts_combination import HARD_NEGATIVE_FEWSHOT, build_hard_negative_pair_prompt


def half_negative_rules(gold: str, pred: str) -> str:
    return (
        "Half-negative rules:\n"
        f"- Item 1 must be unambiguously domain **{gold}**.\n"
        f"- Item 2 must be unambiguously domain **{pred}**.\n"
        "- Use similar general topic wording but contrasting technical vocabulary.\n"
        "- Before JSON, list exactly 3 discriminative cues between the domains.\n"
    )


def build_hard_negative_enhanced_prompt(
    failure_row: dict,
    domain_labels: list[str],
    *,
    n_items: int = 2,
    include_cot: bool = False,
) -> tuple[str, str]:
    gold = str(failure_row.get("gold", ""))
    pred = str(failure_row.get("pred", ""))
    system, user = build_hard_negative_pair_prompt(failure_row, domain_labels, n_items=n_items)
    extra = half_negative_rules(gold, pred)
    if include_cot:
        user = extra + "\n" + user
    else:
        user = user.replace(
            "Use similar general topic words",
            extra.split("\n")[0] + "\nUse similar general topic words",
            1,
        )
    return system, user


def build_contrastive_draft_prompt(
    failure_row: dict,
    domain_labels: list[str],
) -> tuple[str, str]:
    system, user = build_hard_negative_enhanced_prompt(
        failure_row, domain_labels, n_items=2, include_cot=True
    )
    return system, user


def build_contrastive_critique_prompt(
    failure_row: dict,
    draft_json: str,
) -> tuple[str, str]:
    gold = str(failure_row.get("gold", ""))
    pred = str(failure_row.get("pred", ""))
    system = (
        "You critique synthetic domain-classification training pairs.\n"
        "Return ONLY valid JSON with these keys:\n"
        '{"bullet_label_alignment":"...","bullet_distractor_realism":"...",'
        '"bullet_vocabulary_contrast":"...","bullet_mcq_format":"...",'
        '"bullet_ambiguity_risk":"...","thought":"...","next_steps":"..."}\n'
    )
    user = (
        f"Gold domain: {gold}\nPredicted (confusion) domain: {pred}\n\n"
        "Evaluate the DRAFT pair against 5 bullets:\n"
        "1. Label alignment — each item's domain field matches its content.\n"
        "2. Distractor realism — wrong options use plausible industry terms but are definitively wrong.\n"
        "3. Vocabulary contrast — shell similar, technical terms separate domains.\n"
        "4. MCQ format — valid context, question, pipe-separated options.\n"
        "5. Ambiguity risk — could another domain also be correct?\n\n"
        f"Draft JSON:\n{draft_json}\n"
    )
    return system, user


def build_contrastive_refine_prompt(
    failure_row: dict,
    domain_labels: list[str],
    draft_json: str,
    critique_json: str,
) -> tuple[str, str]:
    gold = str(failure_row.get("gold", ""))
    pred = str(failure_row.get("pred", ""))
    system = (
        "You refine synthetic RouterArena training examples.\n"
        f"{_labels_block(domain_labels)}\n"
        "Return ONLY valid JSON:\n"
        '{"items": [{"context":"...","question":"...","options":"...","domain":"<label>"}]}\n'
    )
    user = (
        f"Produce exactly 2 final items (one **{gold}**, one **{pred}**) using draft + critique.\n"
        f"{HARD_NEGATIVE_FEWSHOT}\n\n"
        f"Draft:\n{draft_json}\n\nCritique:\n{critique_json}\n\n"
        f"{_length_instruction(2)}\n"
        "Misclassified reference:\n\n"
        + format_row_for_llm(failure_row)
    )
    return system, user


def build_skip_gate_prompt(
    failure_row: dict,
    domain_labels: list[str],
) -> tuple[str, str]:
    gold = str(failure_row.get("gold", ""))
    pred = str(failure_row.get("pred", ""))
    system = (
        "You audit RouterArena training labels before synthetic generation.\n"
        f"{_labels_block(domain_labels)}\n"
        "Return ONLY valid JSON:\n"
        '{"action":"SKIP"|"GENERATE","reason":"...","items":[...]}\n'
        'Each item must use key "domain" (not gold_domain): '
        '{"context":"...","question":"...","options":"...","domain":"<label>"}\n'
        "If GENERATE, items must be exactly 2 (one gold, one pred domain).\n"
    )
    user = (
        f"Primary rule: gold **{gold}** is the single correct domain without ambiguity.\n"
        f"If the reference could reasonably belong to another domain, output "
        f'{{"action":"SKIP","reason":"..."}} immediately.\n'
        f"Otherwise output "
        f'{{"action":"GENERATE","items":[...]}} with one **{gold}** and one **{pred}** item.\n'
        f"{HARD_NEGATIVE_FEWSHOT}\n\n"
        f"{half_negative_rules(gold, pred)}\n"
        f"{_length_instruction(2)}\n"
        "Reference:\n\n"
        + format_row_for_llm(failure_row)
    )
    return system, user


def build_parallel_hard_negative_gen_prompt(
    refs: list[dict],
    domain_labels: list[str],
    *,
    n_pairs: int = 2,
) -> tuple[str, str]:
    system = (
        "You generate formal RouterArena hard-negative PAIRS for domain classification.\n"
        f"{_labels_block(domain_labels)}\n"
        "Return ONLY valid JSON:\n"
        '{"pairs": [{"pair_id":"pair_1","items":[{"context":"...","question":"...","options":"...","domain":"..."},'
        '{"context":"...","question":"...","options":"...","domain":"..."}]}, ...]}\n'
        "Each pair: item A = gold domain of its reference failure, item B = pred domain.\n"
    )
    blocks = []
    for i, r in enumerate(refs, 1):
        blocks.append(
            f"Reference {i}: gold={r.get('gold')} pred={r.get('pred')}\n"
            + format_row_for_llm(r, i)
        )
    user = (
        f"From the {len(refs)} misclassified references below, generate exactly {n_pairs} PAIRS "
        f"({n_pairs * 2} items total). Each pair_id must be pair_1, pair_2, ...\n"
        f"{HARD_NEGATIVE_FEWSHOT}\n\n"
        + "\n".join(blocks)
    )
    return system, user


def build_pair_judge_prompt(
    pairs_payload: list[dict],
    domain_a: str,
    domain_b: str,
) -> tuple[str, str]:
    system = (
        "You judge PAIRS of synthetic multiple-choice questions for domain routing.\n"
        "Return ONLY valid JSON:\n"
        '{"winner_pair_id":"<id>","rejected_pair_ids":["..."],"rationale":"<one sentence>"}\n'
    )
    user = (
        f"Domains in play: **{domain_a}** vs **{domain_b}**.\n\n"
        "You are judging PAIRS of questions. To win, Question A and Question B in the pair must:\n"
        "- Use highly similar contexts / general wording (same topic shell).\n"
        "- Use technical distractors that prove flawless separation between the two domains.\n"
        "- Both be high quality: if Question A is great but Question B is sloppy, REJECT the pair.\n"
        "- Have plausible but definitively wrong distractors.\n"
        "- Be valid MCQs (context, question, pipe-separated options).\n\n"
        "Candidates:\n"
        + json.dumps(pairs_payload, indent=2, ensure_ascii=False)
        + "\n\nPick exactly one winner_pair_id."
    )
    return system, user


def build_diversity_path_prompt(
    refs: list[dict],
    domain_labels: list[str],
    path: str,
    confusing_class: str,
) -> tuple[str, str]:
    if not refs:
        raise ValueError("diversity path requires at least 1 reference")
    n = len(refs)
    system = (
        "You generate RouterArena training examples.\n"
        f"{_labels_block(domain_labels)}\n"
        f'Return ONLY valid JSON: {{"items": [{{"id":"q1",...}}, ...]}}\n'
        f"Include exactly {n} items with ids q1..q{n} in order.\n"
    )
    blocks = [format_row_for_llm(r, i + 1) for i, r in enumerate(refs)]
    if path == "A":
        instruction = (
            "Path A (Complexity): Rewrite to be twice as long, adding complex, "
            "postgraduate-level terminology from the gold domain."
        )
    elif path == "B":
        instruction = (
            "Path B (Scenario): Keep core logic, but completely change the real-world scenario "
            "or industry framing."
        )
    else:
        instruction = (
            f"Path C (Distractor Hardening): Keep Question and Context like the reference gold class. "
            f"Rewrite Options so incorrect distractors use technical keywords from neighbor class "
            f"**{confusing_class}**, forcing discrimination from anchoring text."
        )
    user = instruction + "\n\n" + _length_instruction(n) + "\n\nReferences:\n\n" + "\n".join(blocks)
    return system, user


def build_diversity_judge_prompt(candidates: list[dict], *, n_pick: int = 3) -> tuple[str, str]:
    system = (
        "You select training questions by objective domain-depth criteria.\n"
        f'Return ONLY valid JSON: {{"winner_ids":[...],"rationale":"..."}} with exactly {n_pick} ids.\n'
    )
    user = (
        "Here are synthetic multiple-choice questions (each has an id).\n"
        f"Select exactly {n_pick} winner_ids that require the deepest domain-specific knowledge to solve.\n"
        "Do not use vague terms like 'informative'. Prefer precise technical discrimination.\n\n"
        + json.dumps(candidates, indent=2, ensure_ascii=False)
    )
    return system, user


def build_expansion_path_prompt(
    refs: list[dict],
    domain_labels: list[str],
    path: str,
) -> tuple[str, str]:
    n = len(refs)
    system = (
        "You generate RouterArena training examples.\n"
        f"{_labels_block(domain_labels)}\n"
        "Return ONLY valid JSON with an items array.\n"
    )
    blocks = [format_row_for_llm(r, i + 1) for i, r in enumerate(refs)]
    if path == "A":
        note = "Make each item harder and more complex (postgraduate terminology), same gold domain."
    elif path == "B":
        note = "Low-confidence style: clarify gold domain with contrasting neighbor vocabulary, same gold."
    else:
        note = (
            "Hard-negative style per reference: if mislabeled, one item gold + one item pred domain; "
            "else two items sharpening gold vs confusing neighbor."
        )
    user = (
        f"{note}\nGenerate exactly {n} items (one per reference, same order).\n"
        f"{_length_instruction(n)}\n\nReferences:\n\n"
        + "\n".join(blocks)
    )
    return system, user
