"""SetFit synthetic generation — max 3 items per LLM request."""

from __future__ import annotations

import uuid
from collections import defaultdict
from typing import Any, Literal

from loguru import logger

from routers.core.constants import (
    GENERATION_TEMPERATURE,
    SETFIT_MAX_ITEMS_PER_REQUEST,
    SETFIT_SYNTHETIC_CAP,
)
from routers.synthetic.batching import fixed_size_batches
from routers.synthetic.generator import _items_from_batch, _parse_batch, _record_batch
from routers.synthetic.llm_client import LlmBackend, chat_json
from routers.synthetic.prompts_setfit import prompt_for_mode

GenerationMode = Literal[
    "label_failure",
    "diversity_triplet",
    "hard_rewrite",
    "diverse_rewrite",
    "mixed_rewrite",
]


def _batch_size_for_mode(mode: str) -> int:
    if mode == "diversity_triplet":
        return 3
    if mode in ("hard_rewrite", "diverse_rewrite"):
        return 1
    return SETFIT_MAX_ITEMS_PER_REQUEST


def _batches_for_mode(reference_pool: list[dict], mode: str) -> list[list[dict]]:
    if mode == "diversity_triplet":
        return fixed_size_batches(reference_pool, 3)
    if mode == "mixed_rewrite":
        half = len(reference_pool) // 2
        hard = reference_pool[:half]
        diverse = reference_pool[half:]
        batches = fixed_size_batches(hard, 1) + fixed_size_batches(diverse, 1)
        return batches
    return fixed_size_batches(reference_pool, _batch_size_for_mode(mode))


def generate_from_references(
    reference_pool: list[dict],
    domain_labels: list[str],
    *,
    mode: GenerationMode = "label_failure",
    cap: int = SETFIT_SYNTHETIC_CAP,
    temperature: float = GENERATION_TEMPERATURE,
    stats: dict[str, Any] | None = None,
    backend: LlmBackend = "ollama",
    generation_model: str | None = None,
) -> list[dict]:
    if not reference_pool:
        logger.warning("No reference rows — skipping generation")
        return []

    pool = reference_pool[:cap]
    synthetics: list[dict] = []

    if mode == "label_failure":
        by_gold: dict[str, list[dict]] = defaultdict(list)
        for r in pool:
            by_gold[str(r["gold"])].append(r)
        for gold_label, rows in by_gold.items():
            for batch in fixed_size_batches(rows, SETFIT_MAX_ITEMS_PER_REQUEST):
                synthetics.extend(
                    _run_batch(
                        batch,
                        domain_labels,
                        mode="label_failure",
                        gold_label=gold_label,
                        temperature=temperature,
                        stats=stats,
                        backend=backend,
                        generation_model=generation_model,
                    )
                )
    elif mode == "mixed_rewrite":
        half = len(pool) // 2
        hard_refs = pool[:half]
        diverse_refs = pool[half:]
        for ref in hard_refs:
            synthetics.extend(
                _run_batch(
                    [ref],
                    domain_labels,
                    mode="hard_rewrite",
                    gold_label=str(ref.get("gold")),
                    temperature=temperature,
                    stats=stats,
                    backend=backend,
                    generation_model=generation_model,
                )
            )
        for ref in diverse_refs:
            synthetics.extend(
                _run_batch(
                    [ref],
                    domain_labels,
                    mode="diverse_rewrite",
                    gold_label=str(ref.get("gold")),
                    temperature=temperature,
                    stats=stats,
                    backend=backend,
                    generation_model=generation_model,
                )
            )
    else:
        for batch in _batches_for_mode(pool, mode):
            if mode == "diversity_triplet" and len(batch) < 3:
                continue
            gold_label = str(batch[0].get("gold", "")) if batch else None
            synthetics.extend(
                _run_batch(
                    batch,
                    domain_labels,
                    mode=mode,
                    gold_label=gold_label,
                    temperature=temperature,
                    stats=stats,
                    backend=backend,
                    generation_model=generation_model,
                )
            )

    synthetics = synthetics[:cap]
    if stats is not None:
        stats["max_items_per_llm_request"] = SETFIT_MAX_ITEMS_PER_REQUEST
        stats["n_synthetic_target"] = cap
        stats["n_llm_calls"] = len(stats.get("generation_batches", []))
    logger.info(
        "Generated {} SetFit synthetic rows (mode={}, backend={}, target={})",
        len(synthetics),
        mode,
        backend,
        cap,
    )
    return synthetics


def _run_batch(
    batch: list[dict],
    domain_labels: list[str],
    *,
    mode: str,
    gold_label: str | None,
    temperature: float,
    stats: dict[str, Any] | None,
    backend: LlmBackend,
    generation_model: str | None,
) -> list[dict]:
    n = len(batch)
    system, user = prompt_for_mode(mode, batch, domain_labels, target_domain=gold_label)

    def _pf(text: str, _n=n) -> Any:
        return _parse_batch(text, _n, domain_labels)

    chat = chat_json(
        system,
        user,
        temperature=temperature,
        parse_fn=_pf,
        backend=backend,
        model=generation_model,
        skip_on_failure=True,
    )
    record_extra = {
        "generation_mode": mode,
        "llm_backend": backend,
        "generation_model": generation_model,
        "n_requested": n,
    }
    if gold_label:
        record_extra["target_domain"] = gold_label
    _record_batch(
        stats,
        mode=mode,
        system=system,
        user=user,
        chat=chat,
        exemplar_ids=[ex.get("id") for ex in batch],
        extra=record_extra,
    )
    parsed = chat.parsed
    batch_out = parsed if parsed is not None and hasattr(parsed, "items") else None
    rows = _items_from_batch(
        batch_out,
        batch,
        gold_label=gold_label if mode == "label_failure" else None,
        mode=mode,
        stats=stats,
    )
    for row in rows:
        row["llm_backend"] = backend
        if generation_model:
            row["generation_model"] = generation_model
    return rows
