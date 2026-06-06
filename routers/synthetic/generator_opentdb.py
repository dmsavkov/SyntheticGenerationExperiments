"""Gemini generation for ensemble/combination (up to 8 items per request)."""

from __future__ import annotations

from typing import Any

from loguru import logger

from routers.core.constants import (
    COMBINATION_GOOGLE_THINKING_LEVEL,
    ENSEMBLE_MAX_ITEMS_PER_REQUEST,
    GENERATION_TEMPERATURE,
)
from routers.synthetic.batching import fixed_size_batches
from routers.synthetic.generator import _items_from_batch, _parse_batch, _record_batch
from routers.synthetic.llm_client import LlmBackend, chat_json
from routers.synthetic.prompts import build_label_failure_prompt
from routers.synthetic.prompts_combination import build_hard_negative_pair_prompt
from routers.synthetic.prompts_ensemble import build_ovr_mutation_prompt


def generate_label_failure_opentdb(
    reference_pool: list[dict],
    domain_labels: list[str],
    *,
    cap: int,
    stats: dict[str, Any] | None = None,
    generation_model: str | None = None,
    max_per_request: int = ENSEMBLE_MAX_ITEMS_PER_REQUEST,
) -> list[dict]:
    from collections import defaultdict

    if not reference_pool:
        return []
    synthetics: list[dict] = []
    by_gold: dict[str, list[dict]] = defaultdict(list)
    for r in reference_pool[:cap]:
        by_gold[str(r["gold"])].append(r)
    for gold_label, rows in by_gold.items():
        for batch in fixed_size_batches(rows, max_per_request):
            n = len(batch)
            system, user = build_label_failure_prompt(batch, domain_labels, gold_label)

            def _pf(text: str, _n=n) -> Any:
                return _parse_batch(text, _n, domain_labels)

            chat = chat_json(
                system,
                user,
                temperature=GENERATION_TEMPERATURE,
                parse_fn=_pf,
                backend="google",
                model=generation_model,
                skip_on_failure=True,
                thinking_level=COMBINATION_GOOGLE_THINKING_LEVEL,
            )
            _record_batch(
                stats,
                mode="label_failure",
                system=system,
                user=user,
                chat=chat,
                exemplar_ids=[ex.get("id") for ex in batch],
                extra={"target_domain": gold_label, "llm_backend": "google"},
            )
            parsed = chat.parsed
            batch_out = parsed if parsed is not None and hasattr(parsed, "items") else None
            synthetics.extend(
                _items_from_batch(
                    batch_out,
                    batch,
                    gold_label=gold_label,
                    mode="label_failure",
                    stats=stats,
                )
            )
            if len(synthetics) >= cap:
                break
        if len(synthetics) >= cap:
            break
    return synthetics[:cap]


def generate_hard_negative_pairs(
    failures: list[dict],
    domain_labels: list[str],
    *,
    stats: dict[str, Any] | None = None,
    generation_model: str | None = None,
    items_per_failure: int = 2,
) -> list[dict]:
    synthetics: list[dict] = []
    for fail in failures:
        system, user = build_hard_negative_pair_prompt(
            fail, domain_labels, n_items=items_per_failure
        )

        def _pf(text: str, _n=items_per_failure) -> Any:
            return _parse_batch(text, _n, domain_labels)

        chat = chat_json(
            system,
            user,
            temperature=GENERATION_TEMPERATURE,
            parse_fn=_pf,
            backend="google",
            model=generation_model,
            skip_on_failure=True,
            thinking_level=COMBINATION_GOOGLE_THINKING_LEVEL,
        )
        _record_batch(
            stats,
            mode="hard_negative_pair",
            system=system,
            user=user,
            chat=chat,
            exemplar_ids=[fail.get("id")],
            extra={"gold": fail.get("gold"), "pred": fail.get("pred")},
        )
        parsed = chat.parsed
        batch_out = parsed if parsed is not None and hasattr(parsed, "items") else None
        rows = _items_from_batch(
            batch_out,
            [fail] * items_per_failure,
            gold_label=None,
            mode="hard_negative_pair",
            stats=stats,
        )
        for row in rows:
            row["source_ids"] = [fail.get("id")]
            row["confusion_pair"] = [str(fail.get("gold")), str(fail.get("pred"))]
        synthetics.extend(rows)
    logger.info("Generated {} hard-negative synthetics from {} failures", len(synthetics), len(failures))
    return synthetics


def generate_ovr_mutation(
    exemplars: list[dict],
    domain_labels: list[str],
    target_domain: str,
    *,
    cap: int,
    stats: dict[str, Any] | None = None,
    generation_model: str | None = None,
    max_per_request: int = ENSEMBLE_MAX_ITEMS_PER_REQUEST,
) -> list[dict]:
    synthetics: list[dict] = []
    for batch in fixed_size_batches(exemplars[:cap], max_per_request):
        n = len(batch)
        system, user = build_ovr_mutation_prompt(
            batch, domain_labels, target_domain, n_items=n
        )

        def _pf(text: str, _n=n) -> Any:
            return _parse_batch(text, _n, domain_labels)

        chat = chat_json(
            system,
            user,
            temperature=GENERATION_TEMPERATURE,
            parse_fn=_pf,
            backend="google",
            model=generation_model,
            skip_on_failure=True,
        )
        _record_batch(
            stats,
            mode="ovr_mutation",
            system=system,
            user=user,
            chat=chat,
            exemplar_ids=[ex.get("id") for ex in batch],
            extra={"target_domain": target_domain},
        )
        parsed = chat.parsed
        batch_out = parsed if parsed is not None and hasattr(parsed, "items") else None
        rows = _items_from_batch(
            batch_out,
            batch,
            gold_label=target_domain,
            mode="ovr_mutation",
            stats=stats,
        )
        for row in rows:
            row["target_domain"] = target_domain
        synthetics.extend(rows)
    return synthetics[:cap]


def generate_from_cascade_session(
    session: Any,
    user_prompt: str,
    domain_labels: list[str],
    *,
    n_expected: int,
    stats: dict[str, Any] | None = None,
    mode: str = "cascade",
) -> list[dict]:
    def _pf(text: str) -> Any:
        return _parse_batch(text, n_expected, domain_labels)

    chat = session.complete_json(user_prompt, parse_fn=_pf)
    _record_batch(
        stats,
        mode=mode,
        system="(session)",
        user=user_prompt,
        chat=chat,
        exemplar_ids=[],
        extra={"n_expected": n_expected},
    )
    parsed = chat.parsed
    batch_out = parsed if parsed is not None and hasattr(parsed, "items") else None
    return _items_from_batch(
        batch_out, [], gold_label=None, mode=mode, stats=stats
    )[:n_expected]
