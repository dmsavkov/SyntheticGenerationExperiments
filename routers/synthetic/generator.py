"""Synthetic sample generation (label_failure + confusion_pair modes)."""

from __future__ import annotations

import random
import uuid
from collections import Counter, defaultdict
from typing import Any

from loguru import logger
from tqdm import tqdm

from routers.core.constants import GENERATION_TEMPERATURE, SYNTHETIC_CAP_DEFAULT
from routers.synthetic.batching import dynamic_batches
from routers.synthetic.ollama_client import ChatResult, chat_json
from routers.synthetic.parse import SyntheticBatch, parse_synthetic_batch, parse_validation_batch
from routers.synthetic.prompts import (
    build_confusion_pair_prompt,
    build_label_failure_prompt,
    build_validator_prompt,
    synthetic_item_to_train_row,
    validation_items_to_map,
)


def _parse_batch(text: str, n: int, labels: list[str]) -> SyntheticBatch:
    return parse_synthetic_batch(text, n_expected=n, domain_labels=labels)


def _llm_item_dict(item: Any) -> dict[str, str]:
    return {
        "context": item.context,
        "question": item.question,
        "options": item.options,
        "domain": item.domain,
    }


def _record_batch(
    stats: dict[str, Any] | None,
    *,
    mode: str,
    system: str,
    user: str,
    chat: ChatResult,
    exemplar_ids: list[Any],
    extra: dict[str, Any] | None = None,
) -> None:
    if stats is None:
        return
    batches: list[dict] = stats.setdefault("generation_batches", [])
    parsed_items = None
    if chat.parsed is not None and hasattr(chat.parsed, "items"):
        items = chat.parsed.items
        if items:
            if isinstance(items[0], dict):
                parsed_items = list(items)
            else:
                parsed_items = [_llm_item_dict(it) for it in items]
    record: dict[str, Any] = {
        "mode": mode,
        "success": chat.parsed is not None,
        "attempts": chat.attempts,
        "error": chat.error,
        "system_prompt": system,
        "user_prompt": user,
        "raw_llm_response": chat.raw_response,
        "parsed_items": parsed_items,
        "exemplar_ids": exemplar_ids,
    }
    if extra:
        record.update(extra)
    batches.append(record)


def _items_from_batch(
    batch_out: SyntheticBatch | None,
    batch: list[dict],
    *,
    gold_label: str | None,
    mode: str,
    extra_fields: dict | None = None,
    stats: dict[str, Any] | None = None,
) -> list[dict]:
    if batch_out is None:
        if stats is not None:
            stats["skipped_batches"] = int(stats.get("skipped_batches", 0)) + 1
        return []
    items = batch_out.items
    if not items:
        return []
    refs = list(batch)
    if not refs:
        refs = [{}]
    if len(refs) < len(items):
        refs = refs + [refs[-1]] * (len(items) - len(refs))
    rows: list[dict] = []
    for item, ex in zip(items, refs):
        sid = f"synth_{uuid.uuid4().hex[:12]}"
        llm_generated = _llm_item_dict(item)
        row = synthetic_item_to_train_row(item, sid)
        row["llm_generated"] = llm_generated
        row["generation_mode"] = mode
        row["source_ids"] = [ex.get("id")]
        if gold_label is not None:
            row["target_domain"] = gold_label
            if row["gold"] != gold_label:
                from routers.core.data import build_text_from_parts

                row["gold"] = gold_label
                row["prompt"] = build_text_from_parts(row["context"], row["question"], row["options"])
        if extra_fields:
            row.update(extra_fields)
        rows.append(row)
    return rows


def _run_generation_batch(
    system: str,
    user: str,
    batch: list[dict],
    domain_labels: list[str],
    *,
    mode: str,
    temperature: float,
    stats: dict[str, Any] | None,
    gold_label: str | None = None,
    extra_fields: dict | None = None,
    record_extra: dict | None = None,
) -> list[dict]:
    n = len(batch)

    def _pf(text: str, _n=n) -> SyntheticBatch:
        return _parse_batch(text, _n, domain_labels)

    chat = chat_json(system, user, temperature=temperature, parse_fn=_pf, skip_on_failure=True)
    _record_batch(
        stats,
        mode=mode,
        system=system,
        user=user,
        chat=chat,
        exemplar_ids=[ex.get("id") for ex in batch],
        extra=record_extra,
    )
    return _items_from_batch(
        chat.parsed if isinstance(chat.parsed, SyntheticBatch) else None,
        batch,
        gold_label=gold_label,
        mode=mode,
        extra_fields=extra_fields,
        stats=stats,
    )


def generate_label_failure(
    failure_rows: list[dict],
    domain_labels: list[str],
    *,
    cap: int | None = None,
    temperature: float = GENERATION_TEMPERATURE,
    stats: dict[str, Any] | None = None,
) -> list[dict]:
    if not failure_rows:
        logger.warning("No failure rows — skipping generation")
        return []
    cap = cap if cap is not None else min(len(failure_rows), SYNTHETIC_CAP_DEFAULT)
    pool = failure_rows[:cap]
    by_gold: dict[str, list[dict]] = defaultdict(list)
    for r in pool:
        by_gold[str(r["gold"])].append(r)

    synthetics: list[dict] = []
    for gold_label, rows in by_gold.items():
        for batch in dynamic_batches(rows):
            system, user = build_label_failure_prompt(batch, domain_labels, gold_label)
            synthetics.extend(
                _run_generation_batch(
                    system,
                    user,
                    batch,
                    domain_labels,
                    mode="label_failure",
                    temperature=temperature,
                    stats=stats,
                    gold_label=gold_label,
                    record_extra={"target_domain": gold_label},
                )
            )
    logger.info(
        "Generated {} synthetic rows (label_failure), skipped_batches={}",
        len(synthetics),
        (stats or {}).get("skipped_batches", 0),
    )
    return synthetics


def generate_confusion_pair_boundary(
    failure_rows: list[dict],
    train_rows: list[dict],
    domain_labels: list[str],
    *,
    cap: int | None = None,
    min_pair_errors: int = 3,
    temperature: float = GENERATION_TEMPERATURE,
    seed: int = 42,
    stats: dict[str, Any] | None = None,
) -> list[dict]:
    if not failure_rows:
        return []
    cap = cap if cap is not None else min(len(failure_rows), SYNTHETIC_CAP_DEFAULT)
    pool = failure_rows[:cap]
    by_pair: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in pool:
        pair = (str(r["gold"]), str(r["pred"]))
        by_pair[pair].append(r)

    rng = random.Random(seed)
    by_domain_train: dict[str, list[dict]] = defaultdict(list)
    for r in train_rows:
        by_domain_train[str(r["gold"])].append(r)

    synthetics: list[dict] = []
    pairs = [(p, rs) for p, rs in by_pair.items() if len(rs) >= min_pair_errors]
    for (gold_a, gold_b), fail_rows in tqdm(pairs, desc="confusion_pairs"):
        if len(by_domain_train[gold_a]) < 1 or len(by_domain_train[gold_b]) < 1:
            logger.warning("Skipping pair {} vs {} — insufficient few-shot", gold_a, gold_b)
            continue
        few_a = rng.sample(by_domain_train[gold_a], min(5, len(by_domain_train[gold_a])))
        few_b = rng.sample(by_domain_train[gold_b], min(5, len(by_domain_train[gold_b])))
        for batch in dynamic_batches(fail_rows):
            system, user = build_confusion_pair_prompt(
                (gold_a, gold_b), batch, few_a, few_b, domain_labels, len(batch)
            )
            synthetics.extend(
                _run_generation_batch(
                    system,
                    user,
                    batch,
                    domain_labels,
                    mode="confusion_pair_boundary",
                    temperature=temperature,
                    stats=stats,
                    extra_fields={"confusion_pair": [gold_a, gold_b]},
                    record_extra={"confusion_pair": [gold_a, gold_b]},
                )
            )
    logger.info("Generated {} synthetic rows (confusion_pair)", len(synthetics))
    return synthetics


def allocate_proportional(total: int, failure_rows: list[dict]) -> dict[str, int]:
    counts = Counter(str(r["gold"]) for r in failure_rows)
    if not counts:
        return {}
    keys = list(counts.keys())
    raw = {k: total * counts[k] / sum(counts.values()) for k in keys}
    allocated = {k: int(raw[k]) for k in keys}
    remainder = total - sum(allocated.values())
    for k in sorted(keys, key=lambda x: raw[x] - allocated[x], reverse=True)[:remainder]:
        allocated[k] += 1
    return allocated


def generate_proportional_label_failure(
    failure_rows: list[dict],
    domain_labels: list[str],
    total_synthetic: int = 500,
    *,
    temperature: float = GENERATION_TEMPERATURE,
    stats: dict[str, Any] | None = None,
) -> list[dict]:
    if not failure_rows:
        logger.warning("No holdout failures — proportional generation skipped")
        return []
    quota = allocate_proportional(total_synthetic, failure_rows)
    by_gold: dict[str, list[dict]] = defaultdict(list)
    for r in failure_rows:
        by_gold[str(r["gold"])].append(r)

    synthetics: list[dict] = []
    for gold_label, n_need in quota.items():
        if n_need <= 0:
            continue
        rows = by_gold.get(gold_label, [])
        if not rows:
            logger.warning("No failures for domain {!r} — skip quota {}", gold_label, n_need)
            continue
        rng = random.Random(hash(gold_label) % 2**32)
        picked: list[dict] = []
        while len(picked) < n_need:
            picked.extend(rng.sample(rows, min(len(rows), n_need - len(picked))))
        picked = picked[:n_need]
        for batch in dynamic_batches(picked):
            system, user = build_label_failure_prompt(batch, domain_labels, gold_label)
            synthetics.extend(
                _run_generation_batch(
                    system,
                    user,
                    batch,
                    domain_labels,
                    mode="label_failure_proportional",
                    temperature=temperature,
                    stats=stats,
                    gold_label=gold_label,
                    record_extra={"target_domain": gold_label, "quota": n_need},
                )
            )
    synthetics = synthetics[:total_synthetic]
    logger.info("Generated {} proportional synthetic rows (target {})", len(synthetics), total_synthetic)
    return synthetics


def validate_synthetics(
    synthetics: list[dict],
    domain_labels: list[str],
    *,
    temperature: float | None = None,
    stats: dict[str, Any] | None = None,
) -> list[dict]:
    from routers.core.constants import VALIDATOR_TEMPERATURE

    temperature = VALIDATOR_TEMPERATURE if temperature is None else temperature
    results: list[dict] = []
    batch_size = 10
    for start in range(0, len(synthetics), batch_size):
        chunk = synthetics[start : start + batch_size]
        system, user = build_validator_prompt(chunk, domain_labels)
        n = len(chunk)

        def _pf(text: str, _n=n) -> Any:
            return parse_validation_batch(text, n_expected=_n)

        chat = chat_json(system, user, temperature=temperature, parse_fn=_pf, skip_on_failure=True)
        _record_batch(
            stats,
            mode="validator",
            system=system,
            user=user,
            chat=chat,
            exemplar_ids=[syn["id"] for syn in chunk],
        )
        if chat.parsed is None:
            if stats is not None:
                stats["skipped_validator_batches"] = int(stats.get("skipped_validator_batches", 0)) + 1
            for syn in chunk:
                results.append(
                    {
                        "synthetic_id": syn["id"],
                        "gold_domain": syn.get("gold"),
                        "validator_verdict": "NO",
                        "validator_reason": "validator_batch_failed",
                        "validator_raw_response": chat.raw_response,
                    }
                )
            continue
        vbatch = chat.parsed
        vmap = validation_items_to_map(vbatch.items)
        for syn in chunk:
            vid = syn["id"]
            vi = vmap.get(vid)
            results.append(
                {
                    "synthetic_id": vid,
                    "gold_domain": syn.get("gold"),
                    "validator_verdict": vi.verdict if vi else "NO",
                    "validator_reason": vi.reason if vi else "missing",
                    "validator_raw_response": chat.raw_response,
                }
            )
    return results
