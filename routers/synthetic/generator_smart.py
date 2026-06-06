"""Smart experiment synthetic generation orchestration."""

from __future__ import annotations

import json
import uuid
from typing import Any

from loguru import logger

from routers.core.constants import (
    COMBINATION_GOOGLE_THINKING_LEVEL,
    ENSEMBLE_MAX_ITEMS_PER_REQUEST,
    GENERATION_TEMPERATURE,
    GOOGLE_FLASH_MODEL_DEFAULT,
    SMART_CREATIVE_TEMPERATURE,
    SMART_CREATIVE_TOP_P,
    SMART_PARSE_MAX_RETRIES,
    SMART_PAIRS_PER_STREAM,
    SMART_PARALLEL_STREAMS,
    SMART_SYNTHETIC_CAP,
)
from routers.synthetic.async_llm import LlmRequest, chat_json_parallel
from routers.synthetic.generator import _items_from_batch, _parse_batch, _record_batch
from routers.synthetic.llm_client import chat_json
from routers.synthetic.parse import parse_synthetic_batch
from routers.synthetic.parse_smart import (
    parse_critique,
    parse_judge_pair_pick,
    parse_judge_question_pick,
    parse_pairs_batch,
    parse_skip_or_generate,
)
from routers.synthetic.prompts_smart import (
    build_contrastive_critique_prompt,
    build_contrastive_draft_prompt,
    build_contrastive_refine_prompt,
    build_diversity_judge_prompt,
    build_diversity_path_prompt,
    build_parallel_hard_negative_gen_prompt,
    build_pair_judge_prompt,
    build_skip_gate_prompt,
)
from routers.synthetic.smart_selection import neighbor_class_for_row


def _llm_kw(
    *,
    creative: bool = False,
    thinking: bool = True,
) -> dict[str, Any]:
    if creative:
        return {
            "temperature": SMART_CREATIVE_TEMPERATURE,
            "top_p": SMART_CREATIVE_TOP_P,
            "thinking_level": None,
            "max_retries": SMART_PARSE_MAX_RETRIES,
        }
    return {
        "temperature": GENERATION_TEMPERATURE,
        "top_p": None,
        "thinking_level": COMBINATION_GOOGLE_THINKING_LEVEL if thinking else None,
        "max_retries": SMART_PARSE_MAX_RETRIES,
    }


def _rows_from_items(
    batch_out: Any,
    refs: list[dict],
    *,
    mode: str,
    stats: dict | None,
    gold_label: str | None = None,
) -> list[dict]:
    parsed = batch_out if batch_out is not None and hasattr(batch_out, "items") else None
    rows = _items_from_batch(parsed, refs, gold_label=gold_label, mode=mode, stats=stats)
    return rows


def generate_hard_negative_pairs_smart(
    failures: list[dict],
    domain_labels: list[str],
    *,
    stats: dict[str, Any] | None = None,
    generation_model: str | None = None,
    items_per_failure: int = 2,
    include_cot: bool = False,
) -> list[dict]:
    from routers.synthetic.prompts_smart import build_hard_negative_enhanced_prompt

    synthetics: list[dict] = []
    kw = _llm_kw(creative=False)
    for fail in failures:
        system, user = build_hard_negative_enhanced_prompt(
            fail, domain_labels, n_items=items_per_failure, include_cot=include_cot
        )

        def _pf(text: str, _n=items_per_failure) -> Any:
            return _parse_batch(text, _n, domain_labels)

        chat = chat_json(
            system,
            user,
            parse_fn=_pf,
            backend="google",
            model=generation_model or GOOGLE_FLASH_MODEL_DEFAULT,
            skip_on_failure=True,
            **kw,
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
        batch_out = chat.parsed
        rows = _rows_from_items(
            batch_out, [fail] * items_per_failure, mode="hard_negative_pair", stats=stats
        )
        for row in rows:
            row["source_ids"] = [fail.get("id")]
            row["confusion_pair"] = [str(fail.get("gold")), str(fail.get("pred"))]
        synthetics.extend(rows)
    logger.info("Smart hard-negative: {} items from {} failures", len(synthetics), len(failures))
    return synthetics


def generate_contrastive_pipeline(
    failures: list[dict],
    domain_labels: list[str],
    *,
    stats: dict[str, Any] | None = None,
    generation_model: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """3-step per failure: draft → critique → refine. Returns (synthetics, traces)."""
    synthetics: list[dict] = []
    traces: list[dict] = []
    kw = _llm_kw(creative=False)
    model = generation_model or GOOGLE_FLASH_MODEL_DEFAULT

    for fail in failures:
        trace: dict[str, Any] = {"source_id": fail.get("id"), "gold": fail.get("gold"), "pred": fail.get("pred")}
        system, user = build_contrastive_draft_prompt(fail, domain_labels)

        def _pf_draft(text: str) -> Any:
            return _parse_batch(text, 2, domain_labels)

        draft_chat = chat_json(
            system, user, parse_fn=_pf_draft, backend="google", model=model, skip_on_failure=True, **kw
        )
        _record_batch(stats, mode="contrastive_draft", system=system, user=user, chat=draft_chat, exemplar_ids=[fail.get("id")])
        if draft_chat.parsed is None:
            trace["failed_at"] = "draft"
            traces.append(trace)
            continue
        draft_json = json.dumps(
            {"items": [{"context": i.context, "question": i.question, "options": i.options, "domain": i.domain} for i in draft_chat.parsed.items]},
            ensure_ascii=False,
        )
        trace["draft"] = draft_json

        sys_c, usr_c = build_contrastive_critique_prompt(fail, draft_json)
        crit_chat = chat_json(
            sys_c, usr_c, parse_fn=parse_critique, backend="google", model=model, skip_on_failure=True, **kw
        )
        _record_batch(stats, mode="contrastive_critique", system=sys_c, user=usr_c, chat=crit_chat, exemplar_ids=[fail.get("id")])
        if crit_chat.parsed is None:
            trace["failed_at"] = "critique"
            traces.append(trace)
            continue
        critique_json = crit_chat.parsed.model_dump_json()
        trace["critique"] = critique_json

        sys_r, usr_r = build_contrastive_refine_prompt(fail, domain_labels, draft_json, critique_json)

        def _pf_refine(text: str) -> Any:
            return _parse_batch(text, 2, domain_labels)

        ref_chat = chat_json(
            sys_r, usr_r, parse_fn=_pf_refine, backend="google", model=model, skip_on_failure=True, **kw
        )
        _record_batch(stats, mode="contrastive_refine", system=sys_r, user=usr_r, chat=ref_chat, exemplar_ids=[fail.get("id")])
        if ref_chat.parsed is not None:
            rows = _rows_from_items(ref_chat.parsed, [fail, fail], mode="contrastive_refine", stats=stats)
            for row in rows:
                row["source_ids"] = [fail.get("id")]
                row["confusion_pair"] = [str(fail.get("gold")), str(fail.get("pred"))]
            synthetics.extend(rows)
            trace["success"] = True
        else:
            trace["failed_at"] = "refine"
        traces.append(trace)

    logger.info("Contrastive pipeline: {} items, {} traces", len(synthetics), len(traces))
    return synthetics, traces


def generate_hard_negative_skip(
    failures: list[dict],
    domain_labels: list[str],
    *,
    stats: dict[str, Any] | None = None,
    generation_model: str | None = None,
) -> tuple[list[dict], list[dict]]:
    synthetics: list[dict] = []
    skip_log: list[dict] = []
    kw = _llm_kw(creative=False)
    model = generation_model or GOOGLE_FLASH_MODEL_DEFAULT

    for fail in failures:
        system, user = build_skip_gate_prompt(fail, domain_labels)
        chat = chat_json(
            system,
            user,
            parse_fn=parse_skip_or_generate,
            backend="google",
            model=model,
            skip_on_failure=True,
            **kw,
        )
        _record_batch(stats, mode="hard_negative_skip", system=system, user=user, chat=chat, exemplar_ids=[fail.get("id")])
        if chat.parsed is None:
            skip_log.append({"id": fail.get("id"), "action": "ERROR", "reason": chat.error})
            continue
        out = chat.parsed
        if out.action == "SKIP":
            skip_log.append({"id": fail.get("id"), "action": "SKIP", "reason": out.reason})
            continue
        if not out.items:
            skip_log.append({"id": fail.get("id"), "action": "SKIP", "reason": "empty items"})
            continue
        n_expected = len(out.items)
        batch = parse_synthetic_batch(
            json.dumps({"items": out.items}), n_expected=n_expected, domain_labels=domain_labels
        )
        rows = _rows_from_items(batch, [fail] * len(out.items), mode="hard_negative_skip", stats=stats)
        for row in rows:
            row["source_ids"] = [fail.get("id")]
            row["confusion_pair"] = [str(fail.get("gold")), str(fail.get("pred"))]
        synthetics.extend(rows)
        skip_log.append({"id": fail.get("id"), "action": "GENERATE", "n_items": len(rows)})

    logger.info("Skip gate: {} generated, {} skip/error log entries", len(synthetics), len(skip_log))
    return synthetics, skip_log


def generate_parallel_hard_negative_judge_for_failure(
    fail: dict,
    domain_labels: list[str],
    *,
    stats: dict[str, Any] | None = None,
    generation_model: str | None = None,
) -> tuple[list[dict], dict[str, Any]]:
    """One mislabeled ref: 3 parallel gens (2 pairs each) → 1 pair judge → 2 items."""
    refs = [fail]
    domain_a = str(fail.get("gold", ""))
    domain_b = str(fail.get("pred", ""))
    kw = _llm_kw(creative=True)
    model = generation_model or GOOGLE_FLASH_MODEL_DEFAULT
    log_entry: dict[str, Any] = {"source_id": fail.get("id"), "gold": domain_a, "pred": domain_b}

    gen_requests: list[LlmRequest] = []
    for i in range(SMART_PARALLEL_STREAMS):
        system, user = build_parallel_hard_negative_gen_prompt(
            refs, domain_labels, n_pairs=SMART_PAIRS_PER_STREAM
        )
        gen_requests.append(
            LlmRequest(
                system=system,
                user=user,
                parse_fn=parse_pairs_batch,
                request_id=f"gen_stream_{i}",
                model=model,
                **kw,
            )
        )

    gen_results = chat_json_parallel(gen_requests)
    all_pairs: list[dict] = []
    for stream_id, chat in gen_results.items():
        _record_batch(
            stats,
            mode="parallel_hard_negative_gen",
            system="(parallel)",
            user=f"stream={stream_id}",
            chat=chat,
            exemplar_ids=[fail.get("id")],
            extra={"stream": stream_id},
        )
        if chat.parsed is None:
            continue
        for p in chat.parsed.pairs:
            all_pairs.append({"pair_id": f"{p.pair_id}_{stream_id}", "items": p.items})

    if not all_pairs:
        log_entry["error"] = "no pairs generated"
        return [], log_entry

    system_j, user_j = build_pair_judge_prompt(all_pairs, domain_a, domain_b)
    judge_chat = chat_json(
        system_j,
        user_j,
        parse_fn=parse_judge_pair_pick,
        backend="google",
        model=model,
        skip_on_failure=True,
        **kw,
    )
    _record_batch(
        stats,
        mode="pair_judge",
        system=system_j,
        user=user_j,
        chat=judge_chat,
        exemplar_ids=[fail.get("id")],
    )
    if judge_chat.parsed is None:
        log_entry["error"] = judge_chat.error
        return [], log_entry

    winner = judge_chat.parsed.winner_pair_id
    log_entry["winner_pair_id"] = winner
    log_entry["rationale"] = judge_chat.parsed.rationale
    accepted: list[dict] = []
    for pair in all_pairs:
        if pair.get("pair_id") != winner:
            continue
        for it in pair.get("items", []):
            sid = f"synth_{uuid.uuid4().hex[:12]}"
            from routers.synthetic.prompts import synthetic_item_to_train_row

            batch = parse_synthetic_batch(
                json.dumps({"items": [it]}), n_expected=1, domain_labels=domain_labels
            )
            row = synthetic_item_to_train_row(batch.items[0], sid)
            row["generation_mode"] = "parallel_judge"
            row["pair_id"] = winner
            row["source_ids"] = [fail.get("id")]
            row["confusion_pair"] = [domain_a, domain_b]
            accepted.append(row)
    return accepted, log_entry


def generate_parallel_hard_negative_judge_pool(
    failures: list[dict],
    domain_labels: list[str],
    *,
    stats: dict[str, Any] | None = None,
    generation_model: str | None = None,
    judge_log: list[dict] | None = None,
) -> list[dict]:
    from tqdm import tqdm

    all_rows: list[dict] = []
    jlog = judge_log if judge_log is not None else []
    for fail in tqdm(failures, desc="smart06 parallel+judge"):
        rows, entry = generate_parallel_hard_negative_judge_for_failure(
            fail, domain_labels, stats=stats, generation_model=generation_model
        )
        all_rows.extend(rows)
        jlog.append(entry)
    logger.info("Parallel+judge pool: {} items from {} failures", len(all_rows), len(failures))
    return all_rows


def generate_parallel_hard_negative_judge(
    refs: list[dict],
    domain_labels: list[str],
    *,
    stats: dict[str, Any] | None = None,
    generation_model: str | None = None,
    judge_log: list[dict] | None = None,
) -> list[dict]:
    return generate_parallel_hard_negative_judge_pool(
        refs, domain_labels, stats=stats, generation_model=generation_model, judge_log=judge_log
    )


def generate_diversity_paths_judge_for_ref(
    ref: dict,
    domain_labels: list[str],
    *,
    stats: dict[str, Any] | None = None,
    generation_model: str | None = None,
) -> tuple[list[dict], dict[str, Any]]:
    kw = _llm_kw(creative=True)
    model = generation_model or GOOGLE_FLASH_MODEL_DEFAULT
    confusing = neighbor_class_for_row(ref)
    rid = str(ref.get("id", "ref"))[:12]
    log_entry: dict[str, Any] = {"source_id": ref.get("id"), "gold": ref.get("gold")}

    gen_requests: list[LlmRequest] = []
    for path in ("A", "B", "C"):
        system, user = build_diversity_path_prompt([ref], domain_labels, path, confusing)
        gen_requests.append(
            LlmRequest(
                system=system,
                user=user,
                parse_fn=lambda text, _labels=domain_labels: _parse_batch(text, 1, _labels),
                request_id=f"path_{path}",
                model=model,
                **kw,
            )
        )

    gen_results = chat_json_parallel(gen_requests)
    candidates: list[dict] = []
    for path_key, chat in gen_results.items():
        path = path_key.replace("path_", "")
        if chat.parsed is None:
            continue
        item = chat.parsed.items[0]
        from routers.synthetic.prompts import synthetic_item_to_train_row

        row = synthetic_item_to_train_row(item, f"synth_{uuid.uuid4().hex[:8]}")
        candidates.append(
            {
                "id": f"{path}_{rid}",
                "context": row["context"],
                "question": row["question"],
                "options": row["options"],
                "domain": row["gold"],
                "path": path,
            }
        )

    if not candidates:
        log_entry["error"] = "no candidates"
        return [], log_entry

    system_j, user_j = build_diversity_judge_prompt(candidates, n_pick=1)
    judge_chat = chat_json(
        system_j,
        user_j,
        parse_fn=lambda text: parse_judge_question_pick(text, n_pick=1),
        backend="google",
        model=model,
        skip_on_failure=True,
        **kw,
    )
    _record_batch(
        stats,
        mode="diversity_judge",
        system=system_j,
        user=user_j,
        chat=judge_chat,
        exemplar_ids=[ref.get("id")],
    )
    if judge_chat.parsed is None:
        log_entry["error"] = judge_chat.error
        rows = _rows_from_items_simple(candidates[:1], domain_labels, stats, mode="diversity_judge")
        for row in rows:
            row["source_ids"] = [ref.get("id")]
        return rows, log_entry

    winner_ids = set(judge_chat.parsed.winner_ids)
    log_entry["winner_ids"] = list(winner_ids)
    log_entry["rationale"] = judge_chat.parsed.rationale
    picked = [c for c in candidates if c["id"] in winner_ids]
    rows = _rows_from_items_simple(picked, domain_labels, stats, mode="diversity_judge")
    for row in rows:
        row["source_ids"] = [ref.get("id")]
    return rows, log_entry


def generate_diversity_paths_judge_pool(
    refs: list[dict],
    domain_labels: list[str],
    *,
    stats: dict[str, Any] | None = None,
    generation_model: str | None = None,
    judge_log: list[dict] | None = None,
) -> list[dict]:
    from tqdm import tqdm

    all_rows: list[dict] = []
    jlog = judge_log if judge_log is not None else []
    for ref in tqdm(refs, desc="smart07 diversity+judge"):
        rows, entry = generate_diversity_paths_judge_for_ref(
            ref, domain_labels, stats=stats, generation_model=generation_model
        )
        all_rows.extend(rows)
        jlog.append(entry)
    logger.info("Diversity+judge pool: {} items from {} refs", len(all_rows), len(refs))
    return all_rows


def generate_diversity_paths_judge(
    refs: list[dict],
    domain_labels: list[str],
    *,
    stats: dict[str, Any] | None = None,
    generation_model: str | None = None,
    judge_log: list[dict] | None = None,
) -> list[dict]:
    return generate_diversity_paths_judge_pool(
        refs, domain_labels, stats=stats, generation_model=generation_model, judge_log=judge_log
    )


def _rows_from_items_simple(
    items: list[dict],
    domain_labels: list[str],
    stats: dict | None,
    *,
    mode: str,
) -> list[dict]:
    from routers.synthetic.prompts import synthetic_item_to_train_row
    from routers.core.data import build_text_from_parts
    from routers.synthetic.parse import SyntheticItem

    rows: list[dict] = []
    for it in items:
        item = SyntheticItem(
            context=str(it.get("context", "")),
            question=str(it.get("question", "")),
            options=str(it.get("options", "")),
            domain=str(it.get("domain", "")),
        )
        sid = f"synth_{uuid.uuid4().hex[:12]}"
        row = synthetic_item_to_train_row(item, sid)
        row["generation_mode"] = mode
        row["prompt"] = build_text_from_parts(row["context"], row["question"], row["options"])
        rows.append(row)
    return rows


def generate_expansion_hard_negatives(
    buckets: dict[str, list[dict]],
    domain_labels: list[str],
    *,
    stats: dict[str, Any] | None = None,
    generation_model: str | None = None,
    cap: int = SMART_SYNTHETIC_CAP,
) -> list[dict]:
    """Per reference: 2 hard-negative items (gold + pred/neighbor), like combination02."""
    from tqdm import tqdm

    synthetics: list[dict] = []
    all_refs: list[tuple[str, dict]] = []
    for bucket_key, refs in buckets.items():
        for ref in refs:
            all_refs.append((bucket_key, ref))

    for bucket_key, ref in tqdm(all_refs, desc="smart08 hard-negative"):
        fail = dict(ref)
        if ref.get("correct"):
            fail["pred"] = neighbor_class_for_row(ref)
        elif not fail.get("pred"):
            fail["pred"] = neighbor_class_for_row(ref)
        batch = generate_hard_negative_pairs_smart(
            [fail],
            domain_labels,
            stats=stats,
            generation_model=generation_model,
            items_per_failure=2,
        )
        for row in batch:
            row["expansion_bucket"] = bucket_key
        synthetics.extend(batch)
        if len(synthetics) >= cap:
            break

    logger.info("Expansion hard-negative: {} items (cap={})", len(synthetics[:cap]), cap)
    return synthetics[:cap]


def generate_dataset_expansion(
    buckets: dict[str, list[dict]],
    domain_labels: list[str],
    *,
    stats: dict[str, Any] | None = None,
    generation_model: str | None = None,
    batch_n: int = 5,
    cap: int = SMART_SYNTHETIC_CAP,
) -> list[dict]:
    """Deprecated alias — use generate_expansion_hard_negatives."""
    del batch_n
    return generate_expansion_hard_negatives(
        buckets, domain_labels, stats=stats, generation_model=generation_model, cap=cap
    )
