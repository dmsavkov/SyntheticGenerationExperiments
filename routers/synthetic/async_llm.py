"""Bounded parallel LLM calls (max 15 in flight globally per process)."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

from loguru import logger

from routers.core.constants import SMART_MAX_IN_FLIGHT, SMART_PARALLEL_WORKERS
from routers.synthetic.llm_client import chat_json
from routers.synthetic.ollama_client import ChatResult

T = TypeVar("T")

_semaphore = threading.Semaphore(SMART_MAX_IN_FLIGHT)


@dataclass(frozen=True)
class LlmRequest:
    """One chat_json invocation."""

    system: str
    user: str
    temperature: float
    parse_fn: Callable[[str], Any]
    request_id: str = ""
    model: str | None = None
    skip_on_failure: bool = True
    thinking_level: str | None = None
    top_p: float | None = None
    max_retries: int | None = None


def _run_one(req: LlmRequest) -> tuple[str, ChatResult]:
    with _semaphore:
        result = chat_json(
            req.system,
            req.user,
            temperature=req.temperature,
            parse_fn=req.parse_fn,
            backend="google",
            model=req.model,
            skip_on_failure=req.skip_on_failure,
            thinking_level=req.thinking_level,
            top_p=req.top_p,
            max_retries=req.max_retries,
        )
    return req.request_id, result


def chat_json_parallel(
    requests: list[LlmRequest],
    *,
    max_workers: int | None = None,
) -> dict[str, ChatResult]:
    """
    Run requests in parallel. Global in-flight cap is SMART_MAX_IN_FLIGHT (15).
    Worker pool size defaults to SMART_PARALLEL_WORKERS (5).
    """
    if not requests:
        return {}
    workers = min(max_workers or SMART_PARALLEL_WORKERS, len(requests), SMART_MAX_IN_FLIGHT)
    logger.info(
        "Parallel LLM: {} requests, workers={}, max_in_flight={}",
        len(requests),
        workers,
        SMART_MAX_IN_FLIGHT,
    )
    out: dict[str, ChatResult] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_one, req): req.request_id for req in requests}
        for fut in as_completed(futures):
            rid = futures[fut]
            try:
                key, result = fut.result()
                out[key or rid] = result
            except Exception as e:
                logger.warning("Parallel request {} failed: {}", rid, e)
                out[rid] = ChatResult(parsed=None, raw_response="", attempts=0, error=str(e))
    return out
