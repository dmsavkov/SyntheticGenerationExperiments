"""Ollama chat client with JSON format and parse retries."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, TypeVar

from loguru import logger

from routers.core.constants import PARSE_MAX_RETRIES, OLLAMA_MODEL_DEFAULT
from routers.synthetic.ollama_server import ensure_ollama_server

T = TypeVar("T")


@dataclass
class ChatResult:
    parsed: T | None
    raw_response: str
    attempts: int
    error: str | None = None


def ollama_model() -> str:
    return os.environ.get("OLLAMA_MODEL", OLLAMA_MODEL_DEFAULT)


def chat_json(
    system: str,
    user: str,
    *,
    temperature: float,
    parse_fn: Callable[[str], T],
    max_retries: int = PARSE_MAX_RETRIES,
    ensure_server: bool = True,
    skip_on_failure: bool = False,
) -> ChatResult:
    if ensure_server:
        ensure_ollama_server()

    import ollama

    last_err: Exception | None = None
    last_text = ""
    for attempt in range(1, max_retries + 1):
        user_msg = user
        if attempt > 1:
            user_msg = user + "\n\nYour previous response was invalid. Return ONLY valid JSON matching the schema."
        try:
            resp = ollama.chat(
                model=ollama_model(),
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
                format="json",
                options={"temperature": temperature},
            )
            last_text = (resp.get("message") or {}).get("content") or ""
            parsed = parse_fn(last_text)
            return ChatResult(parsed=parsed, raw_response=last_text, attempts=attempt)
        except Exception as e:
            last_err = e
            logger.warning("Ollama parse attempt {}/{} failed: {}", attempt, max_retries, e)
    err_msg = str(last_err) if last_err else "unknown"
    if skip_on_failure:
        logger.warning("Skipping batch after {} failed attempts: {}", max_retries, err_msg)
        return ChatResult(parsed=None, raw_response=last_text, attempts=max_retries, error=err_msg)
    raise RuntimeError(f"Ollama chat failed after {max_retries} retries") from last_err
