"""Google GenAI chat client (OpenAI-compatible) with JSON parse retries."""

from __future__ import annotations

import json
import os
import re
from typing import Callable, TypeVar

from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI

from routers.core.constants import GOOGLE_FLASH_MODEL_DEFAULT, PARSE_MAX_RETRIES
from routers.synthetic.ollama_client import ChatResult

T = TypeVar("T")


def google_model() -> str:
    return os.environ.get("GOOGLE_GENAI_MODEL") or os.environ.get(
        "GOOGLE_GEMMA_MODEL", GOOGLE_FLASH_MODEL_DEFAULT
    )


def _extract_json_text(text: str) -> str:
    text = text.strip()
    if text.startswith("{"):
        return text
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        return fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


def _client() -> OpenAI:
    load_dotenv()
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY or GEMINI_API_KEY required for Google GenAI backend")
    return OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )


def chat_json(
    system: str,
    user: str,
    *,
    temperature: float,
    parse_fn: Callable[[str], T],
    max_retries: int = PARSE_MAX_RETRIES,
    model: str | None = None,
    skip_on_failure: bool = False,
    thinking_level: str | None = None,
    top_p: float | None = None,
) -> ChatResult:
    client = _client()
    model_name = model or google_model()
    last_err: Exception | None = None
    last_text = ""
    for attempt in range(1, max_retries + 1):
        user_msg = user
        if attempt > 1:
            user_msg = user + "\n\nYour previous response was invalid. Return ONLY valid JSON matching the schema."
        try:
            kwargs: dict = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
                "temperature": temperature,
            }
            if top_p is not None:
                kwargs["top_p"] = top_p
            if "gemini" in model_name.lower():
                kwargs["response_format"] = {"type": "json_object"}
            if thinking_level and "gemini" in model_name.lower():
                # OpenAI-compat endpoint maps reasoning_effort → Gemini thinking budget.
                effort_map = {
                    "minimal": "low",
                    "low": "low",
                    "medium": "medium",
                    "high": "high",
                }
                kwargs["reasoning_effort"] = effort_map.get(
                    thinking_level.lower(), thinking_level
                )
            resp = client.chat.completions.create(**kwargs)
            last_text = (resp.choices[0].message.content or "").strip()
            parsed = parse_fn(_extract_json_text(last_text))
            return ChatResult(parsed=parsed, raw_response=last_text, attempts=attempt)
        except Exception as e:
            last_err = e
            logger.warning("Google GenAI parse attempt {}/{} failed: {}", attempt, max_retries, e)
    err_msg = str(last_err) if last_err else "unknown"
    if skip_on_failure:
        logger.warning("Skipping batch after {} failed attempts: {}", max_retries, err_msg)
        return ChatResult(parsed=None, raw_response=last_text, attempts=max_retries, error=err_msg)
    raise RuntimeError(f"Google GenAI chat failed after {max_retries} retries") from last_err
