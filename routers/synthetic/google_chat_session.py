"""Multi-turn Gemini chat session for cascading generation (combination03)."""

from __future__ import annotations

import re
from typing import Any, Callable, TypeVar

from loguru import logger

from routers.core.constants import (
    GENERATION_TEMPERATURE,
    GOOGLE_FLASH_MODEL_DEFAULT,
    PARSE_MAX_RETRIES,
    COMBINATION_GOOGLE_THINKING_LEVEL,
)
from routers.synthetic.google_client import _client, _extract_json_text
from routers.synthetic.ollama_client import ChatResult

T = TypeVar("T")


class GoogleChatSession:
    def __init__(
        self,
        *,
        system: str,
        model: str | None = None,
        thinking_level: str | None = COMBINATION_GOOGLE_THINKING_LEVEL,
    ) -> None:
        self._client = _client()
        self._model = model or GOOGLE_FLASH_MODEL_DEFAULT
        self._temperature = GENERATION_TEMPERATURE
        self._thinking_level = thinking_level
        self.messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        self.turns: list[dict[str, Any]] = []

    def append_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def append_assistant(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def complete_json(
        self,
        user: str,
        *,
        parse_fn: Callable[[str], T],
        max_retries: int = PARSE_MAX_RETRIES,
    ) -> ChatResult:
        self.append_user(user)
        last_err: Exception | None = None
        last_text = ""
        for attempt in range(1, max_retries + 1):
            user_msg = user
            if attempt > 1:
                user_msg = user + "\n\nReturn ONLY valid JSON matching the schema."
            msgs = list(self.messages[:-1]) + [
                {"role": "user", "content": user_msg if attempt > 1 else user}
            ]
            try:
                kwargs: dict = {
                    "model": self._model,
                    "messages": msgs,
                    "temperature": self._temperature,
                }
                if "gemini" in self._model.lower():
                    kwargs["response_format"] = {"type": "json_object"}
                if self._thinking_level and "gemini" in self._model.lower():
                    effort_map = {
                        "minimal": "low",
                        "low": "low",
                        "medium": "medium",
                        "high": "high",
                    }
                    kwargs["reasoning_effort"] = effort_map.get(
                        self._thinking_level.lower(), self._thinking_level
                    )
                resp = self._client.chat.completions.create(**kwargs)
                text = (resp.choices[0].message.content or "").strip()
                last_text = text
                parsed = parse_fn(_extract_json_text(text))
                self.messages[-1] = {"role": "user", "content": user}
                self.append_assistant(text)
                self.turns.append(
                    {"user": user, "assistant": text, "success": True, "attempts": attempt}
                )
                return ChatResult(parsed=parsed, raw_response=text, attempts=attempt)
            except Exception as e:
                last_err = e
                logger.warning("Chat session attempt {} failed: {}", attempt, e)
        self.turns.append(
            {
                "user": user,
                "assistant": last_text,
                "success": False,
                "error": str(last_err),
            }
        )
        return ChatResult(parsed=None, raw_response=last_text, attempts=max_retries, error=last_err)

    def transcript(self) -> list[dict[str, Any]]:
        return list(self.turns)
