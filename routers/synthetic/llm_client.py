"""Unified LLM chat entry for Ollama and Google GenAI backends."""

from __future__ import annotations

from typing import Callable, Literal, TypeVar

from routers.synthetic.ollama_client import ChatResult

T = TypeVar("T")
LlmBackend = Literal["ollama", "google"]


def chat_json(
    system: str,
    user: str,
    *,
    temperature: float,
    parse_fn: Callable[[str], T],
    backend: LlmBackend = "ollama",
    model: str | None = None,
    skip_on_failure: bool = False,
    thinking_level: str | None = None,
    top_p: float | None = None,
    max_retries: int | None = None,
) -> ChatResult:
    if backend == "google":
        from routers.synthetic.google_client import chat_json as google_chat

        kwargs: dict = {
            "temperature": temperature,
            "parse_fn": parse_fn,
            "model": model,
            "skip_on_failure": skip_on_failure,
            "thinking_level": thinking_level,
            "top_p": top_p,
        }
        if max_retries is not None:
            kwargs["max_retries"] = max_retries
        return google_chat(system, user, **kwargs)
    from routers.synthetic.ollama_client import chat_json as ollama_chat

    return ollama_chat(
        system,
        user,
        temperature=temperature,
        parse_fn=parse_fn,
        skip_on_failure=skip_on_failure,
    )
