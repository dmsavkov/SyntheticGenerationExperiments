"""Bounded parallel LLM wrapper."""

from __future__ import annotations

from unittest.mock import patch

from routers.core.constants import SMART_MAX_IN_FLIGHT
from routers.synthetic.async_llm import LlmRequest, chat_json_parallel
from routers.synthetic.ollama_client import ChatResult


def test_parallel_empty():
    assert chat_json_parallel([]) == {}


def test_max_in_flight_constant():
    assert SMART_MAX_IN_FLIGHT == 15


@patch("routers.synthetic.async_llm.chat_json")
def test_parallel_collects(mock_chat):
    mock_chat.return_value = ChatResult(parsed={"ok": True}, raw_response="{}", attempts=1)

    reqs = [
        LlmRequest(system="s", user=f"u{i}", temperature=0.0, parse_fn=lambda t: t, request_id=f"r{i}")
        for i in range(3)
    ]
    out = chat_json_parallel(reqs, max_workers=2)
    assert len(out) == 3
    assert mock_chat.call_count == 3
