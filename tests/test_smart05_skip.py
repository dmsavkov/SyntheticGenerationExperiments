"""Smart05 skip-gate generation: unit mocks + optional real API smoke."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from routers.core.constants import GOOGLE_FLASH_MODEL_DEFAULT
from routers.synthetic.generator_smart import generate_hard_negative_skip
from routers.synthetic.ollama_client import ChatResult
from routers.synthetic.parse_smart import SkipOrGenerate

_LABELS = [
    "0 Computer science, information, and general works",
    "6 Technology",
]

_FAILURE = {
    "id": "unit_fail_1",
    "gold": _LABELS[0],
    "pred": _LABELS[1],
    "context": "A display standard used in early personal computers.",
    "question": "What resolution is typical for VGA?",
    "options": "640x480|800x600",
    "prompt": (
        "Context: A display standard used in early personal computers. | "
        "Question: What resolution is typical for VGA? | Options: 640x480|800x600"
    ),
}


@patch("routers.synthetic.generator_smart.chat_json")
def test_generate_hard_negative_skip_accepts_gold_domain_items(mock_chat):
    mock_chat.return_value = ChatResult(
        parsed=SkipOrGenerate(
            action="GENERATE",
            reason="clear labels",
            items=[
                {
                    "context": "Network cable impedance standards.",
                    "question": "What is typical coax impedance?",
                    "options": "50 ohm|75 ohm",
                    "gold_domain": _LABELS[0],
                },
                {
                    "context": "Consumer display interface resolution.",
                    "question": "What is VGA resolution?",
                    "options": "640x480|800x600",
                    "gold_domain": _LABELS[1],
                },
            ],
        ),
        raw_response="{}",
        attempts=1,
    )
    rows, log = generate_hard_negative_skip([_FAILURE], _LABELS)
    assert len(rows) == 2
    assert {r["gold"] for r in rows} == set(_LABELS)
    assert log[0]["action"] == "GENERATE"
    assert rows[0]["generation_mode"] == "hard_negative_skip"


@patch("routers.synthetic.generator_smart.chat_json")
def test_generate_hard_negative_skip_records_skip(mock_chat):
    mock_chat.return_value = ChatResult(
        parsed=SkipOrGenerate(action="SKIP", reason="ambiguous gold"),
        raw_response='{"action":"SKIP"}',
        attempts=1,
    )
    rows, log = generate_hard_negative_skip([_FAILURE], _LABELS)
    assert rows == []
    assert log == [{"id": _FAILURE["id"], "action": "SKIP", "reason": "ambiguous gold"}]


@pytest.mark.skipif(
    not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")),
    reason="GOOGLE_API_KEY or GEMINI_API_KEY required",
)
def test_smart05_skip_gate_real_api_smoke():
    rows, log = generate_hard_negative_skip(
        [_FAILURE],
        _LABELS,
        generation_model=GOOGLE_FLASH_MODEL_DEFAULT,
    )
    assert len(log) == 1
    action = log[0]["action"]
    assert action in ("SKIP", "GENERATE", "ERROR")
    if action == "GENERATE":
        assert len(rows) == 2
        assert {str(r["gold"]) for r in rows} == set(_LABELS)
        for row in rows:
            assert row.get("generation_mode") == "hard_negative_skip"
            assert len(str(row.get("prompt", ""))) > 20
