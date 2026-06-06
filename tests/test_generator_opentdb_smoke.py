"""Smoke test for OpenTDB hard-negative pair generation (requires Google API key)."""

from __future__ import annotations

import os

import pytest

from routers.core.constants import GOOGLE_FLASH_MODEL_DEFAULT
from routers.synthetic.generator_opentdb import generate_hard_negative_pairs

pytestmark = pytest.mark.skipif(
    not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")),
    reason="GOOGLE_API_KEY or GEMINI_API_KEY required",
)

_LABELS = [
    "0 Computer science, information, and general works",
    "6 Technology",
]

_FAILURE = {
    "id": "smoke_fail_1",
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


def test_hard_negative_pairs_real_api_and_json_parse():
    rows = generate_hard_negative_pairs(
        [_FAILURE],
        _LABELS,
        generation_model=GOOGLE_FLASH_MODEL_DEFAULT,
        items_per_failure=2,
    )
    assert len(rows) == 2, "Expected one item per target domain"
    domains = {str(r["gold"]) for r in rows}
    assert domains == set(_LABELS)
    for row in rows:
        assert row.get("confusion_pair") == [_LABELS[0], _LABELS[1]]
        assert row.get("generation_mode") == "hard_negative_pair"
        assert len(str(row.get("prompt", ""))) > 20
        assert row.get("source_ids") == [_FAILURE["id"]]
