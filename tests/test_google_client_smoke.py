"""Integration smoke test for Google GenAI (skipped without API key)."""

from __future__ import annotations

import os

import pytest

from routers.core.constants import GOOGLE_FLASH_MODEL_DEFAULT
from routers.synthetic.google_client import chat_json
from routers.synthetic.parse import parse_synthetic_batch

_SMOKE_DOMAIN = "5 Science"


def _parse_smoke_batch(text: str):
    return parse_synthetic_batch(
        text, n_expected=1, domain_labels=[_SMOKE_DOMAIN]
    )


pytestmark = pytest.mark.skipif(
    not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")),
    reason="GOOGLE_API_KEY or GEMINI_API_KEY required",
)


def test_google_flash_json_generation():
    system = (
        'Return ONLY JSON: {"items": [{"context": "c", "question": "q?", '
        '"options": "a|b", "domain": "5 Science"}]}'
    )
    result = chat_json(
        system,
        "Generate exactly 1 training example for domain 5 Science.",
        temperature=0.0,
        parse_fn=_parse_smoke_batch,
        model=GOOGLE_FLASH_MODEL_DEFAULT,
        max_retries=3,
    )
    assert result.parsed is not None, result.error
    assert len(result.parsed.items) >= 1
    assert len(result.raw_response) > 10
