"""Smoke: Gemini accepts top_p + temp=1 without reasoning (skip without API key)."""

from __future__ import annotations

import os

import pytest

from routers.core.constants import GOOGLE_FLASH_MODEL_DEFAULT, SMART_CREATIVE_TEMPERATURE, SMART_CREATIVE_TOP_P
from routers.synthetic.google_client import chat_json

pytestmark = pytest.mark.skipif(
    not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")),
    reason="GOOGLE_API_KEY or GEMINI_API_KEY required",
)


def test_creative_params_no_reasoning():
    result = chat_json(
        "Return JSON only.",
        '{"ok": true}',
        temperature=SMART_CREATIVE_TEMPERATURE,
        top_p=SMART_CREATIVE_TOP_P,
        parse_fn=lambda t: {"ok": True},
        model=GOOGLE_FLASH_MODEL_DEFAULT,
        thinking_level=None,
        max_retries=3,
    )
    assert result.parsed is not None
