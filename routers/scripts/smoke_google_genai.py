"""Smoke test Google GenAI OpenAI-compatible endpoint."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from loguru import logger

from routers.core.constants import GOOGLE_FLASH_MODEL_DEFAULT
from routers.synthetic.google_client import chat_json
from routers.synthetic.parse import parse_synthetic_batch

_SMOKE_DOMAIN = "5 Science"


def _parse_smoke_batch(text: str):
    return parse_synthetic_batch(
        text, n_expected=1, domain_labels=[_SMOKE_DOMAIN]
    )


def main() -> None:
    load_dotenv(ROOT / ".env")
    system = (
        'Return ONLY JSON: {"items": [{"context": "c", "question": "q?", '
        '"options": "a|b", "domain": "5 Science"}]}'
    )
    user = "Generate exactly 1 item for domain 5 Science."
    result = chat_json(
        system,
        user,
        temperature=0.0,
        parse_fn=_parse_smoke_batch,
        model=GOOGLE_FLASH_MODEL_DEFAULT,
        max_retries=3,
    )
    assert result.parsed is not None, result.error or "parse failed"
    items = result.parsed.items
    logger.info("OK model={} attempts={} n_items={} raw_len={}", GOOGLE_FLASH_MODEL_DEFAULT, result.attempts, len(items), len(result.raw_response))
    print("smoke_google_genai: PASS")


if __name__ == "__main__":
    main()
