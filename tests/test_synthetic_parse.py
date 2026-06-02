"""Tests for Ollama JSON parsing (no network)."""

from __future__ import annotations

import pytest

from routers.synthetic.parse import (
    parse_synthetic_batch,
    parse_validation_batch,
    strip_thought,
)

LABELS = ["Math", "Computer Science", "Philosophy"]


def test_parse_synthetic_clean():
    text = '{"items": [{"context": "c", "question": "q", "options": "o", "domain": "Math"}]}'
    batch = parse_synthetic_batch(text, n_expected=1, domain_labels=LABELS)
    assert batch.items[0].domain == "Math"


def test_parse_synthetic_with_thought():
    text = '<thought>hidden</thought>{"items": [{"context": "c", "question": "q", "options": "o", "domain": "Philosophy"}]}'
    batch = parse_synthetic_batch(text, n_expected=1, domain_labels=LABELS)
    assert batch.items[0].domain == "Philosophy"


def test_parse_synthetic_wrong_domain_raises():
    text = '{"items": [{"context": "c", "question": "q", "options": "o", "domain": "NotALabel"}]}'
    with pytest.raises(ValueError, match="not in vocabulary"):
        parse_synthetic_batch(text, n_expected=1, domain_labels=LABELS)


def test_parse_validation_yes_no():
    text = '{"items": [{"id": "s1", "verdict": "yes", "reason": "ok"}]}'
    batch = parse_validation_batch(text, n_expected=1)
    assert batch.items[0].verdict == "YES"


def test_strip_thought():
    assert "hello" in strip_thought("<thought>x</thought>hello")
