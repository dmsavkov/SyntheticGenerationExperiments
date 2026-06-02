"""Tests for frozen synthetic loading and format variants."""

from __future__ import annotations

from routers.core.data import build_text_from_parts
from routers.synthetic.frozen_loader import apply_synth_format, load_frozen_synthetics


def test_load_baseline_exp02():
    rows = load_frozen_synthetics()
    assert len(rows) >= 100
    assert all(r["gold"] for r in rows)
    assert all("Context:" in r["prompt"] for r in rows)


def test_context_in_question_merges_non_empty():
    base = [
        {
            "id": "a",
            "context": "Some passage.",
            "question": "What is X?",
            "options": "[\"A\"]",
            "gold": "5 Science",
        }
    ]
    out = apply_synth_format(base, "context_in_question")[0]
    assert out["context"] == ""
    assert out["question"] == "Some passage. What is X?"
    assert out["prompt"] == build_text_from_parts("", "Some passage. What is X?", "[\"A\"]")


def test_context_in_question_skips_empty_context():
    base = [{"id": "b", "context": "", "question": "Q?", "options": "O", "gold": "5 Science"}]
    out = apply_synth_format(base, "context_in_question")[0]
    assert out["question"] == "Q?"
    assert out["prompt"] == build_text_from_parts("", "Q?", "O")


def test_no_context_clears_context():
    base = [{"id": "c", "context": "Passage.", "question": "Q?", "options": "O", "gold": "5 Science"}]
    out = apply_synth_format(base, "no_context")[0]
    assert out["context"] == ""
    assert out["question"] == "Q?"
    assert out["prompt"] == build_text_from_parts("", "Q?", "O")
