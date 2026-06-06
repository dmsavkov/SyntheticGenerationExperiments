"""Pydantic parsers for smart LLM outputs."""

from __future__ import annotations

import json

from routers.synthetic.parse import parse_synthetic_batch
from routers.synthetic.parse_smart import (
    SkipOrGenerate,
    parse_critique,
    parse_judge_pair_pick,
    parse_judge_question_pick,
    parse_pairs_batch,
)


def test_parse_judge_pair_pick():
    text = '{"winner_pair_id":"pair_2","rejected_pair_ids":["pair_1"],"rationale":"Symmetric quality."}'
    out = parse_judge_pair_pick(text)
    assert out.winner_pair_id == "pair_2"


def test_parse_skip():
    text = '{"action":"SKIP","reason":"ambiguous gold"}'
    out = SkipOrGenerate.model_validate_json(text)
    assert out.action == "SKIP"


def test_parse_pairs_batch():
    text = (
        '{"pairs":[{"pair_id":"pair_1","items":[{"context":"c","question":"q",'
        '"options":"a|b","domain":"6 Technology"},{"context":"c2","question":"q2",'
        '"options":"a|b","domain":"0 Computer science, information, and general works"}]}]}'
    )
    batch = parse_pairs_batch(text)
    assert len(batch.pairs) == 1
    assert len(batch.pairs[0].items) == 2


def test_parse_judge_question_pick_flexible():
    text = '{"winner_ids":["q1"],"rationale":"deepest"}'
    out = parse_judge_question_pick(text, n_pick=1)
    assert out.winner_ids == ["q1"]


def test_parse_critique():
    text = (
        '{"bullet_label_alignment":"ok","bullet_distractor_realism":"ok",'
        '"bullet_vocabulary_contrast":"ok","bullet_mcq_format":"ok",'
        '"bullet_ambiguity_risk":"low","thought":"t","next_steps":"n"}'
    )
    c = parse_critique(text)
    assert c.next_steps == "n"


def test_parse_synthetic_batch_accepts_gold_domain_alias():
    labels = [
        "0 Computer science, information, and general works",
        "6 Technology",
    ]
    text = json.dumps(
        {
            "items": [
                {
                    "context": "c",
                    "question": "q",
                    "options": "a|b",
                    "gold_domain": labels[0],
                },
                {
                    "context": "c2",
                    "question": "q2",
                    "options": "a|b",
                    "gold_domain": labels[1],
                },
            ]
        }
    )
    batch = parse_synthetic_batch(text, n_expected=2, domain_labels=labels)
    assert batch.items[0].domain == labels[0]
    assert batch.items[1].domain == labels[1]
