"""Pydantic parsers for smart experiment LLM outputs."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from routers.synthetic.parse import extract_json_array_or_object, strip_thought


class CritiqueResponse(BaseModel):
    bullet_label_alignment: str
    bullet_distractor_realism: str
    bullet_vocabulary_contrast: str
    bullet_mcq_format: str
    bullet_ambiguity_risk: str
    thought: str = ""
    next_steps: str


class JudgePairPick(BaseModel):
    winner_pair_id: str
    rejected_pair_ids: list[str] = Field(default_factory=list)
    rationale: str


class JudgeQuestionPick(BaseModel):
    winner_ids: list[str]
    rationale: str = ""


class SkipOrGenerate(BaseModel):
    action: Literal["SKIP", "GENERATE"]
    reason: str = ""
    items: list[dict] = Field(default_factory=list)


class PairItems(BaseModel):
    pair_id: str
    items: list[dict]


class PairsBatch(BaseModel):
    pairs: list[PairItems]


def parse_pairs_batch(text: str) -> PairsBatch:
    return PairsBatch.model_validate(_load_json(text))


def _load_json(text: str) -> Any:
    raw = extract_json_array_or_object(strip_thought(text))
    if raw is None:
        raise ValueError("No JSON in response")
    return raw


def parse_critique(text: str) -> CritiqueResponse:
    raw = _load_json(text)
    if isinstance(raw, dict) and "items" in raw:
        raise ValueError("Expected critique object, got items batch")
    return CritiqueResponse.model_validate(raw)


def parse_judge_pair_pick(text: str) -> JudgePairPick:
    return JudgePairPick.model_validate(_load_json(text))


def parse_judge_question_pick(text: str, *, n_pick: int = 3) -> JudgeQuestionPick:
    data = _load_json(text)
    pick = JudgeQuestionPick.model_validate(data)
    if len(pick.winner_ids) != n_pick:
        raise ValueError(f"Expected {n_pick} winner_ids, got {len(pick.winner_ids)}")
    return pick


def parse_skip_or_generate(text: str) -> SkipOrGenerate:
    return SkipOrGenerate.model_validate(_load_json(text))


def parse_draft_pairs_json(text: str) -> dict[str, Any]:
    """Draft/refine steps use standard items array."""
    return _load_json(text)
