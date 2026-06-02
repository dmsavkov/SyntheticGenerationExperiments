"""JSON extraction and Pydantic validation for Ollama responses."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field, field_validator


class SyntheticItem(BaseModel):
    context: str = ""
    question: str = ""
    options: str = ""
    domain: str

    @field_validator("context", "question", "options", mode="before")
    @classmethod
    def _coerce_str(cls, v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, (list, dict)):
            return json.dumps(v, ensure_ascii=False)
        return str(v).strip()

    @field_validator("domain", mode="before")
    @classmethod
    def _coerce_domain(cls, v: Any) -> str:
        return str(v).strip()


class SyntheticBatch(BaseModel):
    items: list[SyntheticItem]


class ValidationItem(BaseModel):
    id: str
    verdict: str
    reason: str = ""

    @field_validator("verdict", mode="before")
    @classmethod
    def _norm_verdict(cls, v: Any) -> str:
        s = str(v).strip().upper()
        if s.startswith("Y"):
            return "YES"
        if s.startswith("N"):
            return "NO"
        return s


class ValidationBatch(BaseModel):
    items: list[ValidationItem]


def strip_thought(text: str) -> str:
    return re.sub(r"<thought>.*?</thought>", "", text or "", flags=re.DOTALL | re.IGNORECASE)


def extract_json_array_or_object(text: str) -> Any:
    text = strip_thought(text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    return None


def parse_synthetic_batch(text: str, *, n_expected: int, domain_labels: list[str]) -> SyntheticBatch:
    raw = extract_json_array_or_object(text)
    if raw is None:
        raise ValueError("No JSON found in model response")
    if isinstance(raw, dict) and "items" in raw:
        items_raw = raw["items"]
    elif isinstance(raw, list):
        items_raw = raw
    else:
        raise ValueError(f"Unexpected JSON shape: {type(raw)}")
    batch = SyntheticBatch.model_validate({"items": items_raw})
    if len(batch.items) != n_expected:
        raise ValueError(f"Expected {n_expected} items, got {len(batch.items)}")
    label_set = set(domain_labels)
    for it in batch.items:
        if it.domain not in label_set:
            raise ValueError(f"Domain {it.domain!r} not in vocabulary")
    return batch


def parse_validation_batch(text: str, *, n_expected: int) -> ValidationBatch:
    raw = extract_json_array_or_object(text)
    if raw is None:
        raise ValueError("No JSON found in validator response")
    if isinstance(raw, dict) and "items" in raw:
        items_raw = raw["items"]
    elif isinstance(raw, list):
        items_raw = raw
    else:
        raise ValueError(f"Unexpected JSON shape: {type(raw)}")
    batch = ValidationBatch.model_validate({"items": items_raw})
    if len(batch.items) != n_expected:
        raise ValueError(f"Expected {n_expected} items, got {len(batch.items)}")
    for it in batch.items:
        if it.verdict not in ("YES", "NO"):
            raise ValueError(f"Invalid verdict {it.verdict!r}")
    return batch
