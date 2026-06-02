"""Tests for SetFit prompt batch sizes."""

from __future__ import annotations

import pytest

from routers.synthetic.batching import fixed_size_batches
from routers.synthetic.prompts_setfit import build_diversity_triplet_prompt


def test_fixed_size_batches_max_three():
    rows = [{"id": str(i)} for i in range(10)]
    batches = fixed_size_batches(rows, 3)
    assert len(batches) == 4
    assert all(len(b) <= 3 for b in batches)
    assert sum(len(b) for b in batches) == 10


def test_diversity_triplet_requires_three():
    refs = [{"gold": "5 Science", "context": "", "question": "q", "options": "o"} for _ in range(3)]
    system, user = build_diversity_triplet_prompt(refs, ["5 Science"])
    assert "exactly 3" in user.lower() or "3 new items" in user
    assert "fill-in-the-blank" in user

    with pytest.raises(ValueError):
        build_diversity_triplet_prompt(refs[:2], ["5 Science"])
