"""Tests for holdout reference selection and cycling."""

from __future__ import annotations

from routers.synthetic.holdout_selection import expand_references, select_holdout_references


def _scored(p_gold: float, correct: bool = True, gold: str = "5 Science") -> dict:
    return {"id": f"x_{p_gold}", "gold": gold, "p_gold": p_gold, "correct": correct, "prompt": "q"}


def test_expand_references_cycles():
    refs = [{"id": "a", "gold": "5 Science"}, {"id": "b", "gold": "5 Science"}]
    pool, stats = expand_references(refs, 6)
    assert len(pool) == 6
    assert stats["cycling_used"] is True
    assert stats["n_unique_refs"] == 2
    assert stats["cycle_multiplier"] == 3.0


def test_uncertainty_band_selection():
    rows = [_scored(0.1), _scored(0.5), _scored(0.9), _scored(0.4, gold="4 Language")]
    pool, stats = select_holdout_references(rows, mode="uncertainty", cap=4, seed=42)
    assert stats["n_ref_slots"] == 4
    assert stats["uncertainty_band"] == [0.2, 0.8]
    assert all(0.2 <= r["p_gold"] <= 0.8 for r in pool if "p_gold" in r)


def test_narrow_acceptance_band():
    rows = [_scored(0.35), _scored(0.5), _scored(0.65), _scored(0.2)]
    pool, stats = select_holdout_references(
        rows,
        mode="uncertainty",
        cap=4,
        seed=42,
        uncertainty_low=0.4,
        uncertainty_high=0.6,
    )
    assert stats["uncertainty_band"] == [0.4, 0.6]
    assert all(0.4 <= r["p_gold"] <= 0.6 for r in pool if "p_gold" in r)


def test_high_conf_top_fraction():
    rows = [_scored(0.95 - i * 0.01, correct=True) for i in range(20)]
    pool, stats = select_holdout_references(rows, mode="high_conf_correct", cap=10, seed=1)
    assert stats["n_ref_slots"] == 10
    assert min(r["p_gold"] for r in pool) >= 0.86
