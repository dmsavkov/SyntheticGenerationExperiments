"""Smart09 iterative selection helper (no API)."""

from __future__ import annotations

from unittest.mock import patch

from routers.synthetic.smart_selection import bottom_correct_rows, mislabeled_rows, score_rows


def _row(gold: str, pred: str, p: float) -> dict:
    return {
        "id": f"{gold}_{pred}_{p}",
        "gold": gold,
        "pred": pred,
        "p_gold": p,
        "correct": pred == gold,
        "pred_probs": {gold: p, "other": 1.0 - p},
    }


def test_mislabel_lowconf_selection_counts():
    scored = [_row("A", "B", 0.9), _row("C", "C", 0.2), _row("D", "D", 0.95)]
    assert len(mislabeled_rows(scored)) == 1
    assert len(bottom_correct_rows(scored)) == 1


@patch("routers.synthetic.generator_smart.generate_hard_negative_pairs_smart")
def test_generate_mislabel_lowconf_synthetics(mock_gen):
    from routers.experiments.smart._common import generate_mislabel_lowconf_synthetics

    mock_gen.side_effect = lambda fails, labels, **kw: [{"id": f"s{i}"} for i in range(len(fails) * 2)]
    scored = [_row("A", "B", 0.9), _row("C", "C", 0.15), _row("D", "D", 0.95)]
    rows, sel = generate_mislabel_lowconf_synthetics(scored, ["A", "B", "C", "D"], cap=50)
    assert sel["n_mislabeled"] == 1
    assert sel["n_bottom_correct"] == 1
    assert mock_gen.call_count == 2
    assert len(rows) <= 50


def test_smart09_module_import():
    from routers.experiments.smart import smart09_iterative_hardneg

    assert callable(smart09_iterative_hardneg.run)


def test_smart_package_exports_all_experiments():
    from routers.experiments import smart

    for name in smart.__all__:
        mod = getattr(smart, name)
        assert callable(getattr(mod, "run"))
