"""Tests for SetFit run config fields (no misleading prob bands)."""

from __future__ import annotations

from routers.experiments.setfit._common import setfit_base_config
from routers.synthetic.holdout_selection import selection_uses_reference_prob_band


def test_selection_modes_prob_band_usage():
    assert selection_uses_reference_prob_band("uncertainty") is True
    assert selection_uses_reference_prob_band("rewrite_split") is True
    assert selection_uses_reference_prob_band("low_conf_correct") is False
    assert selection_uses_reference_prob_band("high_conf_correct") is False


def test_setfit09_style_extra_omits_bands():
    extra = {
        "selection_mode": "low_conf_correct",
        "verification_v2": False,
        "split_variant": "opentdb",
    }
    cfg = setfit_base_config(experiment="setfit09", hypotheses="test", extra=extra)
    assert "reference_prob_band" not in cfg
    assert "verification_prob_band" not in cfg
    assert "uncertainty_band" not in cfg


def test_setfit10_style_extra_includes_both_bands():
    extra = {
        "selection_mode": "uncertainty",
        "verification_v2": True,
        "reference_prob_band": [0.4, 0.6],
        "verification_prob_band": [0.4, 0.6],
    }
    cfg = setfit_base_config(experiment="setfit10", hypotheses="test", extra=extra)
    assert cfg["reference_prob_band"] == [0.4, 0.6]
    assert cfg["verification_prob_band"] == [0.4, 0.6]
    assert "uncertainty_band" not in cfg
