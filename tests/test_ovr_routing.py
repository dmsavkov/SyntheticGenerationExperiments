"""OVR routing: expert override, scream-based confusion."""

from __future__ import annotations

from unittest.mock import MagicMock

from routers.ensemble.routing import (
    resolve_expert_general_prediction,
    resolve_ovr_panel_prediction,
)


def _mock_probe(label: str, prob: float) -> MagicMock:
    p = MagicMock()
    p.positive_domain = label
    p.p_positive.return_value = prob
    return p


def _mock_general(pred: str, probs: dict[str, float]) -> MagicMock:
    g = MagicMock()
    g.predict_extended.return_value = {"pred": pred, "pred_probs": probs}
    return g


def test_expert_override_when_confident():
    general = _mock_general("B", {"A": 0.3, "B": 0.7})
    experts = {"A": _mock_probe("A", 0.9), "B": _mock_probe("B", 0.2)}
    out = resolve_expert_general_prediction(general, experts, {"prompt": "x"})
    assert out["confused"] is False
    assert out["pred"] == "A"
    assert out["routing_case"] == "expert_override"


def test_general_fallback_when_expert_not_screaming():
    general = _mock_general("B", {"A": 0.3, "B": 0.7})
    experts = {"A": _mock_probe("A", 0.6), "B": _mock_probe("B", 0.4)}
    out = resolve_expert_general_prediction(general, experts, {"prompt": "x"})
    assert out["confused"] is False
    assert out["pred"] == "B"
    assert out["routing_case"] == "general_fallback"


def test_confused_when_multiple_models_scream():
    general = _mock_general("B", {"A": 0.3, "B": 0.9})
    experts = {"A": _mock_probe("A", 0.91)}
    out = resolve_expert_general_prediction(general, experts, {"prompt": "x"})
    assert out["confused"] is True
    assert set(out["screaming_labels"]) == {"A", "B"}


def test_panel_single_screamer_not_confused():
    probes = {"A": _mock_probe("A", 0.9), "B": _mock_probe("B", 0.3)}
    out = resolve_ovr_panel_prediction(probes, {"prompt": "x"})
    assert out["confused"] is False
    assert out["pred"] == "A"


def test_panel_two_screamers_confused():
    probes = {"A": _mock_probe("A", 0.9), "B": _mock_probe("B", 0.88)}
    out = resolve_ovr_panel_prediction(probes, {"prompt": "x"})
    assert out["confused"] is True
    assert out["pred"] == "A"
