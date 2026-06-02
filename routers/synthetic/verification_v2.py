"""Post-generation verification: fastembed diversity + SetFit uncertainty filter."""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from routers.core.constants import (
    FASTEMBED_MODEL_ID,
    SETFIT_DIVERSITY_SIM_THRESHOLD,
    SETFIT_UNCERTAINTY_HIGH,
    SETFIT_UNCERTAINTY_LOW,
)
from routers.core.harness import save_json_artifact
from routers.experiments.session import ExperimentSession


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _embed_texts(texts: list[str]) -> list[np.ndarray]:
    from fastembed import TextEmbedding

    model = TextEmbedding(model_name=FASTEMBED_MODEL_ID)
    return [np.array(v, dtype=np.float32) for v in model.embed(texts)]


def filter_verification_v2(
    candidates: list[dict],
    train_rows: list[dict],
    probe: Any,
    *,
    session: ExperimentSession | None = None,
    sim_threshold: float = SETFIT_DIVERSITY_SIM_THRESHOLD,
    uncertainty_low: float = SETFIT_UNCERTAINTY_LOW,
    uncertainty_high: float = SETFIT_UNCERTAINTY_HIGH,
) -> tuple[list[dict], list[dict], dict[str, Any]]:
    if not candidates:
        return [], [], {"n_in": 0, "n_accepted": 0, "n_rejected": 0}

    corpus_rows = list(train_rows)
    corpus_texts = [str(r["prompt"]) for r in corpus_rows]
    corpus_vecs = _embed_texts(corpus_texts) if corpus_texts else []

    accepted: list[dict] = []
    rejected: list[dict] = []

    for cand in candidates:
        text = str(cand["prompt"])
        reasons: list[str] = []

        if corpus_vecs:
            cand_vec = _embed_texts([text])[0]
            max_sim = max(_cosine_sim(cand_vec, v) for v in corpus_vecs)
            if max_sim > sim_threshold:
                reasons.append(f"diversity_fail:max_sim={max_sim:.4f}")

        gold = str(cand.get("gold", ""))
        p_gold = float(probe.predict_extended(cand)["pred_probs"].get(gold, 0.0))
        if not (uncertainty_low <= p_gold <= uncertainty_high):
            reasons.append(f"setfit_prob_fail:p_gold={p_gold:.4f}")

        record = {
            "synthetic_id": cand.get("id"),
            "gold_domain": gold,
            "prompt": text,
            "p_gold": p_gold,
            "accepted": not reasons,
            "reasons": reasons,
        }
        if reasons:
            rejected.append(record)
        else:
            accepted.append(cand)
            corpus_rows.append(cand)
            corpus_vecs.extend(_embed_texts([text]))

    stats = {
        "n_in": len(candidates),
        "n_accepted": len(accepted),
        "n_rejected": len(rejected),
        "sim_threshold": sim_threshold,
        "prob_band": [uncertainty_low, uncertainty_high],
    }
    logger.info(
        "Verification v2: {}/{} accepted (rejected={})",
        stats["n_accepted"],
        stats["n_in"],
        stats["n_rejected"],
    )
    if session is not None:
        session.verification_stats = stats
        save_json_artifact(
            session.out_dir,
            "synthetic_verification_v2.json",
            {"stats": stats, "rejected": rejected},
        )
    return accepted, rejected, stats
