"""One experiment run → one timestamp folder under results/baseline/."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from routers.core.harness import save_json_artifact
from routers.core.splits import project_root

RESULTS_TOPIC = "baseline"


def serialize_synthetic_rows(rows: list[dict]) -> list[dict]:
    """LLM output first: raw model fields + normalized training fields."""
    out: list[dict] = []
    for r in rows:
        llm = r.get("llm_generated") or {
            "context": r.get("context", ""),
            "question": r.get("question", ""),
            "options": r.get("options", ""),
            "domain": r.get("gold"),
        }
        out.append(
            {
                "id": r.get("id"),
                "llm_generated": llm,
                "normalized_for_training": {
                    "context": r.get("context", ""),
                    "question": r.get("question", ""),
                    "options": r.get("options", ""),
                    "domain": r.get("gold"),
                    "prompt": r.get("prompt", ""),
                },
                "source": r.get("source"),
                "generation_mode": r.get("generation_mode"),
                "source_ids": r.get("source_ids"),
                "target_domain": r.get("target_domain"),
                "confusion_pair": r.get("confusion_pair"),
                "seed_domain": r.get("seed_domain"),
                "donor_id": r.get("donor_id"),
                "iteration": r.get("iteration"),
            }
        )
    return out


def metrics_snapshot(metrics: dict[str, Any]) -> dict[str, Any]:
    skip = {"golds", "preds", "rows", "results_path"}
    return {k: v for k, v in metrics.items() if k not in skip}


class ExperimentSession:
    """results/<topic>/<experiment>/<timestamp>/ — all artifacts in one folder."""

    def __init__(self, experiment: str, *, topic: str = RESULTS_TOPIC) -> None:
        self.experiment = experiment
        self.topic = topic
        self.timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.out_dir = project_root() / "results" / topic / experiment / self.timestamp
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.holdout_metrics: dict[str, dict[str, Any]] = {}
        self.selection_stats: dict[str, Any] = {}
        self.verification_stats: dict[str, Any] = {}
        self.generation_stats: dict[str, Any] = {
            "skipped_batches": 0,
            "n_synthetic": 0,
            "generation_batches": [],
        }
        logger.info("Experiment session {} ({}) → {}", experiment, topic, self.out_dir)

    def save_config(self, config: dict[str, Any]) -> None:
        save_json_artifact(self.out_dir, "config.json", config)

    def record_holdout(self, phase: str, metrics: dict[str, Any]) -> None:
        self.holdout_metrics[phase] = metrics_snapshot(metrics)

    def save_synthetics(
        self,
        rows: list[dict],
        filename: str = "synthetic_samples.json",
        *,
        stats: dict[str, Any] | None = None,
    ) -> None:
        payload = serialize_synthetic_rows(rows)
        save_json_artifact(self.out_dir, filename, payload)
        self.generation_stats["n_synthetic"] = len(payload)
        if stats is not None:
            self.generation_stats["skipped_batches"] = stats.get("skipped_batches", 0)
            self.generation_stats["generation_batches"] = stats.get("generation_batches", [])
        self._write_generation_log(stats or self.generation_stats)
        logger.info(
            "Saved {} samples + generation log to {}",
            len(payload),
            self.out_dir,
        )

    def save_generation_log(self, stats: dict[str, Any] | None = None) -> None:
        self._write_generation_log(stats or self.generation_stats)

    def _write_generation_log(self, stats: dict[str, Any]) -> None:
        batches = stats.get("generation_batches", [])
        if not batches:
            return
        self.generation_stats["skipped_batches"] = stats.get("skipped_batches", 0)
        self.generation_stats["generation_batches"] = batches
        save_json_artifact(
            self.out_dir,
            "llm_generation_log.json",
            {
                "n_batches": len(batches),
                "n_success": sum(1 for b in batches if b.get("success")),
                "n_failed": sum(1 for b in batches if not b.get("success")),
                "batches": batches,
            },
        )

    def patch_metrics(self, supplemental: dict[str, Any]) -> None:
        path = self.out_dir / "metrics.json"
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        data.update(supplemental)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def supplemental_metrics(self) -> dict[str, Any]:
        log_summary = {
            "n_batches": len(self.generation_stats.get("generation_batches", [])),
            "skipped_batches": self.generation_stats.get("skipped_batches", 0),
            "n_synthetic": self.generation_stats.get("n_synthetic", 0),
        }
        out: dict[str, Any] = {
            "experiment": self.experiment,
            "topic": self.topic,
            "run_timestamp": self.timestamp,
            "holdout_metrics": self.holdout_metrics,
            "generation_stats": log_summary,
        }
        if self.selection_stats:
            out["selection_stats"] = self.selection_stats
        if self.verification_stats:
            out["verification_stats"] = self.verification_stats
        return out
