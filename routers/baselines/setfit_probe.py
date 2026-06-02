"""SetFit classifier for low-data Domain routing experiments."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from routers.core.constants import (
    SETFIT_CLASSIFIER,
    SETFIT_MODEL_ID,
    TARGET_COL,
)


@dataclass
class SetFitProbe:
    model: Any
    classes_: list[str]
    id2label: dict[int, str]
    label2id: dict[str, int]

    def predict(self, row: dict) -> str:
        text = str(row["prompt"])
        raw = self.model.predict([text])
        arr = np.asarray(raw)
        idx = int(arr.flat[0])
        return self.id2label.get(idx, str(idx))

    def predict_proba_dict(self, row: dict) -> dict[str, float]:
        text = str(row["prompt"])
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba([text])[0]
            return {c: round(float(p), 6) for c, p in zip(self.classes_, probs)}
        pred = self.predict(row)
        return {c: 1.0 if c == pred else 0.0 for c in self.classes_}

    def predict_extended(self, row: dict) -> dict[str, Any]:
        pred = self.predict(row)
        return {"pred": pred, "pred_probs": self.predict_proba_dict(row)}


def default_training_args() -> dict[str, Any]:
    return {
        "batch_size": 16,
        "num_epochs": 2,
        "num_iterations": 20,
        "body_learning_rate": 2e-5,
        "eval_strategy": "no",
        "sampling_strategy": "num_iterations",
        "show_progress_bar": True,
    }


def train_setfit_probe(
    train_rows: list[dict],
    *,
    model_id: str = SETFIT_MODEL_ID,
    training_args: dict[str, Any] | None = None,
) -> tuple[SetFitProbe, float]:
    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments

    t0 = time.perf_counter()
    labels = sorted({str(r["gold"]) for r in train_rows})
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    ds = Dataset.from_dict(
        {
            "text": [str(r["prompt"]) for r in train_rows],
            "label": [label2id[str(r["gold"])] for r in train_rows],
        }
    )
    args_dict = {**default_training_args(), **(training_args or {})}
    model = SetFitModel.from_pretrained(model_id, labels=list(range(len(labels))))
    args = TrainingArguments(**args_dict)
    Trainer(model=model, args=args, train_dataset=ds).train()
    probe = SetFitProbe(
        model=model,
        classes_=labels,
        id2label=id2label,
        label2id=label2id,
    )
    return probe, time.perf_counter() - t0


def probe_config_extra() -> dict[str, Any]:
    return {
        "classifier": SETFIT_CLASSIFIER,
        "setfit_model_id": SETFIT_MODEL_ID,
        "target_col": TARGET_COL,
        "setfit_training": default_training_args(),
    }
