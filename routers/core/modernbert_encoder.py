"""Frozen ModernBERT-base mean-pooled embeddings."""

from __future__ import annotations

import numpy as np
from loguru import logger

from routers.core.constants import CONTEXT_MAX_CHARS, MODERNBERT_MAX_LENGTH, MODERNBERT_MODEL_ID

_encoder: "ModernBertEncoder | None" = None


class ModernBertEncoder:
    def __init__(self, model_id: str = MODERNBERT_MODEL_ID, max_length: int = MODERNBERT_MAX_LENGTH):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_id = model_id
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading ModernBERT encoder: {} max_length={}", model_id, max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device)
        self.model.eval()

    @classmethod
    def get(cls, max_length: int = MODERNBERT_MAX_LENGTH) -> "ModernBertEncoder":
        global _encoder
        if _encoder is None or _encoder.max_length != max_length:
            _encoder = cls(max_length=max_length)
        return _encoder

    def encode_batch(self, texts: list[str], batch_size: int = 8) -> np.ndarray:
        import torch

        all_vecs: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = [t[:CONTEXT_MAX_CHARS] for t in texts[start : start + batch_size]]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                out = self.model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            summed = (out.last_hidden_state * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            pooled = (summed / counts).cpu().numpy()
            all_vecs.append(pooled.astype(np.float32))
        return np.vstack(all_vecs) if all_vecs else np.zeros((0, 768), dtype=np.float32)
