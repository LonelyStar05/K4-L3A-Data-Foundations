"""Cross-encoder reranking.

A bi-encoder embeds query and chunk separately, so it measures *topic* similarity; a
cross-encoder reads (query, chunk) together and scores whether the chunk *answers* the
query — exactly the failure mode where a same-topic chunk without the number outranks
the chunk that has it. Too slow to score the whole corpus, so it only reorders the
top candidates coming out of hybrid retrieval.

Default model: BAAI/bge-reranker-v2-m3 (multilingual, strong on Vietnamese).
Override with RERANKER_MODEL, e.g. cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (smaller).
"""

from __future__ import annotations

import os


class CrossEncoderReranker:
    def __init__(self, model_name: str | None = None, max_length: int = 1024) -> None:
        self.model_name = model_name or os.getenv("RERANKER_MODEL") or "BAAI/bge-reranker-v2-m3"
        self.max_length = max_length
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            from transformers import AutoConfig

            # Never ask for more tokens than the model has positions (MiniLM: 512, bge-m3: 8k).
            config = AutoConfig.from_pretrained(self.model_name)
            limit = getattr(config, "max_position_embeddings", 512)
            if getattr(config, "model_type", "") in {"xlm-roberta", "roberta"}:
                limit -= 2  # RoBERTa-style models reserve two position ids
            self._model = CrossEncoder(self.model_name, max_length=min(self.max_length, limit))
        return self._model

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        return [float(s) for s in self.model.predict([(query, p) for p in passages], batch_size=8)]
