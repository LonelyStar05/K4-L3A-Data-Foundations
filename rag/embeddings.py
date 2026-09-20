"""OpenAI embeddings with batching and an on-disk cache.

Plugs straight into `src.EmbeddingStore(embedding_fn=...)`: it is callable text -> vector,
and exposes `prefetch(texts)`, which EmbeddingStore.add_documents calls to embed a whole
batch in a few API requests instead of one request per chunk. Re-running the benchmark
costs nothing: every vector is cached by sha256(model + text).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "embeddings"
BATCH_SIZE = 96


class CachedOpenAIEmbedder:
    def __init__(self, model_name: str | None = None) -> None:
        from openai import OpenAI

        self.model_name = model_name or os.getenv("OPENAI_EMBEDDING_MODEL") or "text-embedding-3-small"
        self._backend_name = f"{self.model_name} (cached)"
        self.client = OpenAI()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._path = CACHE_DIR / f"{self.model_name}.jsonl"
        self._cache: dict[str, list[float]] = {}
        if self._path.exists():
            for line in self._path.read_text(encoding="utf-8").splitlines():
                key, vector = json.loads(line)
                self._cache[key] = vector

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model_name}\n{text}".encode()).hexdigest()

    @staticmethod
    def _normalize(vector: list[float]) -> list[float]:
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def prefetch(self, texts: list[str]) -> None:
        missing = list(dict.fromkeys(t for t in texts if self._key(t) not in self._cache))
        with self._path.open("a", encoding="utf-8") as handle:
            for start in range(0, len(missing), BATCH_SIZE):
                batch = missing[start : start + BATCH_SIZE]
                response = self.client.embeddings.create(model=self.model_name, input=batch)
                for text, item in zip(batch, response.data):
                    key = self._key(text)
                    self._cache[key] = self._normalize(item.embedding)
                    handle.write(json.dumps([key, self._cache[key]]) + "\n")

    def __call__(self, text: str) -> list[float]:
        key = self._key(text)
        if key not in self._cache:
            self.prefetch([text])
        return self._cache[key]
