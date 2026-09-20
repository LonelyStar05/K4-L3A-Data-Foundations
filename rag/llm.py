"""OpenAI chat wrapper with a deterministic on-disk cache (temperature 0).

Used for HyDE generation and answer generation. Caching keeps benchmark reruns free
and makes the numbers in ket_qua_benchmark.txt reproducible.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "llm"


class CachedChatLLM:
    def __init__(self, model_name: str | None = None, max_tokens: int = 700) -> None:
        from openai import OpenAI

        self.model_name = model_name or os.getenv("OPENAI_CHAT_MODEL") or "gpt-4.1-mini"
        self.max_tokens = max_tokens
        self.client = OpenAI()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._path = CACHE_DIR / f"{self.model_name}.jsonl"
        self._cache: dict[str, str] = {}
        if self._path.exists():
            for line in self._path.read_text(encoding="utf-8").splitlines():
                key, value = json.loads(line)
                self._cache[key] = value

    def __call__(self, prompt: str) -> str:
        key = hashlib.sha256(f"{self.model_name}\n{self.max_tokens}\n{prompt}".encode()).hexdigest()
        if key not in self._cache:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=self.max_tokens,
            )
            self._cache[key] = (response.choices[0].message.content or "").strip()
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps([key, self._cache[key]], ensure_ascii=False) + "\n")
        return self._cache[key]
