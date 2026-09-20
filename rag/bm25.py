"""BM25 sparse retrieval tuned for Vietnamese regulation text.

Why a lexical retriever next to embeddings: questions about fees hinge on exact tokens
("2.000.000", "2,5%", "IFOM", "Techcombank") that dense vectors blur. Vietnamese words
are multi-syllable ("học phí", "hoàn trả"), so syllable bigrams are indexed too — the
bigram "học_phí" is far more specific than "học" or "phí" alone.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

# Numbers keep their separators ("349.650.000", "2,5%"); everything else is a word token.
_TOKEN = re.compile(r"\d+(?:[.,]\d+)*%?|\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    words = _TOKEN.findall(unicodedata.normalize("NFC", text).lower())
    bigrams = [f"{a}_{b}" for a, b in zip(words, words[1:]) if not a[0].isdigit() and not b[0].isdigit()]
    return words + bigrams


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        df = Counter(term for d in self.docs for term in d)
        n = len(self.docs)
        self.idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}

    def scores(self, query: str, candidates: list[int] | None = None) -> list[tuple[int, float]]:
        terms = [t for t in set(tokenize(query)) if t in self.idf]
        indices = range(len(self.docs)) if candidates is None else candidates
        out = []
        for i in indices:
            doc, length = self.docs[i], self.lengths[i]
            score = 0.0
            for term in terms:
                tf = doc.get(term, 0)
                if tf:
                    score += self.idf[term] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / self.avg_len))
            out.append((i, score))
        out.sort(key=lambda pair: pair[1], reverse=True)
        return out
