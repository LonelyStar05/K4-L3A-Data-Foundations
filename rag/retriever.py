"""HybridRetriever: dense + HyDE + BM25 -> Reciprocal Rank Fusion -> cross-encoder rerank.

Every branch applies the metadata filter *before* ranking (same contract as
src.EmbeddingStore.search_with_filter), so a wrong-audience chunk can never take a slot.

    dense  : EmbeddingStore.search_with_filter(query)           — semantic match
    hyde   : EmbeddingStore.search_with_filter(hypothetical_doc) — query rewritten into
             the *register of the corpus* (regulation prose) before embedding, which closes
             the gap between a short casual question and formal regulation text; one
             passage per corpus language (vi + en), each its own fusion branch
    bm25   : lexical match on exact tokens (amounts, percentages, names)
    RRF    : score(d) = sum over branches of 1 / (rrf_k + rank_branch(d)); rank-based, so
             cosine and BM25 scores never need to be put on the same scale
    rerank : cross-encoder re-scores (query, chunk) pairs jointly over a pool = fused
             top-N + each branch's own top-k (so a specialist branch's winner is never lost)
"""

from __future__ import annotations

from src.models import Document
from src.store import EmbeddingStore

from .bm25 import BM25

# One hypothetical passage per corpus language: the corpus mixes Vietnamese regulations with
# an English guideline, and a Vietnamese-only passage cannot bridge to English text.
# "[X]" placeholders keep invented numbers out of the embedding.
HYDE_PROMPTS = {
    "vi": """Bạn là chuyên viên Phòng Tài chính – Công tác sinh viên của một trường đại học tại Việt Nam.
Hãy viết MỘT đoạn ngắn (3–5 câu) theo đúng văn phong văn bản quy định/chính sách của nhà trường,
như thể được trích nguyên văn từ quy định chính thức, để trả lời câu hỏi dưới đây.
Dùng thuật ngữ hành chính chuẩn (học phí niêm yết, hoàn trả, bảo lưu, học bổng tài năng, hỗ trợ tài chính, điểm trung bình...).
Nếu không chắc con số, tỷ lệ hay mốc thời gian, hãy viết [X] thay vì tự đặt ra.

Câu hỏi: {question}

Đoạn quy định:""",
    "en": """You are an officer at the Finance / Financial Aid Office of a university in Vietnam.
Write ONE short passage (3-5 sentences) in the style of the university's official policy documents,
as if quoted verbatim from an official regulation or guideline, that answers the question below.
The question may be in Vietnamese; write the passage in English. Use standard administrative terms
(listed tuition fee, refund, deferral, merit scholarship, financial aid, GPA, application period...).
If unsure about a number, percentage or date, write [X] instead of inventing it.

Question: {question}

Policy passage:""",
}


class HybridRetriever:
    def __init__(
        self,
        chunks: list[Document],
        embedder,
        *,
        use_dense: bool = True,
        use_bm25: bool = False,
        use_hyde: bool = False,
        reranker=None,
        llm=None,
        candidate_k: int = 20,
        rerank_top_n: int = 20,
        branch_pool: int = 5,
        rrf_k: int = 60,
        name: str = "",
    ) -> None:
        if use_hyde and llm is None:
            raise ValueError("HyDE needs an llm")
        self.name = name
        self.chunks = chunks
        self.use_dense, self.use_bm25, self.use_hyde = use_dense, use_bm25, use_hyde
        self.reranker, self.llm = reranker, llm
        self.candidate_k, self.rerank_top_n, self.rrf_k = candidate_k, rerank_top_n, rrf_k
        self.branch_pool = branch_pool
        self.store = EmbeddingStore(collection_name=name or "hybrid", embedding_fn=embedder)
        self.store.add_documents(chunks)
        self.bm25 = BM25([c.content for c in chunks]) if use_bm25 else None
        languages = sorted({c.metadata.get("language", "vi") for c in chunks} & HYDE_PROMPTS.keys())
        self.hyde_languages = languages or ["vi"]
        self.last_hyde: dict[str, str] = {}

    def hyde_documents(self, question: str) -> dict[str, str]:
        return {lang: self.llm(HYDE_PROMPTS[lang].format(question=question)) for lang in self.hyde_languages}

    def _bm25_ranking(self, query: str, metadata_filter: dict | None) -> list[str]:
        allowed = [i for i, c in enumerate(self.chunks) if not metadata_filter or EmbeddingStore._matches(c.metadata, metadata_filter)]
        ranked = [(i, s) for i, s in self.bm25.scores(query, allowed) if s > 0]
        return [self.chunks[i].id for i, _ in ranked[: self.candidate_k]]

    def retrieve(self, query: str, top_k: int = 3, metadata_filter: dict | None = None) -> list[dict]:
        rankings: dict[str, list[str]] = {}
        dense_scores: dict[str, float] = {}
        if self.use_dense:
            hits = self.store.search_with_filter(query, top_k=self.candidate_k, metadata_filter=metadata_filter)
            rankings["dense"] = [h["id"] for h in hits]
            dense_scores = {h["id"]: h["score"] for h in hits}
        if self.use_hyde:
            self.last_hyde = self.hyde_documents(query)
            for lang, passage in self.last_hyde.items():
                hits = self.store.search_with_filter(passage, top_k=self.candidate_k, metadata_filter=metadata_filter)
                rankings[f"hyde_{lang}"] = [h["id"] for h in hits]
        if self.bm25 is not None:
            rankings["bm25"] = self._bm25_ranking(query, metadata_filter)

        fused: dict[str, float] = {}
        for ranking in rankings.values():
            for rank, chunk_id in enumerate(ranking, start=1):
                fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)
        if len(rankings) == 1 and "dense" in rankings:
            order = rankings["dense"]  # plain dense search: keep cosine order and scores
        else:
            order = sorted(fused, key=fused.get, reverse=True)

        by_id = {c.id: c for c in self.chunks}
        if self.reranker:
            # Rerank pool = fused top-N plus every branch's own top few. Equal-weight RRF dilutes
            # a specialist branch (the English HyDE passage is the only branch that can see the
            # English guideline), so its winners are nominated to the cross-encoder directly.
            nominated = [cid for ranking in rankings.values() for cid in ranking[: self.branch_pool]]
            candidates = list(dict.fromkeys(order[: self.rerank_top_n] + nominated))
        else:
            candidates = order[:top_k]
        rerank_scores: dict[str, float] = {}
        if self.reranker and candidates:
            scores = self.reranker.score(query, [by_id[c].content for c in candidates])
            rerank_scores = dict(zip(candidates, scores))
            candidates = sorted(candidates, key=rerank_scores.get, reverse=True)

        results = []
        for chunk_id in candidates[:top_k]:
            chunk = by_id[chunk_id]
            signals = {name: ranking.index(chunk_id) + 1 for name, ranking in rankings.items() if chunk_id in ranking}
            score = rerank_scores.get(chunk_id) if self.reranker else (dense_scores.get(chunk_id) if len(rankings) == 1 else fused[chunk_id])
            results.append({
                "id": chunk_id,
                "content": chunk.content,
                "metadata": chunk.metadata,
                "score": score,
                "signals": {**signals, "rrf": round(fused.get(chunk_id, 0.0), 4), **({"rerank": round(rerank_scores[chunk_id], 4)} if rerank_scores else {})},
            })
        return results
