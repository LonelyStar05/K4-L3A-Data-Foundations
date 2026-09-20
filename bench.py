#!/usr/bin/env python3
"""Benchmark: 5 group queries x retrieval strategies on data/hoc-phi-vinuni.

What it does (lab §6 "bench.py"):
  1. Read every .md: frontmatter -> metadata, body -> content              (rag.corpus)
  2. Chunk OUTSIDE the store; one Document per chunk, id "<file>#<i>",
     frontmatter spread into every chunk, metadata["doc_id"] = source file (rag.chunkers)
  3. Load into src.EmbeddingStore and query through search_with_filter()   (rag.retriever)
  4. Print top-3 with score + doc_id, score each query at TWO levels:
       - naive  : is a gold doc_id in the top-3?           (inflates results)
       - content: does a top-3 chunk from a gold doc contain the answer fact,
                  and does the generated answer state it?  (docs/SCORING.md)
     2 = relevant chunk at top-1 + correct answer, 1 = relevant chunk in top-3,
     0 = no relevant chunk in top-3.
  5. A/B: every query that needs {"audience": "student"} is also run without it.

Usage:
    python bench.py                      # all strategies -> ket_qua_benchmark.txt
    python bench.py --only structure_tree          # one strategy
    python bench.py --no-rerank                    # skip the cross-encoder (no model download)
    python bench.py --rerankers BAAI/bge-reranker-v2-m3,cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
"""

from __future__ import annotations

import argparse
import sys
import time
import unicodedata
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from rag.chunkers import BaselineChunker, StructureAwareChunker, chunk_corpus
from rag.corpus import load_corpus
from rag.embeddings import CachedOpenAIEmbedder
from rag.llm import CachedChatLLM
from rag.pipeline import RAGPipeline
from rag.rerank import CrossEncoderReranker
from rag.retriever import HybridRetriever

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "data" / "hoc-phi-vinuni"
OUTPUT = ROOT / "ket_qua_benchmark.txt"
STUDENT = {"audience": "student"}

# Each *_keys entry is a group of alternatives; every group must be matched.
QUERIES = [
    {
        "id": "Q1", "type": "tra số liệu",
        "question": "Học phí niêm yết một năm của chương trình Cử nhân Điều dưỡng là bao nhiêu?",
        "gold": "349.650.000 VND/năm (174.825.000 VND/kỳ; 9.780.000 VND/tín chỉ).",
        "gold_docs": ["quy-dinh-tai-chinh-bieu-phi", "hoc-phi-cu-nhan", "faq-hoc-phi-hoc-bong"],
        "context_keys": [["349.650.000"]],
        "answer_keys": [["349.650.000", "349,65 triệu", "349.65"]],
        "filter": None,
    },
    {
        "id": "Q2", "type": "điều kiện",
        "question": "Nếu thôi học trong vòng 2 tuần đầu của học kỳ thì được hoàn trả bao nhiêu phần trăm học phí?",
        "gold": "Hoàn trả 50% học phí thực nộp của học kỳ (trước ngày học đầu tiên: 80%; sau 2 tuần: không hoàn).",
        "gold_docs": ["quy-dinh-tai-chinh-bieu-phi"],
        "context_keys": [["Hoàn trả 50%"]],
        "answer_keys": [["50%"]],
        "filter": None,
    },
    {
        "id": "Q3", "type": "điều kiện — cần lọc audience",
        "question": "Học bổng 100% mà điểm trung bình năm học chỉ đạt 2,8 thì có bị hạ học bổng không?",
        "gold": "Không bị hạ ngay: GPA năm học 2,50–3,19 được 'duy trì học bổng có điều kiện', gia hạn thêm 1 học kỳ "
                "để cải thiện kết quả và thể hiện E.X.C.E.L; chỉ tự động hạ 1 bậc khi GPA 0,0–2,49 (GDL-SAM-004 V2.1). "
                "Bẫy: FAQ tuyển sinh (audience=all) nói chung chung là 'giảm 1 bậc (10%)'.",
        "gold_docs": ["duy-tri-hoc-bong-ho-tro-tai-chinh"],
        "context_keys": [["Duy trì học bổng có điều kiện"]],
        "answer_keys": [["có điều kiện"], ["1 học kỳ", "một học kỳ"]],
        "filter": STUDENT,
    },
    {
        "id": "Q4", "type": "liệt kê",
        "question": "Có những chính sách ưu đãi học phí hoặc chiết khấu đóng phí nào?",
        "gold": "Ưu đãi Gia đình giảm 2,5% (từ người thứ 2); ưu đãi Cựu sinh viên 10%; chiết khấu 5% khi đóng học phí và "
                "phí KTX cả năm đúng hạn. Ưu đãi cựu SV không cộng với ưu đãi gia đình.",
        "gold_docs": ["quy-dinh-tai-chinh-bieu-phi"],
        "context_keys": [["2,5%"], ["10%"], ["5%"]],
        "answer_keys": [["2,5%", "2.5%"], ["10%"], ["5%"]],
        "filter": None,
    },
    {
        "id": "Q5", "type": "mốc thời gian (hỏi tiếng Việt, nguồn tiếng Anh) — cần lọc audience",
        "question": "Hạn nộp hồ sơ xin hỗ trợ tài chính cho học kỳ mùa Thu là khi nào?",
        "gold": "Đợt nộp 20/6 – 10/7 (hạn 10/7), hạn xử lý 02/8, áp dụng cho học kỳ Thu (GDL-FAO-001). "
                "Bẫy: trang tân sinh viên (audience=all) ghi hạn '23:59 ngày 15 của tháng liền kề trước'.",
        "gold_docs": ["huong-dan-de-nghi-ho-tro-tai-chinh", "ho-tro-tai-chinh-sinh-vien-dang-hoc"],
        "context_keys": [["20 June – 10 July", "July 10th", "tháng 7 và tháng 11"]],
        "answer_keys": [["10/7", "10 tháng 7", "10/07", "July 10", "10 July", "ngày 10 tháng 7"]],
        "filter": STUDENT,
    },
]


def norm(text: str) -> str:
    return unicodedata.normalize("NFC", text).lower().replace("–", "-").replace("—", "-")


def has(text: str, alternatives: list[str]) -> bool:
    return any(norm(a) in norm(text) for a in alternatives)


def evaluate(query: dict, result: dict) -> dict:
    chunks = result["chunks"]
    primary = query["context_keys"][0]
    relevant = [
        c["metadata"]["doc_id"] in query["gold_docs"] and has(c["content"], primary)
        for c in chunks
    ]
    doc_hit = any(c["metadata"]["doc_id"] in query["gold_docs"] for c in chunks)
    union = "\n".join(c["content"] for c in chunks)
    coverage = sum(has(union, group) for group in query["context_keys"]) / len(query["context_keys"])
    answer_ok = all(has(result["answer"], group) for group in query["answer_keys"])
    first = relevant.index(True) + 1 if any(relevant) else None
    score = 2 if first == 1 and answer_ok else (1 if first else 0)
    return {"score": score, "first_relevant": first, "doc_hit": doc_hit, "coverage": coverage, "answer_ok": answer_ok}


def build_strategies(chunk_sets: dict, embedder, llm, rerankers: list) -> dict[str, HybridRetriever]:
    specs = {
        # lab baselines: same embedder, dense search only, only the chunker changes
        "fixed_size": ("fixed_size", {}),
        "by_sentences": ("by_sentences", {}),
        "recursive": ("recursive", {}),
        "structure_leaf": ("structure_leaf", {}),
        "structure_tree": ("structure_aware", {}),
        # ablation on the structure-aware (tree) chunks: add one technique at a time
        "tree+bm25": ("structure_aware", {"use_bm25": True}),
        "tree+bm25+hyde": ("structure_aware", {"use_bm25": True, "use_hyde": True, "llm": llm}),
    }
    for reranker in rerankers:  # full pipeline = tree chunks + BM25 + HyDE + cross-encoder
        short = reranker.model_name.split("/")[-1].split("-v")[0]
        specs[f"full[{short}]"] = ("structure_aware", {"use_bm25": True, "use_hyde": True, "llm": llm, "reranker": reranker})
    return {name: HybridRetriever(chunk_sets[chunker], embedder, name=name, **options) for name, (chunker, options) in specs.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="run a single strategy by name")
    parser.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder strategy")
    parser.add_argument("--rerankers", default=None, help="comma-separated cross-encoder models (default: RERANKER_MODEL or bge-reranker-v2-m3)")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env", override=False)
    docs = load_corpus(CORPUS)
    chunkers = [
        BaselineChunker("fixed_size"), BaselineChunker("by_sentences"), BaselineChunker("recursive"),
        StructureAwareChunker(mode="leaf", max_chars=1200), StructureAwareChunker(mode="tree"),
    ]
    chunk_sets = {c.name: chunk_corpus(docs, c) for c in chunkers}

    embedder = CachedOpenAIEmbedder()
    llm = CachedChatLLM()
    rerankers = [] if args.no_rerank else [
        CrossEncoderReranker(name) for name in (args.rerankers.split(",") if args.rerankers else [None])
    ]
    strategies = build_strategies(chunk_sets, embedder, llm, rerankers)
    if args.only:
        strategies = {args.only: strategies[args.only]}

    lines: list[str] = []
    out = lines.append
    out(f"KẾT QUẢ BENCHMARK — Lab 7, chủ đề: Học phí / Học bổng / Hỗ trợ tài chính VinUni ({date.today()})")
    out(f"Corpus: {CORPUS.relative_to(ROOT)} — {len(docs)} tài liệu")
    out(f"Embedding: {embedder._backend_name} | LLM (HyDE + trả lời): {llm.model_name} | "
        f"Reranker: {', '.join(r.model_name for r in rerankers) or 'tắt'}")
    out("Chấm: 2 = chunk liên quan ở top-1 + câu trả lời đúng; 1 = chunk liên quan trong top-3; 0 = không có.")
    out("'Liên quan' = chunk thuộc tài liệu gold VÀ chứa dữ kiện trả lời (không chỉ đúng doc_id).")

    out("\n" + "=" * 100 + "\n1. THỐNG KÊ CHUNK (baseline analysis)\n" + "=" * 100)
    for name, chunks in chunk_sets.items():
        lens = [len(c.content) for c in chunks]
        out(f"{name:16} số chunk={len(chunks):4}  dài TB={sum(lens) / len(lens):6.0f}  min={min(lens):4}  max={max(lens):5}")

    summary: dict[str, list[dict]] = {name: [] for name in strategies}
    ab_rows: list[str] = []
    started = time.time()
    for query in QUERIES:
        out("\n" + "=" * 100)
        out(f"{query['id']} [{query['type']}] {query['question']}")
        out(f"   filter: {query['filter']}")
        out(f"   gold  : {query['gold']}")
        out("=" * 100)
        for name, retriever in strategies.items():
            pipeline = RAGPipeline(retriever, llm, top_k=3)
            result = pipeline.answer(query["question"], metadata_filter=query["filter"])
            verdict = evaluate(query, result)
            summary[name].append(verdict)
            out(f"\n-- {name}: điểm {verdict['score']}/2 | chunk liên quan đầu tiên: {verdict['first_relevant'] or '-'} | "
                f"doc_id gold trong top-3: {'có' if verdict['doc_hit'] else 'không'} | độ phủ dữ kiện: {verdict['coverage']:.0%} | "
                f"trả lời đúng: {'có' if verdict['answer_ok'] else 'không'}")
            if retriever.use_hyde:
                for lang, passage in retriever.last_hyde.items():
                    out(f"   HyDE[{lang}]: {passage[:200].replace(chr(10), ' ')}...")
            for rank, chunk in enumerate(result["chunks"], start=1):
                meta = chunk["metadata"]
                mark = "*" if meta["doc_id"] in query["gold_docs"] and has(chunk["content"], query["context_keys"][0]) else " "
                where = meta.get("section_path", "")[:70]
                out(f"   {mark}{rank}. score={chunk['score']:.4f} {chunk['id']:42} [{meta.get('audience')}] {where}")
                out(f"      signals={chunk.get('signals', {})}")
            out(f"   Trả lời: {result['answer'].replace(chr(10), ' ')[:600]}")

            if query["filter"]:
                unfiltered = pipeline.answer(query["question"], metadata_filter=None)
                v2 = evaluate(query, unfiltered)
                tops = lambda r: ", ".join(f"{c['id']}[{c['metadata'].get('audience')}]" for c in r["chunks"])  # noqa: E731
                ab_rows.append(
                    f"{query['id']} | {name:32} | có filter  : điểm {verdict['score']} | {tops(result)}\n"
                    f"{'':3}| {'':32} | không filter: điểm {v2['score']} | {tops(unfiltered)}\n"
                    f"{'':3}| {'':32} | trả lời không filter: {unfiltered['answer'].replace(chr(10), ' ')[:300]}"
                )

    out("\n" + "=" * 100 + "\n2. TỔNG HỢP (điểm /10 theo docs/SCORING.md)\n" + "=" * 100)
    width = max(len(name) for name in summary) + 2
    out(f"{'strategy':{width}}" + "".join(f"{q['id']:>5}" for q in QUERIES) + "   tổng   doc-hit@3   relevant@3   MRR    trả lời đúng")
    for name, verdicts in summary.items():
        total = sum(v["score"] for v in verdicts)
        doc_hits = sum(v["doc_hit"] for v in verdicts)
        rel = sum(bool(v["first_relevant"]) for v in verdicts)
        mrr = sum(1 / v["first_relevant"] for v in verdicts if v["first_relevant"]) / len(verdicts)
        answers = sum(v["answer_ok"] for v in verdicts)
        out(f"{name:{width}}" + "".join(f"{v['score']:>5}" for v in verdicts)
            + f"   {total:>2}/10   {doc_hits}/5         {rel}/5          {mrr:.2f}   {answers}/5")

    out("\n" + "=" * 100 + "\n3. A/B METADATA FILTER {'audience': 'student'} (các câu cần lọc)\n" + "=" * 100)
    lines.extend(ab_rows)
    out(f"\n(thời gian chạy: {time.time() - started:.0f}s)")

    report = "\n".join(lines)
    print(report)
    if not args.only:
        OUTPUT.write_text(report + "\n", encoding="utf-8")
        print(f"\n-> đã ghi {OUTPUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
