"""Retrieve -> grounded, cited answer."""

from __future__ import annotations

ANSWER_PROMPT = """Bạn là trợ lý tư vấn học phí, học bổng và hỗ trợ tài chính của Trường Đại học VinUni.
Chỉ dùng thông tin trong NGỮ CẢNH. Mỗi ý trả lời phải trích dẫn số đoạn nguồn dạng [1], [2].
Giữ nguyên con số, tỷ lệ, mốc thời gian như trong nguồn. Nếu NGỮ CẢNH không đủ để trả lời,
nói rõ: "Không tìm thấy thông tin trong tài liệu được cung cấp." — tuyệt đối không suy đoán.
Trả lời ngắn gọn bằng tiếng Việt.

NGỮ CẢNH:
{context}

CÂU HỎI: {question}

TRẢ LỜI:"""

NOT_FOUND = "Không tìm thấy thông tin trong tài liệu được cung cấp."


def format_context(chunks: list[dict]) -> str:
    """Number chunks and label each with its provenance, so every claim is traceable."""
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk["metadata"]
        label = " | ".join(filter(None, [
            f"nguồn: {meta.get('doc_id')}",
            f"đối tượng: {meta.get('audience')}" if meta.get("audience") else "",
            f"phiên bản: {meta.get('document_version')}" if meta.get("document_version") else "",
        ]))
        blocks.append(f"[{i}] ({label})\n{chunk['content']}")
    return "\n\n".join(blocks)


class RAGPipeline:
    def __init__(self, retriever, llm, top_k: int = 3) -> None:
        self.retriever, self.llm, self.top_k = retriever, llm, top_k

    def answer(self, question: str, metadata_filter: dict | None = None) -> dict:
        chunks = self.retriever.retrieve(question, top_k=self.top_k, metadata_filter=metadata_filter)
        if not chunks:
            return {"answer": NOT_FOUND, "chunks": []}
        answer = self.llm(ANSWER_PROMPT.format(context=format_context(chunks), question=question))
        return {"answer": answer, "chunks": chunks}
