from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    NOT_FOUND = "Không tìm thấy thông tin liên quan trong cơ sở tri thức."

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    @staticmethod
    def build_context(chunks: list[dict]) -> str:
        """Number every chunk [1], [2]... with its source so the answer can cite it."""
        blocks = []
        for i, chunk in enumerate(chunks, start=1):
            meta = chunk.get("metadata", {})
            source = meta.get("doc_id", chunk.get("id", "?"))
            section = meta.get("section_path") or meta.get("title") or ""
            header = f"[{i}] (nguồn: {source}{' — ' + section if section else ''})"
            blocks.append(f"{header}\n{chunk['content']}")
        return "\n\n".join(blocks)

    def answer(self, question: str, top_k: int = 3) -> str:
        chunks = self.store.search(question, top_k=top_k)
        if not chunks:
            return self.NOT_FOUND  # empty store: do not waste an LLM call
        prompt = (
            "Bạn là trợ lý trả lời câu hỏi về quy định và dịch vụ của trường đại học.\n"
            "Chỉ dùng thông tin trong NGỮ CẢNH bên dưới. Mỗi ý trong câu trả lời phải trích dẫn "
            "số đoạn nguồn dạng [1], [2]. Nếu ngữ cảnh không chứa câu trả lời, hãy nói rõ "
            f"\"{self.NOT_FOUND}\" và không suy đoán.\n\n"
            f"NGỮ CẢNH:\n{self.build_context(chunks)}\n\n"
            f"CÂU HỎI: {question}\n\nTRẢ LỜI:"
        )
        return self.llm_fn(prompt)
