"""Chunking strategies that turn SourceDocs into `src.models.Document` chunks.

StructureAwareChunker is built for regulation-style documents: the authors already
split the text into numbered clauses ("A. > II. > 6. Phí ở Ký túc xá"), and each
clause is one self-contained rule. So:

1. Parse the Markdown heading tree. A clause whose whole subtree fits the budget is
   one chunk (sub-clauses that refer to each other stay together); a bigger clause
   recurses into its children; runs of tiny sibling clauses are grouped.
2. Prefix each chunk with a contextual header — document title + heading path —
   so "Mức phí: 20.000.000 đồng" still says *which* fee it is once cut out of the file.
3. A single clause longer than `max_chars` is split on block boundaries (paragraph,
   list item, table row) — never inside a table row — with the header repeated on
   every part; a table that must be split repeats its column header row.
4. A heading with no text of its own produces no chunk of its own; its title lives on
   in the heading path of the chunks below it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.chunking import FixedSizeChunker, RecursiveChunker, SentenceChunker
from src.models import Document

from .corpus import SourceDoc

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_TOP_LIST_ITEM = re.compile(r"^(?:[-*]|\d+\.)\s")


def _split_blocks(text: str, max_chars: int) -> list[str]:
    """Atomic units: blank-line separated blocks. A table or list that fits the budget stays
    whole; a longer one is split per row (header repeated) or per top-level item."""
    units: list[str] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.splitlines()
        if len(block) <= max_chars:
            units.append(block)
        elif lines and lines[0].startswith("|"):
            header, rows = lines[:2], lines[2:]
            units.append("\n".join(header + rows[:1]) if rows else block)
            units.extend("\n".join(header + [row]) for row in rows[1:])  # keep header on every row unit
        elif lines and _TOP_LIST_ITEM.match(lines[0]):
            item: list[str] = []
            for line in lines:
                if _TOP_LIST_ITEM.match(line) and item:
                    units.append("\n".join(item))
                    item = []
                item.append(line)
            units.append("\n".join(item))
        else:
            units.append(block)
    return units


def _pack(units: list[str], max_chars: int) -> list[str]:
    """Greedily pack units into parts <= max_chars; merge consecutive table-row units back
    into one table (dropping the repeated header) while they fit."""
    parts: list[str] = []
    current = ""
    for unit in units:
        if len(unit) > max_chars:  # one paragraph/item longer than the budget: last-resort split
            if current:
                parts.append(current)
                current = ""
            parts.extend(RecursiveChunker(chunk_size=max_chars).chunk(unit))
            continue
        if current and unit.startswith("|") and current.split("\n\n")[-1].startswith("|"):
            candidate = current + "\n" + unit.split("\n", 2)[2]  # same table: append the row only
        elif current and _TOP_LIST_ITEM.match(unit) and _TOP_LIST_ITEM.match(current.split("\n\n")[-1].split("\n")[0]):
            candidate = current + "\n" + unit  # same list: no blank line between items
        else:
            candidate = f"{current}\n\n{unit}" if current else unit
        if len(candidate) <= max_chars:
            current = candidate
        else:
            parts.append(current)
            current = unit
    if current:
        parts.append(current)
    return parts


@dataclass
class Node:
    """A section in the heading tree: its own text plus its sub-sections."""

    level: int
    title: str
    path: list[str]
    lines: list[str] = field(default_factory=list)
    children: list["Node"] = field(default_factory=list)

    @property
    def own_text(self) -> str:
        return "\n".join(self.lines).strip()

    def full_text(self, with_heading: bool = False) -> str:
        """The whole subtree as Markdown (sub-headings kept, so structure survives inside a chunk)."""
        parts = [f"{'#' * self.level} {self.title}"] if with_heading and self.title else []
        if self.own_text:
            parts.append(self.own_text)
        parts += [child.full_text(with_heading=True) for child in self.children]
        return "\n\n".join(parts)


def build_tree(body: str) -> Node:
    root = Node(level=1, title="", path=[])
    stack = [root]
    in_code = False
    for line in body.splitlines():
        if line.startswith("```"):
            in_code = not in_code
        match = None if in_code else _HEADING.match(line)
        if match and len(match.group(1)) == 1:
            continue  # H1 = document title, carried in the contextual header
        if match:
            level = len(match.group(1))
            while len(stack) > 1 and stack[-1].level >= level:
                stack.pop()
            node = Node(level=level, title=match.group(2), path=stack[-1].path + [match.group(2)])
            stack[-1].children.append(node)
            stack.append(node)
        else:
            stack[-1].lines.append(line)
    return root


class StructureAwareChunker:
    """mode="tree" (default): a section whose whole subtree fits `max_chars` is ONE chunk
    (a numbered clause with its sub-clauses, e.g. "2. Ưu đãi" + 2.1–2.4, stays together:
    2.4 refers back to 2.1–2.3); bigger sections recurse into their children, and runs of
    tiny sibling sections (< min_chars) are grouped under their parent's path.
    mode="leaf": every section's own text is a chunk (finest granularity)."""

    def __init__(self, max_chars: int = 2200, min_chars: int = 400, mode: str = "tree", contextual_header: bool = True) -> None:
        self.max_chars, self.min_chars, self.mode = max_chars, min_chars, mode
        self.contextual_header = contextual_header
        self.name = "structure_aware" if mode == "tree" else "structure_leaf"

    def header(self, doc: SourceDoc, path: list[str], part: int = 1, n_parts: int = 1) -> str:
        lines = [f"Tài liệu: {doc.title}"]
        if path:
            lines.append("Mục: " + " > ".join(path))
        if n_parts > 1:
            lines[-1] += f" (phần {part}/{n_parts})"
        return "\n".join(lines)

    def _budget(self, doc: SourceDoc, path: list[str]) -> int:
        return self.max_chars - (len(self.header(doc, path, 9, 9)) + 2 if self.contextual_header else 0)

    def _units(self, doc: SourceDoc, node: Node) -> list[tuple[list[str], str]]:
        """(heading path, text) units for one subtree, following the tree/leaf policy."""
        budget = self._budget(doc, node.path)
        if self.mode == "leaf":
            out = [(node.path, node.own_text)] if node.own_text else []
            for child in node.children:
                out += self._units(doc, child)
            return out

        whole = node.full_text()
        if whole and len(whole) <= budget:
            return [(node.path, whole)]
        out = [(node.path, node.own_text)] if node.own_text else []
        group: list[Node] = []

        def flush() -> None:
            if len(group) == 1:
                out.extend(self._units(doc, group[0]))
            elif group:
                out.append((node.path, "\n\n".join(c.full_text(with_heading=True) for c in group)))
            group.clear()

        for child in node.children:
            size = len(child.full_text(with_heading=True))
            if size >= self.min_chars:
                flush()
                out.extend(self._units(doc, child))
                continue
            if group and len("\n\n".join(c.full_text(with_heading=True) for c in group + [child])) > budget:
                flush()
            group.append(child)
        flush()
        return out

    def split(self, doc: SourceDoc) -> list[tuple[str, dict]]:
        out: list[tuple[str, dict]] = []
        for path, text in self._units(doc, build_tree(doc.body)):
            budget = self._budget(doc, path)
            parts = [text] if len(text) <= budget else _pack(_split_blocks(text, budget), budget)
            for i, part in enumerate(parts, start=1):
                body = f"{self.header(doc, path, i, len(parts))}\n\n{part}" if self.contextual_header else part
                out.append((body, {
                    "section_path": " > ".join(path) or "(mở đầu)",
                    "section_title": path[-1] if path else doc.title,
                    "heading_level": len(path) + 1,
                    "part": i,
                    "n_parts": len(parts),
                }))
        return out


class BaselineChunker:
    """The lab's src chunkers applied to the document body (frontmatter already removed)."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._chunker = {
            "fixed_size": FixedSizeChunker(chunk_size=500, overlap=50),
            "by_sentences": SentenceChunker(max_sentences_per_chunk=3),
            "recursive": RecursiveChunker(chunk_size=500),
        }[name]

    def split(self, doc: SourceDoc) -> list[tuple[str, dict]]:
        return [(chunk, {}) for chunk in self._chunker.chunk(doc.body)]


def chunk_corpus(docs: list[SourceDoc], chunker) -> list[Document]:
    """One Document per chunk: id "<doc_id>#<i>", frontmatter spread into every chunk's
    metadata (so search_with_filter can filter), doc_id pointing at the source file."""
    chunks: list[Document] = []
    for doc in docs:
        for i, (text, extra) in enumerate(chunker.split(doc)):
            metadata = {**doc.metadata, **extra, "doc_id": doc.doc_id, "chunk_index": i, "strategy": chunker.name}
            chunks.append(Document(id=f"{doc.doc_id}#{i}", content=text, metadata=metadata))
    return chunks
