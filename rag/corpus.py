"""Load the Markdown corpus: YAML frontmatter becomes metadata, the body becomes content."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_FRONTMATTER = re.compile(r"^---\n(.*?)\n---\n", re.S)


@dataclass
class SourceDoc:
    doc_id: str
    title: str
    body: str  # Markdown without frontmatter, starts with "# Title"
    metadata: dict[str, str]


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    match = _FRONTMATTER.match(text)
    if not match:
        return {}, text
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        key, sep, value = line.partition(":")
        if sep:
            meta[key.strip()] = value.split(" #", 1)[0].strip().strip('"').strip("'")
    return meta, text[match.end():].lstrip("\n")


def load_corpus(directory: str | Path) -> list[SourceDoc]:
    docs = []
    for path in sorted(Path(directory).glob("*.md")):
        meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        meta.setdefault("doc_id", path.stem)
        docs.append(SourceDoc(doc_id=meta["doc_id"], title=meta.get("title", path.stem), body=body, metadata=meta))
    return docs
