"""Convert raw HTML / PDF sources into section-structured Markdown.

Both parsers emit the same intermediate representation (a flat list of
`Block`s); `render_markdown` then applies one formatting contract, so every
corpus file looks the same regardless of where it came from:

* one `# Title`, sections as `##`, sub-sections one level deeper each time,
  never skipping a level;
* original numbering kept in heading text ("## A. ...", "### I. ...");
* lists as `-` / `1.` with 2-space nesting;
* short tables as GFM pipe tables, tables with long/multi-part cells expanded
  into one sub-section per row ("record" layout) so no chunk has to cut a row;
* no navigation, images, emoji, empty paragraphs or page numbers.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- IR


@dataclass
class Block:
    kind: str  # heading | para | list | table
    text: str = ""
    level: int = 0
    items: list[tuple[int, str]] = field(default_factory=list)  # (depth, text)
    ordered: bool = False
    rows: list[list[str]] = field(default_factory=list)  # rows[0] is the header


# --------------------------------------------------------------------------- text

_EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u26FF\u2B50\u2B55\uFE0F]")
_SPACE = re.compile("[ \t\u00a0\u2009\u202f\u200b\ufeff]+")


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = _EMOJI.sub("", text)
    text = _SPACE.sub(" ", text.replace("\r", " ").replace("\n", " "))
    text = re.sub(r"\s+([,.;:!?)])", r"\1", text)
    text = re.sub(r"([(])\s+", r"\1", text)
    text = re.sub(r"[”\"“]{3,}", "", text)  # stray quote runs left in the CMS source
    text = re.sub(r"\[\s*([^\[\]]*?\(https?://[^)\s]+\))\s*\]", r"\1", text)  # "[ tại đây (url) ]"
    return text.strip()


def decode_cf_email(hex_string: str) -> str:
    """Undo Cloudflare's e-mail obfuscation (data-cfemail = XOR key + payload)."""
    key = int(hex_string[:2], 16)
    return "".join(chr(int(hex_string[i : i + 2], 16) ^ key) for i in range(2, len(hex_string), 2))


# --------------------------------------------------------------------------- HTML

_SKIP_TAGS = {"script", "style", "noscript", "svg", "img", "picture", "figure", "iframe", "form", "nav"}
_BLOCK_TAGS = {"p", "ul", "ol", "table", "h1", "h2", "h3", "h4", "h5", "h6"}
_DASH_ITEM = re.compile(r"^\s*[–—\-•]\s+")
_POINTER_LINK = re.compile(r"(?i)^\[?\s*(tại đây|đây|here|link|click here|bấm vào đây)\s*\]?([.,;:]?)$")


def _inline_text(node) -> str:
    return clean_text(node.get_text(" "))


def _is_strong_only(p) -> bool:
    strong = p.find(["strong", "b"])
    return bool(strong) and clean_text(strong.get_text(" ")) == clean_text(p.get_text(" "))


def _list_items(list_node, depth: int = 0) -> list[tuple[int, str]]:
    items: list[tuple[int, str]] = []
    for li in list_node.find_all("li", recursive=False):
        nested = li.find_all(["ul", "ol"], recursive=False)
        for sub in nested:
            sub.extract()
        text = _inline_text(li)
        if text:
            items.append((depth, text))
        for sub in nested:
            items.extend(_list_items(sub, depth + 1))
    return items


def _cell_lines(node) -> list[str]:
    """Cell content in document order: text runs as lines, list items as "- item" lines."""
    lines: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        text = clean_text(" ".join(buffer))
        if text:
            lines.append(text)
        buffer.clear()

    def visit(n) -> None:
        for child in n.children:
            name = getattr(child, "name", None)
            if name in {"ul", "ol"}:
                flush()
                lines.extend(f"- {clean_text(li.get_text(' '))}" for li in child.find_all("li") if clean_text(li.get_text(" ")))
            elif name in {"p", "div"}:
                flush()
                visit(child)
                flush()
            elif name is None:
                buffer.append(str(child))
            else:
                buffer.append(child.get_text(" "))

    visit(node)
    flush()
    return lines


def _table_rows(table) -> list[list[str]]:
    """Cells with several parts keep them as newline-separated lines (rendered as nested bullets)."""
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells = ["\n".join(_cell_lines(cell)) for cell in tr.find_all(["th", "td"], recursive=False)]
        if any(cells):
            rows.append(cells)
    return rows


def html_to_blocks(root) -> list[Block]:
    """Walk a BeautifulSoup subtree and emit blocks in document order."""
    for node in root.select("[data-cfemail]"):
        node.replace_with(decode_cf_email(node["data-cfemail"]))
    for node in root.select("h1 button, h2 button, h3 button, h4 button, h5 button, h6 button"):
        node.unwrap()  # FAQ accordions put the question inside <h3><button>
    for node in root.find_all([*_SKIP_TAGS, "button"]):
        node.decompose()
    for link in root.find_all("a", href=True):
        # "xem [tại đây]" only makes sense with the target; keep it for traceability.
        pointer = _POINTER_LINK.match(clean_text(link.get_text(" ")))
        if pointer and link["href"].startswith("http"):
            link.replace_with(f"{pointer.group(1)} ({link['href']}){pointer.group(2)}")
    for br in root.find_all("br"):
        br.replace_with("\n")

    blocks: list[Block] = []
    last_real_heading = 1

    def emit_para(p) -> None:
        segments = [clean_text(s) for s in p.get_text(" ").split("\n")]
        segments = [s for s in segments if s]
        # <br>-separated "– item" lines inside one <p> are really a list.
        if len(segments) > 1 and sum(bool(_DASH_ITEM.match(s)) for s in segments) >= 2:
            lead = [s for s in segments if not _DASH_ITEM.match(s)]
            if lead:
                blocks.append(Block("para", text=" ".join(lead)))
            blocks.append(Block("list", items=[(0, _DASH_ITEM.sub("", s)) for s in segments if _DASH_ITEM.match(s)]))
            return
        # A <br>-separated block of "Label: value" lines (contact cards) is a list too.
        labelled = sum(bool(re.match(r"^[^:]{2,40}:", s)) for s in segments)
        if (len(segments) >= 3 and labelled >= 2) or (len(segments) == 2 and labelled == 2):
            head, *rest = segments
            if re.match(r"^[^:]{2,40}:", head):
                head, rest = "", segments
            if head:
                blocks.append(Block("para", text=head))
            blocks.append(Block("list", items=[(0, s) for s in rest]))
            return
        text = " ".join(segments)
        if text:
            blocks.append(Block("para", text=text))

    def walk(node) -> None:
        nonlocal last_real_heading
        for child in node.children:
            name = getattr(child, "name", None)
            if name is None:
                text = clean_text(str(child))
                if text and node is root:
                    blocks.append(Block("para", text=text))
                continue
            if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
                text = _inline_text(child)
                if not text:
                    continue
                # A long heading ending in ":" is a lead-in sentence, not a section title;
                # a heading that only points somewhere else ("xem tại ...") is a sentence too.
                if (text.endswith(":") and len(text) > 40) or re.match(r"(?i)thông tin chi tiết xem tại", text):
                    blocks.append(Block("para", text=text))
                    continue
                last_real_heading = int(name[1])
                blocks.append(Block("heading", text=text.rstrip(":"), level=last_real_heading))
            elif name == "p":
                text = _inline_text(child)
                if not text:
                    continue
                if _is_strong_only(child):
                    if text.endswith(":"):
                        blocks.append(Block("para", text=f"**{text}**"))
                    elif len(text) <= 100:
                        # <p><strong> pseudo-headings are siblings of the last real heading, not children
                        blocks.append(Block("heading", text=text, level=max(last_real_heading, 2)))
                    else:
                        blocks.append(Block("para", text=text))
                else:
                    emit_para(child)
            elif name in {"ul", "ol"}:
                items = _list_items(child)
                if items:
                    blocks.append(Block("list", items=items, ordered=name == "ol"))
            elif name == "table":
                rows = _table_rows(child)
                if rows:
                    blocks.append(Block("table", rows=rows))
            else:
                if child.find(_BLOCK_TAGS):
                    walk(child)
                else:
                    text = _inline_text(child)
                    if text:
                        blocks.append(Block("para", text=text))

    walk(root)
    return blocks


# --------------------------------------------------------------------------- PDF

_BULLET_MARKERS = {"•", "", "▪", "✓", "", "o", "§", "-", "–", "❖", "➢", "►"}
_NUMBERED_HEADING = re.compile(r"^(?P<num>[A-Z]|[IVX]{1,4}|\d+(?:\.\d+)*)\.?\s+(?P<title>\S.*)$")


def _heading_level(number: str, context: dict[str, int]) -> int:
    """Map a numbering token to a Markdown level, relative to enclosing sections."""
    if re.fullmatch(r"[A-Z]", number) and not re.fullmatch(r"[IVX]", number):
        context["letter"], context["roman"] = 2, 0
        return 2
    if re.fullmatch(r"[IVX]{1,4}", number):
        context["roman"] = (context.get("letter") or 1) + 1
        return context["roman"]
    parent = context.get("roman") or context.get("letter") or 1
    return min(6, parent + number.count(".") + 1)


def pdf_to_blocks(path, body_start: str | None = None) -> list[Block]:
    """Parse a Word-exported PDF: bold numbered lines are headings, bullets become lists.

    `body_start` (regex) drops everything before the first matching line — the
    cover page, table of contents and revision history belong in frontmatter.
    """
    import pymupdf

    doc = pymupdf.open(path)
    events: list[tuple[str, object]] = []
    for page in doc:
        tables = page.find_tables().tables
        rects = [pymupdf.Rect(t.bbox) for t in tables]
        items: list[tuple[float, float, str, object]] = []
        for table in tables:
            rows = [[clean_text(c or "") for c in row] for row in table.extract()]
            items.append((table.bbox[1], table.bbox[0], "table", rows))
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                bbox = pymupdf.Rect(line["bbox"])
                if any(r.contains(pymupdf.Point((bbox.x0 + bbox.x1) / 2, (bbox.y0 + bbox.y1) / 2)) for r in rects):
                    continue
                spans = [s for s in line["spans"] if s["text"]]
                text = "".join(s["text"] for s in spans)
                if not text.strip():
                    continue
                bold_chars = sum(len(s["text"].strip()) for s in spans if s["flags"] & 16 or "Bold" in s["font"])
                total = sum(len(s["text"].strip()) for s in spans) or 1
                # A bullet glyph is either its own span ("✓", " ", "text") or its own line ("- ").
                marker = spans[0]["text"].strip() if len(spans) > 1 or len(text.strip()) == 1 else ""
                items.append((bbox.y0, bbox.x0, "line", {
                    "text": text, "bold": bold_chars / total >= 0.6, "x": bbox.x0, "block": (page.number, block["number"]),
                    "marker": marker if marker in _BULLET_MARKERS else "",
                }))
        items.sort(key=lambda it: (round(it[0] / 3), it[1]))

        # Merge fragments sharing a baseline (bullet glyph + text, tab-aligned amounts).
        merged: list[tuple[float, float, str, object]] = []
        for it in items:
            prev = merged[-1] if merged else None
            if prev and it[2] == prev[2] == "line" and abs(it[0] - prev[0]) < 2.5:
                prev[3]["text"] += " " + it[3]["text"]
                prev[3]["bold"] = prev[3]["bold"] and it[3]["bold"]
                continue
            merged.append(it)
        for y, _x, kind, payload in merged:
            if kind == "line" and payload["text"].strip().isdigit() and y > page.rect.height - 70:
                continue  # page number
            events.append((kind, payload))

    # Bullet depth is relative *within one list*: the same glyph sits at different x
    # offsets depending on context (✓ at 72, 76.6, 81 or 108pt), so keep a stack of
    # marker x positions — deeper indent pushes a level, shallower pops back.
    x_stack: list[float] = []

    def depth_of(x: float) -> int:
        while x_stack and x < x_stack[-1] - 4:
            x_stack.pop()
        if not x_stack or x > x_stack[-1] + 4:
            x_stack.append(x)
        return len(x_stack) - 1

    blocks: list[Block] = []
    context: dict[str, int] = {}
    started = body_start is None
    current: Block | None = None
    last_line: dict | None = None

    for kind, payload in events:
        if kind == "table":
            if not started:
                continue
            rows = [r for r in payload if any(r)]
            if rows and sum(any(r[c] for r in rows) for c in range(max(len(r) for r in rows))) == 1:
                # Degenerate "table": a wrapped paragraph whose ruling lines fooled the table finder.
                text = clean_text(" ".join(c for r in rows for c in r if c))
                if current is not None and current.kind == "list":
                    current.items.append((current.items[-1][0], text))
                else:
                    current = Block("para", text=text)
                    blocks.append(current)
                continue
            prev = blocks[-1] if blocks else None
            if prev and prev.kind == "table" and rows and rows[0] == prev.rows[0]:
                prev.rows.extend(rows[1:])  # table continued on next page, header repeated
            elif rows:
                blocks.append(Block("table", rows=rows))
            current = None
            continue

        text = clean_text(payload["text"])
        if not started:
            if re.search(body_start, text):
                started = True
            else:
                continue

        marker = payload["marker"] or ("-" if re.match(r"^[-–]\s", text) else "")
        if marker:
            body = clean_text(text[len(marker):]) if text.startswith(marker) else text
            if current is None or current.kind != "list":
                current = Block("list")
                blocks.append(current)
                x_stack.clear()
            current.items.append((depth_of(payload["x"]), body))
            # glyph in its own span: text hangs right of it; "- text" in one span: wraps align with it
            payload["marker_x"] = payload["x"] if payload["marker"] else payload["x"] - 8
            last_line = payload
            continue

        match = _NUMBERED_HEADING.match(text)
        if payload["bold"] and match and len(text) <= 160:
            level = _heading_level(match.group("num"), context)
            blocks.append(Block("heading", text=text, level=level))
            current, last_line = None, payload
            continue

        continues = (
            current is not None
            and last_line is not None
            and (
                # a wrapped list line is indented past the marker of the item it belongs to,
                # and either stays in the same text block or follows an unfinished sentence
                (
                    current.kind == "list"
                    and payload["x"] > last_line.get("marker_x", last_line["x"]) + 4
                    and (payload["block"] == last_line["block"] or not re.search(r"[.;!?]\s*$", current.items[-1][1]))
                )
                or (current.kind == "para" and payload["block"] == last_line["block"])
                or (current.kind == "para" and not re.search(r"[.:;!?)]\s*$", current.text))
            )
        )
        if continues and current.kind == "list":
            depth, prev_text = current.items[-1]
            current.items[-1] = (depth, f"{prev_text} {text}")
            payload["marker_x"] = last_line.get("marker_x", last_line["x"])
        elif continues:
            current.text = f"{current.text} {text}"
        else:
            label = payload["bold"] and len(text) <= 80
            current = Block("para", text=f"**{text}**" if label else text)
            blocks.append(current)
        last_line = payload

    for block in blocks:
        if block.kind == "para":
            block.text = clean_text(block.text)
        elif block.kind == "list":
            block.items = [(d, clean_text(t)) for d, t in block.items]
    return blocks


# --------------------------------------------------------------------------- render

RECORD_CELL_LIMIT = 150


def normalize_headings(blocks: list[Block], title: str) -> list[Block]:
    """Drop a leading heading that repeats the document title, make levels contiguous
    from 2, and nest a list under the previous list item when that item ends in ":"."""
    def key(s: str) -> str:
        return re.sub(r"[^0-9a-zà-ỹđ]+", "", s.lower())

    first_heading = next((i for i, b in enumerate(blocks) if b.kind == "heading"), None)
    if first_heading is not None:
        head = key(blocks[first_heading].text)
        if head and head in key(title) and len(head) >= 0.6 * len(key(title)):
            blocks = blocks[:first_heading] + blocks[first_heading + 1 :]

    merged: list[Block] = []
    for block in blocks:
        prev = merged[-1] if merged else None
        if block.kind == "list" and prev and prev.kind == "list" and prev.items and block.items:
            if prev.items[-1][1].endswith(":"):
                base = prev.items[-1][0] + 1
                prev.items.extend((base + d, t) for d, t in block.items)
                continue
            if min(d for d, _ in block.items) > 0:  # orphan nested list (<ul><li><ul>...) continues the previous one
                prev.items.extend(block.items)
                continue
        merged.append(block)
    blocks = merged
    for block in blocks:  # a list always starts at depth 0 and never skips a level
        if block.kind == "list" and block.items:
            base = min(d for d, _ in block.items)
            fixed, prev_depth = [], -1
            for depth, text in block.items:
                depth = min(depth - base, prev_depth + 1)
                fixed.append((depth, text))
                prev_depth = depth
            block.items = fixed

    levels = [b.level for b in blocks if b.kind == "heading"]
    if not levels:
        return blocks
    shift = min(levels) - 2
    prev = 1
    for block in blocks:
        if block.kind == "heading":
            block.level = max(2, min(block.level - shift, prev + 1))
            prev = block.level
    return blocks


def _table_is_simple(rows: list[list[str]]) -> bool:
    cells = [cell for row in rows for cell in row]
    return all(len(c) <= RECORD_CELL_LIMIT and "\n" not in c for c in cells) and len(rows[0]) <= 6


def _cell_markdown(label: str | None, cell: str) -> list[str]:
    lines = cell.split("\n")
    lead = [l for l in lines if not l.startswith("- ")]
    items = [l for l in lines if l.startswith("- ")]
    if label is None:  # the row has a single field: render it as plain content, in source order
        out: list[str] = []
        for line in lines:
            if out and out[-1].startswith("- ") != line.startswith("- "):
                out.append("")
            elif out and not line.startswith("- "):
                out.append("")
            out.append(line)
        return out
    out = [f"- **{label}:** {' '.join(lead)}".rstrip()]
    return out + [f"  {item}" for item in items]


def _render_table(rows: list[list[str]], heading_level: int) -> list[str]:
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    header, body = rows[0], rows[1:]
    if _table_is_simple(rows):
        esc = lambda s: s.replace("|", "\\|")  # noqa: E731
        out = ["| " + " | ".join(esc(h) for h in header) + " |", "|" + "---|" * width]
        out += ["| " + " | ".join(esc(c) for c in row) + " |" for row in body]
        return out
    # Record layout: each row becomes its own sub-section, so a chunk never splits a row.
    level = min(6, heading_level + 1)
    out: list[str] = []
    title_cols = [i for i, h in enumerate(header) if not re.fullmatch(r"(?i)(stt|nr\.?|no\.?|steps?|#)", h.strip())]
    title_col = title_cols[0] if title_cols else 0
    for row in body:
        number = row[0] if title_col != 0 and row[0] else ""
        head = f"{number}. {row[title_col]}" if number and not row[title_col].startswith(number) else row[title_col]
        out += ["#" * level + " " + (head or "(không tiêu đề)"), ""]
        fields = [
            (h, cell) for i, (h, cell) in enumerate(zip(header, row))
            if cell and not (i == title_col or (i == 0 and (not h or number)))
        ]
        for h, cell in fields:
            out += _cell_markdown(None if len(fields) == 1 else h, cell)
        out.append("")
    return out[:-1] if out and out[-1] == "" else out


def render_markdown(blocks: list[Block], title: str) -> str:
    out = [f"# {title}", ""]
    current_level = 1
    for block in blocks:
        if block.kind == "heading":
            current_level = block.level
            out += ["#" * block.level + " " + block.text, ""]
        elif block.kind == "para":
            out += [block.text, ""]
        elif block.kind == "list":
            counters: dict[int, int] = {}
            for depth, text in block.items:
                if block.ordered and depth == 0:
                    counters[depth] = counters.get(depth, 0) + 1
                    out.append(f"{counters[depth]}. {text}")
                else:
                    out.append("  " * depth + f"- {text}")
            out.append("")
        elif block.kind == "table":
            out += _render_table(block.rows, current_level) + [""]
    text = "\n".join(out).rstrip() + "\n"
    return re.sub(r"\n{3,}", "\n\n", text)
