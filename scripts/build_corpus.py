#!/usr/bin/env python3
"""Build the `data/hoc-phi-vinuni/` corpus from official VinUni sources.

Pipeline (each stage can be run on its own):

    fetch    -> download raw HTML/PDF into .cache/raw/ (robots.txt-aware, >=1.5s delay)
    inspect  -> profile the raw files (structure, noise, tables) before converting
    convert  -> raw -> draft Markdown in .cache/draft/ (section-structured)
    publish  -> draft -> data/hoc-phi-vinuni/*.md + sources.csv (frontmatter added)
    validate -> check every published file against the corpus format contract

Raw HTML/PDF never goes into data/ (see docs/DATA_COLLECTION.md).

Usage:
    python scripts/build_corpus.py fetch
    python scripts/build_corpus.py inspect
    python scripts/build_corpus.py convert
    python scripts/build_corpus.py publish [--force]
    python scripts/build_corpus.py validate
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from datetime import date
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "urls.csv"
RAW_DIR = ROOT / ".cache" / "raw"
DRAFT_DIR = ROOT / ".cache" / "draft"
OUT_DIR = ROOT / "data" / "hoc-phi-vinuni"

USER_AGENT = "Day7DataFoundationsCourse/1.0 (+educational-lab)"
REQUEST_DELAY_SECONDS = 1.5

REQUIRED_FIELDS = ["doc_id", "title", "source_url", "retrieved_at", "document_version", "audience"]
FRONTMATTER_ORDER = REQUIRED_FIELDS + ["department", "category", "program_level", "language", "doc_type"]
SOURCES_FIELDS = ["doc_id", "file_path", "title", "source_url", "retrieved_at", "document_version", "license_or_permission"]


def load_manifest() -> list[dict[str, str]]:
    with MANIFEST.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# --------------------------------------------------------------------------- fetch


class RobotsRules:
    """robots.txt rules for `User-agent: *`, with `*` / `$` wildcard support.

    urllib.robotparser only does prefix matching, so a rule such as
    `Disallow: */students/` would silently never match.
    """

    def __init__(self, text: str) -> None:
        self.rules: list[tuple[bool, re.Pattern[str], int]] = []
        applies = False
        for line in text.splitlines():
            line = line.split("#", 1)[0].strip()
            if ":" not in line:
                continue
            key, value = (part.strip() for part in line.split(":", 1))
            key = key.lower()
            if key == "user-agent":
                applies = value == "*"
            elif applies and key in {"allow", "disallow"} and value:
                pattern = re.escape(value).replace(r"\*", ".*")
                if pattern.endswith(r"\$"):
                    pattern = pattern[:-2] + "$"
                self.rules.append((key == "allow", re.compile("^" + pattern), len(value)))

    def allowed(self, url: str) -> tuple[bool, str | None]:
        parsed = urlparse(url)
        path = (parsed.path or "/") + (f"?{parsed.query}" if parsed.query else "")
        best: tuple[bool, str | None, int] = (True, None, -1)
        for is_allow, pattern, length in self.rules:
            if pattern.match(path) and length > best[2]:
                best = (is_allow, pattern.pattern, length)
        return best[0], best[1]


def http_get(url: str) -> tuple[bytes, str, str]:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        return response.read(), response.headers.get("Content-Type", ""), response.geturl()


def cmd_fetch(_: argparse.Namespace) -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RAW_DIR / "fetch_log.json"
    log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {}
    robots_cache: dict[str, RobotsRules] = {}
    failures = 0

    for row in load_manifest():
        url, doc_id, kind = row["url"], row["doc_id"], row["kind"]
        host = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
        if host not in robots_cache:
            body, _, _ = http_get(f"{host}/robots.txt")
            robots_cache[host] = RobotsRules(body.decode("utf-8", errors="replace"))
            time.sleep(REQUEST_DELAY_SECONDS)
        ok, rule = robots_cache[host].allowed(url)
        if not ok:
            print(f"SKIP  {doc_id:38} disallowed by robots.txt ({rule})")
            failures += 1
            continue

        body, content_type, final_url = http_get(url)
        time.sleep(REQUEST_DELAY_SECONDS)
        if final_url.rstrip("/") != url.rstrip("/"):
            print(f"FAIL  {doc_id:38} redirected to {final_url}")
            failures += 1
            continue
        expected = "application/pdf" if kind == "pdf" else "text/html"
        if expected not in content_type:
            print(f"FAIL  {doc_id:38} unexpected content-type {content_type!r}")
            failures += 1
            continue

        target = RAW_DIR / f"{doc_id}.{kind}"
        target.write_bytes(body)
        log[doc_id] = {
            "url": url,
            "kind": kind,
            "content_type": content_type,
            "bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
            "retrieved_at": date.today().isoformat(),
        }
        print(f"OK    {doc_id:38} {len(body):>8} B  {content_type}")

    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    return 1 if failures else 0


# --------------------------------------------------------------------------- per-source config

ADMISSIONS_EDITOR = "div.col-lg-8 div.editor"

PARSERS: dict[str, dict] = {
    "quy-dinh-tai-chinh-bieu-phi": {"parser": "pdf", "body_start": r"^A\.\s+QUY ĐỊNH TÀI CHÍNH VÀ BIỂU PHÍ HỆ ĐẠI HỌC CHÍNH QUY$"},
    "duy-tri-hoc-bong-ho-tro-tai-chinh": {"parser": "pdf", "body_start": r"^1\.\s+Mục đích$"},
    "huong-dan-de-nghi-ho-tro-tai-chinh": {"parser": "html", "selector": "article.single_content"},
    "faq-hoc-phi-hoc-bong": {"parser": "faq"},
}

# One source page that mixes audiences is split into one file per audience
# (docs/DATA_COLLECTION.md §4) so that `metadata_filter={"audience": ...}` has real work to do.
SPLITS: dict[str, list[dict]] = {
    "ho-tro-tai-chinh": [
        {
            "doc_id": "ho-tro-tai-chinh-tan-sinh-vien",
            "title": "Hỗ trợ tài chính dành cho tân sinh viên (ứng viên mới)",
            "audience": "all",
            "until": r"DÀNH CHO SINH VIÊN ĐANG HỌC",
        },
        {
            "doc_id": "ho-tro-tai-chinh-sinh-vien-dang-hoc",
            "title": "Hỗ trợ tài chính dành cho sinh viên đang học",
            "audience": "student",
            "from": r"DÀNH CHO SINH VIÊN ĐANG HỌC",
        },
    ],
}


def parse_source(row: dict[str, str]):
    from bs4 import BeautifulSoup

    import md_convert as mc

    doc_id = row["doc_id"]
    config = PARSERS.get(doc_id, {"parser": "html", "selector": ADMISSIONS_EDITOR})
    raw_path = RAW_DIR / f"{doc_id}.{row['kind']}"
    if config["parser"] == "pdf":
        return mc.pdf_to_blocks(raw_path, body_start=config.get("body_start"))

    soup = BeautifulSoup(raw_path.read_text(encoding="utf-8"), "lxml")
    if config["parser"] == "faq":
        blocks = []
        for tab in soup.select('[data-bs-toggle="tab"], [data-toggle="tab"], [role="tab"]'):
            pane = soup.select_one(tab.get("data-bs-target") or tab["href"])
            if pane is None:
                continue
            blocks.append(mc.Block("heading", text=mc.clean_text(tab.get_text(" ")), level=2))
            # question = h3 (level 3 under the tab group)
            blocks.extend(mc.html_to_blocks(pane))
        return blocks

    roots = soup.select(config["selector"])
    if len(roots) != 1:
        raise RuntimeError(f"{doc_id}: expected 1 content container for {config['selector']!r}, got {len(roots)}")
    return mc.html_to_blocks(roots[0])


def split_blocks(blocks, spec: dict):
    """Keep the blocks from the `from` heading (inclusive) up to the `until` heading (exclusive)."""
    start = 0
    end = len(blocks)
    for i, block in enumerate(blocks):
        if block.kind != "heading":
            continue
        if spec.get("from") and re.search(spec["from"], block.text):
            start = i
        if spec.get("until") and re.search(spec["until"], block.text):
            end = i
            break
    return blocks[start:end]


def output_docs(row: dict[str, str]) -> list[dict[str, str]]:
    """Manifest row -> one or more output document metadata dicts."""
    base = {k: v for k, v in row.items() if k not in {"url", "kind"}}
    base["source_url"] = row["url"]
    specs = SPLITS.get(row["doc_id"])
    if not specs:
        return [base]
    return [{**base, **{k: v for k, v in spec.items() if k not in {"from", "until"}}, "_split": spec} for spec in specs]


# --------------------------------------------------------------------------- inspect


def cmd_inspect(_: argparse.Namespace) -> int:
    """Profile raw sources: what structure is there, what noise must go."""
    from collections import Counter

    for row in load_manifest():
        raw_path = RAW_DIR / f"{row['doc_id']}.{row['kind']}"
        raw = raw_path.read_bytes()
        noise = Counter()
        if row["kind"] == "html":
            html_text = raw.decode("utf-8", errors="replace")
            noise["cf-email"] = html_text.count("data-cfemail")
            noise["img"] = len(re.findall(r"<img\b", html_text))
            noise["empty-p"] = len(re.findall(r"<p[^>]*>\s*(&nbsp;)?\s*</p>", html_text))
        blocks = parse_source(row)
        kinds = Counter(b.kind for b in blocks)
        headings = [f"h{b.level}:{b.text[:40]}" for b in blocks if b.kind == "heading"]
        tables = [f"{len(b.rows)}x{max(len(r) for r in b.rows)}" for b in blocks if b.kind == "table"]
        print(f"== {row['doc_id']} ({row['kind']}, {len(raw):,} B)")
        print(f"   blocks={dict(kinds)} tables={tables} noise={dict(noise)}")
        print(f"   headings({len(headings)}): {headings[:12]}{' ...' if len(headings) > 12 else ''}")
    return 0


# --------------------------------------------------------------------------- convert

OVERRIDES_DIR = Path(__file__).resolve().parent / "overrides"


def apply_overrides(doc_id: str, body: str) -> str:
    """Replace whole sections with hand-transcribed Markdown from scripts/overrides/<doc_id>.md.

    Used only where automatic extraction is provably wrong (e.g. a PDF table whose merged,
    wrapped cells the table finder splits into one row per printed line). Each override
    section replaces the section with the same heading line, up to the next heading of the
    same or higher rank.
    """
    path = OVERRIDES_DIR / f"{doc_id}.md"
    if not path.exists():
        return body
    override = path.read_text(encoding="utf-8").strip()
    for section in re.split(r"(?m)^(?=## )", override):
        section = section.strip()
        if not section:
            continue
        heading = section.splitlines()[0]
        level = len(heading) - len(heading.lstrip("#"))
        pattern = re.compile(rf"(?ms)^{re.escape(heading)}\n.*?(?=^#{{1,{level}}} |\Z)")
        if not pattern.search(body):
            raise RuntimeError(f"override for {doc_id}: heading not found: {heading}")
        body = pattern.sub(lambda _: section + "\n\n", body, count=1)
    return re.sub(r"\n{3,}", "\n\n", body).rstrip() + "\n"


def cmd_convert(_: argparse.Namespace) -> int:
    import md_convert as mc

    DRAFT_DIR.mkdir(parents=True, exist_ok=True)
    log = json.loads((RAW_DIR / "fetch_log.json").read_text(encoding="utf-8"))
    for row in load_manifest():
        blocks = parse_source(row)
        for meta in output_docs(row):
            spec = meta.pop("_split", None)
            doc_blocks = split_blocks(blocks, spec) if spec else blocks
            doc_blocks = mc.normalize_headings(list(doc_blocks), meta["title"])
            body = apply_overrides(meta["doc_id"], mc.render_markdown(doc_blocks, meta["title"]))
            meta["retrieved_at"] = log[row["doc_id"]]["retrieved_at"]
            (DRAFT_DIR / f"{meta['doc_id']}.md").write_text(body, encoding="utf-8")
            (DRAFT_DIR / f"{meta['doc_id']}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"draft {meta['doc_id']:38} {len(body):>6} chars  {body.count(chr(10) + '#')} headings")
    return 0


# --------------------------------------------------------------------------- publish


def frontmatter(meta: dict[str, str]) -> str:
    lines = ["---"]
    for key in FRONTMATTER_ORDER:
        value = meta.get(key, "")
        if not value:
            continue
        needs_quotes = any(ch in value for ch in ":#,[]{}\"'") or value[0] in "-?|>*&!%@`" or re.fullmatch(r"[\d.\-]+", value)
        lines.append(f'{key}: "{value}"' if needs_quotes else f"{key}: {value}")
    lines.append("---")
    return "\n".join(lines) + "\n\n"


def cmd_publish(args: argparse.Namespace) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sources = []
    skipped = 0
    for meta_path in sorted(DRAFT_DIR.glob("*.meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        body = (DRAFT_DIR / f"{meta['doc_id']}.md").read_text(encoding="utf-8")
        target = OUT_DIR / f"{meta['doc_id']}.md"
        if target.exists() and not args.force:
            skipped += 1  # keep manual edits; re-run with --force to regenerate
        else:
            target.write_text(frontmatter(meta) + body, encoding="utf-8")
            print(f"wrote {target.relative_to(ROOT)}")
        sources.append({**{k: meta.get(k, "") for k in SOURCES_FIELDS}, "file_path": target.relative_to(ROOT).as_posix()})

    with (OUT_DIR / "sources.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCES_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(sources, key=lambda r: r["doc_id"]))
    print(f"sources.csv: {len(sources)} rows" + (f" ({skipped} existing files kept, use --force to overwrite)" if skipped else ""))
    return 0


# --------------------------------------------------------------------------- validate

SHORT_DOC_CHARS = 1500  # a short single-topic page is one section: no headings are invented for it
BOILERPLATE = re.compile(r"(?i)(trang chủ|chuyển đến nội dung|\[email[^\]]*protected\]|đăng ký ngay|xem thêm tin|cookie)")


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    match = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not match:
        return {}, text
    meta = {}
    for line in match.group(1).splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip().strip('"')
    return meta, text[match.end():]


def check_file(path: Path) -> list[str]:
    """The corpus format contract. Returns a list of violations (empty = OK)."""
    problems = []
    meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED_FIELDS if not meta.get(k)]
    if missing:
        problems.append(f"missing metadata {missing}")
    if meta.get("doc_id") != path.stem:
        problems.append("doc_id != file name")
    if meta.get("audience") not in {"student", "faculty", "staff", "all"}:
        problems.append(f"bad audience {meta.get('audience')!r}")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", meta.get("retrieved_at", "")):
        problems.append("retrieved_at not YYYY-MM-DD")
    if not any(meta.get(k) for k in ("department", "category", "language")):
        problems.append("needs one extra filter field")

    lines = body.splitlines()
    headings = [(len(m.group(1)), m.group(2)) for m in (re.match(r"^(#{1,6}) (.+)$", l) for l in lines) if m]
    if [lvl for lvl, _ in headings].count(1) != 1 or not headings or headings[0][0] != 1:
        problems.append("must have exactly one H1, first")
    if not any(lvl == 2 for lvl, _ in headings) and len(body) >= SHORT_DOC_CHARS:
        problems.append(f"no H2 sections (only documents under {SHORT_DOC_CHARS} chars may be a single section)")
    prev = 1
    for lvl, text in headings:
        if lvl > prev + 1:
            problems.append(f"heading level jump h{prev}->h{lvl}: {text[:40]}")
        prev = lvl
    # empty sections: a heading immediately followed by a heading of the same or higher rank
    for (lvl_a, text_a), (lvl_b, _) in zip(headings, headings[1:]):
        idx_a = lines.index("#" * lvl_a + " " + text_a)
        between = [l for l in lines[idx_a + 1 :] if l.strip()]
        if between and between[0].startswith("#") and lvl_b <= lvl_a:
            problems.append(f"empty section: {text_a[:40]}")
    for i, line in enumerate(lines, 1):
        if BOILERPLATE.search(line):
            problems.append(f"boilerplate on line {i}: {line[:60]}")
    tables = re.findall(r"((?:^\|.*\|\n)+)", body, re.M)
    for table in tables:
        widths = {row.count("|") - row.count("\\|") for row in table.strip().splitlines()}
        if len(widths) != 1:
            problems.append(f"ragged table starting {table[:40]!r}")
    if re.search(r"\n{3,}", body):
        problems.append("more than one blank line in a row")
    return problems


def cmd_validate(_: argparse.Namespace) -> int:
    files = sorted(OUT_DIR.glob("*.md"))
    rows = list(csv.DictReader((OUT_DIR / "sources.csv").open(encoding="utf-8")))
    failures = 0
    audiences: dict[str, int] = {}
    for path in files:
        problems = check_file(path)
        meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        audiences[meta.get("audience", "?")] = audiences.get(meta.get("audience", "?"), 0) + 1
        status = "OK" if not problems else "FAIL"
        failures += bool(problems)
        print(f"{status:4} {path.name:42} {len(body):>6} chars  audience={meta.get('audience')}")
        for problem in problems:
            print(f"       - {problem}")
    ids_match = sorted(r["doc_id"] for r in rows) == sorted(p.stem for p in files)
    print(f"files: {len(files)} (need 5-10) | sources.csv: {'1-1 match' if ids_match else 'MISMATCH'} | audience: {audiences}")
    ok = not failures and ids_match and 5 <= len(files) <= 10 and len(audiences) >= 2
    return 0 if ok else 1


# --------------------------------------------------------------------------- cli


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch").set_defaults(func=cmd_fetch)
    sub.add_parser("inspect").set_defaults(func=cmd_inspect)
    sub.add_parser("convert").set_defaults(func=cmd_convert)
    publish = sub.add_parser("publish")
    publish.add_argument("--force", action="store_true", help="overwrite files already in data/ (discards manual edits)")
    publish.set_defaults(func=cmd_publish)
    sub.add_parser("validate").set_defaults(func=cmd_validate)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
