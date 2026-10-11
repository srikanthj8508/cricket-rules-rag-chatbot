"""Stage 3: extracted JSON -> chunks. Reads data/extracted only, never the web.

Two strategies:
  fixed   512 tokens, 64 overlap, split on sentence boundaries
  struct  one chunk per heading section (cap 800), each table its own chunk,
          heading path prepended

Chunk id: {doc_id}:{strategy}-v{n}:{seq:04d}
"""
from __future__ import annotations

import json
import re
from typing import Iterator

from cricketrag import config

_SENT = re.compile(r"(?<=[.!?])\s+")
_TOK = re.compile(r"\w+|[^\w\s]")


def n_tokens(text: str) -> int:
    """Cheap, dependency-free token estimate (words + punctuation)."""
    return len(_TOK.findall(text))


def chunk_id(doc_id: str, version: str, seq: int) -> str:
    return f"{doc_id}:{version}:{seq:04d}"


def _pack(sentences: list[str], max_tokens: int, overlap: int) -> list[str]:
    """Greedy sentence packing with a token overlap carried between chunks."""
    out, cur, cur_n = [], [], 0
    for s in sentences:
        n = n_tokens(s)
        if cur and cur_n + n > max_tokens:
            out.append(" ".join(cur))
            keep, keep_n = [], 0
            for prev in reversed(cur):
                if keep_n + n_tokens(prev) > overlap:
                    break
                keep.insert(0, prev)
                keep_n += n_tokens(prev)
            cur, cur_n = keep, keep_n
        cur.append(s)
        cur_n += n
    if cur:
        out.append(" ".join(cur))
    return out


def _sentences(text: str) -> list[str]:
    return [s for s in _SENT.split(text) if s.strip()]


def chunk_fixed(doc: dict) -> list[dict]:
    sentences, pages = [], []
    for b in doc["blocks"]:
        if b["type"] == "table":
            sentences.append(b["text"])  # a table is never cut mid-row
        elif b["type"] == "heading" and not b["text"].startswith("Page "):
            sentences.append(b["text"] + ".")
        elif b["type"] == "text":
            sentences.extend(_sentences(b["text"]))
        pages.append(b.get("page"))
    texts = _pack(sentences, config.FIXED_TOKENS, config.FIXED_OVERLAP)
    return [{"text": t, "content_type": "text", "heading_path": "", "page": None} for t in texts]


def chunk_struct(doc: dict) -> list[dict]:
    chunks: list[dict] = []
    path: list[tuple[int, str]] = []
    buf: list[str] = []
    page = None

    def heading_path() -> str:
        return " > ".join(t for _, t in path)

    def flush() -> None:
        nonlocal buf
        if not buf:
            return
        prefix = heading_path()
        budget = config.STRUCT_MAX_TOKENS - n_tokens(prefix)
        for piece in _pack([s for t in buf for s in _sentences(t)], budget, 0):
            chunks.append({"text": f"{prefix}\n{piece}".strip(), "content_type": "text",
                           "heading_path": prefix, "page": page})
        buf = []

    for b in doc["blocks"]:
        if b["type"] == "heading":
            flush()
            page = b.get("page", page)
            while path and path[-1][0] >= b["level"]:
                path.pop()
            path.append((b["level"], b["text"]))
        elif b["type"] == "table":
            flush()
            head = "\n".join(x for x in (heading_path(), b.get("caption", "")) if x)
            chunks.append({"text": f"{head}\n{b['text']}".strip(), "content_type": "table",
                           "heading_path": heading_path(), "page": b.get("page", page)})
        else:
            page = b.get("page", page)
            buf.append(b["text"])
    flush()
    return chunks


STRATEGIES = {"fixed": chunk_fixed, "struct": chunk_struct}


def chunk_document(doc: dict, strategy: str, entry: dict | None = None) -> list[dict]:
    """entry is this document's catalog row (title, edition_year, format, gender, section)."""
    version = config.version_tag(strategy)
    entry = entry or {}
    title = entry.get("title") or doc.get("title", "")
    out = []
    for seq, c in enumerate(STRATEGIES[strategy](doc)):
        if strategy == "struct" and title:  # the document name tells formats and editions apart
            c = {**c, "text": f"{title}\n{c['text']}"}
        out.append({
            "chunk_id": chunk_id(doc["doc_id"], version, seq),
            "doc_id": doc["doc_id"],
            "version": version,
            "seq": seq,
            "url": doc["url"],
            "title": title,
            "edition_year": entry.get("edition_year"),
            "format": entry.get("format", ""),
            "gender": entry.get("gender", ""),
            "section": entry.get("section", ""),
            "content_hash": doc["content_hash"],
            **c,
        })
    return out


def iter_documents() -> Iterator[dict]:
    for p in sorted(config.EXTRACT_DIR.glob("*.json")):
        yield json.loads(p.read_text())
