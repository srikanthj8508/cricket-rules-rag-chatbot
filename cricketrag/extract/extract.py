"""Stage 2: raw HTML / PDF -> one normalized JSON document per source.

Run:  python -m cricketrag.extract.extract

Output schema (data/extracted/{doc_id}.json):
  {doc_id, url, title, source_type, content_hash, fetched_at,
   blocks: [{type: "heading"|"text"|"table", text, level?, page?}]}
Tables are stored as Markdown so the header row survives chunking.
"""
from __future__ import annotations

import json
import re

from cricketrag import config

_WS = re.compile(r"\s+")


def clean(text: str) -> str:
    return _WS.sub(" ", text or "").strip()


def table_to_markdown(rows: list[list[str]]) -> str:
    rows = [[clean(c or "") for c in r] for r in rows if r and any(c for c in r)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "| " + " | ".join(["---"] * width) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def extract_html(html: str) -> tuple[str, list[dict]]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    title = clean(soup.title.get_text()) if soup.title else ""
    root = soup.find("main") or soup.find(attrs={"role": "main"}) or soup.body or soup
    for tag in root.find_all(["nav", "script", "style", "footer", "header", "form", "aside"]):
        tag.decompose()

    blocks: list[dict] = []
    for el in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "table"]):
        if el.name != "table" and el.find_parent("table"):
            continue  # cell content is captured with its table
        if el.name == "table":
            rows = [[c.get_text(" ") for c in tr.find_all(["th", "td"])] for tr in el.find_all("tr")]
            md = table_to_markdown(rows)
            if md:
                cap = el.find("caption")
                blocks.append({"type": "table", "text": md,
                               "caption": clean(cap.get_text()) if cap else ""})
        elif el.name.startswith("h"):
            text = clean(el.get_text(" "))
            if text:
                blocks.append({"type": "heading", "level": int(el.name[1]), "text": text})
        else:
            text = clean(el.get_text(" "))
            if len(text) > 1:
                blocks.append({"type": "text", "text": text})
    return title, blocks


def extract_pdf(path) -> tuple[str, list[dict]]:
    import fitz  # PyMuPDF
    import pdfplumber

    blocks: list[dict] = []
    doc = fitz.open(path)
    title = clean(doc.metadata.get("title") or "") or path.stem
    with pdfplumber.open(path) as plumb:
        for i, page in enumerate(doc):
            n = i + 1
            blocks.append({"type": "heading", "level": 2, "text": f"Page {n}", "page": n})
            # sort=True gives reading order for multi-column layouts
            for b in page.get_text("blocks", sort=True):
                text = clean(b[4])
                if b[6] == 0 and len(text) > 1:
                    blocks.append({"type": "text", "text": text, "page": n})
            try:  # the two PDF libraries can disagree on page count or choke on a page
                tables = plumb.pages[i].extract_tables() if i < len(plumb.pages) else []
            except Exception:
                tables = []
            for tbl in tables:
                md = table_to_markdown(tbl)
                if md:
                    blocks.append({"type": "table", "text": md, "caption": "", "page": n})
    return title, blocks


def main() -> None:
    config.EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    stats = {"html": 0, "pdf": 0, "tables": 0, "failed": 0}
    for meta_path in sorted(config.RAW_DIR.glob("*.meta.json")):
        meta = json.loads(meta_path.read_text())
        raw = config.RAW_DIR / f"{meta['doc_id']}.{meta['source_type']}"
        out_path = config.EXTRACT_DIR / f"{meta['doc_id']}.json"
        if out_path.exists():  # already extracted from this exact content: reuse it
            prev = json.loads(out_path.read_text())
            if prev.get("content_hash") == meta["content_hash"]:
                stats[meta["source_type"]] += 1
                stats["tables"] += sum(b["type"] == "table" for b in prev["blocks"])
                continue
        try:
            if meta["source_type"] == "pdf":
                title, blocks = extract_pdf(raw)
            else:
                title, blocks = extract_html(raw.read_text(errors="ignore"))
        except Exception as exc:  # one bad file must not stop the run
            print(f"FAILED {meta['url']}: {exc}")
            stats["failed"] += 1
            continue
        out = {**{k: meta[k] for k in ("doc_id", "url", "source_type", "content_hash", "fetched_at")},
               "title": title, "blocks": blocks}
        (config.EXTRACT_DIR / f"{meta['doc_id']}.json").write_text(json.dumps(out, indent=1))
        stats[meta["source_type"]] += 1
        stats["tables"] += sum(b["type"] == "table" for b in blocks)
    print("Corpus stats (use these numbers for the section 2 minimums):", stats)


if __name__ == "__main__":
    main()
