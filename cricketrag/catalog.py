"""Document catalog: a title, edition, format and in/out-of-scope decision per document.

Run:  python -m cricketrag.catalog

Reads the raw HTML (for the link text that points at each PDF) and the extracted
JSON. Writes data/catalog.json (used by chunking and retrieval) and
data/manifest.csv (the crawl manifest: one row per fetched document).
No network access.
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from urllib.parse import urljoin

from cricketrag import config

CATALOG_FILE = config.DATA / "catalog.json"
MANIFEST_FILE = config.DATA / "manifest.csv"

_GENERIC = re.compile(r"^(download( pdf)?|click here|pdf|here|read more|view)$", re.I)
_YEAR = re.compile(r"\b(20[0-3]\d)\b")
_EFFECTIVE = re.compile(r"effective[^.]{0,40}?\b(20[0-3]\d)\b", re.I)
# Decisions and awards about named people or boards: excluded on ethics grounds.
_NAMED_CASE = re.compile(r"in the matter of|\bdecision (of|in)\b|\baward\b|\bsanctions?\b|\bv\.? [A-Z]{2,}|"
                         r"\bMR\.? [A-Z]|\bMS\.? [A-Z]|\bbreaches\b|\btribunal\b|\bbetween\b|\bdemerit\b", re.I)
# Publications that are not rules, regulations or governance text.
_OFF_TOPIC = re.compile(r"media[- ]guide|annual[- ]report|financial statements|commercial-opportunities|"
                        r"working-at-icc|\bballot\b|\bticket", re.I)


def detect_year(title: str, head: str) -> int | None:
    m = _EFFECTIVE.search(title) or _EFFECTIVE.search(head)
    if m:
        return int(m.group(1))
    years = _YEAR.findall(title)
    if years:
        return max(int(y) for y in years)
    years = _YEAR.findall(head)
    return int(Counter(years).most_common(1)[0][0]) if years else None


def detect_format(text: str) -> str:
    t = text.lower()
    if re.search(r"t20|twenty20|twenty 20", t):
        return "T20I"
    if re.search(r"one[- ]day|\bodi\b|world cup(?! league)|champions trophy", t):
        return "ODI"
    if re.search(r"\btest\b|wtc|world test championship|four[- ]day", t):
        return "Test"
    return ""


def detect_gender(text: str) -> str:
    t = text.lower().replace("’", "'")
    if "women" in t:
        return "women"
    return "men" if re.search(r"\bmen", t) else ""


def exclusion_reason(title: str, section: str, head: str) -> str:
    if _NAMED_CASE.search(title) or re.search(r"IN THE MATTER OF|\bMR\.? [A-Z]{3,}|\bAWARD\b|TRIBUNAL", head[:600]):
        return "named disciplinary or dispute decision"
    if _OFF_TOPIC.search(section) or _OFF_TOPIC.search(title):
        return "not rules, regulations or governance"
    return ""


_MONTHS = r"january|february|march|april|may|june|july|august|september|october|november|december"


def family(title: str) -> str:
    """Title with dates stripped, so editions of one document share a key."""
    t = title.lower().replace("\u2019", "'")
    t = re.sub(r"\(.*?\)|effective|amended|as from|from|eff\b|updated|" + _MONTHS, " ", t)
    return " ".join(re.sub(r"[^a-z' ]", " ", t).split())


def mark_superseded(catalog: dict[str, dict]) -> None:
    """An edition is superseded when a newer edition of the same document exists, or when it is
    an old annual handbook (their contents are republished as standalone playing conditions)."""
    newest: dict[str, int] = {}
    for e in catalog.values():
        if e["edition_year"] and not e["excluded"]:
            fam = family(e["title"])
            newest[fam] = max(newest.get(fam, 0), e["edition_year"])
    for e in catalog.values():
        year = e["edition_year"]
        old_handbook = "handbook" in e["title"].lower() and bool(year) and year < config.CURRENT_YEAR - 1
        e["superseded"] = bool(year) and (year < newest.get(family(e["title"]), 0) or old_handbook)


def link_texts() -> dict[str, tuple[str, str]]:
    """PDF url -> (best link text, path of the page that links to it)."""
    from bs4 import BeautifulSoup

    out: dict[str, tuple[str, str]] = {}
    for meta_path in config.RAW_DIR.glob("*.meta.json"):
        meta = json.loads(meta_path.read_text())
        if meta["source_type"] != "html":
            continue
        html = (config.RAW_DIR / f"{meta['doc_id']}.html").read_text(errors="ignore")
        section = meta["url"].split("icc-cricket.com", 1)[-1]
        for a in BeautifulSoup(html, "lxml").find_all("a", href=True):
            url = urljoin(meta["url"], a["href"]).split("#")[0]
            if not url.lower().endswith(".pdf"):
                continue
            text = " ".join(a.get_text(" ").split())
            if _GENERIC.match(text) or len(text) < 4:
                text = ""
            if url not in out or len(text) > len(out[url][0]):
                out[url] = (text, section)
    return out


def build() -> dict[str, dict]:
    links = link_texts()
    catalog: dict[str, dict] = {}
    rows = []
    for path in sorted(config.EXTRACT_DIR.glob("*.json")):
        doc = json.loads(path.read_text())
        body = [b for b in doc["blocks"] if b["type"] != "heading"]
        texts = [b["text"] for b in body if b["type"] == "text"]
        head = " ".join(texts[:40])[:4000]
        link_text, section = links.get(doc["url"], ("", ""))
        if doc["source_type"] == "html":
            section = doc["url"].split("icc-cricket.com", 1)[-1]
            title = doc["title"].split("|")[0].strip() or section
        else:
            first_lines = " ".join(t for t in texts[:3])[:110].strip()
            title = link_text or first_lines or doc["doc_id"]
        entry = {
            "title": title,
            "section": section,
            "edition_year": detect_year(title, head) if doc["source_type"] == "pdf" else None,
            "format": detect_format(title),
            "gender": detect_gender(title),
            "excluded": exclusion_reason(title, section, head) if body else "no extractable text",
        }
        catalog[doc["doc_id"]] = entry
        rows.append({"doc_id": doc["doc_id"], "url": doc["url"], "type": doc["source_type"],
                     "fetched_at": doc["fetched_at"], "content_hash": doc["content_hash"][:16],
                     "text_chars": sum(len(b["text"]) for b in body),
                     "tables": sum(b["type"] == "table" for b in body)})
    mark_superseded(catalog)
    rows = [{**r, **catalog[r["doc_id"]]} for r in rows]
    CATALOG_FILE.write_text(json.dumps(catalog, indent=1))
    with MANIFEST_FILE.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return catalog


def load() -> dict[str, dict]:
    return json.loads(CATALOG_FILE.read_text()) if CATALOG_FILE.exists() else {}


if __name__ == "__main__":
    cat = build()
    kept = [c for c in cat.values() if not c["excluded"]]
    print(f"{len(cat)} documents: {len(kept)} in scope, {len(cat) - len(kept)} excluded")
    for reason, n in Counter(c["excluded"] for c in cat.values() if c["excluded"]).most_common():
        print(f"  excluded, {reason}: {n}")
    print(f"  in scope but superseded by a newer edition: {sum(c['superseded'] for c in kept)}")
    print(f"Wrote {CATALOG_FILE.name} and {MANIFEST_FILE.name}")
