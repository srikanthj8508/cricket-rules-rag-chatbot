"""Central settings. Every stage reads paths and parameters from here."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW_DIR = DATA / "raw"            # unmodified HTML / PDF bytes + .meta.json
EXTRACT_DIR = DATA / "extracted"  # one normalized JSON per document
STATE_FILE = DATA / "index_state.json"
LOG_FILE = DATA / "chat_log.jsonl"

# --- crawl -----------------------------------------------------------------
SITE_HOST = "www.icc-cricket.com"
PDF_HOST = "images.icc-cricket.com"   # where the ICC hosts its rule PDFs
SITEMAPS = [f"https://{SITE_HOST}/sitemap.xml"]
if not os.environ.get("CRICKET_SKIP_NEWS"):  # news index is large; skipped on the first run
    SITEMAPS.append(f"https://{SITE_HOST}/sitemap-article-index.xml")
# HTML pages in scope: the whole /about/ section (rules, integrity, rankings,
# the ICC itself) plus news articles whose slug looks rules-related.
ALLOW_PREFIXES = ("/about/",)
NEWS_PREFIX = "/news/"
# whole hyphen-separated words in the slug (an optional plural "s" is allowed)
NEWS_KEYWORDS = ("playing-condition", "law", "rule", "regulation", "code-of-conduct", "explained",
                 "faq", "dls", "over-rate", "drs", "umpire", "governance", "anti-corruption",
                 "eligibility", "policy")
# disciplinary news about named people or teams is out of scope
NEWS_EXCLUDE = ("fined", "reprimanded", "suspended", "banned", "charged", "guilty", "sanctioned",
                "breach", "breaching")
CONTACT_EMAIL = os.environ.get("CRICKET_CONTACT_EMAIL", "srikanthreddy8508@gmail.com")
USER_AGENT = f"CricketRAG-CourseProject/0.1 (student project; contact: {CONTACT_EMAIL})"
DOWNLOAD_DELAY_S = 2.0
MAX_PAGES = 900

# --- chunking --------------------------------------------------------------
FIXED_TOKENS = 512
FIXED_OVERLAP = 64
STRUCT_MAX_TOKENS = 800
STRATEGY_VERSIONS = {"fixed": 1, "struct": 2}  # bump when parameters change

# --- models / index --------------------------------------------------------
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
RERANK_MODEL = "BAAI/bge-reranker-base"
LLM_MODEL = "claude-haiku-4-5-20251001"   # used only when ANTHROPIC_API_KEY is set
# Free local model served by Ollama (https://ollama.com); used when there is no API key.
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
OLLAMA_CONTEXT = 8192   # tokens; must hold five passages plus the question
QDRANT_PATH = DATA / "qdrant"   # embedded Qdrant: no server or Docker needed
COLLECTION = "cricket"

# --- retrieval / refusal ---------------------------------------------------
CANDIDATES = 30          # per retriever, and after fusion
TOP_K = 5                # chunks passed to the LLM
RRF_K = 60
REFUSAL_THRESHOLD = 0.30  # reranker score (sigmoid); tune on the dev split
REFUSAL_TOKEN = "INSUFFICIENT_CONTEXT"
# Added to the rerank score of the newest editions (scaled 2010..CURRENT_YEAR), so a current
# rule outranks a near-identical superseded one. Tune on the dev split; 0 switches it off.
RECENCY_BONUS = 0.10
CURRENT_YEAR = 2026


def version_tag(strategy: str) -> str:
    return f"{strategy}-v{STRATEGY_VERSIONS[strategy]}"


def get_active_version() -> str:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())["active_version"]
    return version_tag("struct")


def set_active_version(tag: str) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps({"active_version": tag}, indent=2))
