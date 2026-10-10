"""URL scope rules and document ids (no Scrapy import, so tests stay light)."""
from __future__ import annotations

import hashlib
import re
from urllib.parse import urlparse

from cricketrag import config

_NEWS_OK = re.compile("(?:^|-)(?:" + "|".join(config.NEWS_KEYWORDS) + ")s?(?:-|$)")
_NEWS_NO = re.compile("(?:^|-)(?:" + "|".join(config.NEWS_EXCLUDE) + ")(?:-|$)")


def doc_id_for(url: str) -> str:
    """Stable 8-hex id from the canonical URL (no fragment, no trailing slash)."""
    canonical = url.split("#")[0].rstrip("/")
    return hashlib.sha256(canonical.encode()).hexdigest()[:8]


def is_pdf_url(url: str) -> bool:
    p = urlparse(url)
    return p.netloc == config.PDF_HOST and p.path.lower().endswith(".pdf")


def in_scope(url: str) -> bool:
    p = urlparse(url)
    if is_pdf_url(url):
        return True
    if p.netloc != config.SITE_HOST or p.query:
        return False
    if p.path.startswith(config.ALLOW_PREFIXES):
        return True
    if not p.path.startswith(config.NEWS_PREFIX):
        return False
    slug = p.path[len(config.NEWS_PREFIX):]
    return bool(_NEWS_OK.search(slug)) and not _NEWS_NO.search(slug)
