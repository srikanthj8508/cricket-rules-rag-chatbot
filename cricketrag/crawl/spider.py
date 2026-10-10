"""Stage 1: polite sitemap crawl of icc-cricket.com into the raw store.

Run:  python -m cricketrag.crawl.spider            (full crawl)
      python -m cricketrag.crawl.spider --limit 50 (dry run)

Every response is saved byte-for-byte with a .meta.json sidecar, so no later
stage ever needs the network.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone

import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.spiders import SitemapSpider

from cricketrag import config
from cricketrag.crawl_scope import doc_id_for, in_scope, is_pdf_url

class IccSpider(SitemapSpider):
    name = "icc"
    sitemap_urls = config.SITEMAPS
    allowed_domains = [config.SITE_HOST, config.PDF_HOST]

    def sitemap_filter(self, entries):
        for entry in entries:
            loc = entry["loc"]
            # keep nested sitemap indexes, filter page URLs by the allowlist
            if loc.endswith(".xml") or "sitemap" in loc or in_scope(loc):
                yield entry

    def parse(self, response):
        is_pdf = is_pdf_url(response.url) or b"%PDF" in response.body[:8]
        self._save(response, "pdf" if is_pdf else "html")
        if is_pdf:
            return
        # follow rule-document PDF links (playing conditions, codes, regulations)
        for href in response.css("a::attr(href)").getall():
            url = response.urljoin(href)
            if is_pdf_url(url):
                yield scrapy.Request(url, callback=self.parse)

    def _save(self, response, kind: str) -> None:
        config.RAW_DIR.mkdir(parents=True, exist_ok=True)
        did = doc_id_for(response.url)
        (config.RAW_DIR / f"{did}.{kind}").write_bytes(response.body)
        meta = {
            "doc_id": did,
            "url": response.url,
            "source_type": kind,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "content_hash": hashlib.sha256(response.body).hexdigest(),
            "etag": response.headers.get("ETag", b"").decode(),
        }
        (config.RAW_DIR / f"{did}.meta.json").write_text(json.dumps(meta, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=config.MAX_PAGES)
    args = ap.parse_args()
    process = CrawlerProcess(settings={
        "USER_AGENT": config.USER_AGENT,
        "ROBOTSTXT_OBEY": True,
        "CONCURRENT_REQUESTS": 1,
        "DOWNLOAD_DELAY": config.DOWNLOAD_DELAY_S,
        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_START_DELAY": config.DOWNLOAD_DELAY_S,
        "RETRY_HTTP_CODES": [429, 500, 502, 503, 504],
        "RETRY_TIMES": 4,
        "HTTPCACHE_ENABLED": True,          # resumable; re-runs hit the cache
        "HTTPCACHE_DIR": str(config.DATA / "httpcache"),
        "CLOSESPIDER_PAGECOUNT": args.limit,
        "LOG_LEVEL": "INFO",
    })
    process.crawl(IccSpider)
    process.start()


if __name__ == "__main__":
    main()
