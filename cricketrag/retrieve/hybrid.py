"""Stages 6-7: hybrid retrieval (dense + BM25, RRF) and cross-encoder reranking."""
from __future__ import annotations

import math
import re
from functools import lru_cache

from cricketrag import config

_WORD = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def rrf(rankings: list[list[str]], k: int = config.RRF_K, top_n: int = config.CANDIDATES) -> list[str]:
    """Reciprocal rank fusion over lists of chunk ids, best first."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda c: (-scores[c], c))[:top_n]


def passage_key(chunk: dict) -> str:
    """Fingerprint of a passage's wording, ignoring its document title and page label."""
    text = chunk["text"]
    for prefix in (chunk.get("title") or "", chunk.get("heading_path") or ""):
        text = text.replace(prefix, "", 1) if prefix else text
    return re.sub(r"[^a-z]", "", text.lower())[:200]


def edition_bonus(year: int | None, asked: set[int]) -> float:
    """Prefer the edition the question names; otherwise prefer the most recent edition."""
    if not year:
        return 0.0
    if asked:
        return config.RECENCY_BONUS if year in asked else 0.0
    span = config.CURRENT_YEAR - 2010
    return config.RECENCY_BONUS * max(0.0, min(1.0, (year - 2010) / span))


class Retriever:
    """mode: 'dense' | 'bm25' | 'hybrid'. rerank=False gives the B1-B3 baselines."""

    def __init__(self, mode: str = "hybrid", rerank: bool = True, version: str | None = None):
        from cricketrag.index.build import get_client
        self.mode, self.use_rerank = mode, rerank
        self.version = version or config.get_active_version()
        self.client = get_client()
        self.chunks = self._load_chunks()
        from cricketrag import catalog
        self.catalog = catalog.load()

    def _in_scope(self, cid: str) -> bool:
        """False for documents the catalog excludes but an older index may still hold."""
        return not self.catalog.get(cid.split(":")[0], {}).get("excluded")

    def _allowed(self, cid: str, asked: set) -> bool:
        """Edition filter. A question that names a year searches that year's editions (plus
        undated pages); any other question skips editions a newer one has replaced."""
        entry = self.catalog.get(cid.split(":")[0], {})
        year = entry.get("edition_year")
        if asked:
            return year is None or year in asked
        return not entry.get("superseded")

    def _filter(self):
        from qdrant_client import models as m
        return m.Filter(must=[m.FieldCondition(key="version", match=m.MatchValue(value=self.version))])

    def _load_chunks(self) -> dict[str, dict]:
        out, offset = {}, None
        while True:
            points, offset = self.client.scroll(config.COLLECTION, scroll_filter=self._filter(),
                                                limit=1000, offset=offset, with_payload=True)
            out.update({p.payload["chunk_id"]: p.payload for p in points})
            if offset is None:
                return out

    @property
    @lru_cache(maxsize=1)
    def _bm25(self):
        from rank_bm25 import BM25Okapi
        ids = list(self.chunks)
        return ids, BM25Okapi([tokenize(self.chunks[i]["text"]) for i in ids])

    @property
    @lru_cache(maxsize=1)
    def _embedder(self):
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(config.EMBED_MODEL)

    @property
    @lru_cache(maxsize=1)
    def _reranker(self):
        from sentence_transformers import CrossEncoder
        return CrossEncoder(config.RERANK_MODEL)

    def dense(self, query: str, asked: set = frozenset(), filtered: bool = True) -> list[str]:
        # bge models expect this instruction prefix on the query side only
        q = "Represent this sentence for searching relevant passages: " + query
        vec = self._embedder.encode(q, normalize_embeddings=True).tolist()
        hits = self.client.query_points(config.COLLECTION, query=vec, query_filter=self._filter(),
                                        limit=config.CANDIDATES * 8, with_payload=["chunk_id"]).points
        ids = [h.payload["chunk_id"] for h in hits]
        return [c for c in ids if self._in_scope(c) and (not filtered or self._allowed(c, asked))][: config.CANDIDATES]

    def bm25(self, query: str, asked: set = frozenset(), filtered: bool = True) -> list[str]:
        ids, index = self._bm25
        scores = index.get_scores(tokenize(query))
        out = []
        for i in sorted(range(len(ids)), key=lambda i: -scores[i]):
            if scores[i] <= 0 or len(out) == config.CANDIDATES:
                break
            if self._in_scope(ids[i]) and (not filtered or self._allowed(ids[i], asked)):
                out.append(ids[i])
        return out

    def _candidates(self, query: str, asked: set, filtered: bool) -> list[str]:
        if self.mode == "dense":
            return self.dense(query, asked, filtered)
        if self.mode == "bm25":
            return self.bm25(query, asked, filtered)
        return rrf([self.dense(query, asked, filtered), self.bm25(query, asked, filtered)])

    def search_multi(self, question: str, sub_queries: list[str], top_k: int = config.TOP_K + 1) -> list[dict]:
        """Comparisons and multi-part questions: search each part separately and take turns
        picking from each result list, so every part gets evidence. One extra slot is allowed."""
        lists = [self.search(q, top_k) for q in sub_queries] + [self.search(question, top_k)]
        picked, seen_ids, seen_keys = [], set(), set()
        for rank in range(top_k):
            for results in lists:
                if rank < len(results) and len(picked) < top_k:
                    c = results[rank]
                    key = passage_key(c)
                    if c["chunk_id"] not in seen_ids and key not in seen_keys:
                        seen_ids.add(c["chunk_id"])
                        seen_keys.add(key)
                        picked.append(c)
        # strongest evidence first, so the refusal gate reads the best score
        return sorted(picked, key=lambda c: -(c.get("score") or 0))

    def search(self, query: str, top_k: int = config.TOP_K) -> list[dict]:
        """Returns top_k chunk payloads, each with a 'score' in [0, 1] when reranked."""
        asked = {int(y) for y in re.findall(r"\b(20[0-3]\d)\b", query)}
        cands = self._candidates(query, asked, filtered=True)
        if len(cands) < top_k:  # the edition filter left too little: search everything
            cands = self._candidates(query, asked, filtered=False)
        if not cands:
            return []
        if not self.use_rerank:
            return [{**self.chunks[c], "score": None} for c in cands[:top_k]]
        logits = self._reranker.predict([(query, self.chunks[c]["text"]) for c in cands])
        rows = []
        for c, s in zip(cands, logits):
            score = 1 / (1 + math.exp(-float(s)))
            rows.append((score + edition_bonus(self.chunks[c].get("edition_year"), asked), score, c))
        rows.sort(key=lambda r: -r[0])
        # The same clause appears almost word for word in many playing conditions. Keep the best
        # copy of each passage so the five slots hold five different pieces of evidence.
        picked, seen = [], set()
        for _, score, c in rows:
            key = passage_key(self.chunks[c])
            if key in seen:
                continue
            seen.add(key)
            picked.append({**self.chunks[c], "score": score})
            if len(picked) == top_k:
                break
        # 'score' stays the plain reranker score: the refusal gate must not be inflated by the bonus
        return picked
