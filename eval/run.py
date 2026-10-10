"""Evaluation harness: baselines B0-B3 and the full system on the golden set.

Run:  python -m eval.run --split dev --strategy struct

Golden set format (eval/golden.jsonl), one JSON object per line:
  {"id", "question", "reference", "gold_urls": [...], "category"}
category "unanswerable" has empty gold_urls and expects a refusal.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

from cricketrag import config

GOLDEN = Path(__file__).parent / "golden.jsonl"
CONFIGS = {  # as defined in section 8 of the proposal
    "B0": {"mode": None, "rerank": False, "gates": False},                        # closed book
    "B1": {"mode": "bm25", "rerank": False, "gates": False},                      # BM25 only
    "B2": {"mode": "dense", "rerank": False, "gates": False, "strategy": "fixed"},  # naive dense
    "B3": {"mode": "hybrid", "rerank": True, "gates": True},                      # final system
}


def load_golden() -> list[dict]:
    return [json.loads(l) for l in GOLDEN.read_text().splitlines() if l.strip()]


def split(items: list[dict], dev_frac: float = 0.6, seed: int = 13) -> dict[str, list[dict]]:
    """Stratified by category; questions sharing a gold document stay in one split."""
    by_cat = defaultdict(lambda: defaultdict(list))
    for it in items:
        group = (it["gold_urls"] or [it["id"]])[0]
        by_cat[it["category"]][group].append(it)
    rng = random.Random(seed)
    out = {"dev": [], "test": []}
    for cat in sorted(by_cat):
        groups = sorted(by_cat[cat])
        rng.shuffle(groups)
        target = round(dev_frac * sum(len(v) for v in by_cat[cat].values()))
        n = 0
        for g in groups:
            side = "dev" if n < target else "test"
            out[side] += by_cat[cat][g]
            n += len(by_cat[cat][g]) if side == "dev" else 0
    return out


def recall_at_k(retrieved: list[str], gold: list[str], k: int = 5) -> float:
    return float(any(u in gold for u in retrieved[:k])) if gold else float("nan")


def mrr(retrieved: list[str], gold: list[str]) -> float:
    for rank, u in enumerate(retrieved, 1):
        if u in gold:
            return 1.0 / rank
    return 0.0


def mean(xs: list[float]) -> float:
    xs = [x for x in xs if x == x]
    return round(sum(xs) / len(xs), 3) if xs else float("nan")


def run(name: str, items: list[dict], version: str) -> dict:
    from cricketrag.generate.answer import answer
    from cricketrag.retrieve.hybrid import Retriever

    cfg = CONFIGS[name]
    if "strategy" in cfg:  # B2 always runs on fixed-size chunks (index both with --keep-old)
        version = config.version_tag(cfg["strategy"])
    retriever = Retriever(cfg["mode"], cfg["rerank"], version) if cfg["mode"] else None
    rows, lat = [], []
    for it in items:
        t0 = time.time()
        res = answer(it["question"], retriever, use_gates=cfg["gates"])
        lat.append(time.time() - t0)
        urls = [c["url"] for c in res["sources"]]
        rows.append({"id": it["id"], "category": it["category"], "answer": res["answer"],
                     "refused": res["refused"], "retrieved_urls": urls,
                     "recall5": recall_at_k(urls, it["gold_urls"]), "mrr": mrr(urls, it["gold_urls"])})
    ans = [r for r in rows if r["category"] != "unanswerable"]
    una = [r for r in rows if r["category"] == "unanswerable"]
    summary = {
        "config": name, "version": version, "n": len(rows),
        "recall@5": mean([r["recall5"] for r in ans]) if retriever else None,
        "mrr": mean([r["mrr"] for r in ans]) if retriever else None,
        "refusal_recall": mean([float(r["refused"]) for r in una]),
        "false_refusal_rate": mean([float(r["refused"]) for r in ans]),
        "median_latency_s": round(sorted(lat)[len(lat) // 2], 2) if lat else None,
    }
    out = Path(__file__).parent / "results" / f"{name}_{version}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--strategy", choices=["fixed", "struct"], default="struct")
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    a = ap.parse_args()
    items = split(load_golden())[a.split]
    for name in a.configs:
        print(run(name, items, config.version_tag(a.strategy)))
    # TODO (eval lead): answer correctness + faithfulness via LLM judge, nDCG@10, token counts
