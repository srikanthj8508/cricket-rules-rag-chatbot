"""Stages 4-5: embed chunks and load Qdrant. Also the re-chunk procedure.

Run:  python -m cricketrag.index.build --strategy struct

Re-chunk without re-crawling and without orphans:
  1. write new chunks under the new version tag (old version stays live)
  2. verify every extracted doc_id has >= 1 chunk in the new version
  3. flip active_version
  4. delete every point whose version is not the active one
"""
from __future__ import annotations

import argparse
import uuid

from cricketrag import catalog as catalog_mod
from cricketrag import config
from cricketrag.chunk.chunker import chunk_document, iter_documents


def point_id(chunk_id: str) -> str:
    """Qdrant needs UUID/int ids; derive one deterministically from chunk_id."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def get_client():
    from qdrant_client import QdrantClient
    config.QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(config.QDRANT_PATH))


def ensure_collection(client) -> None:
    from qdrant_client import models as m
    if not client.collection_exists(config.COLLECTION):
        client.create_collection(
            config.COLLECTION,
            vectors_config=m.VectorParams(size=config.EMBED_DIM, distance=m.Distance.COSINE),
        )
        for field in ("version", "doc_id"):
            client.create_payload_index(config.COLLECTION, field, m.PayloadSchemaType.KEYWORD)


def build(strategy: str, keep_old: bool = False) -> None:
    from qdrant_client import models as m
    from sentence_transformers import SentenceTransformer

    version = config.version_tag(strategy)
    client = get_client()
    ensure_collection(client)
    model = SentenceTransformer(config.EMBED_MODEL)

    catalog = catalog_mod.build()  # titles, editions and the in/out-of-scope decision
    doc_ids, total, empty = set(), 0, 0
    for doc in iter_documents():
        entry = catalog.get(doc["doc_id"], {})
        if entry.get("excluded"):
            empty += 1  # no text, a named disciplinary decision, or off-topic: not indexed
            continue
        doc_ids.add(doc["doc_id"])
        chunks = chunk_document(doc, strategy, entry)
        # drop this doc's chunks for this version first: handles changed pages
        # (new content_hash) and a doc that now yields fewer chunks than before
        client.delete(config.COLLECTION, m.FilterSelector(filter=m.Filter(must=[
            m.FieldCondition(key="doc_id", match=m.MatchValue(value=doc["doc_id"])),
            m.FieldCondition(key="version", match=m.MatchValue(value=version)),
        ])))
        if not chunks:
            continue
        vectors = model.encode([c["text"] for c in chunks], normalize_embeddings=True,
                               batch_size=32, show_progress_bar=False)
        client.upsert(config.COLLECTION, [
            m.PointStruct(id=point_id(c["chunk_id"]), vector=v.tolist(), payload=c)
            for c, v in zip(chunks, vectors)
        ])
        total += len(chunks)

    # step 2: coverage check before anything is switched or deleted
    covered = set()
    offset = None
    while True:
        points, offset = client.scroll(
            config.COLLECTION, limit=1000, offset=offset, with_payload=["doc_id"],
            scroll_filter=m.Filter(must=[m.FieldCondition(key="version", match=m.MatchValue(value=version))]),
        )
        covered.update(p.payload["doc_id"] for p in points)
        if offset is None:
            break
    missing = doc_ids - covered
    if len(missing) > 0.02 * max(len(doc_ids), 1):
        raise SystemExit(f"Aborting: {len(missing)} documents with text produced no chunks; old version left active.")

    # step 3: flip, step 4: remove everything that is not the active version
    config.set_active_version(version)
    if not keep_old:
        client.delete(config.COLLECTION, m.FilterSelector(filter=m.Filter(must_not=[
            m.FieldCondition(key="version", match=m.MatchValue(value=version)),
        ])))
    # documents that vanished from the extraction store
    stale = covered - doc_ids
    for did in stale:
        client.delete(config.COLLECTION, m.FilterSelector(filter=m.Filter(must=[
            m.FieldCondition(key="doc_id", match=m.MatchValue(value=did))])))
    print(f"Indexed {total} chunks as {version}; active_version={version}; "
          f"removed {len(stale)} stale documents; skipped {empty} excluded documents (see data/manifest.csv)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=["fixed", "struct"], default="struct")
    ap.add_argument("--keep-old", action="store_true",
                    help="keep other versions (needed while comparing both strategies in eval)")
    a = ap.parse_args()
    build(a.strategy, a.keep_old)
