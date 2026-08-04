#!/usr/bin/env python3
"""Give existing text collections a BM25 sparse vector (Phase 39).

Qdrant cannot add a sparse vector to a collection that has none — verified
against 1.18.3 on 2026-08-03, `400 Wrong input: Not existing vector name`. So
every text collection that predates this phase has to be rebuilt.

**Rebuilt, not re-embedded.** The dense vectors are already in Qdrant and
scroll back out with `with_vectors=True`, so this copies them across verbatim
and computes the sparse vector from `payload.text`, which is stored on every
point. The embedder is never called, no GPU is touched, and no source file is
re-read — which matters because for text uploads the source bytes are long
gone.

## Order of operations, and why

For each collection:

  1. build `<name>__bm25` with the same dense config plus sparse config
  2. copy every point into it — dense verbatim, sparse computed, payload byte
     for byte
  3. **verify the copy has the same number of points**
  4. only then delete the original
  5. recreate the original, correctly configured, and copy back
  6. delete the scratch

Two copies rather than one, because Qdrant has no rename. The alternative is
an alias, which would leave the real collection name pointing at a differently
named collection forever — a thing that reads as a mistake to whoever finds it
next, and that every `collection_exists` call in the codebase would have to
agree with.

The original is deleted only after its replacement has been counted. If this
dies at any point before step 4, nothing has been lost: rerun it. If it dies
between 4 and 5, the scratch collection still holds every point and the script
picks it up on the next run rather than starting over.

## Running it

Audrey keeps serving throughout. Reads against a collection mid-rebuild will
find it missing for the seconds between steps 4 and 5 and return no hits from
it — annoying, not damaging, and the reason to run this when nobody is asking
questions. It is safe to run repeatedly; a collection that already has sparse
config is skipped.

    # from the laptop, in the repo root
    .venv/bin/python scripts/migrate_bm25.py --host 192.168.1.11 --dry-run
    .venv/bin/python scripts/migrate_bm25.py --host 192.168.1.11
"""

from __future__ import annotations

import argparse
import sys

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from audrey.kb import bm25
from audrey.kb.qdrant import SPARSE_NAME, TEXT_DIM
from audrey.kb.user_store import USER_TEXT_PREFIX

# Only the collections the hybrid query path actually searches:
# `_search_text_hybrid` reads `qdrant.text_collection` and the caller's own
# `kb_user_text_*`, and nothing else.
#
# `kb_chat_archive` and `kb_memory` are deliberately NOT here. They are text
# collections, and migrating them would look consistent — but they belong to
# `tools-server`, which creates them itself and upserts bare-list dense
# vectors through code this phase does not touch. No query path reads them
# lexically, so rebuilding them buys nothing today and takes on the risk of
# reshaping another service's storage to do it. Add them here when something
# actually searches them, and test that service's writes first.
#
# Image collections are excluded for a different reason: image points carry a
# `caption`, not `text`, and lexical search over captions is its own idea.
TEXT_COLLECTIONS = ("kb_text",)
SCRATCH_SUFFIX = "__bm25"
PAGE = 256


def is_text_collection(name: str) -> bool:
    return name in TEXT_COLLECTIONS or name.startswith(USER_TEXT_PREFIX)


def has_sparse(client: QdrantClient, name: str) -> bool:
    info = client.get_collection(name)
    return SPARSE_NAME in (info.config.params.sparse_vectors or {})


def dense_params(client: QdrantClient, name: str) -> qm.VectorParams:
    """The collection's existing dense config, so the copy is identical.

    Read rather than assumed. A collection built with a different dim or
    distance would otherwise be silently rebuilt as a 768-d cosine one and
    every vector in it would become meaningless.
    """
    params = client.get_collection(name).config.params.vectors
    if not isinstance(params, qm.VectorParams):
        raise SystemExit(
            f"{name} has named dense vectors ({params!r}); this script only "
            "handles the unnamed dense vector every Audrey collection uses.")
    return params


def copy_points(
    client: QdrantClient, src: str, dst: str, *, add_sparse: bool,
) -> int:
    """Stream `src` into `dst`, one page at a time. Returns points written."""
    written = 0
    offset = None
    while True:
        points, offset = client.scroll(
            src, limit=PAGE, offset=offset, with_payload=True, with_vectors=True)
        if not points:
            break
        batch = []
        for p in points:
            payload = p.payload or {}
            # The dense vector comes back as a bare list for an unnamed vector
            # and as a dict once the collection has named ones — the scratch
            # collection is the second case on the way back.
            dense = p.vector if isinstance(p.vector, list) else (p.vector or {}).get("")
            if dense is None:
                raise SystemExit(f"{src} point {p.id} has no unnamed dense vector")
            vectors: dict = {"": dense}
            if add_sparse:
                idx, val = bm25.document_vector(str(payload.get("text") or ""))
                vectors[SPARSE_NAME] = qm.SparseVector(indices=idx, values=val)
            else:
                existing = (p.vector or {}).get(SPARSE_NAME) if isinstance(p.vector, dict) else None
                if existing is not None:
                    vectors[SPARSE_NAME] = existing
            batch.append(qm.PointStruct(id=p.id, vector=vectors, payload=payload))
        client.upsert(dst, points=batch, wait=True)
        written += len(batch)
        print(f"      {written} points", end="\r", flush=True)
        if offset is None:
            break
    return written


def copy_payload_indexes(client: QdrantClient, src_schema: dict, dst: str) -> None:
    """Recreate the `user` / `file_id` keyword indexes the copy would lose.

    Without these, `delete_by_file_id` and the per-user file list fall back to
    a full scan — correct but slow, and slow in a way nobody would connect
    back to this script months later.
    """
    for field, schema in (src_schema or {}).items():
        try:
            client.create_payload_index(
                collection_name=dst, field_name=field,
                field_schema=getattr(schema, "data_type", None) or qm.PayloadSchemaType.KEYWORD)
        except Exception as e:  # noqa: BLE001 — an index we cannot recreate is worth naming
            print(f"      WARNING: could not recreate index on {field!r}: {e}")


def migrate(client: QdrantClient, name: str, *, dry_run: bool) -> bool:
    scratch = f"{name}{SCRATCH_SUFFIX}"
    original_exists = client.collection_exists(name)
    scratch_exists = client.collection_exists(scratch)

    # Which collection is authoritative depends on how far a previous run got.
    # Between steps 4 and 5 the original does not exist and the scratch holds
    # every point — reading the count from the original there would compare
    # the copy-back against 0 and report a successful rebuild as a failure.
    if original_exists:
        source_of_truth = name
    elif scratch_exists:
        source_of_truth = scratch
    else:
        print(f"  SKIP  {name} — no such collection")
        return True
    count = client.count(source_of_truth, exact=True).count

    if original_exists and has_sparse(client, name):
        print(f"  SKIP  {name} — already has a {SPARSE_NAME} vector")
        return True
    if dry_run:
        print(f"  WOULD MIGRATE  {name} ({count} points)")
        return True

    print(f"  {name} ({count} points, from {source_of_truth})")
    params = dense_params(client, source_of_truth)
    schema = client.get_collection(source_of_truth).payload_schema

    # Steps 1-3. Skipped when a previous run already got this far and died
    # before recreating the original.
    if not scratch_exists:
        client.create_collection(
            scratch, vectors_config=params,
            sparse_vectors_config={
                SPARSE_NAME: qm.SparseVectorParams(modifier=qm.Modifier.IDF)},
        )
        moved = copy_points(client, name, scratch, add_sparse=True)
        if moved != count:
            raise SystemExit(
                f"    ABORT: copied {moved} of {count} points from {name}. "
                f"The original is untouched; {scratch} can be deleted and this rerun.")
        print(f"    copied {moved} points to {scratch}, verified")
    else:
        print(f"    {scratch} already exists from an earlier run, resuming")

    # Step 4-5. The only destructive moment in the script, and it happens
    # after the replacement has been counted.
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        name, vectors_config=params,
        sparse_vectors_config={
            SPARSE_NAME: qm.SparseVectorParams(modifier=qm.Modifier.IDF)},
    )
    back = copy_points(client, scratch, name, add_sparse=False)
    copy_payload_indexes(client, schema, name)
    if back != count:
        print(f"    WARNING: copied back {back} of {count}; leaving {scratch} in place")
        return False
    client.delete_collection(scratch)
    print(f"    rebuilt {name} with {back} points and a {SPARSE_NAME} vector")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=6333)
    ap.add_argument("--dry-run", action="store_true",
                    help="List what would be rebuilt and stop.")
    ap.add_argument("--only", help="Migrate just this one collection.")
    args = ap.parse_args()

    client = QdrantClient(host=args.host, port=args.port, timeout=120)
    print(f"qdrant {client.info().version} at {args.host}:{args.port}\n")

    names = sorted(c.name for c in client.get_collections().collections)
    targets = [n for n in names
               if is_text_collection(n) and not n.endswith(SCRATCH_SUFFIX)]
    if args.only:
        targets = [n for n in targets if n == args.only]
        if not targets:
            print(f"no text collection named {args.only!r}")
            return 1

    print(f"{len(targets)} text collection(s) to consider "
          f"(dim {TEXT_DIM} expected):\n")
    ok = all(migrate(client, n, dry_run=args.dry_run) for n in targets)
    print("\n" + ("done" if ok else "finished with warnings — read the output"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
