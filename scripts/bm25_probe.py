#!/usr/bin/env python3
"""Settle whether Phase 39's backfill is possible before building it.

The phase plan rests on one claim: sparse vectors can be added to Audrey's
*existing* collections, so no dense vector is ever recomputed and the embedder
is never called. That was checked against the qdrant-client method list, which
proves only that the call exists — not that a server accepts it.

The specific doubt is `kb_text`'s **unnamed** dense vector. It was created with
`vectors_config=VectorParams(...)`, no name, so Qdrant stores it under `""`.
Adding named sparse vectors to a collection like that may be refused outright,
and qdrant-client's local mode cannot answer the question — its
`update_sparse_vectors_config` only edits a sparse vector that already exists.

If any of these fail, the backfill is not a scroll-and-update: it is a
recreate-and-re-embed of every point in the KB, which is a different phase.

Touches nothing that exists. Creates one scratch collection, exercises the
mechanic, deletes it. Read-only against real data.

    # from the laptop, in the repo root — the venv's interpreter, not a bare
    # `python3`, which has no qdrant_client and fails on import
    .venv/bin/python scripts/bm25_probe.py --host 192.168.1.11
"""

from __future__ import annotations

import argparse
import sys

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

SCRATCH = "phase39_bm25_probe"
DIM = 768


def _report(step: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {step}{'  — ' + detail if detail else ''}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=6333)
    args = ap.parse_args()

    client = QdrantClient(host=args.host, port=args.port)
    print(f"qdrant {client.info().version} at {args.host}:{args.port}\n")

    if client.collection_exists(SCRATCH):
        client.delete_collection(SCRATCH)

    # Built exactly like kb_text and every kb_user_text_*: one unnamed dense
    # vector, no sparse config. If the probe passed against a *named* dense
    # vector it would prove nothing about the collections we actually have.
    client.create_collection(
        SCRATCH,
        vectors_config=qm.VectorParams(size=DIM, distance=qm.Distance.COSINE),
    )
    dense_a = [0.0] * DIM
    dense_a[0] = 1.0
    dense_b = [0.0] * DIM
    dense_b[1] = 1.0
    client.upsert(SCRATCH, points=[
        qm.PointStruct(id=1, vector=dense_a, payload={"text": "the quick brown fox"}),
        qm.PointStruct(id=2, vector=dense_b, payload={"text": "a slow green turtle"}),
    ])
    print(f"scratch collection {SCRATCH!r}: 2 points, unnamed dense only\n")

    ok = True
    try:
        # 1. THE question. Everything else is moot if this fails.
        try:
            client.update_collection(SCRATCH, sparse_vectors_config={
                "bm25": qm.SparseVectorParams(modifier=qm.Modifier.IDF)})
            ok &= _report("add a sparse vector to an existing collection", True)
        except Exception as e:  # noqa: BLE001 — the whole point is to report it
            ok &= _report(
                "add a sparse vector to an existing collection", False,
                f"{type(e).__name__}: {str(e)[:200]}")
            print("\n  -> The backfill would require recreating every collection")
            print("     and re-embedding every point. Re-plan the phase.\n")
            return 1

        info = client.get_collection(SCRATCH)
        sparse_cfg = info.config.params.sparse_vectors
        ok &= _report("the sparse config is readable afterwards", bool(sparse_cfg),
                      str(sparse_cfg))

        # 2. The dense vector must still be usable *unnamed* after the change.
        #    If adding a sparse vector forces every existing point to be
        #    addressed by name, the dense search path breaks on deploy.
        try:
            r = client.query_points(SCRATCH, query=dense_a, limit=2)
            ok &= _report("dense search still works with no `using=`", True,
                          f"top id={r.points[0].id}")
        except Exception as e:  # noqa: BLE001
            ok &= _report("dense search still works with no `using=`", False,
                          f"{type(e).__name__}: {str(e)[:200]}")

        # 3. Backfill mechanic: write a sparse vector onto a point that already
        #    has a dense one, without touching its payload.
        try:
            client.update_vectors(SCRATCH, points=[qm.PointVectors(
                id=1, vector={"bm25": qm.SparseVector(indices=[10, 20], values=[1.5, 0.7])})])
            got = client.retrieve(SCRATCH, ids=[1], with_payload=True, with_vectors=True)[0]
            kept = (got.payload or {}).get("text") == "the quick brown fox"
            ok &= _report("update_vectors adds sparse without touching payload", kept,
                          f"vectors={sorted((got.vector or {}).keys())}")
        except Exception as e:  # noqa: BLE001
            ok &= _report("update_vectors adds sparse without touching payload", False,
                          f"{type(e).__name__}: {str(e)[:200]}")

        # 4. New writes must carry both at once. `""` is the unnamed dense
        #    vector's real name — if this is wrong, ingest breaks after the
        #    collection gains sparse config.
        wrote_both = False
        for name in ("", "dense"):
            try:
                client.upsert(SCRATCH, points=[qm.PointStruct(
                    id=3, payload={"text": "a fast red fox"},
                    vector={name: dense_a, "bm25": qm.SparseVector(
                        indices=[10, 30], values=[1.1, 0.9])})])
                wrote_both = _report(
                    "upsert one point with dense + sparse together", True,
                    f"dense key {name!r}")
                break
            except Exception as e:  # noqa: BLE001
                _report(f"upsert with dense key {name!r}", False,
                        f"{type(e).__name__}: {str(e)[:120]}")
        ok &= wrote_both

        # 5. Sparse-only search — the lexical retriever itself.
        try:
            r = client.query_points(
                SCRATCH, query=qm.SparseVector(indices=[10], values=[1.0]),
                using="bm25", limit=5)
            hits = [(p.id, round(p.score, 4)) for p in r.points]
            ok &= _report("sparse search returns only points that share a term",
                          bool(hits) and all(i in (1, 3) for i, _ in hits), str(hits))
        except Exception as e:  # noqa: BLE001
            ok &= _report("sparse search", False, f"{type(e).__name__}: {str(e)[:200]}")

        # 6. Server-side RRF. Not required — the merge may end up client-side so
        #    the junk rule can see each retriever's own score — but knowing
        #    whether it works decides whether that is a choice or a constraint.
        try:
            r = client.query_points(SCRATCH, prefetch=[
                qm.Prefetch(query=dense_a, limit=5),
                qm.Prefetch(query=qm.SparseVector(indices=[10], values=[1.0]),
                            using="bm25", limit=5),
            ], query=qm.FusionQuery(fusion=qm.Fusion.RRF), limit=5)
            ok &= _report("server-side RRF fusion over both retrievers", True,
                          str([(p.id, round(p.score, 4)) for p in r.points]))
        except Exception as e:  # noqa: BLE001
            ok &= _report("server-side RRF fusion over both retrievers", False,
                          f"{type(e).__name__}: {str(e)[:200]}")

        # 7. A point with no sparse vector must stay findable by dense search.
        #    This is the half-indexed state the whole backfill runs in.
        try:
            r = client.query_points(SCRATCH, query=dense_b, limit=2)
            found = any(p.id == 2 for p in r.points)
            ok &= _report("a not-yet-backfilled point is still dense-searchable",
                          found, f"ids={[p.id for p in r.points]}")
        except Exception as e:  # noqa: BLE001
            ok &= _report("a not-yet-backfilled point is still dense-searchable",
                          False, f"{type(e).__name__}: {str(e)[:200]}")
    finally:
        client.delete_collection(SCRATCH)
        print(f"\nscratch collection {SCRATCH!r} deleted")

    print("\n" + ("ALL PASS — the phase 39 backfill plan holds"
                  if ok else "SOMETHING FAILED — re-read before building"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
