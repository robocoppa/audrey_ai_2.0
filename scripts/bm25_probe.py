#!/usr/bin/env python3
"""Settle how Phase 39's lexical index can be built, before building it.

**Round 1 answered the plan's original question: no.** Qdrant 1.18.3 refuses
`update_collection(sparse_vectors_config=...)` on a collection that has no
sparse vector yet — `400 Wrong input: Not existing vector name error: bm25`.
That call only edits the params of one that already exists. The phase plan's
"no collection is recreated and no dense vector is recomputed" was checked
against the client's method list, which proves the method exists, not that a
server accepts it. Every existing collection must therefore be rebuilt.

**Rebuilt does not have to mean re-embedded.** The dense vectors are already
stored in Qdrant and can be scrolled back out, so a migration can copy them
into a new collection and attach freshly-computed sparse vectors without ever
calling the embedder or re-reading a source file. That is the plan's real
saving, and it survives.

What is now in doubt is the *shape* of the new collection. Audrey's dense
vector is unnamed — created as `vectors_config=VectorParams(...)`, stored by
Qdrant under `""` — and every search in `kb/qdrant.py` relies on that by
passing no `using=`. If a collection can be created with an unnamed dense
vector *and* named sparse vectors, the migration is invisible to all existing
code. If Qdrant demands that the dense vector be named once sparse vectors
exist, then every dense read and write in the codebase changes too, and the
migration has to swap collection and code at the same instant.

That is the question this probe now answers.

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
from qdrant_client.http.exceptions import UnexpectedResponse

# The real vectoriser, not a stand-in. Whether `document_vector` emits
# something Qdrant will accept — sorted, unique, uint32 indices — is a genuine
# integration risk, and a hand-made sparse vector here would hide it.
from audrey.kb import bm25

SCRATCH = "phase39_bm25_probe"      # stands in for today's kb_text
SCRATCH_V2 = "phase39_bm25_probe_v2"  # stands in for what it migrates to
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

    for name in (SCRATCH, SCRATCH_V2):
        if client.collection_exists(name):
            client.delete_collection(name)

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
        # 0. Re-confirm the dead end, so this script keeps carrying the
        #    evidence for why the migration exists at all.
        try:
            client.update_collection(SCRATCH, sparse_vectors_config={
                "bm25": qm.SparseVectorParams(modifier=qm.Modifier.IDF)})
            _report("in-place: add sparse to an existing collection", True,
                    "UNEXPECTED — round 1 said this fails; re-read the plan")
        except Exception as e:  # noqa: BLE001 — a refusal here is the expected result
            detail = ((e.content or b"").decode("utf-8", errors="replace")
                      if isinstance(e, UnexpectedResponse) else str(e))
            _report("in-place add is refused (expected, this is why we migrate)",
                    True, detail.strip()[:110])

        # 1. THE question. Can the replacement collection keep the dense vector
        #    unnamed while gaining named sparse vectors? If yes, every existing
        #    dense read and write in kb/qdrant.py is untouched by this phase.
        try:
            client.create_collection(
                SCRATCH_V2,
                vectors_config=qm.VectorParams(size=DIM, distance=qm.Distance.COSINE),
                sparse_vectors_config={
                    "bm25": qm.SparseVectorParams(modifier=qm.Modifier.IDF)},
            )
            ok &= _report("create: unnamed dense AND named sparse together", True)
        except Exception as e:  # noqa: BLE001 — the whole point is to report it
            ok &= _report("create: unnamed dense AND named sparse together", False,
                          f"{type(e).__name__}: {str(e)[:200]}")
            print("\n  -> The dense vector must be NAMED once sparse vectors exist.")
            print("     Every dense read/write in kb/qdrant.py changes too, and")
            print("     collection and code have to swap at the same instant.\n")
            return 1

        # 2. The migration itself: read the old points back WITH their vectors.
        #    This is what makes the rebuild cheap — the embedder is never
        #    called, because the dense vectors already exist in Qdrant.
        old, _ = client.scroll(SCRATCH, limit=100, with_payload=True, with_vectors=True)
        shape = type(old[0].vector).__name__
        ok &= _report("old points scroll back with their dense vectors",
                      bool(old) and old[0].vector is not None,
                      f"{len(old)} points, vector is {shape}")

        # 3. Write them into the new collection, dense carried over verbatim,
        #    sparse computed from payload.text. `""` is the unnamed dense
        #    vector's real name — if that is wrong, the migration cannot write.
        def _sparse(text: str) -> qm.SparseVector:
            idx, val = bm25.document_vector(text)
            return qm.SparseVector(indices=idx, values=val)

        migrated = False
        for name in ("", "dense"):
            try:
                client.upsert(SCRATCH_V2, points=[qm.PointStruct(
                    id=p.id, payload=p.payload,
                    vector={name: p.vector,
                            "bm25": _sparse(str((p.payload or {}).get("text", "")))},
                ) for p in old])
                migrated = _report("migrate: dense carried over + sparse attached",
                                   True, f"dense key {name!r}, real bm25 vectors")
                break
            except Exception as e:  # noqa: BLE001
                _report(f"migrate with dense key {name!r}", False,
                        f"{type(e).__name__}: {str(e)[:120]}")
        ok &= migrated

        # 4. Dense search must still work with no `using=`, or every existing
        #    call site breaks on deploy.
        try:
            r = client.query_points(SCRATCH_V2, query=dense_a, limit=2)
            ok &= _report("dense search still works with no `using=`",
                          bool(r.points), f"top id={r.points[0].id}")
        except Exception as e:  # noqa: BLE001
            ok &= _report("dense search still works with no `using=`", False,
                          f"{type(e).__name__}: {str(e)[:200]}")

        # 5. The lexical retriever itself, end to end through the real
        #    tokenizer. "brown fox" is in point 1 and shares nothing with
        #    point 2, so a correct lexical search returns exactly one hit —
        #    which a dense search on this data could not do.
        try:
            q_idx, q_val = bm25.query_vector("brown fox")
            r = client.query_points(
                SCRATCH_V2, query=qm.SparseVector(indices=q_idx, values=q_val),
                using="bm25", limit=5)
            hits = [(p.id, round(p.score, 4)) for p in r.points]
            ok &= _report("sparse search matches only the point sharing terms",
                          [i for i, _ in hits] == [1], str(hits))
        except Exception as e:  # noqa: BLE001
            ok &= _report("sparse search", False, f"{type(e).__name__}: {str(e)[:200]}")

        # 6. A point whose sparse vector has not been written yet must stay
        #    dense-searchable. This is the state the whole migration runs in.
        try:
            client.upsert(SCRATCH_V2, points=[qm.PointStruct(
                id=99, vector={"": dense_b}, payload={"text": "no sparse yet"})])
            r = client.query_points(SCRATCH_V2, query=dense_b, limit=3)
            ok &= _report("a point with no sparse vector is still dense-searchable",
                          any(p.id == 99 for p in r.points),
                          f"ids={[p.id for p in r.points]}")
        except Exception as e:  # noqa: BLE001
            ok &= _report("a point with no sparse vector is still dense-searchable",
                          False, f"{type(e).__name__}: {str(e)[:200]}")

        # 7. Server-side RRF. Not required — the merge is client-side so the
        #    junk rule can see each retriever's own score — but knowing whether
        #    it works decides whether that stays a choice.
        try:
            q_idx, q_val = bm25.query_vector("brown fox")
            r = client.query_points(SCRATCH_V2, prefetch=[
                qm.Prefetch(query=dense_a, limit=5),
                qm.Prefetch(query=qm.SparseVector(indices=q_idx, values=q_val),
                            using="bm25", limit=5),
            ], query=qm.FusionQuery(fusion=qm.Fusion.RRF), limit=5)
            _report("server-side RRF fusion (informational)", True,
                    str([(p.id, round(p.score, 4)) for p in r.points]))
        except Exception as e:  # noqa: BLE001
            _report("server-side RRF fusion (informational)", False,
                    f"{type(e).__name__}: {str(e)[:160]}")
    finally:
        for name in (SCRATCH, SCRATCH_V2):
            if client.collection_exists(name):
                client.delete_collection(name)
        print(f"\nscratch collections {SCRATCH!r}, {SCRATCH_V2!r} deleted")

    print("\n" + ("ALL PASS — migrate by copying dense vectors, no re-embed"
                  if ok else "SOMETHING FAILED — re-read before building"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
