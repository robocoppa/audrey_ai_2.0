"""Tests for the Phase 39 BM25 migration.

This script deletes collections. That is the entire reason these tests exist:
it is the only code in the phase that can lose data, and the loss would be
silent — a rebuilt collection with half its points looks exactly like a
correct one until somebody searches for the missing half.

Run against `QdrantClient(":memory:")`, which implements everything the
migration uses. It cannot stand in for the server on the one thing the server
was needed for — adding a sparse vector to an existing collection, which both
refuse identically — but the migration deliberately never does that.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from audrey.kb import bm25
from audrey.kb.qdrant import SPARSE_NAME

_SPEC = importlib.util.spec_from_file_location(
    "migrate_bm25",
    Path(__file__).resolve().parent.parent / "scripts" / "migrate_bm25.py",
)
assert _SPEC and _SPEC.loader
migrate_bm25 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(migrate_bm25)

DIM = 8


def _dense(seed: int) -> list[float]:
    v = [0.0] * DIM
    v[seed % DIM] = 1.0
    return v


@pytest.fixture
def client() -> QdrantClient:
    return QdrantClient(":memory:")


def _make_old_collection(client: QdrantClient, name: str, texts: list[str]) -> None:
    """A collection shaped exactly like one that predates Phase 39."""
    client.create_collection(
        name, vectors_config=qm.VectorParams(size=DIM, distance=qm.Distance.COSINE))
    client.upsert(name, points=[
        qm.PointStruct(id=i + 1, vector=_dense(i),
                       payload={"text": t, "source": f"/d/{i}.md", "chunk_idx": i,
                                "user": "a@b.c", "kind": "text"})
        for i, t in enumerate(texts)
    ])


class TestCollectionSelection:
    def test_text_collections_are_selected(self):
        assert migrate_bm25.is_text_collection("kb_text")
        assert migrate_bm25.is_text_collection("kb_user_text_bart_proton_me")

    def test_image_collections_are_left_alone(self):
        """Image points carry a caption, not text. Rebuilding them would be a
        long destructive operation that bought nothing."""
        assert not migrate_bm25.is_text_collection("kb_images")
        assert not migrate_bm25.is_text_collection("kb_user_images_bart_proton_me")

    def test_tools_server_collections_are_left_alone(self):
        """`kb_chat_archive` and `kb_memory` are text, and migrating them would
        look consistent — but tools-server owns them, creates them itself, and
        upserts bare-list dense vectors through code this phase does not
        touch. No query path reads them lexically, so the rebuild would buy
        nothing and risk another service's storage to do it."""
        assert not migrate_bm25.is_text_collection("kb_chat_archive")
        assert not migrate_bm25.is_text_collection("kb_memory")


class TestMigration:
    def test_every_point_survives(self, client: QdrantClient):
        """The failure that must never happen quietly."""
        _make_old_collection(client, "kb_text", [f"document number {i}" for i in range(50)])

        assert migrate_bm25.migrate(client, "kb_text", dry_run=False)

        assert client.count("kb_text", exact=True).count == 50

    def test_the_dense_vectors_are_carried_over_unchanged(self, client: QdrantClient):
        """The saving that makes this affordable. If the dense vectors were
        not preserved byte for byte, the only fix would be re-embedding
        15,913 points — and for text uploads the sources are long gone."""
        _make_old_collection(client, "kb_text", ["alpha", "beta", "gamma"])
        before = {p.id: p.vector for p in client.retrieve(
            "kb_text", ids=[1, 2, 3], with_vectors=True)}

        migrate_bm25.migrate(client, "kb_text", dry_run=False)

        after = client.retrieve("kb_text", ids=[1, 2, 3], with_vectors=True)
        for p in after:
            assert p.vector[""] == before[p.id], f"point {p.id} lost its dense vector"

    def test_payloads_are_preserved(self, client: QdrantClient):
        _make_old_collection(client, "kb_text", ["hello world"])

        migrate_bm25.migrate(client, "kb_text", dry_run=False)

        payload = client.retrieve("kb_text", ids=[1], with_payload=True)[0].payload
        assert payload["text"] == "hello world"
        assert payload["source"] == "/d/0.md"
        assert payload["user"] == "a@b.c"

    def test_the_collection_gains_a_working_lexical_index(self, client: QdrantClient):
        """Not just sparse *config* — an actual searchable index, built from
        payload.text. This is the point of the whole migration."""
        _make_old_collection(client, "kb_text", [
            "and watch us play some baseball this year",
            "quarterly revenue figures for the third quarter",
        ])

        migrate_bm25.migrate(client, "kb_text", dry_run=False)

        idx, val = bm25.query_vector("watch us play baseball")
        r = client.query_points(
            "kb_text", query=qm.SparseVector(indices=idx, values=val),
            using=SPARSE_NAME, limit=5)
        assert [p.id for p in r.points] == [1]

    def test_the_dense_path_still_works_afterwards(self, client: QdrantClient):
        """Every existing search passes no `using=`. If the rebuild named the
        dense vector, all of them break at once."""
        _make_old_collection(client, "kb_text", ["alpha", "beta"])

        migrate_bm25.migrate(client, "kb_text", dry_run=False)

        r = client.query_points("kb_text", query=_dense(0), limit=2)
        assert r.points[0].id == 1

    def test_the_scratch_collection_is_cleaned_up(self, client: QdrantClient):
        _make_old_collection(client, "kb_text", ["alpha"])

        migrate_bm25.migrate(client, "kb_text", dry_run=False)

        assert not client.collection_exists("kb_text" + migrate_bm25.SCRATCH_SUFFIX)

    def test_an_empty_collection_migrates(self, client: QdrantClient):
        """A user who registered but never uploaded. The scroll returns
        nothing and the count comparison is 0 == 0, which must not read as a
        failed copy."""
        client.create_collection(
            "kb_user_text_new_user",
            vectors_config=qm.VectorParams(size=DIM, distance=qm.Distance.COSINE))

        assert migrate_bm25.migrate(client, "kb_user_text_new_user", dry_run=False)
        assert migrate_bm25.has_sparse(client, "kb_user_text_new_user")

    def test_a_point_with_empty_text_is_migrated_with_an_empty_sparse_vector(
        self, client: QdrantClient,
    ):
        """An empty transcript from a silent video. It must not abort the run."""
        _make_old_collection(client, "kb_text", ["", "real words here"])

        assert migrate_bm25.migrate(client, "kb_text", dry_run=False)
        assert client.count("kb_text", exact=True).count == 2


class TestIdempotence:
    def test_an_already_migrated_collection_is_skipped(self, client: QdrantClient):
        _make_old_collection(client, "kb_text", ["alpha"])
        migrate_bm25.migrate(client, "kb_text", dry_run=False)
        first = client.retrieve("kb_text", ids=[1], with_vectors=True)[0].vector

        assert migrate_bm25.migrate(client, "kb_text", dry_run=False)

        assert client.retrieve("kb_text", ids=[1], with_vectors=True)[0].vector == first

    def test_running_it_twice_does_not_lose_points(self, client: QdrantClient):
        _make_old_collection(client, "kb_text", [f"doc {i}" for i in range(20)])
        migrate_bm25.migrate(client, "kb_text", dry_run=False)
        migrate_bm25.migrate(client, "kb_text", dry_run=False)
        assert client.count("kb_text", exact=True).count == 20

    def test_a_dry_run_changes_nothing(self, client: QdrantClient):
        _make_old_collection(client, "kb_text", ["alpha"])

        assert migrate_bm25.migrate(client, "kb_text", dry_run=True)

        assert not migrate_bm25.has_sparse(client, "kb_text")


class TestResume:
    def test_a_leftover_scratch_collection_is_resumed_not_restarted(
        self, client: QdrantClient,
    ):
        """The window between deleting the original and recreating it is the
        only moment the data lives in one place. If the script died there, a
        rerun must finish the job from the scratch collection rather than
        rebuild from an original that no longer exists."""
        _make_old_collection(client, "kb_text", ["alpha", "beta", "gamma"])
        scratch = "kb_text" + migrate_bm25.SCRATCH_SUFFIX
        client.create_collection(
            scratch,
            vectors_config=qm.VectorParams(size=DIM, distance=qm.Distance.COSINE),
            sparse_vectors_config={
                SPARSE_NAME: qm.SparseVectorParams(modifier=qm.Modifier.IDF)},
        )
        migrate_bm25.copy_points(client, "kb_text", scratch, add_sparse=True)
        client.delete_collection("kb_text")  # died right here

        assert migrate_bm25.migrate(client, "kb_text", dry_run=False)

        assert client.count("kb_text", exact=True).count == 3
        assert migrate_bm25.has_sparse(client, "kb_text")
        assert not client.collection_exists(scratch)


class TestSafety:
    def test_a_named_dense_vector_is_refused_rather_than_rebuilt(
        self, client: QdrantClient,
    ):
        """This script only understands the unnamed dense vector every Audrey
        collection uses. Silently rebuilding a differently-shaped collection
        as a 768-d cosine one would make every vector in it meaningless."""
        client.create_collection(
            "kb_text",
            vectors_config={"dense": qm.VectorParams(size=DIM, distance=qm.Distance.COSINE)})

        with pytest.raises(SystemExit, match="named dense vectors"):
            migrate_bm25.migrate(client, "kb_text", dry_run=False)

    def test_the_original_survives_a_failed_copy(self, client: QdrantClient, monkeypatch):
        """The count check exists so a partial copy aborts *before* the
        original is deleted. Nothing is lost and the run is repeatable."""
        _make_old_collection(client, "kb_text", ["alpha", "beta", "gamma"])
        monkeypatch.setattr(migrate_bm25, "copy_points", lambda *a, **k: 1)

        with pytest.raises(SystemExit, match="copied 1 of 3"):
            migrate_bm25.migrate(client, "kb_text", dry_run=False)

        assert client.count("kb_text", exact=True).count == 3
