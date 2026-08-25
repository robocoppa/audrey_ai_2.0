"""Private KB reads remain isolated when two users share a collection name.

Per-user collection names are intentionally stable and lossy: punctuation is
collapsed to underscores. The raw authenticated user stored in every point is
therefore the final read boundary. These tests use Qdrant's in-memory client so
they exercise the real dense, sparse, file/artifact, and image filters rather
than a fake that merely records a scope argument.
"""

from __future__ import annotations

import pytest
from qdrant_client import QdrantClient

from audrey.kb import qdrant as qdrant_module
from audrey.kb.qdrant import (
    IMAGE_DIM,
    TEXT_DIM,
    QdrantKB,
    SearchScope,
    build_image_point,
    build_text_point,
)
from audrey.kb.user_store import user_image_collection, user_text_collection
from audrey.routes.kb import (
    _search_images_merged,
    _search_text_hybrid,
    _search_text_merged,
)

USER_A = "a.b@example.com"
USER_B = "a-b@example.com"
USERS = ((USER_A, "/private/a"), (USER_B, "/private/b"))
HYBRID_CFG = {"rrf_k": 60, "min_term_overlap": 0.0}


def _vector(dim: int) -> list[float]:
    return [1.0, *([0.0] * (dim - 1))]


@pytest.fixture
def colliding_kb(monkeypatch: pytest.MonkeyPatch) -> QdrantKB:
    client = QdrantClient(":memory:")
    monkeypatch.setattr(qdrant_module, "QdrantClient", lambda **_: client)
    kb = QdrantKB(host="unused", port=0)

    assert user_text_collection(USER_A) == user_text_collection(USER_B)
    assert user_image_collection(USER_A) == user_image_collection(USER_B)

    kb._ensure_sync()
    text_collection = user_text_collection(USER_A)
    image_collection = user_image_collection(USER_A)
    kb._ensure_named_sync(text_collection, TEXT_DIM)
    kb._ensure_named_sync(image_collection, IMAGE_DIM)

    text_points = []
    image_points = []
    for user, source in USERS:
        common = {
            "user": user,
            # Deliberately identical: neither file nor artifact scope can be
            # allowed to stand in for the raw-user condition.
            "file_id": "same-file-id",
            "filename": "same-name.mp4",
            "artifact": "transcript",
        }
        text_points.append(
            build_text_point(
                source=f"{source}.txt",
                chunk_idx=0,
                text=f"shared topic evidence belonging to {user}",
                vector=_vector(TEXT_DIM),
                mtime=0.0,
                extra=common,
                sparse=True,
            )
        )
        image_points.append(
            build_image_point(
                source=f"{source}.jpg",
                chunk_idx=0,
                caption=f"shared image evidence belonging to {user}",
                vector=_vector(IMAGE_DIM),
                mtime=0.0,
                extra=common,
            )
        )

    client.upsert(text_collection, points=text_points, wait=True)
    client.upsert(image_collection, points=image_points, wait=True)
    return kb


def _assert_only_user(hits, user: str, source_prefix: str) -> None:
    assert hits
    assert {hit.payload["user"] for hit in hits} == {user}
    assert all(hit.source.startswith(source_prefix) for hit in hits)


@pytest.mark.parametrize("user,source_prefix", USERS)
@pytest.mark.parametrize("min_score", [0.0, 0.53])
async def test_dense_private_reads_filter_the_raw_user_with_or_without_floor(
    colliding_kb: QdrantKB,
    user: str,
    source_prefix: str,
    min_score: float,
) -> None:
    hits, had_user = await _search_text_merged(
        colliding_kb,
        _vector(TEXT_DIM),
        top_k=10,
        user=user,
        min_score=min_score,
    )

    assert had_user
    _assert_only_user(hits, user, source_prefix)


@pytest.mark.parametrize("user,source_prefix", USERS)
@pytest.mark.parametrize("min_score", [0.0, 0.53])
async def test_hybrid_private_reads_filter_dense_and_sparse_results(
    colliding_kb: QdrantKB,
    user: str,
    source_prefix: str,
    min_score: float,
) -> None:
    hits, had_user = await _search_text_hybrid(
        colliding_kb,
        _vector(TEXT_DIM),
        query="shared topic evidence",
        top_k=10,
        user=user,
        min_score=min_score,
        cfg=HYBRID_CFG,
    )

    assert had_user
    _assert_only_user(hits, user, source_prefix)


@pytest.mark.parametrize("user,source_prefix", USERS)
@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize(
    "scope",
    [
        SearchScope(file_ids=["same-file-id"]),
        SearchScope(artifact="transcript"),
        SearchScope(file_ids=["same-file-id"], artifact="transcript"),
    ],
)
async def test_file_and_artifact_scopes_compose_with_the_raw_user_filter(
    colliding_kb: QdrantKB,
    user: str,
    source_prefix: str,
    hybrid: bool,
    scope: SearchScope,
) -> None:
    if hybrid:
        hits, had_user = await _search_text_hybrid(
            colliding_kb,
            _vector(TEXT_DIM),
            query="shared topic evidence",
            top_k=10,
            user=user,
            min_score=0.53,
            cfg=HYBRID_CFG,
            scope=scope,
        )
    else:
        hits, had_user = await _search_text_merged(
            colliding_kb,
            _vector(TEXT_DIM),
            top_k=10,
            user=user,
            min_score=0.53,
            scope=scope,
        )

    assert had_user
    _assert_only_user(hits, user, source_prefix)


@pytest.mark.parametrize("user,source_prefix", USERS)
async def test_private_image_reads_filter_the_raw_user(
    colliding_kb: QdrantKB,
    user: str,
    source_prefix: str,
) -> None:
    hits, had_user = await _search_images_merged(
        colliding_kb,
        _vector(IMAGE_DIM),
        top_k=10,
        user=user,
    )

    assert had_user
    _assert_only_user(hits, user, source_prefix)
