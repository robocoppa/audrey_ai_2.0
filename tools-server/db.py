"""Qdrant-backed memory store.

Stores each memory as a single Qdrant point whose payload holds `key`,
`value`, `tags`, `user`, `created_at`, `updated_at`. The vector is a
`nomic-embed-text` embedding of
`f"{key}: {value} [tags: {tags}]"` — concatenating tags into the embedded
text gives tag keywords some semantic weight even though we primarily filter
by the `user` payload field.

Point IDs are deterministic: `uuid5(user, key)` — re-storing the same
`(user, key)` overwrites the previous point rather than creating duplicates.

`recall(key)` is a payload-only scroll (no vector search); `search(user,
query, top_k)` is a vector search filtered by `user == <id>` and threshold
`MEMORY_SIMILARITY_THRESHOLD`.

Startup: if a legacy `memory.db` SQLite file exists, read every row, embed,
and upsert into Qdrant. Rename the file to `memory.db.migrated` only after
every row succeeds. Deterministic point ids make a partial retry idempotent.
"""

from __future__ import annotations

import datetime as _dt
import logging
import time as _time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiosqlite
import httpx
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm

log = logging.getLogger(__name__)

# The user tag lives inside the free-form `tags` string as `user:<id>`. We
# pull it out at write time and duplicate it into a dedicated payload field
# so filters can match exactly (substring filters on `tags` are fragile —
# `user:al` would match `user:alice`).
_USER_TAG_PREFIX = "user:"


@dataclass(slots=True, frozen=True)
class MemoryEntry:
    key: str
    value: str
    tags: str
    created_at: str
    updated_at: str


def _parse_user(tags: str) -> str:
    """Extract the `user:<id>` value from a comma/space-separated tag string."""
    for raw in tags.replace(",", " ").split():
        if raw.startswith(_USER_TAG_PREFIX):
            return raw[len(_USER_TAG_PREFIX):]
    return ""


def _point_id(user: str, key: str) -> str:
    """Deterministic UUIDv5 so re-stores overwrite."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{user}|{key}"))


def _embedding_text(key: str, value: str, tags: str) -> str:
    """Text that goes to the embedder.

    Leading `key` gives short queries like "threadripper workstation" some
    lexical hooks; trailing tags (minus the user-scope noise) add topical
    signal without overwhelming the value text.
    """
    stripped_tags = ",".join(
        t for t in tags.replace(",", " ").split()
        if not t.startswith(_USER_TAG_PREFIX)
    )
    if stripped_tags:
        return f"{key}: {value} [tags: {stripped_tags}]"
    return f"{key}: {value}"


def _public_tags(tags: str) -> str:
    """Return user-authored tags without the internal identity token."""
    return ",".join(
        token
        for token in tags.replace(",", " ").split()
        if not token.startswith(_USER_TAG_PREFIX)
    )


def _scoped_tags(user: str, tags: str) -> str:
    """Attach exactly one server-owned user tag to public tag text."""
    public = _public_tags(tags)
    return f"{_USER_TAG_PREFIX}{user},{public}" if public else f"{_USER_TAG_PREFIX}{user}"


def _user_filter(user: str, *, visible_after: str = "") -> qm.Filter:
    """Build the exact-user filter, optionally hiding a purge snapshot.

    Memories written or corrected after the cutoff are new activity and remain
    visible. Missing or malformed legacy timestamps do not satisfy the range
    condition, so they stay hidden once a cutoff exists.
    """
    must: list[qm.FieldCondition] = [
        qm.FieldCondition(key="user", match=qm.MatchValue(value=user)),
    ]
    if visible_after:
        must.append(qm.FieldCondition(
            key="updated_at",
            range=qm.DatetimeRange(gt=visible_after),
        ))
    return qm.Filter(must=must)


class EmbedError(RuntimeError):
    """Raised when Ollama refuses to produce an embedding."""


# Startup warm-up budget. Wide because it covers a cold model load (~4.2s
# measured for nomic-embed-text on the box, and far more if Ollama has to pull
# the model first) and nobody is waiting on it.
_WARM_TIMEOUT_S = 120.0


class MemoryStore:
    """Async Qdrant-backed memory store scoped by `user:<id>` tag."""

    def __init__(
        self,
        *,
        qdrant_url: str,
        ollama_url: str,
        collection: str,
        embed_model: str,
        embed_dim: int,
        similarity_threshold: float,
        embed_timeout_s: float,
        legacy_sqlite_path: Path,
        embed_keep_alive: str = "",
    ) -> None:
        self._qdrant = AsyncQdrantClient(url=qdrant_url)
        self._http = httpx.AsyncClient(base_url=ollama_url, timeout=embed_timeout_s)
        self._collection = collection
        self._embed_model = embed_model
        self._embed_dim = embed_dim
        self._threshold = similarity_threshold
        self._legacy_sqlite_path = legacy_sqlite_path
        self._embed_keep_alive = embed_keep_alive

    async def aclose(self) -> None:
        await self._qdrant.close()
        await self._http.aclose()

    # ─── Lifecycle ────────────────────────────────────────────────────

    async def init(self) -> None:
        """Ensure the collection exists and migrate any legacy SQLite rows."""
        await self._ensure_collection()
        await self._migrate_sqlite_if_present()
        await self.warm_embedder()

    async def warm_embedder(self) -> None:
        """Pay the embedder's cold load at startup instead of on a request.

        `keep_alive` keeps the model resident once it is loaded, but a restart
        starts from nothing — so without this the FIRST recall after every
        deploy is the one that eats 4.18s and blows its 4s budget. Loading it
        here moves that cost to a moment when nobody is waiting.

        Never raises: an unreachable Ollama must not stop custom-tools serving
        the tools that don't need it.
        """
        started = _time.monotonic()
        try:
            # Generous, and deliberately unrelated to the recall budget — this
            # call is the cold load, so it must be allowed to take longer than
            # the deadline that exists to avoid paying for one.
            await self._embed("warm", timeout_s=_WARM_TIMEOUT_S)
        except Exception as e:  # noqa: BLE001 — best-effort warm-up
            log.warning("memory: embedder warm-up failed in %.2fs: %s",
                        _time.monotonic() - started, e)
            return
        log.info("memory: embedder warm in %.2fs (keep_alive=%s)",
                 _time.monotonic() - started, self._embed_keep_alive or "ollama default")

    async def _ensure_collection(self) -> None:
        existing = {c.name for c in (await self._qdrant.get_collections()).collections}
        if self._collection in existing:
            return
        await self._qdrant.create_collection(
            collection_name=self._collection,
            vectors_config=qm.VectorParams(
                size=self._embed_dim,
                distance=qm.Distance.COSINE,
            ),
        )
        # Index on `user` so per-user filters don't scan every point.
        await self._qdrant.create_payload_index(
            collection_name=self._collection,
            field_name="user",
            field_schema=qm.PayloadSchemaType.KEYWORD,
        )
        await self._qdrant.create_payload_index(
            collection_name=self._collection,
            field_name="key",
            field_schema=qm.PayloadSchemaType.KEYWORD,
        )
        log.info("memory: created Qdrant collection %r (dim=%d)", self._collection, self._embed_dim)

    async def _migrate_sqlite_if_present(self) -> None:
        src = self._legacy_sqlite_path
        if not src.exists():
            return
        log.info("memory: migrating legacy SQLite store at %s", src)
        migrated = 0
        failed = 0
        async with aiosqlite.connect(src) as db:
            rows = await db.execute_fetchall(
                "SELECT key, value, tags, created_at, updated_at FROM memory"
            )
        for (key, value, tags, created_at, updated_at) in rows:
            user = _parse_user(tags)
            if not user:
                log.warning("memory: skipping legacy row %r (no user tag)", key)
                failed += 1
                continue
            try:
                vector = await self._embed(_embedding_text(key, value, tags))
            except EmbedError as e:
                log.warning("memory: embed failed for legacy %r: %s", key, e)
                failed += 1
                continue
            try:
                await self._qdrant.upsert(
                    collection_name=self._collection,
                    points=[
                        qm.PointStruct(
                            id=_point_id(user, key),
                            vector=vector,
                            payload={
                                "key": key, "value": value, "tags": tags,
                                "user": user, "created_at": created_at,
                                "updated_at": updated_at,
                            },
                        )
                    ],
                )
            except Exception as e:  # noqa: BLE001 — leave source retryable
                log.warning("memory: upsert failed for legacy %r: %s", key, e)
                failed += 1
                continue
            migrated += 1

        if failed:
            log.warning(
                "memory: migrated %d rows, failed %d; retaining %s for retry",
                migrated, failed, src.name,
            )
            return

        migrated_path = src.with_suffix(src.suffix + ".migrated")
        if migrated_path.exists():
            sequence = 1
            while True:
                candidate = migrated_path.with_name(f"{migrated_path.name}.{sequence}")
                if not candidate.exists():
                    migrated_path = candidate
                    break
                sequence += 1
        src.rename(migrated_path)
        log.info(
            "memory: migrated all %d rows, renamed %s -> %s",
            migrated, src.name, migrated_path.name,
        )

    # ─── Operations ────────────────────────────────────────────────────

    async def store(self, key: str, value: str, tags: str = "") -> MemoryEntry:
        user = _parse_user(tags)
        if not user:
            # Refuse to write untagged memories — otherwise they can't be
            # recalled (search requires a user filter) and they leak across
            # scopes. Callers should always include `user:<id>` in tags.
            raise ValueError("memory_store requires a 'user:<id>' token in tags")

        now = _dt.datetime.now(_dt.UTC).isoformat(timespec="microseconds")
        vector = await self._embed(_embedding_text(key, value, tags))
        point_id = _point_id(user, key)

        # Preserve created_at on overwrites by reading the existing point first.
        created_at = now
        try:
            existing = await self._qdrant.retrieve(
                collection_name=self._collection, ids=[point_id], with_payload=True,
            )
            if existing:
                created_at = existing[0].payload.get("created_at", now) or now
        except Exception as e:  # noqa: BLE001 — created_at preservation is best-effort; any retrieve failure just falls back to `now`
            log.debug("memory: retrieve for created_at failed: %s", e)

        await self._qdrant.upsert(
            collection_name=self._collection,
            points=[
                qm.PointStruct(
                    id=point_id, vector=vector,
                    payload={
                        "key": key, "value": value, "tags": tags,
                        "user": user, "created_at": created_at, "updated_at": now,
                    },
                )
            ],
        )
        return MemoryEntry(
            key=key, value=value, tags=tags,
            created_at=created_at, updated_at=now,
        )

    async def recall(
        self,
        key: str,
        *,
        user: str,
        visible_after: str = "",
    ) -> MemoryEntry | None:
        """Exact-key lookup, scoped to a single user.

        Both `key` AND `user` must match — never relax the `user` filter,
        or memories leak across accounts. If multiple points match (rare,
        since UUIDv5 point-id collapses re-stores into one point) we return
        the newest by `updated_at`.
        """
        result = await self._qdrant.scroll(
            collection_name=self._collection,
            scroll_filter=qm.Filter(must=[
                *_user_filter(user, visible_after=visible_after).must,
                qm.FieldCondition(key="key", match=qm.MatchValue(value=key)),
            ]),
            limit=10, with_payload=True, with_vectors=False,
        )
        points = result[0] if isinstance(result, tuple) else []
        if not points:
            return None
        points.sort(key=lambda p: p.payload.get("updated_at", ""), reverse=True)
        p = points[0].payload or {}
        return MemoryEntry(
            key=p.get("key", key),
            value=p.get("value", ""),
            tags=p.get("tags", ""),
            created_at=p.get("created_at", ""),
            updated_at=p.get("updated_at", ""),
        )

    async def list_user(
        self,
        *,
        user: str,
        limit: int = 100,
        cursor: str | None = None,
        visible_after: str = "",
    ) -> tuple[list[MemoryEntry], str | None]:
        """List one user memory scope without embedding or semantic search.

        The Qdrant scroll offset is already a stable point-id cursor. Memory
        point ids are deterministic UUIDs, so validating the incoming cursor
        before forwarding it gives callers a clean error instead of an opaque
        Qdrant failure. The internal user identity tag is an authorization
        implementation detail and is not part of the user-facing inventory.
        """
        if not user:
            raise ValueError("list_user requires a non-empty user")
        if limit < 1 or limit > 200:
            raise ValueError("limit must be between 1 and 200")
        offset: str | None = None
        if cursor:
            try:
                offset = str(uuid.UUID(cursor))
            except (ValueError, AttributeError) as e:
                raise ValueError("invalid memory cursor") from e

        points, next_offset = await self._qdrant.scroll(
            collection_name=self._collection,
            scroll_filter=_user_filter(user, visible_after=visible_after),
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        entries: list[MemoryEntry] = []
        for point in points:
            payload = point.payload or {}
            entries.append(MemoryEntry(
                key=str(payload.get("key", "")),
                value=str(payload.get("value", "")),
                tags=_public_tags(str(payload.get("tags", ""))),
                created_at=str(payload.get("created_at", "")),
                updated_at=str(payload.get("updated_at", "")),
            ))
        return entries, str(next_offset) if next_offset is not None else None

    async def update_user(
        self,
        *,
        user: str,
        key: str,
        value: str,
        tags: str | None = None,
    ) -> MemoryEntry | None:
        """Correct an existing current-user memory without creating a new key."""
        if not user:
            raise ValueError("update_user requires a non-empty user")
        if not key:
            raise ValueError("update_user requires a non-empty key")

        point_id = _point_id(user, key)
        existing = await self._qdrant.retrieve(
            collection_name=self._collection,
            ids=[point_id],
            with_payload=True,
        )
        if not existing:
            return None
        payload = existing[0].payload or {}
        if str(payload.get("user", "")) != user or str(payload.get("key", "")) != key:
            return None

        public_tags = _public_tags(
            str(payload.get("tags", "")) if tags is None else tags,
        )
        stored_tags = _scoped_tags(user, public_tags)
        now = _dt.datetime.now(_dt.UTC).isoformat(timespec="microseconds")
        created_at = str(payload.get("created_at", "") or now)
        vector = await self._embed(_embedding_text(key, value, stored_tags))
        await self._qdrant.upsert(
            collection_name=self._collection,
            points=[
                qm.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "key": key,
                        "value": value,
                        "tags": stored_tags,
                        "user": user,
                        "created_at": created_at,
                        "updated_at": now,
                    },
                )
            ],
            wait=True,
        )
        return MemoryEntry(
            key=key,
            value=value,
            tags=public_tags,
            created_at=created_at,
            updated_at=now,
        )

    async def delete_user(self, *, user: str, key: str) -> bool:
        """Delete one exact current-user memory and wait for Qdrant's ack."""
        if not user:
            raise ValueError("delete_user requires a non-empty user")
        if not key:
            raise ValueError("delete_user requires a non-empty key")

        point_id = _point_id(user, key)
        existing = await self._qdrant.retrieve(
            collection_name=self._collection,
            ids=[point_id],
            with_payload=True,
        )
        if not existing:
            return False
        payload = existing[0].payload or {}
        if str(payload.get("user", "")) != user or str(payload.get("key", "")) != key:
            return False
        await self._qdrant.delete(
            collection_name=self._collection,
            points_selector=qm.PointIdsList(points=[point_id]),
            wait=True,
        )
        return True

    async def delete_user_before(self, *, user: str, cutoff_at: str) -> None:
        """Delete one user purge snapshot while preserving newer activity.

        `must_not updated_at > cutoff` deliberately includes legacy points
        whose timestamp is absent or malformed. The selector is evaluated by
        Qdrant at deletion time, so a same-key post-cutoff correction cannot be
        deleted from a stale client-side id list.
        """
        if not user:
            raise ValueError("delete_user_before requires a non-empty user")
        if not cutoff_at:
            raise ValueError("delete_user_before requires a non-empty cutoff_at")
        await self._qdrant.delete(
            collection_name=self._collection,
            points_selector=qm.FilterSelector(filter=qm.Filter(
                must=[
                    qm.FieldCondition(
                        key="user",
                        match=qm.MatchValue(value=user),
                    ),
                ],
                must_not=[
                    qm.FieldCondition(
                        key="updated_at",
                        range=qm.DatetimeRange(gt=cutoff_at),
                    ),
                ],
            )),
            wait=True,
        )

    async def search(
        self,
        *,
        user: str,
        query: str,
        top_k: int = 5,
        visible_after: str = "",
    ) -> list[MemoryEntry]:
        """Semantic search scoped to a user.

        Embeds `query`, vector-searches with `user == <user>` payload filter,
        drops results below `MEMORY_SIMILARITY_THRESHOLD`.

        Raises `EmbedError` when the query can't be embedded. It used to
        swallow that and return `[]`, which is a different claim: "you have no
        memories about this" rather than "I could not look". The caller — the
        orchestrator's auto-recall, or a model that called `memory_search`
        itself — has no way back from an empty list, so a stalled embedder read
        as an empty memory store on every turn it affected.
        """
        if not query.strip():
            return []
        qvec = await self._embed(query)
        result = await self._qdrant.query_points(
            collection_name=self._collection,
            query=qvec,
            limit=top_k,
            score_threshold=self._threshold,
            query_filter=_user_filter(user, visible_after=visible_after),
            with_payload=True,
        )
        out: list[MemoryEntry] = []
        for point in result.points:
            p = point.payload or {}
            out.append(MemoryEntry(
                key=p.get("key", ""),
                value=p.get("value", ""),
                tags=p.get("tags", ""),
                created_at=p.get("created_at", ""),
                updated_at=p.get("updated_at", ""),
            ))
        return out

    # ─── Internals ────────────────────────────────────────────────────

    async def _embed(self, text: str, *, timeout_s: float | None = None) -> list[float]:
        """Call Ollama /api/embed and return a single 768-d vector.

        `keep_alive` holds the embedder in VRAM between calls. Without it
        Ollama evicts after 5 minutes and the next recall pays a ~4.2s cold
        load — which does not fit the 4s budget this call runs under, so recall
        failed on every turn that followed a quiet spell.

        `timeout_s` overrides the client default. Only the warm-up uses it: a
        cold load is LONGER than the hot-path budget by design, so a warm-up
        held to that budget would time out every single time and never warm
        anything.
        """
        payload: dict[str, Any] = {"model": self._embed_model, "input": [text]}
        if self._embed_keep_alive:
            payload["keep_alive"] = self._embed_keep_alive
        try:
            r = await self._http.post(
                "/api/embed",
                json=payload,
                timeout=httpx.Timeout(timeout_s) if timeout_s else httpx.USE_CLIENT_DEFAULT,
            )
        except httpx.HTTPError as e:
            raise EmbedError(f"transport error: {type(e).__name__}: {e}") from e
        if r.status_code >= 400:
            raise EmbedError(f"/api/embed -> {r.status_code}: {r.text[:200]}")
        body = r.json()
        vecs = body.get("embeddings") or []
        if not vecs or not isinstance(vecs[0], list):
            raise EmbedError(f"unexpected embed response shape: {body!r}")
        return vecs[0]
