"""Local FTS projection bounds, source replacement and trustworthy text offsets."""

from __future__ import annotations

import hashlib
import sqlite3
import stat
from dataclasses import replace

import pytest

from audrey import content_search
from audrey.content_search import (
    CHUNK_CHARS,
    EXCERPT_CHARS,
    ContentSearchIndex,
    ContentSearchQueryError,
    ContentSearchQuotaError,
    ContentSearchUnavailableError,
    SearchDocument,
)


def document(**changes):
    return replace(
        SearchDocument(
            owner_id="alice",
            source_id="message:one",
            kind="message",
            revision=1,
            title="History notes",
            text="A timeline of crusades and their aftermath.",
            updated_at="2026-10-29T20:00:00Z",
            target={"type": "message", "message_id": "one"},
        ),
        **changes,
    )


@pytest.fixture
def index(tmp_path):
    projection = ContentSearchIndex(tmp_path / "search" / "content.sqlite")
    yield projection
    projection.close()


async def test_content_search_returns_native_identity_metadata_and_exact_excerpt(index):
    source = document(private_required=True, availability="retained_text")
    await index.replace_source(source)
    [result] = await index.search("alice", "crusades")
    assert result["source_id"] == source.source_id
    assert result["owner_id"] == "alice"
    assert result["revision"] == 1
    assert result["target"] == source.target
    assert result["private_required"] is True
    assert result["availability"] == "retained_text"
    assert result["body_hash"] == hashlib.sha256(source.text.encode()).hexdigest()
    start = result["offset"]
    assert source.text[start : start + len(result["excerpt"])] == result["excerpt"]
    assert [result["excerpt"][r["start"] : r["end"]] for r in result["highlights"]] == ["crusades"]


async def test_replacement_removes_old_tokens_and_updates_revision_and_target(index):
    await index.replace_source(document())
    updated = document(
        revision="new-revision",
        title="Changed title",
        text="Trade across oceans.",
        target={"type": "message", "message_id": "one", "revision": "new"},
    )
    await index.replace_source(updated)
    assert await index.search("alice", "crusades") == []
    [result] = await index.search("alice", "oceans")
    assert result["revision"] == "new-revision"
    assert result["target"] == updated.target
    assert result["title"] == updated.title
    assert len(await index.list_sources("alice")) == 1


async def test_removal_is_idempotent_and_does_not_delete_another_owner(index):
    await index.replace_source(document())
    await index.replace_source(document(owner_id="bob"))
    await index.remove_source("alice", "message:one")
    await index.remove_source("alice", "message:one")
    assert await index.search("alice", "crusades") == []
    assert await index.list_sources("alice") == []
    assert len(await index.search("bob", "crusades")) == 1


async def test_owner_filter_applies_before_candidate_page(index):
    for number in range(5):
        await index.replace_source(document(owner_id="bob", source_id=f"bob:{number}"))
    await index.replace_source(document())
    [result] = await index.search("alice", "crusades", limit=1)
    assert result["owner_id"] == "alice"
    assert await index.search("unknown", "crusades") == []


async def test_owner_reconciliation_inventory_and_atomic_cleanup(index):
    await index.replace_source(document())
    await index.replace_source(document(source_id="agent:one", kind="agent"))
    await index.replace_source(document(owner_id="bob"))
    assert await index.list_owners() == ["alice", "bob"]
    await index.remove_owner("alice")
    await index.remove_owner("alice")
    assert await index.list_owners() == ["bob"]
    assert await index.list_sources("alice") == []
    assert await index.search("alice", "crusades") == []
    assert len(await index.search("bob", "crusades")) == 1


async def test_owner_inventory_limit_is_also_enforced_on_writes(index, monkeypatch):
    monkeypatch.setattr(content_search, "MAX_OWNERS", 1)
    await index.replace_source(document())
    with pytest.raises(ContentSearchQuotaError):
        await index.replace_source(document(owner_id="bob"))
    assert await index.list_owners() == ["alice"]


async def test_kind_project_and_archive_filters(index):
    await index.replace_source(document(project_id="public-project"))
    await index.replace_source(
        document(
            source_id="file:one",
            kind="file",
            project_id="private-project",
            archived=True,
            private_required=True,
        )
    )
    await index.replace_source(document(source_id="agent:one", kind="agent"))
    assert len(await index.search("alice", "crusades")) == 2
    [file] = await index.search(
        "alice", "crusades", kind="file", archived=True, project_id="private-project"
    )
    assert file["source_id"] == "file:one"
    assert file["private_required"] is True
    assert await index.search("alice", "crusades", project_id="missing") == []
    assert len(await index.search("alice", "crusades", archived=None)) == 3


@pytest.mark.parametrize("query", ["alpha OR beta", 'alpha "OR" beta', "{alpha} OR (beta)"])
async def test_fts_syntax_is_treated_as_literal_terms(index, query):
    await index.replace_source(document(source_id="literal", text="alpha or beta"))
    await index.replace_source(document(source_id="other", text="alpha"))
    await index.replace_source(document(source_id="third", text="beta"))
    results = await index.search("alice", query)
    assert [result["source_id"] for result in results] == ["literal"]


async def test_user_text_cannot_supply_fts_column_or_sql_expression(index):
    await index.replace_source(document())
    assert await index.search("alice", "owner_key: bob OR 1=1 --") == []
    assert await index.search("alice", 'crusades"; DROP TABLE search_sources; --') == []
    assert len(await index.list_sources("alice")) == 1


@pytest.mark.parametrize("query", ["", "   ", "*", "a" * 257, "one " * 13, "word\x00other"])
async def test_query_bounds_are_rejected_before_fts_execution(index, query):
    with pytest.raises(ContentSearchQueryError):
        await index.search("alice", query)


@pytest.mark.parametrize(
    "filters",
    [
        {"kind": "tool"},
        {"project_id": ""},
        {"archived": "false"},
        {"limit": 0},
        {"limit": 201},
        {"limit": True},
        {"offset": -1},
        {"offset": 10_001},
        {"offset": False},
    ],
)
async def test_filter_and_page_bounds(index, filters):
    with pytest.raises(ContentSearchQueryError):
        await index.search("alice", "crusades", **filters)


async def test_unicode_character_offsets_and_accent_highlights(index):
    text = "🌳 " * 2_500 + "The Café cafe\u0301 had a résumé. " + "end " * 300
    await index.replace_source(document(text=text))
    [result] = await index.search("alice", "cafe")
    assert result["offset"] > CHUNK_CHARS
    assert len(result["excerpt"]) <= EXCERPT_CHARS
    assert result["excerpt"] == text[result["offset"] : result["offset"] + len(result["excerpt"])]
    words = [result["excerpt"][r["start"] : r["end"]] for r in result["highlights"]]
    assert words == ["Café", "cafe\u0301"]


async def test_excerpt_is_plain_untrusted_text_with_ranges_not_html(index):
    text = '<script>alert("crusades")</script> <b>crusades</b>'
    await index.replace_source(document(text=text))
    [result] = await index.search("alice", "crusades")
    assert result["excerpt"] == text
    assert "<script>" in result["excerpt"]
    assert all(result["excerpt"][r["start"] : r["end"]] == "crusades" for r in result["highlights"])


async def test_long_sources_do_not_dominate_and_pages_are_distinct(index):
    await index.replace_source(document(source_id="long", text="crusades " * 5_000))
    for number in range(5):
        await index.replace_source(document(source_id=f"short:{number}"))
    first = await index.search("alice", "crusades", limit=3)
    second = await index.search("alice", "crusades", limit=3, offset=3)
    ids = [result["source_id"] for result in first + second]
    assert len(ids) == len(set(ids)) == 6


async def test_chunk_overlap_preserves_a_word_near_boundary(index):
    text = "padding " * 498 + "crusades " + "after " * 500
    await index.replace_source(document(text=text))
    [result] = await index.search("alice", "crusades")
    assert "crusades" in result["excerpt"]
    assert result["excerpt"] == text[result["offset"] : result["offset"] + len(result["excerpt"])]


async def test_title_is_searchable_even_when_body_is_empty(index):
    await index.replace_source(document(title="Crusades source", text=""))
    [result] = await index.search("alice", "crusades")
    assert result["title"] == "Crusades source"
    assert result["excerpt"] == ""
    assert result["offset"] == 0
    assert result["highlights"] == []


async def test_list_sources_is_bounded_metadata_and_retains_revision_type(index):
    await index.replace_source(document(revision="v2"))
    await index.replace_source(document(owner_id="bob"))
    [metadata] = await index.list_sources("alice")
    assert metadata["revision"] == "v2"
    assert "text" not in metadata
    assert "excerpt" not in metadata
    assert "body_hash" in metadata


async def test_reopen_preserves_index_and_external_content_deletes(tmp_path):
    path = tmp_path / "content.sqlite"
    first = ContentSearchIndex(path)
    await first.replace_source(document())
    first.close()
    second = ContentSearchIndex(path)
    assert len(await second.search("alice", "crusades")) == 1
    await second.remove_source("alice", "message:one")
    assert await second.search("alice", "crusades") == []
    second.close()


async def test_storage_quota_failure_rolls_back_previous_source(index, monkeypatch):
    await index.replace_source(document(text="cat"))
    monkeypatch.setattr(content_search, "MAX_OWNER_CHARS", 4)
    with pytest.raises(ContentSearchQuotaError):
        await index.replace_source(document(text="longer", revision=2))
    [previous] = await index.search("alice", "cat")
    assert previous["revision"] == 1
    assert await index.search("alice", "longer") == []


async def test_owner_quota_does_not_block_different_owner(index, monkeypatch):
    monkeypatch.setattr(content_search, "MAX_OWNER_SOURCES", 1)
    await index.replace_source(document())
    with pytest.raises(ContentSearchQuotaError):
        await index.replace_source(document(source_id="second"))
    await index.replace_source(document(owner_id="bob"))
    assert len(await index.list_sources("bob")) == 1


async def test_source_size_is_rejected_without_replacing_existing_source(index, monkeypatch):
    await index.replace_source(document(text="cat"))
    monkeypatch.setattr(content_search, "MAX_SOURCE_CHARS", 5)
    with pytest.raises(ContentSearchQuotaError):
        await index.replace_source(document(text="longer"))
    assert len(await index.search("alice", "cat")) == 1


async def test_bad_metadata_never_creates_partial_source(index):
    with pytest.raises(ContentSearchQueryError):
        await index.replace_source(document(target={"bad": object()}))
    assert await index.list_sources("alice") == []


@pytest.mark.parametrize(
    "changes",
    [
        {"text": "invalid\ud800"},
        {"title": "invalid\ud800"},
        {"target": {"invalid": "\ud800"}},
        {"archived": "false"},
        {"private_required": 1},
        {"owner_id": "invalid\ud800"},
    ],
)
async def test_invalid_source_encoding_or_flags_does_not_break_next_transaction(index, changes):
    with pytest.raises(ContentSearchQueryError):
        await index.replace_source(document(**changes))
    await index.replace_source(document())
    assert len(await index.search("alice", "crusades")) == 1


async def test_closed_projection_fails_cleanly_and_close_is_idempotent(index):
    index.close()
    index.close()
    with pytest.raises(ContentSearchUnavailableError):
        await index.search("alice", "crusades")
    with pytest.raises(ContentSearchUnavailableError):
        await index.replace_source(document())


def test_fts5_unavailable_has_explicit_clean_failure(tmp_path, monkeypatch):
    connect = sqlite3.connect
    closed = []

    class NoFtsConnection(sqlite3.Connection):
        def executescript(self, sql):
            raise sqlite3.OperationalError("no such module: fts5")

        def close(self):
            closed.append(True)
            return super().close()

    monkeypatch.setattr(
        content_search.sqlite3,
        "connect",
        lambda *a, **kw: connect(*a, **kw, factory=NoFtsConnection),
    )
    with pytest.raises(ContentSearchUnavailableError, match="unavailable"):
        ContentSearchIndex(tmp_path / "unsupported.sqlite")
    assert closed == [True]


def test_unknown_projection_schema_requires_rebuild_without_touching_file(tmp_path):
    path = tmp_path / "future.sqlite"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            "CREATE TABLE future_data (text); INSERT INTO future_data VALUES ('saved'); PRAGMA user_version=9;"
        )
    with pytest.raises(ContentSearchUnavailableError):
        ContentSearchIndex(path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT text FROM future_data").fetchone()[0] == "saved"


async def test_projection_files_are_private_and_raw_deletion_is_enabled(tmp_path):
    path = tmp_path / "dedicated" / "index.sqlite"
    projection = ContentSearchIndex(path)
    try:
        await projection.replace_source(document(private_required=True))
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
        for file in (path, path.with_name(path.name + "-wal"), path.with_name(path.name + "-shm")):
            if file.exists():
                assert stat.S_IMODE(file.stat().st_mode) == 0o600
        assert projection._connection.execute("PRAGMA secure_delete").fetchone()[0] == 1
        await projection.remove_owner("alice")
        wal = path.with_name(path.name + "-wal")
        assert not wal.exists() or wal.stat().st_size == 0
    finally:
        projection.close()


def test_existing_parent_permissions_are_preserved(tmp_path):
    parent = tmp_path / "shared"
    parent.mkdir(mode=0o755)
    original_mode = stat.S_IMODE(parent.stat().st_mode)
    projection = ContentSearchIndex(parent / "index.sqlite")
    projection.close()
    assert stat.S_IMODE(parent.stat().st_mode) == original_mode
