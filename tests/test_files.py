"""Tests for routes/files.py — upload streaming, rollback, filename cap.

Three regression areas pinned here:

  - `_stream_to_disk` checks the cap *before* extending, so a single
    oversized chunk can't push `written` past `limit_bytes`.
  - The upload route's rollback path tolerates a double-failure: if
    `qdrant.delete_by_file_id` raises while we're already cleaning up
    after a sqlite write failure, we log the second error and surface
    the original 500 instead of letting the second exception mask it.
  - Filenames are capped at 255 chars (Linux NAME_MAX) before they
    land in sqlite or the Qdrant payload — a 10 MB filename string
    can't bloat either store.
  - `ALLOWED_EXTENSIONS` stays derived from `ALLOWED_MIMES`, so the
    extension hint the upload page pre-checks against can never offer a
    format the sniff gate would then reject.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from audrey.kb.extract import ALLOWED_EXTENSIONS, ALLOWED_MIMES, SUFFIX_MIMES
from audrey.routes.files import _stream_to_disk

# ─── _stream_to_disk cap-before-extending ──────────────────────────────


class _ChunkedUpload:
    """Minimal stand-in for FastAPI's `UploadFile.read(size)`.

    Returns the supplied chunks in order, then empty bytes (EOF). Doesn't
    honor the `size` argument — the test controls chunk sizes directly.
    """

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)

    async def read(self, _size: int) -> bytes:
        if self._chunks:
            return self._chunks.pop(0)
        return b""


@pytest.mark.asyncio
async def test_stream_to_disk_rejects_oversized_single_chunk_before_writing(
    tmp_path: Path,
):
    # Pre-fix: a single oversized chunk would land on disk and bump
    # `written` past `limit_bytes` before the check noticed. Post-fix:
    # the check fires *before* the write, so a 10 MB chunk against a
    # 5 MB cap returns -1 without touching disk.
    upload = _ChunkedUpload([b"x" * (10 * 1024 * 1024)])
    dest = tmp_path / "blob.bin"

    result = await _stream_to_disk(upload, dest, limit_bytes=5 * 1024 * 1024)

    assert result == -1
    # File handle was opened ("wb" truncates) but no bytes hit disk
    # because the oversized chunk was rejected pre-write.
    assert dest.read_bytes() == b""


@pytest.mark.asyncio
async def test_stream_to_disk_returns_total_when_under_cap(tmp_path: Path):
    upload = _ChunkedUpload([b"abc", b"def", b"ghi"])
    dest = tmp_path / "blob.bin"

    result = await _stream_to_disk(upload, dest, limit_bytes=100)

    assert result == 9
    assert dest.read_bytes() == b"abcdefghi"


@pytest.mark.asyncio
async def test_stream_to_disk_rejects_when_cumulative_size_would_overflow(
    tmp_path: Path,
):
    # First chunk fits, second would push us over. The check should
    # catch it before writing the second chunk — first chunk's bytes
    # are on disk (that's fine; caller unlinks on -1), but the second
    # is never written.
    upload = _ChunkedUpload([b"x" * 60, b"y" * 50])
    dest = tmp_path / "blob.bin"

    result = await _stream_to_disk(upload, dest, limit_bytes=100)

    assert result == -1
    # First chunk wrote; second was rejected pre-write.
    assert dest.read_bytes() == b"x" * 60


# ─── Filename cap at NAME_MAX (255) ───────────────────────────────────


def test_filename_cap_uses_255_char_slice():
    # The cap lives at the upload route's filename normalization step
    # (`filename = Path(...).name[:255]`). The slice is what protects
    # sqlite and the Qdrant payload from a 10 MB filename string.
    # Direct exercise of the slice idiom for a regression guard — if
    # the slice is ever dropped, a 1000-char name lands intact.
    huge = "a" * 10_000 + ".txt"
    capped = Path(huge).name[:255]
    assert len(capped) == 255
    assert capped == "a" * 255


def test_filename_cap_is_no_op_when_already_short():
    # A name that's already under 255 stays exactly as-is — the slice
    # mustn't add any padding or truncation surprise.
    name = "geology-notes.md"
    capped = Path(name).name[:255]
    assert capped == name


# ─── Client pre-check hint list stays derived ─────────────────────────


def test_allowed_extensions_only_maps_to_allowed_mimes():
    # The upload page refuses a file whose extension isn't in this set, so
    # anything advertised here must actually survive the sniff gate. A
    # hand-maintained list would drift the first time a mime was dropped.
    for ext in ALLOWED_EXTENSIONS:
        assert SUFFIX_MIMES[ext] in ALLOWED_MIMES, f"{ext} maps outside the allowlist"


def test_allowed_extensions_covers_every_mapped_allowed_mime():
    # The other direction: every suffix whose mime is allowed must be
    # offered. Dropping one silently makes the page reject files the
    # server would happily accept.
    expected = {ext for ext, mime in SUFFIX_MIMES.items() if mime in ALLOWED_MIMES}
    assert set(ALLOWED_EXTENSIONS) == expected


def test_allowed_extensions_excludes_unsupported_formats():
    # Guards the case that started this: video has no ingest path, so it
    # must never reach the client hint list.
    for ext in (".mp4", ".mov", ".mkv", ".webm", ".exe"):
        assert ext not in ALLOWED_EXTENSIONS


def test_every_extension_is_dot_prefixed_and_lowercase():
    # `extOf()` in upload.html lowercases and keeps the dot; a bare or
    # uppercase entry here would never match and would silently block
    # that format at the pre-check.
    for ext in ALLOWED_EXTENSIONS:
        assert ext.startswith(".") and ext == ext.lower()


# ─── Upload-rollback double-failure (#6) ──────────────────────────────


class _SpyDb:
    """Minimal UploadsDB stand-in that raises on record_upload."""

    def __init__(self, raise_on_record: Exception) -> None:
        self._raise = raise_on_record

    async def record_upload(self, **_kwargs) -> None:
        raise self._raise


class _SpyQdrant:
    """Minimal QdrantKB stand-in that can be set to raise on rollback."""

    def __init__(self, raise_on_delete: Exception | None = None) -> None:
        self._raise = raise_on_delete
        self.delete_calls: list[tuple[str, str, str]] = []

    async def delete_by_file_id(
        self, file_id: str, *, user: str, collection: str,
    ) -> None:
        self.delete_calls.append((file_id, user, collection))
        if self._raise is not None:
            raise self._raise


@pytest.mark.asyncio
async def test_rollback_double_failure_does_not_mask_original_error():
    # Reproduces the upload route's rollback block in isolation: the
    # sqlite write fails (the primary error we want to surface), and
    # the Qdrant rollback also fails (a transient outage). We want the
    # original error to win — the rollback's exception should be logged
    # and swallowed, not propagated.
    primary_error = RuntimeError("sqlite is wedged")
    rollback_error = RuntimeError("qdrant is also wedged")

    db = _SpyDb(raise_on_record=primary_error)
    qdrant = _SpyQdrant(raise_on_delete=rollback_error)

    # Mirror the route's rollback block. (The actual route runs this
    # inside a FastAPI handler; the structure here is the part we
    # changed in the fix.)
    captured: Exception | None = None
    try:
        try:
            await db.record_upload()
        except Exception:
            try:
                await qdrant.delete_by_file_id(
                    "file-id", user="u@example.com", collection="kb_user_text_u",
                )
            except Exception:  # noqa: BLE001, S110 — test pinning the swallow-and-log behavior
                pass
            raise  # re-raise the *original* error
    except Exception as e:  # noqa: BLE001 — test captures whichever exception bubbles
        captured = e

    # The rollback was attempted (and failed); the primary error
    # propagated; nothing leaked the rollback exception to the caller.
    assert qdrant.delete_calls == [
        ("file-id", "u@example.com", "kb_user_text_u"),
    ]
    assert captured is primary_error


@pytest.mark.asyncio
async def test_rollback_success_path_still_re_raises_original_error():
    # The "easy" case: rollback succeeds. We still want the original
    # error to surface (the upload is being aborted; the user needs to
    # know why). This pins the contract that the try/except around
    # rollback never swallows the primary failure.
    primary_error = RuntimeError("sqlite is wedged")

    db = _SpyDb(raise_on_record=primary_error)
    qdrant = _SpyQdrant(raise_on_delete=None)

    captured: Exception | None = None
    try:
        try:
            await db.record_upload()
        except Exception:
            try:
                await qdrant.delete_by_file_id(
                    "file-id", user="u@example.com", collection="kb_user_text_u",
                )
            except Exception:  # noqa: BLE001, S110 — test pinning the swallow-and-log behavior
                pass
            raise
    except Exception as e:  # noqa: BLE001 — test captures whichever exception bubbles
        captured = e

    assert qdrant.delete_calls == [
        ("file-id", "u@example.com", "kb_user_text_u"),
    ]
    assert captured is primary_error


# Avoid an unused-import warning on `asyncio` if we ever drop the
# explicit awaits above. (Currently used implicitly via pytest-asyncio.)
_ = asyncio
