"""Supported audio uploads enter the durable media queue."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from audrey.kb.storage_lifecycle import StorageReservation
from audrey.routes import files as files_routes
from audrey.routes.files import _validate_and_ingest


class _Storage:
    def __init__(self) -> None:
        self.commits: list[dict] = []

    async def commit_upload(self, reservation, **kwargs) -> None:
        self.commits.append({"reservation": reservation, **kwargs})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("suffix", "mime"),
    [
        (".mp3", "audio/mpeg"),
        (".wav", "audio/x-wav"),
        (".m4a", "audio/x-m4a"),
        (".flac", "audio/flac"),
    ],
)
async def test_supported_audio_format_becomes_a_pending_audio_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
    mime: str,
):
    dest = tmp_path / f"interview{suffix}"
    content = b"real container bytes are covered by the decoder tests"
    dest.write_bytes(content)
    storage = _Storage()
    reservation = StorageReservation(
        "r1", "alice@example.com", "single_shot", len(content),
    )
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                cfg=SimpleNamespace(raw={"kb": {"upload_root": str(tmp_path)}}),
            ),
        ),
    )
    monkeypatch.setattr(files_routes, "sniff_mime", lambda _path: mime)

    response = await _validate_and_ingest(
        request,
        dest,
        user="alice@example.com",
        file_id="audio1",
        filename=f"interview{suffix}",
        written=len(content),
        max_total=1000,
        qdrant=object(),
        text_embedder=object(),
        image_embedder=None,
        storage=storage,
        reservation=reservation,
        text_col="kb_user_text",
        image_col="kb_user_images",
    )

    assert response.status == "pending"
    assert response.kind == "audio"
    assert response.mime == mime
    assert response.collection == ""
    assert response.chunks == 0
    assert dest.is_file()
    assert storage.commits[0]["status"] == "pending"
    assert storage.commits[0]["kind"] == "audio"

