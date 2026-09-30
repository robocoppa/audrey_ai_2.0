"""MP3 upload classification enters the durable media queue."""

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
async def test_mp3_becomes_a_pending_audio_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    dest = tmp_path / "interview.mp3"
    dest.write_bytes(b"ID3 real bytes are covered by the smoke fixture")
    storage = _Storage()
    reservation = StorageReservation("r1", "alice@example.com", "single_shot", 43)
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                cfg=SimpleNamespace(raw={"kb": {"upload_root": str(tmp_path)}}),
            ),
        ),
    )
    monkeypatch.setattr(files_routes, "sniff_mime", lambda _path: "audio/mpeg")

    response = await _validate_and_ingest(
        request,
        dest,
        user="alice@example.com",
        file_id="audio1",
        filename="interview.mp3",
        written=43,
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
    assert response.mime == "audio/mpeg"
    assert response.collection == ""
    assert response.chunks == 0
    assert dest.is_file()
    assert storage.commits[0]["status"] == "pending"
    assert storage.commits[0]["kind"] == "audio"

