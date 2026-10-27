"""Owned media playback streams unchanged originals with safe byte-range seeking."""
from __future__ import annotations

import io
import wave
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from test_generated_files import httpx_request
from test_generated_files import output_state as output_state

from audrey.routes import files as uploads
from audrey.routes.app import files as native_files
from audrey.routes.app import router


def _wave() -> bytes:
    output = io.BytesIO()
    with wave.open(output, 'wb') as source:
        source.setnchannels(1)
        source.setsampwidth(2)
        source.setframerate(8000)
        source.writeframes(b'\x00\x00' * 128)
    return output.getvalue()


async def _media(state, *, mime='audio/x-wav', filename='recording.wav', private=False, status='ready'):
    file_id = str(uuid4())
    path = uploads._source_path(httpx_request(state.app), {
        'user': state.owner.storage_namespace, 'file_id': file_id, 'filename': filename,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    content = _wave() if mime == 'audio/x-wav' else b'original stored media bytes ' * 12
    path.write_bytes(content)
    with state.db._lock:
        state.db._record_locked(file_id, state.owner.storage_namespace, filename,
            mime, len(content), 'video' if mime.startswith('video/') else 'audio', 'existing-index', 1,
            '2026-10-27T12:00:00+00:00', status, private_required=private)
    return file_id, path, content


@pytest.mark.parametrize('private', [False, True])
@pytest.mark.parametrize('mime,filename,browser_mime', [
    ('video/mp4', 'clip.mp4', 'video/mp4'),
    ('audio/mpeg', 'recording.mp3', 'audio/mpeg'),
    ('audio/x-wav', 'recording.wav', 'audio/wav'),
    ('audio/x-m4a', 'recording.m4a', 'audio/mp4'),
    ('audio/flac', 'recording.flac', 'audio/flac'),
    ('audio/ogg', 'recording.opus', 'audio/ogg'),
    ('audio/x-hx-aac-adts', 'recording.aac', 'audio/aac'),
])
async def test_media_streams_original_inline_and_real_ranges_without_mutating_catalog(
    output_state, private, mime, filename, browser_mime,
):
    state = output_state
    file_id, path, content = await _media(state, mime=mime, filename=filename, private=private)
    before = await state.db.get_upload(file_id)
    route = f'/api/files/{file_id}/media'
    full = await state.client.get(route)
    assert full.status_code == 200, full.text
    assert full.content == content
    assert full.headers['content-type'] == browser_mime
    assert full.headers['content-disposition'].startswith('inline;')
    assert full.headers['accept-ranges'] == 'bytes'
    partial = await state.client.get(route, headers={'Range': 'bytes=7-19'})
    assert partial.status_code == 206, partial.text
    assert partial.content == content[7:20]
    assert partial.headers['content-range'] == f'bytes 7-19/{len(content)}'
    suffix = await state.client.get(route, headers={'Range': 'bytes=-9'})
    assert suffix.status_code == 206 and suffix.content == content[-9:]
    head = await state.client.head(route)
    assert head.status_code == 200 and head.content == b''
    assert head.headers['content-length'] == str(len(content))
    for response in (full, partial, suffix, head):
        assert response.headers['cache-control'] == 'private, no-store'
        assert response.headers['x-content-type-options'] == 'nosniff'
    assert await state.db.get_upload(file_id) == before
    assert path.read_bytes() == content
    downloaded = await state.client.get(f'/api/files/{file_id}/download')
    assert downloaded.content == content
    assert downloaded.headers['content-disposition'].startswith('attachment;')
    state.current[0] = state.other
    assert (await state.client.get(route)).status_code == 404
    assert (await state.client.get(route, headers={'Range': 'bytes=0-2'})).status_code == 404
    assert (await state.client.head(route)).status_code == 404
    state.current[0] = state.owner
    assert (await state.client.delete(f'/api/files/{file_id}')).status_code == 200
    assert (await state.client.get(route)).status_code == 404


@pytest.mark.parametrize('cause', ['missing', 'reclaimed', 'pending', 'failed', 'symlink', 'directory'])
async def test_media_unavailable_originals_fail_without_host_paths(output_state, cause, tmp_path):
    state = output_state
    file_id, path, _ = await _media(state, status=cause if cause in ('pending', 'failed') else 'ready')
    if cause == 'missing':
        path.unlink()
    elif cause == 'reclaimed':
        await state.db.mark_source_freed(file_id, freed_at='2026-10-27T12:01:00+00:00')
    elif cause == 'symlink':
        external = tmp_path / 'other-owner-private.wav'
        external.write_bytes(b'do not expose')
        path.unlink()
        path.symlink_to(external)
    elif cause == 'directory':
        path.unlink()
        path.mkdir()
    response = await state.client.get(f'/api/files/{file_id}/media')
    assert response.status_code == 410
    assert response.json() == {'detail': 'Stored media is unavailable.'}
    assert str(path) not in response.text and 'do not expose' not in response.text
    assert response.headers['cache-control'] == 'private, no-store'
    assert response.headers['x-content-type-options'] == 'nosniff'


@pytest.mark.parametrize('mime', ['text/html', 'application/json', 'image/svg+xml', 'video/x-unrecognized'])
async def test_inline_media_never_trusts_a_media_filename_over_unsupported_mime(output_state, mime):
    file_id, _, _ = await _media(output_state, mime=mime, filename='pretend.mp4')
    response = await output_state.client.get(f'/api/files/{file_id}/media')
    assert response.status_code == 422
    assert response.json() == {'detail': 'Playback is available for audio and video only.'}
    assert 'original stored' not in response.text


async def test_range_errors_are_safe_private_and_do_not_fall_back_to_full_media(output_state):
    file_id, _, _ = await _media(output_state)
    route = f'/api/files/{file_id}/media'
    for value, status in [('bytes=9999-10000', 416), ('bytes=not-a-range', 400)]:
        response = await output_state.client.get(route, headers={'Range': value})
        assert response.status_code == status, response.text
        assert response.headers['cache-control'] == 'private, no-store'
        assert response.headers['x-content-type-options'] == 'nosniff'
        assert not response.content.startswith(b'RIFF')


async def test_media_rechecks_canonical_deletion_before_exposing_file_response(output_state, monkeypatch):
    state = output_state
    file_id, _, _ = await _media(state)
    owned = native_files._owned_row
    count = 0

    async def delete_before_recheck(request, principal, selected):
        nonlocal count
        count += 1
        if count == 2:
            await state.db.request_file_deletion(file_id, user=state.owner.storage_namespace,
                requested_at='2026-10-27T12:01:00+00:00')
        return await owned(request, principal, selected)

    monkeypatch.setattr(native_files, '_owned_row', delete_before_recheck)
    response = await state.client.get(f'/api/files/{file_id}/media')
    assert response.status_code == 404
    assert not response.content.startswith(b'RIFF')


def test_media_get_and_head_require_authentication():
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    assert client.get('/api/files/unknown/media').status_code == 401
    assert client.head('/api/files/unknown/media').status_code == 401
