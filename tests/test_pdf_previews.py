"""Owned uploaded PDFs render local pages without model or ingestion work."""
from __future__ import annotations

import asyncio
import io
import shutil
import threading
from uuid import uuid4

import pytest
from PIL import Image
from pypdf import PdfWriter
from test_generated_files import httpx_request
from test_generated_files import output_state as output_state

from audrey import pdf_previews
from audrey.media import visual_render
from audrey.routes import files as uploads

pytestmark = pytest.mark.skipif(
    not all(shutil.which(tool) for tool in ("pdfinfo", "pdftoppm")),
    reason="PDF render binaries unavailable",
)


async def _pdf(state, *, private=False, encrypted=False, mime="application/pdf"):
    file_id = str(uuid4())
    path = uploads._source_path(httpx_request(state.app), {
        'user': state.owner.storage_namespace, 'file_id': file_id, 'filename': 'two-pages.pdf',
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    book = PdfWriter()
    book.add_blank_page(width=80, height=100)
    book.add_blank_page(width=200, height=100)
    if encrypted:
        book.encrypt('protected')
    book.write(path)
    # Fixture the canonical result of ordinary/private ingestion without a model
    # call. The runtime API must preserve this stored privacy/catalog state.
    with state.db._lock:
        state.db._record_locked(file_id, state.owner.storage_namespace, 'two-pages.pdf',
            mime, path.stat().st_size, 'text', 'existing-index', 1,
            '2026-10-26T12:00:00+00:00', 'ready', private_required=private)
    return file_id, path


@pytest.mark.parametrize('private', [False, True])
async def test_pdf_preview_real_pages_ownership_deletion_and_no_mutations(output_state, private):
    state = output_state
    file_id, path = await _pdf(state, private=private)
    before = await state.db.get_upload(file_id)
    original = path.read_bytes()
    route = f'/api/files/{file_id}/preview'
    response = await state.client.get(route)
    assert response.status_code == 200, response.text
    assert response.json() == {'id': file_id, 'filename': 'two-pages.pdf',
        'format': 'pdf', 'kind': 'pdf', 'page_count': 2, 'truncated': False}
    for page, dimensions in [(1, (1280, 1600)), (2, (1600, 800))]:
        image = await state.client.get(f'{route}/pages/{page}')
        assert image.status_code == 200, image.text
        assert image.headers['content-type'] == 'image/png'
        with Image.open(io.BytesIO(image.content)) as picture:
            assert picture.format == 'PNG' and picture.size == dimensions
        for reply in (response, image):
            assert reply.headers['cache-control'] == 'private, no-store'
            assert reply.headers['x-content-type-options'] == 'nosniff'
            assert 'content-disposition' not in reply.headers
    assert await state.db.get_upload(file_id) == before
    assert path.read_bytes() == original
    assert (await state.client.get(f'/api/files/{file_id}/download')).content == original
    state.current[0] = state.other
    for suffix in ['', '/pages/1']:
        assert (await state.client.get(route + suffix)).status_code == 404
    state.current[0] = state.owner
    assert (await state.client.delete(f'/api/files/{file_id}')).status_code == 200
    for suffix in ['', '/pages/1']:
        assert (await state.client.get(route + suffix)).status_code == 404


@pytest.mark.parametrize('suffix', ['', '/pages/1'])
@pytest.mark.parametrize('kind', ['malformed', 'encrypted', 'too_large', 'too_many_pages'])
async def test_pdf_preview_parser_and_resource_failures_are_safe(output_state, monkeypatch, suffix, kind):
    state = output_state
    file_id, path = await _pdf(state, encrypted=kind == 'encrypted')
    if kind == 'malformed':
        path.write_bytes(b'%PDF-secret-private-parser-content-not-a-valid-PDF')
    elif kind == 'too_large':
        monkeypatch.setattr(pdf_previews, 'MAX_SOURCE_BYTES', 4)
    elif kind == 'too_many_pages':
        monkeypatch.setattr(pdf_previews, 'MAX_PAGES', 1)
    response = await state.client.get(f'/api/files/{file_id}/preview{suffix}')
    assert response.status_code == 422, response.text
    assert response.json() == {'detail': 'Preview is unavailable. You can still download the file.'}
    assert 'secret' not in response.text and str(path) not in response.text
    assert response.headers['cache-control'] == 'private, no-store'
    assert response.headers['x-content-type-options'] == 'nosniff'


@pytest.mark.parametrize('page', [0, 3, 2001])
async def test_pdf_preview_rejects_out_of_range_pages(output_state, page):
    file_id, _ = await _pdf(output_state)
    response = await output_state.client.get(f'/api/files/{file_id}/preview/pages/{page}')
    assert response.status_code == 422


@pytest.mark.parametrize('suffix', ['', '/pages/1'])
async def test_pdf_preview_missing_original_and_non_pdf_are_rejected(output_state, suffix):
    file_id, path = await _pdf(output_state)
    path.unlink()
    assert (await output_state.client.get(f'/api/files/{file_id}/preview{suffix}')).status_code == 410
    other_id, _ = await _pdf(output_state, mime='application/msword')
    assert (await output_state.client.get(f'/api/files/{other_id}/preview{suffix}')).status_code == 422


@pytest.mark.parametrize('suffix', ['', '/pages/1'])
async def test_pdf_preview_missing_renderer_is_safe_503(output_state, monkeypatch, suffix):
    file_id, _ = await _pdf(output_state)
    monkeypatch.setattr(visual_render.shutil, 'which', lambda _: None)
    response = await output_state.client.get(f'/api/files/{file_id}/preview{suffix}')
    assert response.status_code == 503
    assert response.json() == {'detail': 'PDF preview is temporarily unavailable.'}


@pytest.mark.parametrize('suffix', ['', '/pages/1'])
async def test_pdf_preview_rechecks_tombstones_before_returning_bytes(output_state, monkeypatch, suffix):
    state = output_state
    file_id, _ = await _pdf(state)
    render = pdf_previews.pdf_preview

    async def tombstone_after_render(*args, **kwargs):
        result = await render(*args, **kwargs)
        await state.db.request_file_deletion(file_id, user=state.owner.storage_namespace,
            requested_at='2026-10-26T12:01:00+00:00')
        return result

    monkeypatch.setattr(pdf_previews, 'pdf_preview', tombstone_after_render)
    response = await state.client.get(f'/api/files/{file_id}/preview{suffix}')
    assert response.status_code == 404
    assert 'PNG' not in response.text and 'page_count' not in response.text


async def test_cancelled_pdf_preview_holds_file_lock_until_disk_worker_settles(output_state, monkeypatch):
    state = output_state
    file_id, path = await _pdf(state)
    started, release = threading.Event(), threading.Event()
    identity = pdf_previews._source_identity

    def slow_identity(source):
        started.set()
        assert release.wait(5)
        return identity(source)

    monkeypatch.setattr(pdf_previews, '_source_identity', slow_identity)
    reading = asyncio.create_task(state.client.get(f'/api/files/{file_id}/preview/pages/1'))
    assert await asyncio.to_thread(started.wait, 3)
    reading.cancel()
    deleting = asyncio.create_task(state.client.delete(f'/api/files/{file_id}'))
    try:
        await asyncio.sleep(0.05)
        assert not reading.done() and not deleting.done() and path.is_file()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await reading
    assert (await deleting).status_code == 200
    assert not path.exists()


async def test_pdf_preview_detects_source_changes_during_render(output_state, monkeypatch):
    file_id, path = await _pdf(output_state)
    run = visual_render._run

    async def changing_source(*args, **kwargs):
        await run(*args, **kwargs)
        if args[0] == 'pdftoppm':
            path.write_bytes(path.read_bytes() + b'\nchanged')

    monkeypatch.setattr(visual_render, '_run', changing_source)
    response = await output_state.client.get(f'/api/files/{file_id}/preview/pages/1')
    assert response.status_code == 422


def test_pdf_preview_rejects_oversized_raster(tmp_path):
    image = tmp_path / 'oversized.png'
    Image.new('RGB', (pdf_previews.MAX_PAGE_EDGE + 1, 1)).save(image)
    with pytest.raises(pdf_previews.PDFPreviewError, match='pixel limit'):
        pdf_previews._page_image(image)
