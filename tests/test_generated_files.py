"""Actual generated artifacts share the native owned library and download path."""
from __future__ import annotations

import asyncio
import stat
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from audrey.app_state import ApplicationStore
from audrey.auth import require_principal
from audrey.generated_files import create_generated_file
from audrey.kb.file_deletion import FileDeletionWorker, FileOperationLocks
from audrey.kb.storage_lifecycle import StorageLifecycle
from audrey.kb.uploads_db import UploadsDB
from audrey.processing_policy import ProcessingPolicy, processing_scope
from audrey.routes import files as uploads
from audrey.routes.app import router
from audrey.user_data_visibility import block_remote_personal_reads, unblock_remote_personal_reads


@pytest.fixture
async def output_state(tmp_path):
    store = ApplicationStore(tmp_path / 'app.sqlite')
    owner = await store.resolve_external_identity(provider='test', subject='output-owner',
        email='owner@example.com', display_name='Owner', role='user', auth_method='test')
    other = await store.resolve_external_identity(provider='test', subject='output-other',
        email='other@example.com', display_name='Other', role='user', auth_method='test')
    db = UploadsDB(tmp_path / 'uploads.sqlite')
    app = FastAPI()
    app.state.application_store = store
    app.state.uploads_db = db
    app.state.storage_lifecycle = StorageLifecycle(db)
    app.state.file_operation_locks = FileOperationLocks()
    app.state.cfg = SimpleNamespace(raw={'kb': {'upload_root': str(tmp_path / 'files'),
                                              'max_user_bytes': 1_000_000}})
    app.state.file_deletions = FileDeletionWorker(db=db, qdrant=AsyncMock(),
        upload_root=tmp_path / 'files', locks=app.state.file_operation_locks)
    app.include_router(router)
    current = [owner]
    app.dependency_overrides[require_principal] = lambda: current[0]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        yield SimpleNamespace(app=app, store=store, db=db, owner=owner, other=other,
                              client=client, current=current)
    db.close()
    store.close()


async def _run(s, *, private=False, project=False):
    p = await s.store.projects.create(user_id=s.owner.user_id, name='Working outputs',
        private_required=private) if project else None
    conversation = await s.store.conversations.create(user_id=s.owner.user_id,
        default_model_id='private' if private else 'auto', private_required=private,
        project_id=p.project_id if p else None)
    policy = ProcessingPolicy(local_models={'local:latest': 'a' * 64}, ollama_base_url='http://ollama:11434', allowed_tools=frozenset({'create_file'}), tool_servers=frozenset({'http://ollama:11434'})) if private else None
    with processing_scope(policy):
        started = await s.store.conversations.begin_run(user_id=s.owner.user_id,
            conversation_id=conversation.conversation_id, user_content='Create a file',
            expected_private_required=private)
    return conversation, started.run


async def _create(s, conversation, run, **arguments):
    return await create_generated_file(s.app, s.owner, conversation.conversation_id,
                                       run.run_id, arguments)


async def test_generated_file_roundtrip_download_library_project_retry_and_revision(output_state):
    s = output_state
    conversation, run = await _run(s, project=True)
    args = {'filename': 'notes.md', 'format': 'md', 'content': '# Meeting notes\nA useful output.'}
    first, repeat = await asyncio.gather(_create(s, conversation, run, **args),
                                         _create(s, conversation, run, **args))
    assert first['file_id'] == repeat['file_id']
    assert {first['created'], repeat['created']} == {True, False}
    assert first['project_id'] == conversation.project_id
    listing = await s.client.get('/api/files')
    assert listing.status_code == 200, listing.text
    record = listing.json()['items'][0]
    assert record['origin'] == 'generated' and record['chunks'] == 0
    downloaded = await s.client.get(first['download_url'])
    assert downloaded.status_code == 200
    assert downloaded.text == args['content']
    assert downloaded.headers['cache-control'] == 'private, no-store'
    assert 'attachment' in downloaded.headers['content-disposition']
    assert downloaded.headers['x-content-type-options'] == 'nosniff'
    row = await s.db.get_upload(first['file_id'])
    assert row['collection'] == '' and row['private_required'] == 0
    path = uploads._source_path(httpx_request(s.app), row)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert (await s.client.get(f"/api/files/{first['file_id']}/text")).json()['text'] == args['content']
    files = await s.store.projects.list_files(user_id=s.owner.user_id, project_id=conversation.project_id)
    assert [item.file_id for item in files] == [first['file_id']]
    changed = await _create(s, conversation, run, **{**args, 'content': 'Changed output.'})
    assert changed['file_id'] != first['file_id']
    assert (await s.client.get(first['download_url'])).text == args['content']
    assert len(await s.db.list_user(s.owner.storage_namespace)) == 2
    s.current[0] = s.other
    assert (await s.client.get(first['download_url'])).status_code == 404
    assert (await s.client.get('/api/files')).json()['items'] == []


def httpx_request(app):
    from fastapi import Request
    return Request({'type': 'http', 'app': app})


async def test_private_generation_requires_policy_and_keeps_project_output_private(output_state):
    s = output_state
    conversation, run = await _run(s, private=True, project=True)
    args = {'filename': 'budget.csv', 'format': 'csv', 'rows': [['Item', 'Amount'], ['Rent', 1200]]}
    denied = await _create(s, conversation, run, **args)
    assert denied['error'] == 'file_creation_unavailable'
    policy = ProcessingPolicy(local_models={'local:latest': 'a' * 64}, ollama_base_url='http://ollama:11434', allowed_tools=frozenset({'create_file'}), tool_servers=frozenset({'http://ollama:11434'}))
    with processing_scope(policy):
        result = await _create(s, conversation, run, **args)
    assert result['project_id'] == conversation.project_id
    row = await s.db.get_upload(result['file_id'])
    assert row['private_required'] == 1 and row['origin'] == 'generated'
    assert (await s.client.get(result['download_url'])).status_code == 200
    assert len(await s.db.list_user(s.owner.storage_namespace)) == 1


async def test_deleted_output_is_not_resurrected_by_same_retry(output_state):
    s = output_state
    conversation, run = await _run(s)
    args = {'filename': 'notes.txt', 'format': 'txt', 'content': 'Working output'}
    result = await _create(s, conversation, run, **args)
    assert (await s.client.delete(f"/api/files/{result['file_id']}")).status_code == 200
    assert (await s.client.get(result['download_url'])).status_code == 404
    retried = await _create(s, conversation, run, **args)
    assert retried['error'] == 'generated_file_deleted'
    assert await s.db.list_user(s.owner.storage_namespace) == []


async def test_quota_and_purge_refusals_leave_no_unregistered_bytes(output_state):
    s = output_state
    conversation, run = await _run(s)
    s.app.state.cfg.raw['kb']['max_user_bytes'] = 10
    args = {'filename': 'notes.txt', 'format': 'txt', 'content': 'Long enough to exceed quota'}
    assert (await _create(s, conversation, run, **args))['error'] == 'file_quota_exceeded'
    s.app.state.cfg.raw['kb']['max_user_bytes'] = 1000
    block_remote_personal_reads(user=s.owner.storage_namespace, purge_id='test')
    try:
        assert (await _create(s, conversation, run, **args))['error'] == 'file_creation_unavailable'
    finally:
        unblock_remote_personal_reads(user=s.owner.storage_namespace, purge_id='test')
    assert await s.db.list_user(s.owner.storage_namespace) == []
    assert (await s.db.quota_usage(s.owner.storage_namespace)).total_bytes == 0


async def test_deleted_or_foreign_conversation_cannot_write(output_state):
    s = output_state
    conversation, run = await _run(s)
    args = {'filename': 'notes.txt', 'format': 'txt', 'content': 'Working output'}
    result = await create_generated_file(s.app, s.other, conversation.conversation_id, run.run_id, args)
    assert result['error'] == 'file_creation_unavailable'
    await s.store.conversations.finish_run(user_id=s.owner.user_id, run_id=run.run_id,
        outcome='cancelled', assistant_content='')
    await s.store.conversations.delete(user_id=s.owner.user_id, conversation_id=conversation.conversation_id)
    assert (await _create(s, conversation, run, **args))['error'] == 'file_creation_unavailable'
    assert await s.db.list_user(s.owner.storage_namespace) == []


@pytest.mark.parametrize('cancel', [False, True])
async def test_failed_or_cancelled_write_cannot_unlink_concurrent_identical_retry(output_state, monkeypatch, cancel):
    import threading

    import audrey.generated_files as generation

    s = output_state
    conversation, run = await _run(s)
    args = {'filename': 'notes.txt', 'format': 'txt', 'content': 'Concurrent working output'}
    entered = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    real_write = generation._write_new
    writes = 0

    def staged_write(path, data):
        nonlocal writes
        writes += 1
        if writes == 1:
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(3)
        if writes == 2 and not cancel:
            raise OSError('simulated sidecar failure')
        real_write(path, data)

    monkeypatch.setattr(generation, '_write_new', staged_write)
    first = asyncio.create_task(_create(s, conversation, run, **args))
    await asyncio.wait_for(entered.wait(), 3)
    retry = asyncio.create_task(_create(s, conversation, run, **args))
    await asyncio.sleep(0)
    if cancel:
        first.cancel()
    release.set()
    if cancel:
        with pytest.raises(asyncio.CancelledError):
            await first
    else:
        assert (await first)['error'] == 'file_creation_unavailable'
    result = await retry
    assert result['created'] is True
    assert (await s.client.get(result['download_url'])).text == args['content']
    assert (await s.client.get(f"/api/files/{result['file_id']}/text")).json()['text'] == args['content']
    assert len(await s.db.list_user(s.owner.storage_namespace)) == 1
    assert (await s.db.quota_usage(s.owner.storage_namespace)).single_shot_bytes == 0


@pytest.mark.parametrize('format', ['docx', 'xlsx'])
async def test_office_renderer_revision_keeps_old_bytes_and_reuses_new_outputs(output_state, monkeypatch, format):
    from audrey import generated_files
    s = output_state
    conversation, run = await _run(s)
    args = {'filename': f'versioned.{format}', 'format': format}
    args.update({'content': 'A retained output'} if format == 'docx' else {'sheets': [{'name': 'Values', 'rows': [['Amount'], [12.5]]}]})
    monkeypatch.setattr(generated_files, '_OFFICE_RENDERER_REVISION', 1)
    old = await _create(s, conversation, run, **args)
    original = (await s.client.get(old['download_url'])).content
    monkeypatch.setattr(generated_files, '_OFFICE_RENDERER_REVISION', 2)
    current = await _create(s, conversation, run, **args)
    assert current['created'] and current['file_id'] != old['file_id']
    repeated = await _create(s, conversation, run, **args)
    assert not repeated['created'] and repeated['file_id'] == current['file_id']
    assert (await s.client.get(old['download_url'])).content == original
    # Removing an old copy does not invalidate an independently retained new one.
    assert (await s.client.delete(f"/api/files/{old['file_id']}")).status_code == 200
    assert (await _create(s, conversation, run, **args))['file_id'] == current['file_id']


async def test_office_revision_does_not_resurrect_previously_deleted_output(output_state, monkeypatch):
    from audrey import generated_files
    s = output_state
    conversation, run = await _run(s)
    args = {'filename': 'deleted.docx', 'format': 'docx', 'content': 'A retained output'}
    monkeypatch.setattr(generated_files, '_OFFICE_RENDERER_REVISION', 1)
    old = await _create(s, conversation, run, **args)
    assert (await s.client.delete(f"/api/files/{old['file_id']}")).status_code == 200
    monkeypatch.setattr(generated_files, '_OFFICE_RENDERER_REVISION', 2)
    assert (await _create(s, conversation, run, **args))['error'] == 'generated_file_deleted'
    replacement = await _create(s, conversation, run, **{**args, 'filename': 'replacement.docx'})
    assert replacement['created']
