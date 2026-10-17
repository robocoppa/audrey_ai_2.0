"""Contracts for the bounded, private history embedding diagnostic."""
from __future__ import annotations

import importlib.util
import json
import math
import sqlite3
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('history_embedding_probe', ROOT / 'scripts/probes/history_embedding_probe.py')
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)
chat_archive = importlib.import_module('chat_archive')
TARGET = 'test-calculation'


def source_db(path, extras=()):
    with sqlite3.connect(path) as db:
        db.executescript(chat_archive._SCHEMA)
        rows = [(
            'original', TARGET, 'owner', 'm1,m2',
            f'User: {probe.TARGET_QUESTION}\nAssistant: 1350', '2026-10-15T12:00:00+00:00',
        ), *extras]
        db.executemany('INSERT INTO archive_chunks(chunk_id,conversation_id,user,message_ids_json,text,created_at) '
                       'VALUES(?,?,?,?,?,?)', rows)


def row(ident, text, *, owner='owner', conversation=None):
    return (ident, conversation or ident, owner, 'm-' + ident, text, '2026-10-15T12:00:00+00:00')


async def test_snapshot_is_read_only_scoped_and_labels_original_turn_not_later_lookup(tmp_path):
    path = tmp_path / 'private.sqlite'
    source_db(path, [
        row('lookup', 'User: What did I ask about 1250?\nAssistant: No match.', conversation=TARGET),
        row('other', 'User: OTHER_OWNER_PRIVATE', owner='different'),
        row('deleted', 'User: DELETED_PRIVATE'),
    ])
    with sqlite3.connect(path) as db:
        db.execute('INSERT INTO archive_conversation_deletions VALUES(?,?,?,?,?)',
                   ('owner', 'deleted', '2026-10-16T00:00:00+00:00', 'now', None))
    before = path.read_bytes()
    frozen = await probe.snapshot(path, target_conversation=TARGET)
    assert path.read_bytes() == before
    assert len(frozen.texts) == 2 and len(frozen.targets) == 1
    assert 'User: Calculate 1250' in frozen.texts[next(iter(frozen.targets))]
    assert all('PRIVATE' not in text for text in frozen.texts)
    assert len(frozen.digest) == 64


@pytest.mark.parametrize('condition', ['ambiguous_owner', 'too_large', 'purged_target'])
async def test_source_preconditions_fail_closed_without_embeds(tmp_path, condition):
    path = tmp_path / 'private.sqlite'
    extras = []
    if condition == 'ambiguous_owner':
        extras = [row('foreign', 'PRIVATE_FOREIGN', owner='secret-owner', conversation=TARGET)]
    elif condition == 'too_large':
        extras = [row(f'chunk-{index}', 'PRIVATE_TEXT') for index in range(probe.MAX_CHUNKS)]
    source_db(path, extras)
    if condition == 'purged_target':
        with sqlite3.connect(path) as db:
            db.execute('INSERT INTO archive_user_purges(purge_id,user,cutoff_at,requested_at) VALUES(?,?,?,?)',
                       ('purge', 'owner', '2026-10-16T00:00:00+00:00', 'now'))
    with pytest.raises(probe.SetupError) as caught:
        await probe.snapshot(path, target_conversation=TARGET)
    assert 'PRIVATE' not in str(caught.value) and 'secret-owner' not in str(caught.value)


@pytest.mark.parametrize('value', ['', '{}', '[]', '["one", "one"]', '[null]', '[""]'])
def test_invalid_query_lists_are_setup_errors(value):
    with pytest.raises(probe.SetupError):
        probe.parse_queries(value)


def model_transport(seen, *, wrong_candidate=False):
    def handle(request):
        if request.url.path == '/api/tags':
            return httpx.Response(200, json={'models': [
                {'name': 'nomic-embed-text:latest', 'digest': 'b' * 64},
                {'name': 'qwen3-embedding:4b', 'digest': ('c' if wrong_candidate else 'a') * 64},
            ]})
        body = json.loads(request.content)
        seen.append(body)
        assert body['truncate'] is False
        candidate = body['model'] == 'qwen3-embedding:4b'
        width = 3 if candidate else 768
        vectors = []
        for text in body['input']:
            if 'PRIVATE_QUERY' in text or text == 'calculate' or text.endswith('Query: calculate') or text == 'search_query: calculate':
                vector = [1., 0.]
            else:
                original = 'ORIGINAL_PRIVATE' in text
                score = (.5 if original else .7) if candidate else (.8 if original else .3)
                vector = [score, math.sqrt(1 - score * score)]
            vectors.append(vector + [0.] * (width - 2))
        return httpx.Response(200, json={'embeddings': vectors})
    return httpx.MockTransport(handle)


async def test_matched_profiles_rank_original_and_keep_source_and_queries_out_of_report():
    source = probe.Snapshot(('ORIGINAL_PRIVATE', 'DISTRACTOR_PRIVATE'), frozenset({0}), 'd' * 64)
    profile = probe.EmbeddingProfile('qwen3-embedding:4b', 'a' * 64, 3, '', 'Instruct: retrieval\nQuery: ')
    seen = []
    async with httpx.AsyncClient(base_url='http://unused', transport=model_transport(seen)) as client:
        report = await probe.compare(source, ('PRIVATE_QUERY', 'calculate'), profile,
                                     client=client, cutoff=.62, keep_alive='24h')
    assert report['measurement_status'] == 'completed'
    assert report['embedding_calls'] == 6
    assert [arm['arm'] for arm in report['arms']] == ['nomic_raw', 'nomic_prefixed', 'configured']
    assert report['arms'][0]['queries'][0]['original_rank'] == 1
    assert report['arms'][0]['queries'][0]['deployed_cutoff'] is None
    qwen = report['arms'][-1]['queries'][0]
    assert qwen['original_rank'] == 2 and qwen['original_in_top_5']
    assert qwen['original_score'] == pytest.approx(.5) and not qwen['original_passes_deployed_cutoff']
    assert seen[0]['input'] == list(source.texts)
    assert seen[2]['input'] == ['search_document: ' + text for text in source.texts]
    assert seen[-1]['input'][0] == profile.query('PRIVATE_QUERY')
    assert seen[-1]['keep_alive'] == '24h'
    assert 'PRIVATE' not in json.dumps(report)


async def test_all_model_identities_are_checked_before_any_embedding():
    source = probe.Snapshot(('PRIVATE',), frozenset({0}), 'd' * 64)
    profile = probe.EmbeddingProfile('qwen3-embedding:4b', 'a' * 64, 3, '', 'Query: ')
    seen = []
    async with httpx.AsyncClient(base_url='http://unused', transport=model_transport(seen, wrong_candidate=True)) as client:
        with pytest.raises(Exception, match='pinned digest'):
            await probe.compare(source, ('PRIVATE_QUERY',), profile, client=client, cutoff=.62, keep_alive='24h')
    assert seen == []


def test_failure_output_never_prints_provider_exception_text(monkeypatch, capsys):
    async def fail():
        raise ValueError('PRIVATE_PROVIDER_RESPONSE_AND_TOKEN')
    monkeypatch.setattr(probe, 'run', fail)
    assert probe.main() == 3
    output = capsys.readouterr().out
    assert json.loads(output)['error_kind'] == 'ValueError'
    assert 'PRIVATE' not in output
