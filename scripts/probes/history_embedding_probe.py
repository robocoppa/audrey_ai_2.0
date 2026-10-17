#!/usr/bin/env python3
"""Compare local history embeddings in RAM through the existing on-box runner.

Use CONTAINER=custom-tools FOREGROUND=1 and a JSON QUERIES environment variable.
Only visible chunks belonging to the known calculation's owner are read. No
source text, user identifiers or queries are logged. No stored index is changed.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import aiosqlite
import httpx

sidecar = Path('/app') if Path('/app/chat_archive.py').is_file() else Path(__file__).resolve().parents[2] / 'tools-server'
sys.path.insert(0, str(sidecar))
from chat_archive import ChatArchiveStore  # noqa: E402 -- deployed sidecar modules
from settings import Settings  # noqa: E402
from text_embedding import (  # noqa: E402
    EmbeddingProfile,
    load_settings,
    profile_from_mapping,
    verify_digest,
)

TARGET_QUESTION = 'Calculate 1250 × 1.08 using the calculator'
MAX_CHUNKS = 200
BATCH_SIZE = 8


class SetupError(RuntimeError):
    """A safe precondition message; never contains private source data."""


@dataclass(frozen=True, repr=False)
class Snapshot:
    texts: tuple[str, ...]
    targets: frozenset[int]
    digest: str


def parse_queries(value: str) -> tuple[str, ...]:
    try:
        queries = json.loads(value)
    except ValueError:
        raise SetupError('QUERIES must be a JSON list of English search strings') from None
    if (not isinstance(queries, list) or not 1 <= len(queries) <= 8
            or any(not isinstance(q, str) or not q.strip() or len(q) > 1000 for q in queries)
            or len(set(queries)) != len(queries)):
        raise SetupError('QUERIES must contain 1–8 unique nonempty strings, at most 1000 characters each')
    return tuple(queries)


async def snapshot(path: Path, *, target_conversation: str, retention_days: int = 0) -> Snapshot:
    if not target_conversation or len(target_conversation) > 200:
        raise SetupError('TARGET_CONVERSATION_ID must name the existing calculation conversation')
    uri = (await asyncio.to_thread(path.resolve)).as_uri() + '?mode=ro'
    async with aiosqlite.connect(uri, uri=True) as db:
        await db.execute('PRAGMA query_only=ON')
        await db.execute('BEGIN')
        cursor = await db.execute(
            'SELECT DISTINCT user FROM archive_chunks WHERE conversation_id=?',
            (target_conversation,),
        )
        owners = await cursor.fetchall()
        await cursor.close()
        if len(owners) != 1 or not owners[0][0]:
            raise SetupError('The known calculation must resolve to exactly one owner')
        owner = str(owners[0][0])
        cursor = await db.execute('SELECT COUNT(*) FROM archive_chunks WHERE user=?', (owner,))
        count = (await cursor.fetchone())[0]
        await cursor.close()
        if count > MAX_CHUNKS:
            raise SetupError('Owner corpus exceeds the 200-chunk bound; nothing was embedded')
        store = ChatArchiveStore.__new__(ChatArchiveStore)
        store._db = db
        store._db_lock = asyncio.Lock()
        store._retention_days = retention_days
        rows = await store._visible_search_rows(user=owner, date_from=None, date_to=None)
        # Label the original calculation, not later lookup turns in that chat.
        targets = frozenset(
            index for index, row in enumerate(rows)
            if row[1] == target_conversation
            and row[2].partition('\nAssistant:')[0].removeprefix('User:').strip().casefold()
            == TARGET_QUESTION.casefold()
        )
        if not targets:
            raise SetupError('The original calculation is not present in visible source history')
        digest = hashlib.sha256(json.dumps(
            [(row[0], row[2]) for row in rows], ensure_ascii=False,
        ).encode()).hexdigest()
        return Snapshot(tuple(row[2] for row in rows), targets, digest)


async def embed_batch(client, profile, texts, *, query: bool, keep_alive: str):
    await verify_digest(client, profile)
    response = await client.post('/api/embed', json={
        'model': profile.model,
        'input': [profile.query(text) if query else profile.document(text) for text in texts],
        'truncate': False, 'keep_alive': keep_alive,
    })
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict) or body.get('error') or body.get('model') not in (None, profile.model):
        raise RuntimeError('Invalid embedding response')
    vectors = profile.normalize_vectors(body.get('embeddings'), len(texts))
    await verify_digest(client, profile)
    return vectors


async def compare(source: Snapshot, queries, profile, *, client, cutoff: float, keep_alive: str):
    if (not 1 <= len(source.texts) <= MAX_CHUNKS or not source.targets
            or not source.targets.issubset(range(len(source.texts)))
            or not 1 <= len(queries) <= 8):
        raise SetupError('Source or query bounds are invalid; nothing was embedded')
    response = await client.get('/api/tags')
    response.raise_for_status()
    inventory = response.json().get('models', [])
    nomic = [m for m in inventory if isinstance(m, dict) and m.get('name') == 'nomic-embed-text:latest']
    if (len(nomic) != 1 or not isinstance(nomic[0].get('digest'), str)
            or len(nomic[0]['digest']) != 64):
        raise SetupError('Installed Nomic baseline was not found with a valid digest')
    if profile.model == 'nomic-embed-text:latest' or 'cloud' in profile.model.lower():
        raise SetupError('The configured candidate must be a distinct local embedding model')
    if not math.isfinite(cutoff) or not -1 <= cutoff <= 1:
        raise SetupError('Configured history cutoff is invalid')
    arms = [
        ('nomic_raw', EmbeddingProfile('nomic-embed-text:latest', nomic[0]['digest'], 768, '', '')),
        ('nomic_prefixed', EmbeddingProfile('nomic-embed-text:latest', nomic[0]['digest'], 768,
                                          'search_document: ', 'search_query: ')),
        ('configured', profile),
    ]
    # Validate every arm before any embedding request, then leave Qwen warm last.
    for _, arm in arms:
        await verify_digest(client, arm)
    report = {
        'schema': 1, 'status': 'measured', 'measurement_status': 'completed',
        'scope': 'One known positive exchange; matched diagnostic, not a model promotion gate',
        'corpus': {'chunks': len(source.texts), 'sha256': source.digest, 'target_chunks': len(source.targets)},
        'query_count': len(queries), 'embedding_calls': 0, 'arms': [],
        'planned_embedding_calls': 3 * (math.ceil(len(source.texts) / BATCH_SIZE) + 1),
    }
    for name, arm in arms:
        vectors = []
        for offset in range(0, len(source.texts), BATCH_SIZE):
            vectors.extend(await embed_batch(
                client, arm, source.texts[offset:offset + BATCH_SIZE], query=False,
                keep_alive=keep_alive if name == 'configured' else '5m',
            ))
            report['embedding_calls'] += 1
        query_vectors = await embed_batch(
            client, arm, queries, query=True,
            keep_alive=keep_alive if name == 'configured' else '5m',
        )
        report['embedding_calls'] += 1
        results = []
        for index, query_vector in enumerate(query_vectors, 1):
            scores = [math.fsum(a * b for a, b in zip(query_vector, vector, strict=True)) for vector in vectors]
            ranked = sorted(range(len(scores)), key=lambda item: (-scores[item], item))
            rank = next(position for position, item in enumerate(ranked, 1) if item in source.targets)
            score = max(scores[item] for item in source.targets)
            results.append({
                'query': index, 'original_rank': rank, 'original_in_top_5': rank <= 5,
                'original_score': round(score, 5), 'top_score': round(max(scores), 5),
                'deployed_cutoff': cutoff if name == 'configured' else None,
                'original_passes_deployed_cutoff': score >= cutoff if name == 'configured' else None,
            })
        report['arms'].append({'arm': name, 'model': arm.model, 'digest': arm.digest,
                               'dimensions': arm.dimensions, 'queries': results})
    return report


async def run():
    queries = parse_queries(os.environ.get('QUERIES', ''))
    settings = Settings()
    config = load_settings(settings.embedding_config_path)
    profile = profile_from_mapping(config)
    source = await snapshot(settings.chat_archive_db_path,
                            target_conversation=os.environ.get('TARGET_CONVERSATION_ID', ''),
                            retention_days=settings.chat_archive_retention_days)
    async with httpx.AsyncClient(base_url=settings.ollama_url, timeout=120,
                                follow_redirects=False, trust_env=False) as client:
        return await compare(source, queries, profile, client=client,
                             cutoff=float(config.get('history_min_score', .62)),
                             keep_alive=str(config.get('keep_alive', '24h')))


def main():
    try:
        report = asyncio.run(run())
    except SetupError as error:
        print(json.dumps({'schema': 1, 'status': 'setup_failed', 'error': str(error)}))
        return 2
    except KeyboardInterrupt:
        print(json.dumps({'schema': 1, 'status': 'interrupted'}))
        return 130
    except Exception as error:  # noqa: BLE001 -- CLI must redact all provider/private failures
        # Provider errors may contain input text, paths or authentication data.
        print(json.dumps({'schema': 1, 'status': 'failed', 'error_kind': type(error).__name__}))
        return 3
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
