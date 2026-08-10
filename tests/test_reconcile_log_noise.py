"""The reconcile sweep's httpx noise — suppressed for the sweep, not the box.

Every 30 minutes the periodic sweep scrolls both global collections, one
qdrant round-trip per 256 points, and httpx logs each one at INFO:

    INFO httpx: HTTP Request: POST http://qdrant:6333/collections/kb_text/points/scroll "HTTP/1.1 200 OK"

That is hundreds of lines per sweep, and it buried three of four log
pastes during the 2026-08-10 debugging session before either of them
reached a request-path line worth reading.

The obvious fix — raise the `httpx` logger to WARNING while the sweep
runs — is wrong, and wrong in a way that only shows up when something
else is being debugged: it also silences the chat requests that happen to
land during the sweep, which are the lines with the actual diagnostic
value. So the suppression is scoped by ContextVar to the sweep's own
task, and these tests exist mostly to hold that distinction in place.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from audrey.kb.reconcile import KBReconciler, _sweeping, reconcile_once

_HTTPX = logging.getLogger("httpx")
_SCROLL_LINE = 'HTTP Request: POST http://qdrant:6333/collections/kb_text/points/scroll "HTTP/1.1 200 OK"'


class _ChattyQdrant:
    """A qdrant facade that logs like the real client does.

    qdrant-client talks REST over httpx, so its traffic surfaces on the
    `httpx` logger — from inside `scroll_collection`, which is the call
    the sweep makes. Logging from there is what makes this fake useful:
    the records land in the sweep's context, exactly as they do in
    production.
    """

    text_collection = "kb_text"
    image_collection = "kb_images"

    async def collection_exists(self, _name: str) -> bool:
        return True

    async def scroll_collection(self, _collection: str, *, page_size: int = 256):
        for _ in range(3):  # three pages, three round-trips, three log lines
            _HTTPX.info(_SCROLL_LINE)
        return []

    async def delete_by_source(self, _source: str, *, collection: str) -> None:
        return None


@pytest.fixture(autouse=True)
def _reset_sweep_var():
    """The ContextVar is module-global; a test that leaves it set would make
    every later test look like it was running inside a sweep."""
    token = _sweeping.set(False)
    yield
    _sweeping.reset(token)


def _httpx_lines(caplog) -> list[str]:
    return [r.message for r in caplog.records if r.name == "httpx"]


# ─── The suppression ──────────────────────────────────────────────────


async def test_a_quiet_sweep_drops_its_own_scroll_traffic(caplog):
    caplog.set_level(logging.INFO)

    await reconcile_once(_ChattyQdrant(), quiet_httpx=True)

    assert _httpx_lines(caplog) == []


async def test_the_sweeps_own_summary_line_survives(caplog):
    """The suppression is attached to the `httpx` logger specifically. If it
    ever became a global level flip, this is the line that would vanish with
    it — and then a sweep would run every 30 minutes with no trace at all."""
    caplog.set_level(logging.INFO)

    await reconcile_once(_ChattyQdrant(), quiet_httpx=True)

    assert any("kb.reconcile: pass complete" in r.message for r in caplog.records)


async def test_httpx_warnings_are_never_suppressed(caplog):
    """A sweep that starts failing its round-trips has to be able to say so.
    Quieting the routine case must not quiet the interesting one."""
    caplog.set_level(logging.INFO)

    class _FailingQdrant(_ChattyQdrant):
        async def scroll_collection(self, _collection: str, *, page_size: int = 256):
            _HTTPX.warning("Retrying request to /points/scroll after connect error")
            return []

    await reconcile_once(_FailingQdrant(), quiet_httpx=True)

    assert any("connect error" in line for line in _httpx_lines(caplog))


# ─── The scope ────────────────────────────────────────────────────────


async def test_a_concurrent_request_keeps_its_httpx_lines(caplog):
    """⚠️ The reason this is a ContextVar and not a flag.

    A chat turn that lands mid-sweep is precisely the line you were reading
    the log for. Suppressing by logger level would take it out too, and the
    failure is invisible — the log looks quiet, not broken.
    """
    caplog.set_level(logging.INFO)
    request_started = asyncio.Event()

    class _SlowQdrant(_ChattyQdrant):
        async def scroll_collection(self, _collection: str, *, page_size: int = 256):
            _HTTPX.info(_SCROLL_LINE)
            request_started.set()
            await asyncio.sleep(0)  # let the request task run mid-sweep
            return []

    async def a_chat_request():
        await request_started.wait()
        _HTTPX.info('HTTP Request: POST http://ollama:11434/api/chat "HTTP/1.1 200 OK"')

    async with asyncio.TaskGroup() as tg:
        tg.create_task(reconcile_once(_SlowQdrant(), quiet_httpx=True))
        tg.create_task(a_chat_request())

    lines = _httpx_lines(caplog)
    assert any("/api/chat" in line for line in lines)
    assert not any("points/scroll" in line for line in lines)


async def test_the_admin_trigger_stays_verbose(caplog):
    """`POST /v1/admin/kb/reconcile` is the "I am debugging a sweep" path: one
    sweep, on demand, where the round-trips are the thing you asked for. The
    default is verbose so the admin endpoint gets it without opting in."""
    caplog.set_level(logging.INFO)

    await reconcile_once(_ChattyQdrant())

    # Three round-trips per collection, and the sweep covers both.
    assert len(_httpx_lines(caplog)) == 6


async def test_the_periodic_loop_is_the_one_that_opts_in(caplog):
    """The split only helps if the noisy caller is the one that asked. Drives
    `KBReconciler` rather than asserting on the keyword, so a call site that
    forgot to pass it fails here."""
    caplog.set_level(logging.INFO)
    reconciler = KBReconciler(qdrant=_ChattyQdrant(), interval_s=3600)

    await reconciler.start()
    try:
        for _ in range(50):  # let the startup sweep finish
            await asyncio.sleep(0)
    finally:
        await reconciler.stop()

    assert any("kb.reconcile: pass complete" in r.message for r in caplog.records)
    assert _httpx_lines(caplog) == []


# ─── Cleanup ──────────────────────────────────────────────────────────


async def test_the_var_is_cleared_when_a_sweep_raises():
    """`reconcile_collection` swallows its own errors, but the loop around it
    can still die (a cancel, an OOM). A sweep that left the var set would
    silence httpx for the rest of the process's life."""
    class _ExplodingQdrant(_ChattyQdrant):
        async def scroll_collection(self, _collection: str, *, page_size: int = 256):
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await reconcile_once(_ExplodingQdrant(), quiet_httpx=True)

    assert _sweeping.get() is False
