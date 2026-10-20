"""Protected tasks cannot log prompt content or use general saved memories."""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock

import pytest

from audrey.pipeline import memory
from audrey.private_logging import install_private_logging
from audrey.processing_policy import ProcessingPolicy, processing_scope
from audrey.tools.discovery import ToolRegistry, ToolSpec

_PRIVATE_TEXT = "PRIVATE_CONTEXT_CONTENT_8519"
_NORMAL = "ordinary request remains visible"
_LOGGER = logging.getLogger("audrey.private_test.child")


def _policy():
    return ProcessingPolicy(
        local_models={"local:latest": "a" * 64},
        ollama_base_url="http://ollama:11434",
    )


@pytest.fixture(autouse=True)
def _restore_factory():
    original = logging.getLogRecordFactory()
    logging.setLogRecordFactory(logging.LogRecord)
    try:
        yield
    finally:
        logging.setLogRecordFactory(original)


def test_protected_message_arguments_exception_and_stack_are_content_free(caplog):
    with processing_scope(_policy()):
        try:
            raise ValueError(_PRIVATE_TEXT)
        except ValueError:
            _LOGGER.exception("prompt=%s " + _PRIVATE_TEXT, _PRIVATE_TEXT, stack_info=True)
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert _PRIVATE_TEXT not in caplog.text
    assert _PRIVATE_TEXT not in repr(record.__dict__)
    assert record.args == ()
    assert record.exc_info is record.exc_text is record.stack_info is None
    assert record.levelno == logging.ERROR


def test_protected_values_are_not_formatted(caplog):
    class Unformattable:
        def __str__(self):
            pytest.fail("Protected arguments must not be formatted")

    with processing_scope(_policy()):
        _LOGGER.warning("%s", Unformattable())
    assert "Protected processing" in caplog.text


def test_factory_chains_prior_integration_with_sanitized_inputs(caplog):
    seen = []

    def prior(*args, **kwargs):
        seen.append(args)
        record = logging.LogRecord(*args, **kwargs)
        record.integration = "retained"
        return record

    logging.setLogRecordFactory(prior)
    with processing_scope(_policy()):
        _LOGGER.warning("%s", _PRIVATE_TEXT)
    _LOGGER.warning("%s", _NORMAL)
    assert _PRIVATE_TEXT not in repr(seen)
    assert seen[0][5] == ()
    assert seen[1][4:6] == ("%s", (_NORMAL,))
    assert all(record.integration == "retained" for record in caplog.records)
    assert _NORMAL in caplog.text


def test_repeated_scope_entry_is_idempotent_and_rewraps_new_integrations(caplog):
    with processing_scope(_policy()):
        installed = logging.getLogRecordFactory()
        with processing_scope(_policy()):
            assert logging.getLogRecordFactory() is installed
    install_private_logging()
    assert logging.getLogRecordFactory() is installed
    seen = []

    def later(*args, **kwargs):
        seen.append(args)
        return installed(*args, **kwargs)

    logging.setLogRecordFactory(later)
    with processing_scope(_policy()):
        assert logging.getLogRecordFactory() is not later
        _LOGGER.warning("%s", _PRIVATE_TEXT)
    assert len(seen) == 1 and _PRIVATE_TEXT not in repr(seen)
    assert _PRIVATE_TEXT not in caplog.text


def test_direct_descendant_handler_cannot_bypass_factory():
    records = []

    class Collect(logging.Handler):
        def emit(self, record):
            records.append(record)

    logger = logging.getLogger("audrey.private_test.no_propagation")
    handler = Collect()
    previous_propagate, previous_level = logger.propagate, logger.level
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        with processing_scope(_policy()):
            logger.info("%s", _PRIVATE_TEXT)
    finally:
        logger.removeHandler(handler)
        logger.propagate, logger.level = previous_propagate, previous_level
    assert len(records) == 1
    assert _PRIVATE_TEXT not in records[0].getMessage()


async def test_child_thread_and_nested_scope_logging_leave_ordinary_concurrent_logs_unchanged(caplog):
    entered = asyncio.Event()

    async def ordinary():
        await entered.wait()
        _LOGGER.warning("%s", _NORMAL)
        await asyncio.to_thread(_LOGGER.warning, "%s thread", _NORMAL)

    async def private_child():
        _LOGGER.warning("%s child", _PRIVATE_TEXT)
        await asyncio.to_thread(_LOGGER.warning, "%s thread", _PRIVATE_TEXT)

    ordinary_task = asyncio.create_task(ordinary())
    with processing_scope(_policy()):
        entered.set()
        await ordinary_task
        with processing_scope(_policy()):
            await asyncio.create_task(private_child())
        _LOGGER.warning("%s outer", _PRIVATE_TEXT)
    _LOGGER.warning("%s after", _NORMAL)
    assert _PRIVATE_TEXT not in caplog.text
    ordinary = [record.getMessage() for record in caplog.records if _NORMAL in record.getMessage()]
    assert ordinary == [_NORMAL, _NORMAL + " thread", _NORMAL + " after"]
    assert len(caplog.records) == 6


async def test_scope_restores_logging_after_failure(caplog):
    with pytest.raises(RuntimeError), processing_scope(_policy()):
        _LOGGER.warning("%s", _PRIVATE_TEXT)
        raise RuntimeError("unavailable")
    _LOGGER.warning("%s", _NORMAL)
    assert _PRIVATE_TEXT not in caplog.text and _NORMAL in caplog.text


async def test_private_memory_recall_does_not_read_or_dispatch(monkeypatch):
    dispatch = AsyncMock(side_effect=AssertionError("Memory dispatch must not run"))
    monkeypatch.setattr(memory, "dispatch_one", dispatch)
    registry = ToolRegistry()
    registry.by_name["memory_search"] = ToolSpec(
        name="memory_search", description="Search saved memories", parameters={},
        server_url="http://custom-tools:8001", path="/memory_search",
    )
    with processing_scope(_policy()):
        result = await memory.recall_for_request(
            registry, http=object(), user_id="owner",
            messages=[{"role": "user", "content": _PRIVATE_TEXT}],
        )
    assert result == []
    dispatch.assert_not_called()


def test_private_memory_never_injects_hits_or_store_guidance():
    hits = [{"key": "protected fact", "value": _PRIVATE_TEXT}]
    with processing_scope(_policy()):
        assert memory.memory_system_message(
            hits, user_id="owner", include_store_hint=True,
        ) is None
    assert _PRIVATE_TEXT in memory.memory_system_message(hits)["content"]


def test_ordinary_exception_and_stack_logging_remain_available(caplog):
    with processing_scope(_policy()):
        pass
    try:
        raise ValueError(_NORMAL)
    except ValueError:
        _LOGGER.exception("%s", _NORMAL, stack_info=True)
    record = caplog.records[0]
    assert record.getMessage() == _NORMAL
    assert record.exc_info is not None and record.stack_info is not None
    assert "ValueError" in caplog.text


async def test_private_scope_disables_graph_tracing_without_changing_ordinary_context():
    from langsmith.run_helpers import tracing_context
    from langsmith.utils import tracing_is_enabled

    with tracing_context(enabled=True):
        assert tracing_is_enabled()
        with processing_scope(_policy()):
            assert not tracing_is_enabled()
            assert not await asyncio.to_thread(tracing_is_enabled)
        assert tracing_is_enabled()
