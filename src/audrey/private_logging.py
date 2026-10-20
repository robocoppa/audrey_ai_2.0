"""Content-free logging for task-local protected processing.

Install once for the process, then consult the processing ContextVar for every
record. Keeping the factory installed avoids a global on/off switch racing
ordinary and protected tasks. Child tasks and asyncio.to_thread inherit the
scope; unrelated requests keep their original logging behavior.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

_REDACTED_MESSAGE = "Protected processing diagnostic omitted."
_INSTALL_LOCK = threading.Lock()


class _PrivateLogRecordFactory:
    def __init__(self, previous: Callable[..., logging.LogRecord]) -> None:
        self.previous = previous

    def __call__(
        self, name: str, level: int, fn: str, lno: int, msg: Any,
        args: Any, exc_info: Any, func: str | None = None,
        sinfo: str | None = None, **kwargs: Any,
    ) -> logging.LogRecord:
        # Import here because processing_policy installs this factory while
        # opening a scope. Never copy the current policy into process state.
        from audrey.processing_policy import current_processing_policy

        protected = current_processing_policy() is not None
        if protected:
            # Sanitize before chaining: an earlier factory must not receive
            # request content, nor format its exception or stack information.
            msg, args, exc_info, sinfo = _REDACTED_MESSAGE, (), None, None
        record = self.previous(name, level, fn, lno, msg, args, exc_info, func, sinfo, **kwargs)
        if protected:
            # A previous factory may precompute fields used by formatters.
            record.msg = _REDACTED_MESSAGE
            record.message = _REDACTED_MESSAGE
            record.args = ()
            record.exc_info = None
            record.exc_text = None
            record.stack_info = None
        return record


def install_private_logging() -> None:
    """Idempotently wrap the current factory, preserving prior integrations.

    A factory installed by another integration later is wrapped on the next
    protected scope entry. No handler can bypass this record-creation boundary
    merely by attaching directly to a descendant logger.
    """
    with _INSTALL_LOCK:
        current = logging.getLogRecordFactory()
        if not isinstance(current, _PrivateLogRecordFactory):
            logging.setLogRecordFactory(_PrivateLogRecordFactory(current))
