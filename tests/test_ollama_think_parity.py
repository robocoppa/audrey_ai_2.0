"""`chat` and `chat_stream` must accept the same forwarding knobs (2026-08-12).

Written after shipping a real outage. `passthrough.think` threaded `think=`
into both the streaming and non-streaming forwards; `OllamaClient.chat` had the
parameter and `chat_stream` did not. Every passthrough turn raised `TypeError`
inside the streaming generator, which — because the response headers were
already sent — could not become a 500. The socket closed mid-body and the
client reported `RemoteProtocolError: peer closed connection without sending
complete message body (incomplete chunked read)`, an error naming neither the
parameter nor the file.

⚠️ **The unit tests passed the whole time, and that is the part worth fixing.**
`tests/test_passthrough_dispatch.py` stubs the client, and the stub was updated
to accept `think` — so the double accepted precisely what the real client
rejected. A hand-written fake silently stops testing the contract the moment
its signature drifts from the thing it doubles. These tests introspect the real
signatures, so no stub can hide the next drift.
"""

from __future__ import annotations

import inspect

from audrey.models.ollama import OllamaClient

#: Knobs that must exist on BOTH chat paths. Anything a caller can thread
#: through one and not the other is a `TypeError` waiting for a code path that
#: only fires in production.
_SHARED_KNOBS = ("model", "messages", "options", "tools", "timeout_s", "think")


def _params(fn) -> dict[str, inspect.Parameter]:
    return dict(inspect.signature(fn).parameters)


class TestBothChatPathsTakeTheSameKnobs:
    def test_chat_accepts_every_shared_knob(self):
        params = _params(OllamaClient.chat)
        for knob in _SHARED_KNOBS:
            assert knob in params, f"chat is missing {knob!r}"

    def test_chat_stream_accepts_every_shared_knob(self):
        """The one that was missing `think` and took passthrough down."""
        params = _params(OllamaClient.chat_stream)
        for knob in _SHARED_KNOBS:
            assert knob in params, f"chat_stream is missing {knob!r}"

    def test_think_defaults_to_none_on_both(self):
        """⚠️ `None` (omit the field) rather than `False`. Ollama hard-errors on
        `think` for a model that does not declare the capability, so a default
        of False would break every non-thinking model in one edit."""
        for fn in (OllamaClient.chat, OllamaClient.chat_stream):
            assert _params(fn)["think"].default is None, fn.__name__

    def test_think_is_keyword_only_on_both(self):
        """Positional would make the two orders diverge silently."""
        for fn in (OllamaClient.chat, OllamaClient.chat_stream):
            kind = _params(fn)["think"].kind
            assert kind is inspect.Parameter.KEYWORD_ONLY, fn.__name__


class TestThePassthroughHelpersMatchTheClient:
    """The wrappers in between. A knob added to the client but not threaded
    through these is unreachable; one threaded through these but absent from
    the client is the outage above."""

    def test_passthrough_chat_forwards_think(self):
        from audrey.pipeline.passthrough import passthrough_chat
        assert "think" in _params(passthrough_chat)

    def test_passthrough_stream_forwards_think(self):
        from audrey.pipeline.passthrough import passthrough_stream
        assert "think" in _params(passthrough_stream)

    def test_the_sse_wrapper_forwards_think(self):
        from audrey.routes.openai.passthrough import _passthrough_stream_sse
        assert "think" in _params(_passthrough_stream_sse)
