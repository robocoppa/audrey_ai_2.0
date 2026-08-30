"""Fields real agent clients send that the message schemas must not reject.

⚠️ THE FAILURE THIS FILE EXISTS TO PREVENT IS INVISIBLE FROM THE CLIENT SIDE.
`_StrictChatMessage` sets `extra="forbid"`, so an undeclared field is a hard
422 raised by Pydantic BEFORE the route body runs — nothing is logged by
Audrey, no model is dispatched, and the agent on the other end reports its own
generic "the model provider failed". It reads as an outage of the model, and
the box looks healthy because it never saw a request.

Diagnosed 2026-08-29 on a Hermes agent whose plain chat turns worked while
EVERY tool-calling turn failed. The asymmetry is the tell: the first call of a
tool turn carries no `tool` message and validates; the follow-up carrying the
tool result does, and 422s. See `docs/campaign-3/` for the incident write-up.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from audrey.routes.openai.schemas import ChatCompletionRequest

# The exact conversation shape a Hermes agent replays on the second call of a
# tool turn: an assistant turn with `reasoning_content` echoed back, and a tool
# result carrying the tool's `name`. Both 422'd before 2026-08-29.
_TOOL_LOOP = [
    {"role": "user", "content": "what is your fallback model"},
    {
        "role": "assistant",
        "content": None,
        "reasoning_content": " ",
        "tool_calls": [{
            "id": "call_abc123",
            "type": "function",
            "function": {"name": "terminal", "arguments": '{"cmd": "grep -n model cfg"}'},
        }],
    },
    {
        "role": "tool",
        "tool_call_id": "call_abc123",
        "name": "terminal",
        "content": "model: audrey_passthrough/glm-5.3:cloud",
    },
]


def test_a_tool_result_may_carry_the_tool_name():
    """`name` on a tool message is OpenAI-legal and every real client sends it.

    The other four roles declared it; `tool` did not, which is the whole bug.
    """
    req = ChatCompletionRequest(model="audrey_auto", messages=_TOOL_LOOP)
    tool_msg = req.messages[-1]
    assert tool_msg.name == "terminal"


def test_an_assistant_turn_may_echo_reasoning_content():
    """Thinking models return it; agent clients replay it on the next call."""
    req = ChatCompletionRequest(model="audrey_auto", messages=_TOOL_LOOP)
    assert req.messages[1].reasoning_content == " "


def test_reasoning_content_never_reaches_the_provider():
    """Accepted for validation, dropped on the way out.

    It is the client's echo of our own earlier output, not input — forwarding
    it would put a second copy of the reasoning in the model's context.
    """
    req = ChatCompletionRequest(model="audrey_auto", messages=_TOOL_LOOP)
    dumped = req.messages[1].model_dump(exclude_none=True)
    assert "reasoning_content" not in dumped
    assert dumped["tool_calls"], "the rest of the assistant turn must survive"


def test_tool_name_does_reach_the_provider_like_every_other_role():
    """Unlike `reasoning_content`, `name` is ordinary message data.

    The four roles that already declared it forward it, so `tool` matches.
    """
    req = ChatCompletionRequest(model="audrey_auto", messages=_TOOL_LOOP)
    dumped = req.messages[-1].model_dump(exclude_none=True, exclude={"metadata"})
    assert dumped["name"] == "terminal"


def test_the_whole_replayed_tool_loop_validates():
    """The end-to-end regression: this exact body returned 422 three times."""
    req = ChatCompletionRequest(model="audrey_auto", messages=_TOOL_LOOP)
    assert len(req.messages) == 3


def test_genuinely_unknown_fields_are_still_rejected():
    """The fix widens the schema by two known fields — it does not open it.

    `extra="forbid"` is deliberate; a typo'd field name must still fail loudly
    rather than be silently dropped on the floor.
    """
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            model="audrey_auto",
            messages=[{"role": "user", "content": "hi", "nonsense_field": 1}],
        )
