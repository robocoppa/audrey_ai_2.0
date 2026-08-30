"""OpenAI-compatible request schemas.

`ChatMessage` and `ChatCompletionRequest` are the Pydantic models the route
layer validates incoming `/v1/chat/completions` bodies against. Split out of
the monolithic route module so the other submodules (and tests) can import the
schemas without pulling in the streaming machinery.
"""

from __future__ import annotations

import json
import logging
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

log = logging.getLogger(__name__)

ChatContent = str | list[dict[str, Any]]

#: Every field declared by ANY concrete role, populated once the classes below
#: exist. A key in here that appears on the WRONG role is a misplacement, not
#: unfamiliar vocabulary — it demonstrably carries meaning, because another role
#: models it — so it is rejected rather than dropped. `tool_calls` on a `user`
#: message is the case that motivated the split.
_ALL_ROLE_FIELDS: frozenset[str] = frozenset()

#: Seen (role, field) pairs, so an unknown field is reported the FIRST time a
#: client sends it and not on every subsequent request. Process-local and
#: unbounded only in the number of distinct field names a client invents, which
#: is small. Reset per process, which is what you want — a restart after a
#: client upgrade should say so again.
_REPORTED_UNKNOWN_FIELDS: set[tuple[str, str]] = set()


class _StrictChatMessage(BaseModel):
    """Shared extensions accepted on every role without forwarding them.

    ⚠️ `extra="ignore"`, NOT `"forbid"` — changed 2026-08-30, and the reason is
    worth keeping. Forbidding here made every unknown message field a 422 that
    Pydantic raises BEFORE the route body runs: nothing dispatched, and
    **nothing logged by Audrey at all**. The client sees only its own generic
    "the model provider failed", so a two-field vocabulary gap reads as an
    outage of the model on a box that never saw the request. That is exactly
    how `tool.name` and `assistant.reasoning_content` cost an afternoon.

    Dropping an unknown field was always the safe half: messages reach Ollama
    through `model_dump(exclude_none=True, exclude={"metadata"})`, which is
    allow-list-based, so an undeclared field could never have been forwarded
    anyway. Forbidding bought no safety at the provider boundary — it bought
    VISIBILITY, and then delivered it as an outage.

    So the visibility is kept and the outage is not: `_log_unknown_fields`
    reports anything undeclared at WARNING, once per (role, field) per process.
    A new client extension now shows up as one log line the first time it
    arrives instead of as a phantom failure.

    ⚠️ The trade this accepts: a field that SHOULD have been handled is now
    dropped quietly rather than rejected loudly. The log line is the mitigation.
    If a client's content ever goes missing rather than erroring, grep for
    `unknown message field` before suspecting the model.

    ⚠️ This governs VOCABULARY only. The semantic validators are unchanged and
    still reject: `require_content_or_tool_calls` below, and
    `ChatCompletionRequest.validate_tool_result_links`, which catches tool
    results that reference a call that was never made — a real malformed
    history, not an unrecognised field.
    """

    model_config = ConfigDict(extra="ignore")

    # Older OWUI payloads attach conversation metadata to a message. Preserve
    # it through validation for archive identity; route adapters explicitly
    # exclude it at the model-provider boundary.
    metadata: dict[str, Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def _log_unknown_fields(cls, data: Any) -> Any:
        """Report undeclared fields without rejecting them. Never mutates.

        Runs before validation, so it sees the raw payload — by the time
        `extra="ignore"` has done its work the extras are gone and there is
        nothing left to report.
        """
        if not isinstance(data, dict):
            return data
        unknown = set(data) - set(cls.model_fields)
        if not unknown:
            return data
        role = str(data.get("role", "?"))
        misplaced = sorted(unknown & _ALL_ROLE_FIELDS)
        if misplaced:
            raise ValueError(
                f"{role} message carries field(s) belonging to another role: "
                f"{', '.join(misplaced)}. Audrey models these, so dropping them "
                "would silently discard meaning."
            )
        if unknown:
            for field in sorted(unknown):
                if (role, field) in _REPORTED_UNKNOWN_FIELDS:
                    continue
                _REPORTED_UNKNOWN_FIELDS.add((role, field))
                log.warning(
                    "unknown message field dropped: role=%s field=%s — a client "
                    "sends this and Audrey does not model it. Harmless if it is "
                    "client bookkeeping; declare it if it carries meaning.",
                    role, field,
                )
        return data


class SystemChatMessage(_StrictChatMessage):
    role: Literal["system"]
    content: ChatContent
    name: str | None = None


class DeveloperChatMessage(_StrictChatMessage):
    role: Literal["developer"]
    content: ChatContent
    name: str | None = None


class UserChatMessage(_StrictChatMessage):
    role: Literal["user"]
    # A plain string for ordinary text turns, OR the OpenAI multimodal
    # list-of-parts shape OWUI sends when a user attaches an image.
    content: ChatContent
    name: str | None = None


class AssistantToolFunction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    arguments: str


class AssistantToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    type: Literal["function"]
    function: AssistantToolFunction


class AssistantChatMessage(_StrictChatMessage):
    role: Literal["assistant"]
    content: ChatContent | None = None
    name: str | None = None
    tool_calls: list[AssistantToolCall] | None = None

    # Thinking models (DeepSeek, GLM) return their reasoning in a sibling
    # field, and agent clients REPLAY the assistant turn verbatim on the next
    # request — so it arrives back here on every multi-turn tool loop.
    # `exclude=True` (not just undeclared) is the point: accept it so the
    # request validates, then drop it from every `model_dump()` so it reaches
    # neither Ollama nor the archive. It is the client's own echo, not input.
    # ⚠️ 2026-08-29: without this, a Hermes agent 422'd on its SECOND call of
    # any tool turn — see the note on `ToolChatMessage.name` below.
    reasoning_content: str | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def require_content_or_tool_calls(self) -> AssistantChatMessage:
        if self.content is None and not self.tool_calls:
            raise ValueError("assistant message requires content or tool_calls")
        return self


class ToolChatMessage(_StrictChatMessage):
    role: Literal["tool"]
    content: ChatContent
    tool_call_id: str = Field(min_length=1)
    # ⚠️ THE OTHER FOUR ROLES ALL DECLARED `name`; this one did not, and
    # `_StrictChatMessage` forbids extras — so a tool result carrying the
    # tool's name (which OpenAI accepts, and real agent clients send) was a
    # hard 422. Diagnosed 2026-08-29 from a Hermes bot that failed EVERY
    # tool-calling turn while plain chat worked: the first call succeeded and
    # returned `tool_calls`, then the follow-up carrying the result was
    # rejected before reaching a model. The client surfaced it as a generic
    # "model provider failed", so it read as an outage, not a schema gap.
    # ▶ Failure shape to recognise: 422 `extra_forbidden` at
    #   `body.messages[N].tool.name`.
    name: str | None = None


_ALL_ROLE_FIELDS = frozenset(
    field
    for message_cls in (
        SystemChatMessage, DeveloperChatMessage, UserChatMessage,
        AssistantChatMessage, ToolChatMessage,
    )
    for field in message_cls.model_fields
)


ChatMessage = Annotated[
    SystemChatMessage
    | DeveloperChatMessage
    | UserChatMessage
    | AssistantChatMessage
    | ToolChatMessage,
    Field(discriminator="role"),
]


class ChatCompletionRequest(BaseModel):
    # OWUI adds top-level extension fields such as `chat_id`. Keep accepting
    # them for client compatibility; the public compatibility table explicitly
    # records that unmodelled generation controls are ignored.
    model_config = ConfigDict(extra="ignore")

    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    # Client-owned conversation identity extensions. They are accepted and
    # retained for archive stitching but never forwarded to model providers.
    chat_id: str | None = Field(default=None, exclude=True)
    conversation_id: str | None = Field(default=None, exclude=True)
    metadata: dict[str, Any] | None = Field(default=None, exclude=True)
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    tools: list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "OpenAI-spec tools array. **Only honored on the passthrough "
            "path** (`audrey_passthrough/<concrete>`) — Audrey's pipeline "
            "modes (`audrey_fast`, `audrey_deep`, …) use the server-side "
            "tool registry from `tools/discovery.py` and ignore this field. "
            "Forwarded verbatim to Ollama on passthrough so agent clients "
            "(Hermes, OpenClaw) can advertise their own tools."
        ),
    )
    think: bool | None = Field(
        default=None,
        description=(
            "**Vendor extension, not OpenAI-spec.** Overrides "
            "`passthrough.think` for THIS request; honored only on the "
            "passthrough path, like `tools`. Absent (the default) keeps the "
            "configured behaviour exactly, so serving clients are unaffected. "
            "Still routed through `ollama.thinking_flag`, so asking for "
            "thinking on a model that does not declare the capability omits "
            "the field rather than erroring."
        ),
    )
    user: str | None = Field(
        default=None,
        description=(
            "OpenAI-spec passthrough field. Audrey **ignores** this for "
            "identity purposes — the canonical user id comes from the "
            "Authorization header (require_user → AuthedUser.email). Kept "
            "in the schema for OpenAI client compatibility; logged for "
            "debugging client-vs-resolved identity drift but never trusted."
        ),
    )

    @model_validator(mode="after")
    def validate_tool_result_links(self) -> ChatCompletionRequest:
        """Reject tool results Audrey cannot translate to Ollama safely."""
        calls_by_id: dict[str, str] = {}
        answered: set[str] = set()
        for message in self.messages:
            if isinstance(message, AssistantChatMessage):
                for call in message.tool_calls or []:
                    if call.id in calls_by_id:
                        raise ValueError(f"duplicate assistant tool call id: {call.id}")
                    try:
                        arguments = json.loads(call.function.arguments)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"tool call {call.id} arguments must be valid JSON"
                        ) from exc
                    if not isinstance(arguments, dict):
                        raise ValueError(
                            f"tool call {call.id} arguments must decode to an object"
                        )
                    calls_by_id[call.id] = call.function.name
            elif isinstance(message, ToolChatMessage):
                if message.tool_call_id not in calls_by_id:
                    raise ValueError(
                        "tool message references an unknown earlier tool_call_id: "
                        f"{message.tool_call_id}"
                    )
                if message.tool_call_id in answered:
                    raise ValueError(
                        f"duplicate tool result for tool_call_id: {message.tool_call_id}"
                    )
                answered.add(message.tool_call_id)
        return self
