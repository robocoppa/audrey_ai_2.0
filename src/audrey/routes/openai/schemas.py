"""OpenAI-compatible request schemas.

`ChatMessage` and `ChatCompletionRequest` are the Pydantic models the route
layer validates incoming `/v1/chat/completions` bodies against. Split out of
the monolithic route module so the other submodules (and tests) can import the
schemas without pulling in the streaming machinery.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from audrey.net.public_fetch import parse_public_url

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
    skill: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
        description="Optional Audrey skill id for this request.",
    )
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


_RESPONSES_IMAGE_DATA_URL_MAX_CHARS = 8 * 1024 * 1024
_RESPONSES_IMAGE_MIME_TYPES = frozenset({
    "image/jpeg",
    "image/png",
    "image/webp",
})


class ResponseInputText(BaseModel):
    """One Responses input_text part adapted to Chat Completions text."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["input_text"]
    text: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def require_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("input_text must contain text")
        return value


class ResponseInputImage(BaseModel):
    """One inline/public image URL or authenticated Audrey image reference."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["input_image"]
    image_url: str | None = Field(default=None, min_length=1, max_length=_RESPONSES_IMAGE_DATA_URL_MAX_CHARS)
    file_id: str | None = Field(default=None, min_length=1, max_length=200, pattern=r"^[A-Za-z0-9_-]+$")
    detail: Literal["auto", "low", "high", "original"] = "auto"

    @field_validator("image_url")
    @classmethod
    def require_inline_supported_image(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not value.startswith("data:"):
            return str(parse_public_url(value))
        header, separator, payload = value.partition(",")
        mime = header.removeprefix("data:").removesuffix(";base64")
        if (
            not separator
            or not header.startswith("data:")
            or not header.endswith(";base64")
            or mime not in _RESPONSES_IMAGE_MIME_TYPES
            or not payload
        ):
            raise ValueError(
                "input_image image_url must be an inline base64 JPEG, PNG, or WEBP data URL"
            )
        try:
            decoded = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("input_image image_url contains invalid base64 data") from exc
        if not decoded:
            raise ValueError("input_image image_url must contain image bytes")
        return value

    @model_validator(mode="after")
    def require_one_source(self) -> ResponseInputImage:
        if (self.image_url is None) == (self.file_id is None):
            raise ValueError("input_image requires exactly one of image_url or file_id")
        return self


class ResponseInputFile(BaseModel):
    """One ready Audrey document or a temporary public document URL."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["input_file"]
    file_id: str | None = Field(default=None, min_length=1, max_length=200, pattern=r"^[A-Za-z0-9_-]+$")
    file_url: str | None = Field(default=None, min_length=1, max_length=4096)

    @field_validator("file_url")
    @classmethod
    def require_public_url_shape(cls, value: str | None) -> str | None:
        return str(parse_public_url(value)) if value is not None else None

    @model_validator(mode="after")
    def require_one_source(self) -> ResponseInputFile:
        if (self.file_url is None) == (self.file_id is None):
            raise ValueError("input_file requires exactly one of file_url or file_id")
        return self


ResponseInputContentPart = Annotated[
    ResponseInputText | ResponseInputImage | ResponseInputFile,
    Field(discriminator="type"),
]


class ResponseInputMessage(BaseModel):
    """Easy-input message accepted by Audrey's Responses adapter."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "developer", "user", "assistant"]
    content: str | list[ResponseInputContentPart]

    @model_validator(mode="after")
    def validate_content(self) -> ResponseInputMessage:
        if isinstance(self.content, str):
            if not self.content.strip():
                raise ValueError("message content must contain text")
            return self
        if not self.content:
            raise ValueError("message content must contain at least one part")
        if self.role != "user" and any(
            isinstance(part, (ResponseInputImage, ResponseInputFile)) for part in self.content
        ):
            raise ValueError("input_image and input_file parts are supported only on user messages")
        return self


class ResponseFormatText(BaseModel):
    """Ordinary Responses text output."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["text"]


class ResponseFormatJSONObject(BaseModel):
    """Legacy JSON mode, parsed so the route can reject it explicitly."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["json_object"]


class ResponseFormatJSONSchema(BaseModel):
    """Named JSON Schema format used by the Responses API."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    type: Literal["json_schema"]
    name: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
    )
    description: str | None = Field(default=None, max_length=1024)
    schema_: dict[str, Any] = Field(alias="schema")
    strict: bool | None = None


ResponseFormat = Annotated[
    ResponseFormatText | ResponseFormatJSONObject | ResponseFormatJSONSchema,
    Field(discriminator="type"),
]


class ResponseTextConfig(BaseModel):
    """Responses output text configuration."""

    model_config = ConfigDict(extra="forbid")

    format: ResponseFormat = Field(default_factory=lambda: ResponseFormatText(type="text"))


class ResponseCreateRequest(BaseModel):
    """Supported subset of the OpenAI POST /v1/responses request."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    input: str | list[ResponseInputMessage]
    instructions: str | None = None
    stream: bool = False
    background: bool = False
    store: bool | None = None
    previous_response_id: str | None = None
    conversation: str | dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    text: ResponseTextConfig | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = Field(default=None, gt=0)
    metadata: dict[str, Any] | None = None
    user: str | None = None
    skill: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
        description="Optional Audrey skill id for this request.",
    )

    @model_validator(mode="after")
    def require_nonempty_input(self) -> ResponseCreateRequest:
        if isinstance(self.input, str):
            if not self.input.strip():
                raise ValueError("input must contain text")
        elif not self.input:
            raise ValueError("input must contain at least one message")
        return self
