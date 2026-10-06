"""OpenAI response formatting — pure helpers, no I/O.

Builds OpenAI-shaped responses from pipeline output: request to Ollama
options, Chat Completions envelopes, completed and streamed Responses objects, and
Ollama-to-OpenAI tool-call conversion. Leaf module - depends only on the
schemas and stdlib.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from audrey import __version__
from audrey.routes.openai.schemas import ChatCompletionRequest, ResponseCreateRequest
from audrey.routes.openai.structured_outputs import (
    response_text_config,
    validate_structured_output,
)


def _options_from_request(req: ChatCompletionRequest) -> dict[str, Any]:
    """Map OpenAI-shape sampling knobs onto Ollama's options dict.

    Sibling: `pipeline.graph._options_from_state` does the same conceptual
    mapping from the LangGraph state dict. The two helpers stay parallel
    rather than unified because their input shapes genuinely differ
    (Pydantic model vs. dict); keep them in sync if the knob set changes.
    """
    opts: dict[str, Any] = {}
    if req.temperature is not None:
        opts["temperature"] = req.temperature
    if req.top_p is not None:
        opts["top_p"] = req.top_p
    if req.max_tokens is not None:
        opts["num_predict"] = req.max_tokens
    return opts


def _to_openai_response(
    *,
    virtual: str,
    concrete: str,
    content: str,
    prompt_tokens: int,
    completion_tokens: int,
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
        finish_reason = "tool_calls"
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": virtual,
        "system_fingerprint": f"audrey-{__version__}/{concrete}",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _to_responses_api_response(
    chat_response: dict[str, Any],
    request: ResponseCreateRequest,
) -> dict[str, Any]:
    """Translate Audrey's completed chat envelope into a Responses object."""

    choices = chat_response.get("choices") or []
    if len(choices) != 1:
        raise ValueError("generation returned no single completed choice")
    message = choices[0].get("message") or {}
    if message.get("tool_calls"):
        raise ValueError("generation returned unsupported client tool calls")
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("generation returned no text content")
    truncated = choices[0].get("finish_reason") == "length"
    if not truncated:
        if not content.strip():
            raise ValueError("generation completed without an answer")
        validate_structured_output(content, request)

    usage = chat_response.get("usage") or {}
    input_tokens = int(usage.get("prompt_tokens", 0) or 0)
    output_tokens = int(usage.get("completion_tokens", 0) or 0)
    created_at = int(chat_response.get("created") or time.time())
    return _responses_api_response_object(
        request=request,
        response_id=f"resp_{uuid.uuid4().hex}",
        message_id=f"msg_{uuid.uuid4().hex}",
        created_at=created_at,
        completed_at=None if truncated else int(time.time()),
        status="incomplete" if truncated else "completed",
        content=content,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        output_started=bool(content) if truncated else True,
        incomplete_details={"reason": "max_output_tokens"} if truncated else None,
    )


def _responses_api_response_object(
    *,
    request: ResponseCreateRequest,
    response_id: str,
    message_id: str,
    created_at: int,
    status: str,
    content: str,
    completed_at: int | None,
    input_tokens: int | None,
    output_tokens: int | None,
    output_started: bool = True,
    error: dict[str, str] | None = None,
    incomplete_details: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build one stable Responses object for completed and streamed replies."""

    output_status = "completed" if status == "completed" else "incomplete"
    output = (
        [
            {
                "id": message_id,
                "type": "message",
                "status": output_status,
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": content,
                        "annotations": [],
                        "logprobs": [],
                    }
                ],
            }
        ]
        if output_started
        else []
    )
    usage = None
    if input_tokens is not None and output_tokens is not None:
        usage = {
            "input_tokens": input_tokens,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": output_tokens,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": input_tokens + output_tokens,
        }
    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "completed_at": completed_at,
        "status": status,
        "error": error,
        "incomplete_details": incomplete_details,
        "instructions": request.instructions,
        "max_output_tokens": request.max_output_tokens,
        "metadata": request.metadata or {},
        "model": request.model,
        "output": output,
        "output_text": content,
        "parallel_tool_calls": True,
        "previous_response_id": request.previous_response_id,
        "store": request.store is True,
        "text": response_text_config(request),
        "tool_choice": "auto",
        "tools": [],
        "truncation": "disabled",
        "usage": usage,
    }


def _ollama_to_openai_tool_calls(
    ollama_tool_calls: list[dict[str, Any]] | None,
    *,
    streaming: bool = False,
) -> list[dict[str, Any]] | None:
    """Convert Ollama's tool_calls shape to OpenAI's.

    Ollama returns:
        [{"function": {"name": str, "arguments": dict}}, ...]
    OpenAI clients expect:
        [{"id": str, "type": "function",
          "function": {"name": str, "arguments": str (JSON)}}]

    The argument shape difference (dict vs JSON-string) is the main
    thing — clients that parse OpenAI responses will try `json.loads`
    on `arguments` and crash if it's already a dict. Audrey synthesizes
    the `id` since Ollama does not emit one. With `streaming=True`, each
    delta call also receives the required assembly `index`; completed
    non-streaming assistant messages deliberately omit it.
    """
    if not ollama_tool_calls:
        return None
    out: list[dict[str, Any]] = []
    for index, call in enumerate(ollama_tool_calls):
        fn = call.get("function") or {}
        name = fn.get("name") or ""
        raw_args = fn.get("arguments")
        if isinstance(raw_args, str):
            arguments = raw_args
        elif raw_args is None:
            arguments = "{}"
        else:
            arguments = json.dumps(raw_args)
        converted: dict[str, Any] = {
            "id": call.get("id") or f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        }
        if streaming:
            converted["index"] = index
        out.append(converted)
    return out
