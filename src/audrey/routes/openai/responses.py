"""OpenAI response formatting — pure helpers, no I/O.

Builds the OpenAI chat-completion response shape from pipeline output:
request → Ollama `options`, the final non-streaming response envelope, and
Ollama→OpenAI tool_call conversion. Leaf module — depends only on the schemas
and stdlib.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from audrey import __version__
from audrey.routes.openai.schemas import ChatCompletionRequest


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
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    finish_reason = "stop"
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


def _ollama_to_openai_tool_calls(
    ollama_tool_calls: list[dict[str, Any]] | None,
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
    the `id` since Ollama doesn't emit one.
    """
    if not ollama_tool_calls:
        return None
    out: list[dict[str, Any]] = []
    for call in ollama_tool_calls:
        fn = call.get("function") or {}
        name = fn.get("name") or ""
        raw_args = fn.get("arguments")
        if isinstance(raw_args, str):
            arguments = raw_args
        elif raw_args is None:
            arguments = "{}"
        else:
            arguments = json.dumps(raw_args)
        out.append({
            "id": call.get("id") or f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        })
    return out
