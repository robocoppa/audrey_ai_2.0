"""Client-executed Responses function calls on permitted passthrough models.

Provider calls still use Audrey's existing per-user and GPU gates. Function
requests are validated before any executable item is returned. No server tool
registry or dispatcher participates in this protocol.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse

from audrey.auth import AuthedUser
from audrey.metrics import pipeline_seconds, pipeline_total
from audrey.models.ollama import OllamaError
from audrey.pipeline.messages import last_user_text
from audrey.pipeline.passthrough import passthrough_chat, passthrough_stream
from audrey.pipeline.vision import describe_for_text_model
from audrey.routes.openai.client_tools import (
    adapt_tool_history_item,
    merge_adjacent_tool_calls,
    provider_tools,
    validate_generated_calls,
)
from audrey.routes.openai.file_inputs import validate_file_prompt
from audrey.routes.openai.passthrough import _passthrough_think, _resolve_passthrough_model
from audrey.routes.openai.responses import _options_from_request, _responses_api_response_object
from audrey.routes.openai.schemas import (
    ChatCompletionRequest,
    ResponseCreateRequest,
    ResponseInputMessage,
)

MAX_GENERATED_TEXT_CHARS = 250_000
MAX_RAW_CALL_BYTES = 512 * 1024
MAX_RAW_CALLS = 8


async def validate_client_tool_prompt(payload: ResponseCreateRequest) -> None:
    """Count caller text, call arguments/results, and schemas before any fetch."""
    if isinstance(payload.input, str):
        messages = [{"content": payload.input}]
    else:
        messages = []
        for item in payload.input:
            if isinstance(item, ResponseInputMessage):
                messages.append({"content": item.content if isinstance(item.content, str) else [
                    {"type": "text", "text": part.text} for part in item.content if part.type == "input_text"
                ]})
            else:
                messages.append(adapt_tool_history_item(item))
    if payload.instructions:
        messages.append({"content": payload.instructions})
    messages.append({"content": json.dumps(provider_tools(payload) or [], ensure_ascii=False)})
    await validate_file_prompt(messages)


async def prepare_client_tool_model(payload: ResponseCreateRequest, request: Request, me: AuthedUser) -> tuple[str, str]:
    """Check existing passthrough policy and actual tool capability before fetch."""
    app = request.app
    target = _resolve_passthrough_model(payload.model, app.state.cfg, app.state.registry, me)
    effort = payload.reasoning.effort if payload.reasoning is not None else None
    if effort is not None:
        await _passthrough_think(app.state.ollama, app.state.cfg, target[0], effort=effort)
    try:
        capabilities = await app.state.ollama.capabilities(target[0])
    except OllamaError as exc:
        raise HTTPException(status_code=502, detail={
            "error": "responses_tool_model_unavailable",
            "message": "Could not verify the requested model's tool capability.",
        }) from exc
    if "tools" not in capabilities:
        raise HTTPException(status_code=400, detail={
            "error": "responses_feature_unsupported",
            "message": "Client function calls require a model declaring the tools capability.",
        })
    return target


def _message_item(text: str, *, item_id: str, status: str) -> dict[str, Any]:
    return {"id": item_id, "type": "message", "status": status, "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": [], "logprobs": []}]}


def _response(payload, *, response_id, message_id, created, status, text, calls, usage=None, error=None):
    input_tokens = output_tokens = None
    if usage is not None:
        input_tokens, output_tokens = usage
    result = _responses_api_response_object(
        request=payload, response_id=response_id, message_id=message_id, created_at=created,
        completed_at=int(time.time()) if status == "completed" else None,
        status=status, content=text, input_tokens=input_tokens, output_tokens=output_tokens,
        output_started=False, error=error,
        incomplete_details={"reason": "max_output_tokens"} if status == "incomplete" else None,
    )
    result["output"] = ([_message_item(text, item_id=message_id, status="completed" if status == "completed" else "incomplete")] if text else []) + calls
    result["tools"] = [{**tool, "strict": bool(tool.get("strict", False))} for tool in payload.tools or []]
    result["tool_choice"] = payload.tool_choice or "auto"
    result["parallel_tool_calls"] = payload.parallel_tool_calls is not False
    return result


def _check_text(text: str) -> None:
    if len(text) > MAX_GENERATED_TEXT_CHARS:
        raise HTTPException(status_code=502, detail="Generated response exceeds the text limit.")


def _usage(raw: dict) -> tuple[int, int]:
    counts = []
    for key in ("prompt_eval_count", "eval_count"):
        value = raw.get(key)
        if value is None:
            value = 0
        if type(value) is not int or value < 0:
            raise HTTPException(status_code=502, detail="Provider token usage is invalid.")
        counts.append(value)
    return counts[0], counts[1]


def _collect_calls(existing: list[dict], incoming: Any) -> None:
    if incoming is None:
        return
    if not isinstance(incoming, list) or len(existing) + len(incoming) > MAX_RAW_CALLS:
        raise HTTPException(status_code=502, detail="Generated function calls exceed the count limit or have an invalid shape.")
    existing.extend(incoming)
    try:
        encoded = json.dumps(existing, ensure_ascii=False, allow_nan=False).encode()
    except (TypeError, ValueError, RecursionError) as exc:
        raise HTTPException(status_code=502, detail="Generated function calls are not valid JSON.") from exc
    if len(encoded) > MAX_RAW_CALL_BYTES:
        raise HTTPException(status_code=502, detail="Generated function calls exceed the byte limit.")


@dataclass
class _ToolStream:
    payload: ResponseCreateRequest
    response_id: str = field(default_factory=lambda: f"resp_{uuid.uuid4().hex}")
    message_id: str = field(default_factory=lambda: f"msg_{uuid.uuid4().hex}")
    created: int = field(default_factory=lambda: int(time.time()))
    text: str = ""
    sequence: int = -1
    message_started: bool = False
    usage: tuple[int, int] | None = None

    def frame(self, name, **data):
        self.sequence += 1
        event = {"type": name, **data, "sequence_number": self.sequence}
        return f"event: {name}\ndata: {json.dumps(event)}\n\n"

    def response(self, status, *, calls=None, error=None):
        return _response(self.payload, response_id=self.response_id, message_id=self.message_id,
                         created=self.created, status=status, text=self.text, calls=calls or [],
                         usage=self.usage, error=error)

    def start(self):
        response = self.response("in_progress")
        return self.frame("response.created", response=response) + self.frame("response.in_progress", response=response)

    def content(self, delta):
        _check_text(self.text + delta)
        frames = ""
        if not self.message_started:
            self.message_started = True
            item = _message_item("", item_id=self.message_id, status="in_progress")
            item["content"] = []
            frames += self.frame("response.output_item.added", output_index=0, item=item)
            frames += self.frame("response.content_part.added", item_id=self.message_id,
                                 output_index=0, content_index=0,
                                 part={"type": "output_text", "text": "", "annotations": [], "logprobs": []})
        self.text += delta
        return frames + self.frame("response.output_text.delta", item_id=self.message_id,
                                   output_index=0, content_index=0, delta=delta, logprobs=[])

    def finish(self, status, *, calls=None, error=None):
        frames = ""
        if self.message_started:
            item = _message_item(self.text, item_id=self.message_id, status="completed" if status == "completed" else "incomplete")
            frames += self.frame("response.output_text.done", item_id=self.message_id, output_index=0,
                                 content_index=0, text=self.text, logprobs=[])
            frames += self.frame("response.content_part.done", item_id=self.message_id, output_index=0,
                                 content_index=0, part=item["content"][0])
            frames += self.frame("response.output_item.done", output_index=0, item=item)
        # Native Ollama streams complete argument objects. Hold every call
        # until successful termination and batch validation, then emit one
        # argument delta. A client must never receive an unchecked call item.
        for index, call in enumerate(calls or [], start=int(self.message_started)):
            frames += self.frame("response.output_item.added", output_index=index,
                                 item={**call, "arguments": "", "status": "in_progress"})
            fields = {"item_id": call["id"], "output_index": index}
            frames += self.frame("response.function_call_arguments.delta", **fields, delta=call["arguments"])
            frames += self.frame("response.function_call_arguments.done", **fields, arguments=call["arguments"])
            frames += self.frame("response.output_item.done", output_index=index, item=call)
        name = {"completed": "response.completed", "incomplete": "response.incomplete", "failed": "response.failed"}[status]
        return frames + self.frame(name, response=self.response(status, calls=calls, error=error))


def _observe(started: float, outcome: str) -> None:
    pipeline_seconds.labels(mode="responses_tools", task_type="passthrough").observe(time.perf_counter() - started)
    pipeline_total.labels(mode="responses_tools", task_type="passthrough", outcome=outcome).inc()


async def generate_client_tool_response(payload: ResponseCreateRequest, request: Request, me: AuthedUser,
                                        messages: list[dict[str, Any]], target: tuple[str, str]):
    app = request.app
    concrete, location = target
    # Keep the established Chat role and call/result validation at the final
    # provider seam, after adapting adjacent Responses function-call items.
    chat = ChatCompletionRequest(model=payload.model, messages=merge_adjacent_tool_calls(messages),
                                 temperature=payload.temperature, top_p=payload.top_p,
                                 max_tokens=payload.max_output_tokens, tools=provider_tools(payload))
    messages = [message.model_dump(exclude_none=True, exclude={"metadata"}) for message in chat.messages]
    effort = payload.reasoning.effort if payload.reasoning is not None else None
    think = await _passthrough_think(app.state.ollama, app.state.cfg, concrete, effort=effort)
    async with app.state.inflight.slot(me.email):
        messages, _ = await describe_for_text_model(
            messages, ollama=app.state.ollama, registry=app.state.registry,
            health=app.state.health, gate=app.state.gate, cfg=app.state.cfg,
            target_model=concrete, user_question=last_user_text(messages), user_id=me.email,
        )
    # Image descriptions are new provider text; apply the same aggregate
    # budget after that adaptation as after document hydration.
    await validate_file_prompt([
        *messages,
        {"content": json.dumps(chat.tools or [], ensure_ascii=False)},
    ])
    kwargs = dict(concrete=concrete, location=location, messages=messages,
                  options=_options_from_request(chat), user_id=me.email, tools=chat.tools,
                  timeout_s=float(app.state.cfg.timeouts.get("medium", 180)), think=think)
    if payload.stream:
        async def body():
            session = _ToolStream(payload)
            started = time.perf_counter()
            outcome = "error"
            calls: list[dict] = []
            try:
                yield session.start()
                async with app.state.inflight.slot(me.email):
                    async with aclosing(passthrough_stream(app.state.ollama, app.state.gate, **kwargs)) as stream:
                        async for chunk in stream:
                            if not isinstance(chunk, dict) or not isinstance(chunk.get("message", {}), dict):
                                raise HTTPException(status_code=502, detail="Provider response has an invalid shape.")
                            message = chunk.get("message") or {}
                            delta = message.get("content")
                            if delta is None:
                                delta = ""
                            if not isinstance(delta, str):
                                raise HTTPException(status_code=502, detail="Generated response text is invalid.")
                            if delta:
                                yield session.content(delta)
                            _collect_calls(calls, message.get("tool_calls"))
                            if chunk.get("done"):
                                session.usage = _usage(chunk)
                                if chunk.get("done_reason") == "length":
                                    outcome = "truncated"
                                    yield session.finish("incomplete")
                                else:
                                    validated = validate_generated_calls(calls, payload)
                                    if not validated and not session.text.strip():
                                        raise HTTPException(status_code=502, detail="Generation completed without text or function calls.")
                                    outcome = "ok"
                                    yield session.finish("completed", calls=validated)
                                return
                outcome = "truncated"
                yield session.finish("failed", error={
                    "code": "responses_stream_interrupted",
                    "message": "The model stream ended before confirming completion.",
                })
            except (asyncio.CancelledError, GeneratorExit):
                outcome = "cancelled"
                raise
            except Exception:  # noqa: BLE001 - SSE must end with a typed failure
                yield session.finish("failed", error={"code": "responses_tool_output_invalid", "message": "Audrey could not produce valid client tool output."})
            finally:
                _observe(started, outcome)
        return StreamingResponse(body(), media_type="text/event-stream")

    started = time.perf_counter()
    outcome = "error"
    try:
        async with app.state.inflight.slot(me.email):
            raw = await passthrough_chat(app.state.ollama, app.state.gate, **kwargs)
        if not isinstance(raw, dict) or not isinstance(raw.get("message", {}), dict):
            raise HTTPException(status_code=502, detail="Provider response has an invalid shape.")
        message = raw.get("message") or {}
        text = message.get("content")
        if text is None:
            text = ""
        if not isinstance(text, str):
            raise HTTPException(status_code=502, detail="Generated response text is invalid.")
        _check_text(text)
        calls: list[dict] = []
        _collect_calls(calls, message.get("tool_calls"))
        truncated = raw.get("done_reason") == "length"
        validated = [] if truncated else validate_generated_calls(calls, payload)
        if not truncated and not validated and not text.strip():
            raise HTTPException(status_code=502, detail="Generation completed without text or function calls.")
        usage = _usage(raw)
        outcome = "truncated" if truncated else "ok"
        return _response(payload, response_id=f"resp_{uuid.uuid4().hex}", message_id=f"msg_{uuid.uuid4().hex}",
                         created=int(time.time()), status="incomplete" if truncated else "completed", text=text,
                         calls=validated, usage=usage)
    except (OllamaError, ValueError, TypeError, AttributeError, RecursionError) as exc:
        raise HTTPException(status_code=502, detail="Client tool generation failed at the model provider.") from exc
    except (asyncio.CancelledError, GeneratorExit):
        outcome = "cancelled"
        raise
    finally:
        _observe(started, outcome)
