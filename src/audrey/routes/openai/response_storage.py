"""Opt-in, owner-bound Responses retention and bounded text continuation.

Stored history excludes request-level instructions and generation settings.
Each continuation is admitted anew; function execution remains with the client.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
from contextlib import aclosing
from typing import Any

import anyio
from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from audrey.app_state.responses import (
    InvalidResponseStateError,
    ResponseParentNotFoundError,
    ResponseStorageLimitError,
)
from audrey.auth import AuthedUser
from audrey.routes.openai.schemas import (
    ResponseCreateRequest,
    ResponseInputMessage,
    ResponseReplayMessage,
)

MAX_STORED_INPUT_ITEMS = 128
MAX_CHAIN_DEPTH = 32
MAX_STORED_REQUEST_BYTES = 1024 * 1024


def response_not_found() -> HTTPException:
    return HTTPException(status_code=404, detail={
        "error": "responses_not_found", "message": "Response not found.",
    })


def storage_context(request: Request, me: AuthedUser):
    principal = getattr(me, "principal", None)
    if principal is None:
        raise HTTPException(status_code=401, detail="An Audrey account is required for saved responses.")
    store = getattr(request.app.state, "application_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Response storage is unavailable.")
    return store.responses, principal.user_id


def _storage_error(exc: Exception) -> HTTPException:
    if isinstance(exc, ResponseParentNotFoundError):
        return response_not_found()
    if isinstance(exc, ResponseStorageLimitError):
        return HTTPException(status_code=413, detail={
            "error": "responses_storage_limit", "message": str(exc),
        })
    return HTTPException(status_code=503, detail={
        "error": "responses_storage_failed", "message": "Could not save or read the response.",
    })


_STORAGE_ERRORS = (sqlite3.Error, InvalidResponseStateError, ResponseStorageLimitError, ResponseParentNotFoundError)


async def read_response(request: Request, me: AuthedUser, response_id: str) -> dict[str, Any]:
    repository, owner_id = storage_context(request, me)
    try:
        saved = await repository.get(owner_id, response_id)
    except _STORAGE_ERRORS as exc:
        raise _storage_error(exc) from exc
    if saved is None:
        raise response_not_found()
    return saved


async def remove_response(request: Request, me: AuthedUser, response_id: str) -> dict[str, Any]:
    repository, owner_id = storage_context(request, me)
    try:
        deleted = await repository.delete(owner_id, response_id)
    except _STORAGE_ERRORS as exc:
        raise _storage_error(exc) from exc
    if not deleted:
        raise response_not_found()
    return {"id": response_id, "object": "response.deleted", "deleted": True}


async def expand_stored_request(
    payload: ResponseCreateRequest, request: Request, me: AuthedUser,
) -> ResponseCreateRequest:
    """Resolve owned context before file access or starting a provider stream."""
    if payload.store is not True and payload.previous_response_id is None:
        return payload
    storage_context(request, me)
    items = ([ResponseInputMessage(role="user", content=payload.input)]
             if isinstance(payload.input, str) else list(payload.input))
    if any(isinstance(item, ResponseInputMessage) and isinstance(item.content, list)
           and any(part.type != "input_text" for part in item.content) for item in items):
        raise HTTPException(status_code=400, detail={
            "error": "responses_feature_unsupported",
            "message": "Saved and chained Responses support text and client function items only.",
        })
    if payload.previous_response_id is not None:
        saved = await read_response(request, me, payload.previous_response_id)
        if saved["depth"] >= MAX_CHAIN_DEPTH:
            raise HTTPException(status_code=413, detail={
                "error": "responses_storage_limit", "message": "Response chains are limited to 32 turns.",
            })
        try:
            items = ResponseCreateRequest(model=payload.model, input=saved["replay"]).input + items
        except ValidationError as exc:
            raise _storage_error(InvalidResponseStateError("Invalid retained history.")) from exc
    if len(items) > MAX_STORED_INPUT_ITEMS:
        raise HTTPException(status_code=413, detail={
            "error": "responses_storage_limit", "message": "Stored history is limited to 128 items.",
        })
    expanded = payload.model_copy(update={"input": items})
    # Plain saved answer messages must not select the client-function branch.
    # That branch requires tool-capable passthrough even for historical calls.
    if not any(getattr(item, "type", None) in ("function_call", "function_call_output") for item in items):
        items = [ResponseInputMessage(role="assistant", content="".join(part.text for part in item.content))
                 if isinstance(item, ResponseReplayMessage) else item for item in items]
        expanded = payload.model_copy(update={"input": items})
    try:
        encoded = json.dumps(expanded.model_dump(mode="json", by_alias=True),
                             ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise HTTPException(status_code=400, detail="Saved requests must contain standard JSON.") from exc
    if len(encoded) > MAX_STORED_REQUEST_BYTES:
        raise HTTPException(status_code=413, detail={
            "error": "responses_storage_limit", "message": "Retained request exceeds 1 MiB.",
        })
    return expanded


def _replay(payload: ResponseCreateRequest, response: dict[str, Any]) -> list[dict[str, Any]]:
    items = ([{"role": "user", "content": payload.input}]
             if isinstance(payload.input, str) else [item.model_dump(mode="json", exclude_none=True) for item in payload.input])
    replay = items + response["output"]
    if len(replay) > MAX_STORED_INPUT_ITEMS:
        raise ResponseStorageLimitError("Stored history is limited to 128 items including output.")
    return replay


async def _save(payload, response, repository, owner_id):
    """Finish or undo a concurrent SQLite write if the HTTP request is cancelled."""
    replay = _replay(payload, response)
    operation = asyncio.create_task(repository.save(owner_id, response, replay, payload.previous_response_id))
    try:
        await asyncio.shield(operation)
    except asyncio.CancelledError:
        # ASGI cancellation scopes may cancel each awaited cleanup operation;
        # another explicit Task.cancel can also arrive during the SQLite write.
        # Finish that bounded write and undo it before propagating cancellation.
        with anyio.CancelScope(shield=True):
            try:
                await _settle_cleanup(operation)
            except _STORAGE_ERRORS:
                pass
            else:
                deletion = asyncio.create_task(repository.delete(owner_id, response["id"]))
                await _settle_cleanup(deletion)
        raise


async def _settle_cleanup(operation):
    while not operation.done():
        try:
            await asyncio.shield(operation)
        except asyncio.CancelledError:
            continue
    return operation.result()


def _failure(response, error):
    failed = {**response, "status": "failed", "completed_at": None, "incomplete_details": None,
              "error": {"code": error.detail["error"], "message": error.detail["message"]}}
    failed["output"] = [{**item, "status": "incomplete"} for item in response["output"] if item["type"] == "message"]
    return failed


async def retain_response(result, payload: ResponseCreateRequest, request: Request, me: AuthedUser):
    """Commit completion before HTTP/SSE success; incomplete runs are never saved."""
    repository = owner_id = None
    if payload.store is True:
        repository, owner_id = storage_context(request, me)
    if isinstance(result, dict):
        if repository is not None and result["status"] == "completed":
            try:
                await _save(payload, result, repository, owner_id)
            except _STORAGE_ERRORS as exc:
                raise _storage_error(exc) from exc
        return result
    if not isinstance(result, StreamingResponse) or repository is None:
        return result

    upstream = result.body_iterator

    async def retained_stream():
        async with aclosing(upstream):
            async for chunk in upstream:
                text = chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
                # Audrey's adapters yield complete event groups, including any
                # function items and terminal together. Save before forwarding
                # the entire group, so write failure exposes no executable calls.
                blocks = [block for block in text.split("\n\n") if block]
                frames = [json.loads(next(line[6:] for line in block.splitlines() if line.startswith("data: ")))
                          for block in blocks]
                terminal = next((frame for frame in frames if frame["type"] == "response.completed"), None)
                if terminal is not None:
                    try:
                        await _save(payload, terminal["response"], repository, owner_id)
                    except _STORAGE_ERRORS as exc:
                        error = _storage_error(exc)
                        failed = _failure(terminal["response"], error)
                        failure = {"type": "response.failed", "sequence_number": frames[0]["sequence_number"], "response": failed}
                        yield f"event: response.failed\ndata: {json.dumps(failure)}\n\n"
                        return
                yield chunk

    result.body_iterator = retained_stream()
    return result
