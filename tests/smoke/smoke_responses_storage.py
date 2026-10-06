#!/usr/bin/env python3
"""Prove owner-scoped Responses storage, chained history, and restart durability.

``capture`` makes three small model calls and keeps two stored responses in a
private laptop snapshot. Restart Audrey, then run ``verify`` to retrieve both
without generation and delete the parent and its descendant. ``all`` runs the
same protocol checks and deletes immediately; ``cleanup`` recovers a snapshot.
No upload, conversation, application token, or server tool is created.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any

if __package__:
    from . import smoke_responses_client_tools as contract
    from . import smoke_responses_file_inputs as http
else:
    import smoke_responses_client_tools as contract
    import smoke_responses_file_inputs as http

MODEL = os.getenv("AUDREY_RESPONSES_STORAGE_MODEL", "audrey_passthrough/qwen3.8:latest")
DEFAULT_SNAPSHOT = Path("testing-out/smokes/c3-responses-storage.json")
ROOT_ANSWER = "STORED_ROOT_OK"
_TEXT_EVENTS = contract._ALLOWED_EVENTS - {"response.function_call_arguments.delta", "response.function_call_arguments.done"}
_NOT_FOUND = {"detail": {"error": "responses_not_found", "message": "Response not found."}}


def _identities() -> dict[str, str]:
    _, user = http._json("/api/me", token=http.USER_TOKEN)
    _, other = http._json("/api/me", token=http.ADMIN_TOKEN)
    if not user.get("id") or not other.get("id") or user["id"] == other["id"]:
        raise http.SmokeError("the two smoke assertions must resolve to different Audrey accounts")
    return {"owner_id": user["id"], "other_id": other["id"]}


def _root_payload(marker: str, max_output_tokens: int) -> dict[str, Any]:
    return {
        "model": MODEL, "store": True, "stream": False,
        "max_output_tokens": max_output_tokens,
        "instructions": f"For this response only, reply with exactly {ROOT_ANSWER}.",
        "input": f"Remember the storage smoke marker: {marker}. A later question will ask for it.",
    }


def _child_payload(parent_id: str, *, store: bool, stream: bool, max_output_tokens: int) -> dict[str, Any]:
    return {
        "model": MODEL, "store": store, "stream": stream,
        "previous_response_id": parent_id, "max_output_tokens": max_output_tokens,
        "input": "What is the storage smoke marker from the earlier user message? Reply with only that marker.",
    }


def _validate_response(response: dict[str, Any], answer: str, *, store: bool, parent_id: str | None) -> dict[str, Any]:
    contract._validate_common(response)
    if response.get("store") is not store or response.get("previous_response_id") != parent_id:
        raise http.SmokeError("response did not echo its explicit storage and parent selection")
    if response.get("output_text", "").strip() != answer:
        raise http.SmokeError(f"stored history answer was incorrect: {http._response_diagnostics(response)}")
    if any(item["type"] != "message" for item in response["output"]):
        raise http.SmokeError("text-only storage smoke unexpectedly requested a client function")
    return {"http": 200, "store": store, "sentinel": True, **http._response_diagnostics(response)}


def _validate_stream(body: bytes, content_type: str, answer: str, parent_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if not content_type.startswith("text/event-stream"):
        raise http.SmokeError(f"stored stream did not use SSE: {content_type}")
    events = []
    for block in body.decode().split("\n\n"):
        lines = block.splitlines()
        name = next((line[7:] for line in lines if line.startswith("event: ")), "")
        data = next((line[6:] for line in lines if line.startswith("data: ")), "")
        if not data:
            continue
        event = json.loads(data)
        if not isinstance(event, dict) or event.get("type") != name:
            raise http.SmokeError("stored SSE event label does not match its typed event")
        events.append(event)
    if not events or events[0].get("type") != "response.created" or events[-1].get("type") != "response.completed":
        terminal = events[-1] if events else {}
        raise http.SmokeError(f"stored stream did not complete: {terminal.get('type')}; {http._response_diagnostics(terminal.get('response') or {})}")
    if [event.get("sequence_number") for event in events] != list(range(len(events))):
        raise http.SmokeError("stored SSE sequence numbers are not contiguous")
    if any(event["type"] not in _TEXT_EVENTS for event in events):
        raise http.SmokeError("stored stream leaked native tool or progress events")
    response = events[-1]["response"]
    result = _validate_response(response, answer, store=True, parent_id=parent_id)
    if any(event["response"].get("id") != response["id"] for event in events if "response" in event):
        raise http.SmokeError("stored stream changed its response ID")
    added = [event for event in events if event["type"] == "response.output_item.added"]
    done = [event for event in events if event["type"] == "response.output_item.done"]
    if len(added) != 1 or len(done) != 1 or added[0].get("output_index") != 0 or done[0].get("output_index") != 0:
        raise http.SmokeError("stored stream omitted its single assistant item lifecycle")
    item = response["output"][0]
    if added[0].get("item", {}).get("type") != "message" or added[0].get("item", {}).get("status") != "in_progress":
        raise http.SmokeError("stored stream did not start an assistant message")
    if len(response["output"]) != 1 or done[0].get("item") != item or added[0].get("item", {}).get("id") != item["id"]:
        raise http.SmokeError("stored stream item identity differs from its terminal output")
    if sum(event["type"] == "response.output_text.done" for event in events) != 1:
        raise http.SmokeError("stored stream omitted or duplicated text completion")
    deltas = []
    for event in events:
        if event["type"] not in {"response.output_text.delta", "response.output_text.done"}:
            continue
        if event.get("item_id") != item["id"] or event.get("output_index") != 0 or event.get("content_index") != 0:
            raise http.SmokeError("stored text event does not match its assistant item")
        if event["type"].endswith("delta"):
            if not isinstance(event.get("delta"), str):
                raise http.SmokeError("stored text delta is not a string")
            deltas.append(event["delta"])
        elif event.get("text") != response["output_text"]:
            raise http.SmokeError("stored text done differs from terminal text")
    if not deltas or "".join(deltas) != response["output_text"]:
        raise http.SmokeError("stored text deltas differ from terminal text")
    result.update(terminal="response.completed", event_count=len(events), delta_count=len(deltas), progress_hidden=True)
    return response, result


def _get_exact(response: dict[str, Any]) -> dict[str, Any]:
    _, saved = http._json(f"/v1/responses/{response['id']}", token=http.USER_TOKEN)
    if saved != response:
        raise http.SmokeError("retrieved stored response differs from the exact terminal object")
    return {"http": 200, "exact_terminal_object": True}


def _not_found(path: str, *, token: str, method: str = "GET", payload: dict[str, Any] | None = None) -> None:
    _, body, content_type = http._request(
        path, token=token, method=method, expected=frozenset({404}),
        body=json.dumps(payload).encode() if payload is not None else None,
        content_type="application/json" if payload is not None else "",
    )
    if not content_type.startswith("application/json") or json.loads(body) != _NOT_FOUND:
        raise http.SmokeError("missing/foreign/deleted response denial changed or began SSE")


def _ownership_guards(parent_id: str, max_output_tokens: int) -> dict[str, Any]:
    unknown = f"resp_{uuid.uuid4().hex}"
    _not_found(f"/v1/responses/{unknown}", token=http.USER_TOKEN)
    _not_found(f"/v1/responses/{parent_id}", token=http.ADMIN_TOKEN)
    _not_found(f"/v1/responses/{parent_id}", token=http.ADMIN_TOKEN, method="DELETE")
    for reference, token in ((unknown, http.USER_TOKEN), (parent_id, http.ADMIN_TOKEN)):
        _not_found("/v1/responses", token=token, method="POST", payload=_child_payload(
            reference, store=True, stream=True, max_output_tokens=max_output_tokens,
        ))
    return {"unknown_http": 404, "cross_owner_get_http": 404, "cross_owner_delete_http": 404,
            "cross_owner_chain_http": 404, "unknown_chain_http": 404, "same_not_found": True, "before_sse": True}


def _delete_tree(root_id: str, child_id: str, *, allow_missing: bool = False) -> dict[str, Any]:
    status, deleted = http._json(
        f"/v1/responses/{root_id}", token=http.USER_TOKEN, method="DELETE",
        expected=frozenset({200, 404}) if allow_missing else frozenset({200}),
    )
    if status == 200 and deleted != {"id": root_id, "object": "response.deleted", "deleted": True}:
        raise http.SmokeError("response deletion acknowledgement changed")
    if status == 404 and deleted != _NOT_FOUND:
        raise http.SmokeError("response deletion returned an unexpected missing body")
    _not_found(f"/v1/responses/{root_id}", token=http.USER_TOKEN)
    if child_id:
        _not_found(f"/v1/responses/{child_id}", token=http.USER_TOKEN)
    return {"delete_http": status, "root_get_http": 404, "child_get_http": 404 if child_id else None,
            "descendant_cascade": bool(child_id)}


def _write_snapshot(path: Path, snapshot: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(snapshot, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _load_snapshot(path: Path, identity: dict[str, str]) -> dict[str, Any]:
    try:
        snapshot = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise http.SmokeError(f"snapshot does not exist: {path}; run capture first") from exc
    except (ValueError, OSError) as exc:
        raise http.SmokeError(f"snapshot is unreadable or invalid JSON: {path}") from exc
    if not isinstance(snapshot, dict) or snapshot.get("schema") != 1 or snapshot.get("base_url") != http.BASE_URL:
        raise http.SmokeError("snapshot schema or Audrey base URL differs from this run")
    if snapshot.get("identity") != identity:
        raise http.SmokeError("snapshot belongs to a different pair of Audrey accounts")
    for key in ("root", "child"):
        record = snapshot.get(key)
        if not isinstance(record, dict) or not isinstance(record.get("id"), str) or not re.fullmatch(r"resp_[a-f0-9]{32}", record["id"]):
            raise http.SmokeError("snapshot omitted a stored response ID")
    if snapshot["child"].get("previous_response_id") != snapshot["root"]["id"] or snapshot["root"]["id"] == snapshot["child"]["id"]:
        raise http.SmokeError("snapshot does not describe a root and its direct descendant")
    if not isinstance(snapshot.get("marker"), str) or not snapshot["marker"].startswith("STORAGE_"):
        raise http.SmokeError("snapshot omitted its temporary smoke marker")
    return snapshot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("all", "capture", "verify", "cleanup"), nargs="?", default="all")
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    args = parser.parse_args(argv)
    if args.max_output_tokens <= 0:
        parser.error("--max-output-tokens must be positive")
    if not http.BASE_URL or not http.USER_TOKEN or not http.ADMIN_TOKEN:
        print("Set AUDREY_SMOKE_BASE_URL, AUDREY_USER_JWT, and AUDREY_ADMIN_JWT in .env.test.local.", file=sys.stderr)
        return 2
    result: dict[str, Any] = {"schema": 1, "action": args.action, "generation_calls": 0, "uploads_created": 0}
    root_id = child_id = ""
    retain = False
    try:
        if args.action == "capture" and args.snapshot.exists():
            raise http.SmokeError(f"snapshot already exists: {args.snapshot}; run verify or cleanup first")
        identity = _identities()
        result["identity"] = identity
        if args.action in {"verify", "cleanup"}:
            snapshot = _load_snapshot(args.snapshot, identity)
            root, child = snapshot["root"], snapshot["child"]
            if args.action == "verify":
                _validate_response(root, ROOT_ANSWER, store=True, parent_id=None)
                _validate_response(child, snapshot["marker"], store=True, parent_id=root["id"])
                result["restart"] = {"root": _get_exact(root), "child": _get_exact(child), "captured_at": snapshot.get("captured_at")}
            result["cleanup"] = _delete_tree(root["id"], child["id"], allow_missing=True)
            _not_found("/v1/responses", token=http.USER_TOKEN, method="POST", payload=_child_payload(
                root["id"], store=True, stream=True, max_output_tokens=args.max_output_tokens,
            ))
            result["cleanup"]["deleted_chain_http"] = 404
            result["cleanup"]["before_sse"] = True
            args.snapshot.unlink()
            result["status"] = "passed" if args.action == "verify" else "cleaned"
        else:
            marker = f"STORAGE_{uuid.uuid4().hex[:16].upper()}"
            result["generation_calls"] += 1
            _, root = http._json("/v1/responses", token=http.USER_TOKEN, method="POST", payload=_root_payload(marker, args.max_output_tokens))
            root_id = str(root.get("id") or "")
            result["completed"] = _validate_response(root, ROOT_ANSWER, store=True, parent_id=None)
            result["completed"]["retrieval"] = _get_exact(root)
            result["guards"] = _ownership_guards(root_id, args.max_output_tokens)
            result["generation_calls"] += 1
            _, body, content_type = http._request(
                "/v1/responses", token=http.USER_TOKEN, method="POST", content_type="application/json",
                body=json.dumps(_child_payload(root_id, store=True, stream=True, max_output_tokens=args.max_output_tokens)).encode(),
            )
            child, result["streamed"] = _validate_stream(body, content_type, marker, root_id)
            child_id = child["id"]
            result["streamed"].update(retrieval=_get_exact(child), chained_history=True, previous_instructions_not_inherited=True)
            result["generation_calls"] += 1
            _, unretained = http._json("/v1/responses", token=http.USER_TOKEN, method="POST", payload=_child_payload(
                child_id, store=False, stream=False, max_output_tokens=args.max_output_tokens,
            ))
            result["not_stored"] = _validate_response(unretained, marker, store=False, parent_id=child_id)
            _not_found(f"/v1/responses/{unretained['id']}", token=http.USER_TOKEN)
            result["not_stored"]["get_http"] = 404
            if args.action == "capture":
                _write_snapshot(args.snapshot, {
                    "schema": 1, "base_url": http.BASE_URL, "identity": identity,
                    "captured_at": dt.datetime.now(dt.UTC).isoformat(), "root": root, "child": child, "marker": marker,
                })
                retain = True
                result.update(status="captured", snapshot=str(args.snapshot), next="Restart Audrey, then run verify with this same snapshot and credentials.")
            else:
                result["cleanup"] = _delete_tree(root_id, child_id)
                root_id = ""
                _not_found("/v1/responses", token=http.USER_TOKEN, method="POST", payload=_child_payload(
                    root["id"], store=True, stream=True, max_output_tokens=args.max_output_tokens,
                ))
                result["cleanup"].update(deleted_chain_http=404, before_sse=True)
                result["status"] = "passed"
    except Exception as exc:  # noqa: BLE001 - bounded operator evidence
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        if root_id and not retain:
            try:
                result["cleanup"] = _delete_tree(root_id, child_id, allow_missing=True)
            except Exception as exc:  # noqa: BLE001 - report cleanup without hiding the primary failure
                result.update(status="failed", cleanup_error=f"{type(exc).__name__}: {exc}", recovery_root_id=root_id)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in {"passed", "captured", "cleaned"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
