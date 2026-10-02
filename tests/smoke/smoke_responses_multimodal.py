#!/usr/bin/env python3
"""Prove deployed Responses input_text and inline input_image adaptation."""

from __future__ import annotations

import base64
import binascii
import json
import os
import struct
import sys
import zlib
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

_raw_base = os.getenv("AUDREY_SMOKE_BASE_URL", "").rstrip("/")
if not _raw_base:
    _raw_base = os.getenv("AUDREY_EVAL_BASE_URL", "").rstrip("/")
if _raw_base.endswith("/v1"):
    _raw_base = _raw_base[:-3]
BASE_URL = _raw_base
API_KEY = os.getenv("AUDREY_EVAL_API_KEY", "")
MODEL = os.getenv("AUDREY_RESPONSES_MODEL", "audrey_fast")
TIMEOUT_SECONDS = float(os.getenv("AUDREY_RESPONSES_SMOKE_TIMEOUT_SECONDS", "300"))


class SmokeError(RuntimeError):
    """The deployed Responses multimodal contract was not satisfied."""


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    checksum = binascii.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)


def _red_png_data_url() -> str:
    width = height = 32
    rows = b"".join(b"\x00" + (b"\xff\x00\x00" * width) for _ in range(height))
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(rows))
        + _png_chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _request(
    payload: dict[str, Any],
    *,
    expected: frozenset[int] = frozenset({200}),
) -> tuple[int, dict[str, Any]]:
    request = Request(  # noqa: S310 - base URL is operator-controlled
        f"{BASE_URL}/v1/responses",
        data=json.dumps(payload).encode(),
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=TIMEOUT_SECONDS) as response:  # noqa: S310
            status = response.status
            content = response.read()
    except HTTPError as exc:
        status = exc.code
        content = exc.read()
    parsed = json.loads(content) if content else {}
    if status not in expected:
        raise SmokeError(f"POST /v1/responses: HTTP {status}: {parsed}")
    return status, parsed


def main() -> int:
    if not BASE_URL or not API_KEY:
        print(
            "Set AUDREY_SMOKE_BASE_URL and AUDREY_EVAL_API_KEY "
            "(normally by sourcing .env.test.local).",
            file=sys.stderr,
        )
        return 2

    result: dict[str, Any] = {"schema": 1}
    try:
        status, response = _request({
            "model": MODEL,
            "instructions": (
                "Examine the attached image. Follow the user's exact reply format."
            ),
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "### Task:\nThe attached square has one dominant color. "
                                "If it is red, reply exactly RESPONSES_IMAGE_OK."
                            ),
                        },
                        {
                            "type": "input_image",
                            "image_url": _red_png_data_url(),
                            "detail": "low",
                        },
                    ],
                }
            ],
            "max_output_tokens": 32,
        })
        answer = str(response.get("output_text") or "")
        if response.get("status") != "completed":
            raise SmokeError(f"response did not complete: {response}")
        if "RESPONSES_IMAGE_OK" not in answer:
            raise SmokeError(f"image answer omitted the sentinel: {answer[-500:]!r}")
        response_id = str(response.get("id") or "")
        output = response.get("output") or []
        if not response_id.startswith("resp_") or len(output) != 1:
            raise SmokeError(f"invalid completed response identity: {response}")
        usage = response.get("usage") or {}
        if not isinstance(usage.get("input_tokens"), int):
            raise SmokeError(f"invalid token usage: {usage}")
        result["multimodal"] = {
            "http": status,
            "model": response.get("model"),
            "id_prefix": "resp_",
            "output_type": output[0].get("content", [{}])[0].get("type"),
            "sentinel": True,
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage.get("output_tokens"),
        }

        rejected_status, rejected = _request(
            {
                "model": MODEL,
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Do not generate."},
                            {
                                "type": "input_image",
                                "image_url": "https://example.org/image.png",
                            },
                        ],
                    }
                ],
            },
            expected=frozenset({422}),
        )
        if not isinstance(rejected.get("detail"), list):
            raise SmokeError(f"remote image rejection was not validation-shaped: {rejected}")
        result["unsupported"] = {
            "remote_image_http": rejected_status,
            "validation_error": True,
        }
        result["status"] = "passed"
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001 - smoke must print structured failure
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
