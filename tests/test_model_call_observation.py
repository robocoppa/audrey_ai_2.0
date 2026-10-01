"""Run-scoped observation of concrete generation model calls."""

from __future__ import annotations

import asyncio
import json

import httpx

from audrey.models.ollama import OllamaClient, observe_model_calls


async def test_chat_and_chat_stream_report_models_only_inside_observation_context():
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if payload["stream"]:
            return httpx.Response(
                200,
                content=(
                    json.dumps({
                        "message": {"role": "assistant", "content": "streamed"},
                        "done": True,
                    })
                    + "\n"
                ).encode(),
            )
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "complete"}, "done": True},
        )

    client = OllamaClient(
        "http://ollama:11434",
        transport=httpx.MockTransport(handler),
    )
    observed: list[str] = []
    try:
        with observe_model_calls(observed.append):
            await client.chat(
                model="router:latest",
                messages=[{"role": "user", "content": "Route this."}],
            )
            chunks = [
                chunk
                async for chunk in client.chat_stream(
                    model="writer:latest",
                    messages=[{"role": "user", "content": "Answer this."}],
                )
            ]
        await client.chat(
            model="outside:latest",
            messages=[{"role": "user", "content": "Outside a native run."}],
        )
    finally:
        await client.aclose()

    assert chunks[0]["done"] is True
    assert observed == ["router:latest", "writer:latest"]


async def test_concurrent_observation_contexts_do_not_mix_model_calls():
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "ok"}, "done": True},
        )

    client = OllamaClient(
        "http://ollama:11434",
        transport=httpx.MockTransport(handler),
    )

    async def observed_call(model: str) -> list[str]:
        seen: list[str] = []
        with observe_model_calls(seen.append):
            await client.chat(
                model=model,
                messages=[{"role": "user", "content": model}],
            )
        return seen

    try:
        left, right = await asyncio.gather(
            observed_call("left:latest"),
            observed_call("right:latest"),
        )
    finally:
        await client.aclose()

    assert left == ["left:latest"]
    assert right == ["right:latest"]
