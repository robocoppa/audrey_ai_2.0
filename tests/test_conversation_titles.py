from __future__ import annotations

from typing import Any

from audrey.app_state.titles import (
    fallback_conversation_title,
    normalize_generated_title,
)
from audrey.conversation_titles import ConversationTitleGenerator
from audrey.pipeline.fair_gate import FairLocalGate
from audrey.routes.inflight import UserInflightRegistry


class _Ollama:
    def __init__(self, content: str = "") -> None:
        self.content = content
        self.calls: list[dict[str, Any]] = []
        self.thinking_calls: list[tuple[str, bool]] = []
        self.error: Exception | None = None

    async def thinking_flag(self, model: str, want: bool) -> bool | None:
        self.thinking_calls.append((model, want))
        return want

    async def chat(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return {"message": {"content": self.content}}


class _Registry:
    def location_of(self, model: str) -> str:
        assert model == "title-model"
        return "local"


async def test_title_generator_summarizes_and_sanitizes_model_output():
    ollama = _Ollama("Title: **Weekend Hiking Trip Planning.**\nExtra explanation")
    inflight = UserInflightRegistry(max_inflight_per_user=3)
    generator = ConversationTitleGenerator(
        ollama=ollama,  # type: ignore[arg-type]
        registry=_Registry(),  # type: ignore[arg-type]
        gate=FairLocalGate(concurrency=1),
        inflight=inflight,
        model="title-model",
    )

    title = await generator.generate(
        user_id="usr_alice",
        user_content="Plan a weekend hiking trip with a packing list",
    )

    assert title == "Weekend Hiking Trip Planning"
    assert ollama.thinking_calls == [("title-model", False)]
    assert len(ollama.calls) == 1
    call = ollama.calls[0]
    assert call["model"] == "title-model"
    assert call["messages"][-1]["content"] == (
        "Plan a weekend hiking trip with a packing list"
    )
    assert call["options"] == {"temperature": 0, "num_predict": 32}
    assert call["timeout_s"] == 10.0
    assert call["think"] is False
    assert inflight.pressure_snapshot()["in_use"] == 0
    assert inflight.pressure_snapshot()["waiting"] == 0


async def test_title_generator_fails_back_without_blocking_chat():
    prompt = (
        "Explain how a durable automatic conversation title should be generated "
        "from a very long first prompt without cutting the final word badly."
    )
    ollama = _Ollama()
    ollama.error = RuntimeError("provider detail")
    generator = ConversationTitleGenerator(
        ollama=ollama,  # type: ignore[arg-type]
        registry=_Registry(),  # type: ignore[arg-type]
        gate=FairLocalGate(concurrency=1),
        inflight=UserInflightRegistry(max_inflight_per_user=3),
        model="title-model",
    )

    title = await generator.generate(user_id="usr_alice", user_content=prompt)

    assert title == fallback_conversation_title(prompt)
    assert title.endswith("…")
    assert len(title) <= 72


def test_generated_title_normalization_rejects_empty_or_overlong_output():
    fallback = "Fallback title"

    assert normalize_generated_title("```\n```", fallback=fallback) == fallback
    assert normalize_generated_title(
        "one two three four five six seven eight nine ten eleven twelve",
        fallback=fallback,
    ) == "one two three four five six seven eight nine ten"
