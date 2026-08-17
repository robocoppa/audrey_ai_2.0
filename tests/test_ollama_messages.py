"""Tests for the OpenAI → Ollama message-shape conversion.

Ollama's `/api/chat` rejects array-shaped `content` (the OpenAI multimodal
form OWUI sends for image turns) with a 400:

    json: cannot unmarshal array into Go struct field
    ChatRequest.messages.content of type string

`_to_ollama_messages` flattens that array into Ollama's native shape
(`content` string + sibling `images` list) before the payload is built.
"""

from __future__ import annotations

from audrey.models.ollama import _data_uri_to_b64, _to_ollama_messages

_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M8AAAMBAQDJ/pLvAAAAAElFTkSuQmCC"


def test_string_content_passes_through_untouched():
    msgs = [{"role": "user", "content": "hello"}]
    assert _to_ollama_messages(msgs) == msgs


def test_missing_content_passes_through():
    # A tool/assistant message may legitimately have no content key.
    msgs = [{"role": "assistant", "tool_calls": [{"id": "x"}]}]
    assert _to_ollama_messages(msgs) == msgs


def test_array_text_only_flattens_to_string():
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "line one"},
                {"type": "text", "text": "line two"},
            ],
        }
    ]
    (out,) = _to_ollama_messages(msgs)
    assert out["content"] == "line one\nline two"
    assert "images" not in out  # no image parts → no images key


def test_image_turn_lifts_b64_into_images_field():
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this?"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{_PNG}"},
                },
            ],
        }
    ]
    (out,) = _to_ollama_messages(msgs)
    assert out["content"] == "what is this?"
    assert out["images"] == [_PNG]
    assert out["role"] == "user"  # other keys preserved


def test_non_dict_parts_are_ignored():
    msgs = [{"role": "user", "content": ["bare string", {"type": "text", "text": "ok"}]}]
    (out,) = _to_ollama_messages(msgs)
    assert out["content"] == "ok"


def test_remote_image_url_is_dropped():
    # Ollama can't fetch http(s) URLs on /api/chat; only inline base64 works.
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
            ],
        }
    ]
    (out,) = _to_ollama_messages(msgs)
    assert out["content"] == "describe"
    assert "images" not in out


def test_data_uri_helper():
    assert _data_uri_to_b64(f"data:image/png;base64,{_PNG}") == _PNG
    assert _data_uri_to_b64("https://example.com/a.png") is None
    assert _data_uri_to_b64("data:image/png,notbase64") is None
    assert _data_uri_to_b64("") is None


# ─── Leading system messages are merged (2026-08-17) ───────────────────
#
# ⚠️ Ollama's qwen3-family renderer accepts a system message only as the FIRST
# message and raises on any later one — including a second CONSECUTIVE system
# message at index 1. It fails at render time, so the call comes back
# `/api/chat -> 500 {"error":"system message must be at the beginning"}` with
# nothing generated, and the worker reads as silent rather than broken.
#
# `qwen3.8:latest` returned zero usable drafts across two research runs and a
# code run because of this. The asymmetry that located it: fast-path turns
# worked. Eval requests carry no persona (`eval_research.py` sends a bare user
# turn), so a fast-path turn holds ONE system message and renders; a panel
# worker gets the role prompt prepended on top of it, hits two, and dies.


def test_two_leading_system_messages_become_one():
    """The exact shape a deep-panel worker sends: role prompt prepended onto
    the datetime message that `node_datetime` already put there."""
    msgs = [
        {"role": "system", "content": "You are a worker on a panel."},
        {"role": "system", "content": "Current server date and time: X."},
        {"role": "user", "content": "question"},
    ]
    out = _to_ollama_messages(msgs)
    assert [m["role"] for m in out] == ["system", "user"]
    assert out[0]["content"] == (
        "You are a worker on a panel.\n\nCurrent server date and time: X.")


def test_four_leading_system_messages_become_one():
    """Not a stress case — persona + memory recall + chat-history guidance +
    datetime is an ordinary identified-user request before the panel adds a
    role prompt on top."""
    msgs = [{"role": "system", "content": c} for c in ("a", "b", "c", "d")]
    msgs.append({"role": "user", "content": "q"})
    out = _to_ollama_messages(msgs)
    assert [m["role"] for m in out] == ["system", "user"]
    assert out[0]["content"] == "a\n\nb\n\nc\n\nd"


def test_a_single_system_message_is_untouched():
    msgs = [{"role": "system", "content": "one"}, {"role": "user", "content": "q"}]
    assert _to_ollama_messages(msgs) == msgs


def test_no_system_message_is_untouched():
    msgs = [{"role": "user", "content": "q"}]
    assert _to_ollama_messages(msgs) == msgs


def test_only_the_leading_run_is_merged():
    """A system message after a non-system turn is left exactly where it is.
    Position can carry meaning, and relocating it would say an earlier event
    happened later. The one path that produced such messages — react history
    compaction — now keeps its stubs as tool messages instead."""
    msgs = [
        {"role": "system", "content": "a"},
        {"role": "system", "content": "b"},
        {"role": "user", "content": "q"},
        {"role": "system", "content": "mid"},
    ]
    out = _to_ollama_messages(msgs)
    assert [m["role"] for m in out] == ["system", "user", "system"]
    assert out[0]["content"] == "a\n\nb"
    assert out[2]["content"] == "mid"


def test_merging_preserves_the_first_messages_other_keys():
    msgs = [
        {"role": "system", "content": "a", "name": "persona"},
        {"role": "system", "content": "b"},
    ]
    (out,) = _to_ollama_messages(msgs)
    assert out["name"] == "persona" and out["content"] == "a\n\nb"


def test_empty_system_bodies_do_not_leave_blank_separators():
    msgs = [
        {"role": "system", "content": "a"},
        {"role": "system", "content": "   "},
        {"role": "system", "content": "b"},
        {"role": "user", "content": "q"},
    ]
    out = _to_ollama_messages(msgs)
    assert out[0]["content"] == "a\n\nb"


def test_a_system_message_carrying_an_image_blocks_the_merge():
    """Never seen in practice, but merging would silently drop the image.
    Leaving the list alone is the safe failure: the renderer may reject it,
    which is loud, where a dropped image is not."""
    msgs = [
        {"role": "system", "content": [
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_PNG}"}},
        ]},
        {"role": "system", "content": "b"},
    ]
    out = _to_ollama_messages(msgs)
    assert [m["role"] for m in out] == ["system", "system"]
    assert out[0]["images"] == [_PNG]


def test_the_flatten_runs_before_the_merge():
    """Ordering matters: the merge joins strings, so array content has to be
    flattened first or it would be str()'d into the merged body."""
    msgs = [
        {"role": "system", "content": [{"type": "text", "text": "flat"}]},
        {"role": "system", "content": "second"},
        {"role": "user", "content": "q"},
    ]
    out = _to_ollama_messages(msgs)
    assert out[0]["content"] == "flat\n\nsecond"
