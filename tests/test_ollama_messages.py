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
