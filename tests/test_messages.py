"""Hermetic tests for `pipeline/messages.py` — the `last_user_text` helper."""

from __future__ import annotations

from audrey.pipeline.messages import last_user_text


def test_last_user_text_returns_string_content():
    msgs = [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "hello"},
    ]
    assert last_user_text(msgs) == "hello"


def test_last_user_text_returns_most_recent_user_turn():
    msgs = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ack"},
        {"role": "user", "content": "second"},
    ]
    assert last_user_text(msgs) == "second"


def test_last_user_text_handles_list_content():
    """Multi-modal content arrives as a list of typed parts; we flatten
    the text parts with newlines and ignore non-text shapes."""
    msgs = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ack"},
        {"role": "user", "content": [
            {"type": "text", "text": "second"},
            {"type": "image_url", "image_url": {"url": "..."}},
        ]},
    ]
    assert last_user_text(msgs) == "second\n"


def test_last_user_text_returns_empty_when_no_user_turn():
    msgs = [
        {"role": "system", "content": "alone"},
        {"role": "assistant", "content": "no user yet"},
    ]
    assert last_user_text(msgs) == ""


def test_last_user_text_returns_empty_on_empty_messages():
    assert last_user_text([]) == ""


def test_last_user_text_skips_non_dict_parts_in_list_content():
    msgs = [
        {"role": "user", "content": [
            "garbage",
            {"type": "text", "text": "real"},
            None,
        ]},
    ]
    # Non-dict parts contribute nothing; the dict part lands.
    assert last_user_text(msgs) == "real"
