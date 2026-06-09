"""Hermetic tests for `pipeline/messages.py` — `last_user_text` + `has_image_part`."""

from __future__ import annotations

from audrey.pipeline.messages import has_image_part, last_user_text


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


def test_has_image_part_true_for_image_turn():
    msgs = [
        {"role": "user", "content": [
            {"type": "text", "text": "describe this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]},
    ]
    assert has_image_part(msgs) is True


def test_has_image_part_false_for_plain_text():
    assert has_image_part([{"role": "user", "content": "just words"}]) is False


def test_has_image_part_false_for_text_only_list():
    # A list content with only text parts is not an image turn.
    msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    assert has_image_part(msgs) is False


def test_has_image_part_only_inspects_latest_user_turn():
    # An image in an earlier turn doesn't make the current text turn a vision turn.
    msgs = [
        {"role": "user", "content": [
            {"type": "text", "text": "old"},
            {"type": "image_url", "image_url": {"url": "..."}},
        ]},
        {"role": "assistant", "content": "ack"},
        {"role": "user", "content": "follow-up, no image"},
    ]
    assert has_image_part(msgs) is False


def test_has_image_part_false_when_no_user_turn():
    assert has_image_part([{"role": "system", "content": "alone"}]) is False
    assert has_image_part([]) is False
