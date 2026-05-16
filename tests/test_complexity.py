"""Tests for the complexity gate's token counters."""
from __future__ import annotations

from audrey.pipeline.complexity import (
    count_last_user_tokens,
    count_tokens,
    count_tokens_by_role,
    is_complex,
    is_owui_task_request,
)


def test_count_tokens_sums_string_content():
    messages = [
        {"role": "system", "content": "you are a helpful assistant"},
        {"role": "user", "content": "hello there"},
    ]
    assert count_tokens(messages) > 0


def test_count_tokens_handles_multimodal_parts():
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "describe this"},
            {"type": "image_url", "image_url": {"url": "data:..."}},
        ]},
    ]
    n = count_tokens(messages)
    # Image bytes don't count; only the text part does.
    assert n == count_tokens([{"role": "user", "content": "describe this"}])


def test_count_tokens_ignores_non_string_content():
    messages = [{"role": "user", "content": None}]
    assert count_tokens(messages) == 0


def test_count_tokens_by_role_splits_by_role():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hey there"},
        {"role": "tool", "content": "tool result"},
        {"role": "user", "content": "follow up"},
    ]
    by_role = count_tokens_by_role(messages)
    assert set(by_role) == {"system", "user", "assistant", "tool"}
    # Sum matches the total.
    assert sum(by_role.values()) == count_tokens(messages)
    # Two user messages summed under one bucket.
    one_user = count_tokens_by_role([{"role": "user", "content": "hi"}])["user"]
    two_user = count_tokens_by_role([
        {"role": "user", "content": "hi"},
        {"role": "user", "content": "follow up"},
    ])["user"]
    assert two_user > one_user


def test_count_tokens_by_role_buckets_missing_role_as_other():
    messages = [{"content": "rogue message with no role"}]
    by_role = count_tokens_by_role(messages)
    assert "other" in by_role
    assert by_role["other"] > 0


def test_count_last_user_tokens_picks_most_recent_user():
    messages = [
        {"role": "user", "content": "first message"},
        {"role": "assistant", "content": "a long assistant reply with many tokens to inflate the count"},
        {"role": "tool", "content": "tool result data"},
        {"role": "user", "content": "ok"},
    ]
    last = count_last_user_tokens(messages)
    only_last = count_tokens([{"role": "user", "content": "ok"}])
    assert last == only_last
    # The last-user count must be much smaller than the full total.
    assert last < count_tokens(messages)


def test_count_last_user_tokens_zero_when_no_user_message():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "no user here"},
    ]
    assert count_last_user_tokens(messages) == 0


def test_count_last_user_tokens_handles_multimodal_last_user():
    messages = [
        {"role": "user", "content": "older"},
        {"role": "user", "content": [
            {"type": "text", "text": "describe this image"},
            {"type": "image_url", "image_url": {"url": "data:..."}},
        ]},
    ]
    last = count_last_user_tokens(messages)
    expected = count_tokens([{"role": "user", "content": "describe this image"}])
    assert last == expected


def test_is_owui_task_request_detects_title_generation():
    # Captured payload prefix from a real OWUI Title Generation call.
    messages = [
        {"role": "system", "content": "User local date and time: 2026-05-16"},
        {"role": "user", "content": (
            "### Task:\nGenerate a concise, 3-5 word title with an emoji "
            "summarizing the chat history.\n### Guidelines:\n- The title "
            "should clearly represent the main theme..."
        )},
    ]
    assert is_owui_task_request(messages) is True


def test_is_owui_task_request_detects_tags_generation():
    messages = [
        {"role": "user", "content": "### Task:\nGenerate 1-3 broad tags categorizing this chat."},
    ]
    assert is_owui_task_request(messages) is True


def test_is_owui_task_request_tolerates_leading_whitespace():
    # Defensive: some OWUI templates may emit leading newlines/spaces.
    messages = [
        {"role": "user", "content": "\n\n  ### Task:\nGenerate a concise title"},
    ]
    assert is_owui_task_request(messages) is True


def test_is_owui_task_request_ignores_normal_user_message():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "what is the current temperature in istanbul"},
    ]
    assert is_owui_task_request(messages) is False


def test_is_owui_task_request_ignores_task_keyword_inside_message():
    # The header must be the *prefix* — a user mentioning "task" mid-message
    # shouldn't trigger a false positive.
    messages = [
        {"role": "user", "content": "My next task is to write a report. ### Task:..."},
    ]
    assert is_owui_task_request(messages) is False


def test_is_owui_task_request_only_checks_latest_user():
    # An earlier user turn was an OWUI task, but the most recent isn't —
    # follow-ups in a real conversation must not be misclassified.
    messages = [
        {"role": "user", "content": "### Task:\nGenerate a concise title"},
        {"role": "assistant", "content": "Some Title"},
        {"role": "user", "content": "how many plane rides to pitcairn island"},
    ]
    assert is_owui_task_request(messages) is False


def test_is_owui_task_request_handles_multimodal_user_content():
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "### Task:\nGenerate tags"},
        ]},
    ]
    assert is_owui_task_request(messages) is True


def test_is_owui_task_request_no_user_message_returns_false():
    messages = [{"role": "system", "content": "sys"}]
    assert is_owui_task_request(messages) is False


def test_is_complex_returns_tuple():
    messages = [{"role": "user", "content": "short"}]
    complex_, n = is_complex(messages, threshold=500)
    assert complex_ is False
    assert isinstance(n, int)
    assert n > 0
