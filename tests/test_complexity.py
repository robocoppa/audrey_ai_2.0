"""Tests for the complexity gate's token counters."""
from __future__ import annotations

from audrey.pipeline.complexity import (
    count_last_user_tokens,
    count_tokens,
    count_tokens_by_role,
    has_deep_intent,
    is_complex,
    is_owui_task_request,
)

_PHRASES = ["think hard", "deep dive", "comprehensive", "thorough", "step by step"]


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


def test_count_tokens_strips_audrey_banner_from_assistant_history():
    # Real-world assistant message: banner header, separator, real content,
    # tools-used footer. Only the real content should be counted.
    banner_response = (
        "> _Thinking_....... ✅\n"
        "\n"
        "---\n"
        "\n"
        "The next FIFA World Cup will take place from June 11 to July 19, 2026, "
        "and it will be jointly hosted by the United States, Canada, and Mexico.\n"
        "\n"
        "---\n"
        "> _Tools used:_\n"
        "> - **qwen3.6:35b** — `web_search`\n"
    )
    real_content = (
        "\n"
        "\n"
        "\n"
        "\n"
        "The next FIFA World Cup will take place from June 11 to July 19, 2026, "
        "and it will be jointly hosted by the United States, Canada, and Mexico.\n"
        "\n"
        "\n"
    )
    with_banner = count_tokens([{"role": "assistant", "content": banner_response}])
    without_banner = count_tokens([{"role": "assistant", "content": real_content}])
    assert with_banner == without_banner


def test_count_tokens_does_not_strip_blockquote_from_user_messages():
    # Stripping is assistant-only. A user paste with `>` quote characters
    # should keep its content in the gate input.
    quoted_user = "> please ignore the banner format below\n> _Thinking_ this is a user paste"
    n_user = count_tokens([{"role": "user", "content": quoted_user}])
    # Same content under assistant role would be stripped to empty.
    n_assistant = count_tokens([{"role": "assistant", "content": quoted_user}])
    assert n_user > 0
    assert n_assistant == 0
    assert n_user > n_assistant


def test_count_tokens_strips_audrey_planning_and_dispatch_banners():
    # Deep-mode response has multiple banners stacked.
    deep_response = (
        "> _Planning_..... ✅\n"
        "> _Dispatching panel_..  ✅ kimi-k2.6:cloud\n"
        "> _Synthesizing_..... ✅\n"
        "\n"
        "---\n"
        "\n"
        "Real synthesized answer.\n"
    )
    n = count_tokens([{"role": "assistant", "content": deep_response}])
    real_only = count_tokens([{"role": "assistant", "content": "\n\n\n\nReal synthesized answer.\n"}])
    assert n == real_only


def test_count_tokens_by_role_strips_banner_from_assistant_share():
    banner_assistant = (
        "> _Thinking_..... ✅\n\n---\n\nReal model output that should be counted.\n"
    )
    plain_assistant = "\n\n\nReal model output that should be counted.\n"
    by_role_with = count_tokens_by_role([{"role": "assistant", "content": banner_assistant}])
    by_role_clean = count_tokens_by_role([{"role": "assistant", "content": plain_assistant}])
    assert by_role_with["assistant"] == by_role_clean["assistant"]


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


# ─── has_deep_intent (Phase 22 — short-but-demanding prompts) ─────────


def test_deep_intent_matches_the_reported_prompts():
    """The two short prompts that wrongly went to fast now trip the gate."""
    math = [{"role": "user", "content": (
        "List the top 3 most influential mathematical theorems and give me a "
        "concise but thorough explanation of them. Think hard about this"
    )}]
    philo = [{"role": "user", "content": (
        "Give me a comprehensive deep dive into the lost works of philosophy "
        "of the great masters like aristotle and plato"
    )}]
    assert has_deep_intent(math, _PHRASES) is True     # "thorough" / "think hard"
    assert has_deep_intent(philo, _PHRASES) is True    # "comprehensive" / "deep dive"


def test_deep_intent_is_case_insensitive():
    msg = [{"role": "user", "content": "THINK HARD and be COMPREHENSIVE"}]
    assert has_deep_intent(msg, _PHRASES) is True


def test_deep_intent_false_for_casual_short_prompt():
    msg = [{"role": "user", "content": "what is the capital of france"}]
    assert has_deep_intent(msg, _PHRASES) is False


def test_deep_intent_empty_phrases_disables_feature():
    """No configured phrases → always False, even on an obvious depth prompt."""
    msg = [{"role": "user", "content": "think hard and be comprehensive"}]
    assert has_deep_intent(msg, []) is False


def test_deep_intent_only_checks_latest_user_message():
    """A depth cue in an earlier turn must not force a later short follow-up."""
    messages = [
        {"role": "user", "content": "give me a comprehensive deep dive"},
        {"role": "assistant", "content": "...long answer..."},
        {"role": "user", "content": "thanks, and the capital of spain?"},
    ]
    assert has_deep_intent(messages, _PHRASES) is False


def test_deep_intent_matches_across_multimodal_text_parts():
    msg = [{"role": "user", "content": [
        {"type": "text", "text": "please be"},
        {"type": "text", "text": "thorough about this"},
    ]}]
    assert has_deep_intent(msg, _PHRASES) is True


def test_deep_intent_no_user_message_returns_false():
    assert has_deep_intent([{"role": "system", "content": "think hard"}], _PHRASES) is False
