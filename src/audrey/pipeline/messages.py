"""Shared message-navigation helpers.

The OpenAI-compatible chat-completions content field can be a plain
string OR a list of typed parts (`[{"type": "text", "text": "..."}, ...]`)
for multi-modal turns. Callers across the pipeline (classify, planner,
synth, route) need the text-only view of the most recent user turn —
they all reach in here so the reverse-iterate-and-flatten logic lives
in one place. New content-part shapes (tool-use blocks, ref-image
attachments, etc.) get taught here.
"""

from __future__ import annotations

from typing import Any


def last_user_text(messages: list[dict[str, Any]]) -> str:
    """Return the text of the most recent `role: "user"` message.

    Walks from the end backward, returning the first user turn's content.
    String content is returned as-is. List content (multi-modal) is
    flattened by joining every `{"text": ...}` part with `\\n`; non-dict
    parts and dicts without a `text` key contribute nothing. Returns the
    empty string if no user turn exists or if every text part was missing.
    """
    for m in reversed(messages):
        if m.get("role") != "user":
            continue
        content = m.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


__all__ = ["last_user_text"]
