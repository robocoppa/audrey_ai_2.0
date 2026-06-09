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


def has_image_part(messages: list[dict[str, Any]]) -> bool:
    """True if the most recent `role: "user"` turn carries an image.

    OWUI (and any OpenAI-compatible client) sends an attached image as a
    multimodal `content` list with an `{"type": "image_url", ...}` part
    alongside the text part. Detecting this lets the gate force such turns
    onto the vision (`vl`) pool — neutral wording like "describe this"
    wouldn't trip the text-keyword classifier on its own.

    Only the latest user turn matters: that's the turn being answered. A
    plain-string content (the ordinary text case) is never an image turn.
    """
    for m in reversed(messages):
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, list):
            return any(
                isinstance(p, dict) and p.get("type") == "image_url" for p in content
            )
        return False
    return False


__all__ = ["last_user_text", "has_image_part"]
