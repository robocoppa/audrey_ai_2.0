"""Pure normalization helpers for automatic conversation titles."""

from __future__ import annotations

import re

AUTOMATIC_TITLE_MAX_CHARS = 72
_GENERATED_TITLE_MAX_WORDS = 10


def fallback_conversation_title(user_content: str) -> str:
    """Build a stable title from the prompt when semantic generation fails."""

    title = " ".join(str(user_content).split())
    title = re.sub(r"^(?:#{1,6}\s+|>\s+|[-+*]\s+)", "", title).strip()
    title = title.strip("`*_\"'") or "New conversation"
    return _fit_title(title, ellipsis=True)


def normalize_generated_title(value: str, *, fallback: str) -> str:
    """Reduce an untrusted model reply to one short display-safe title."""

    lines = [
        line.strip()
        for line in str(value).splitlines()
        if line.strip() and not line.strip().startswith("```")
    ]
    if not lines:
        return fallback

    title = re.sub(
        r"^(?:conversation\s+title|title)\s*:\s*",
        "",
        lines[0],
        flags=re.IGNORECASE,
    )
    title = re.sub(r"^(?:#{1,6}\s+|>\s+|[-+*]\s+)", "", title).strip()
    title = " ".join(title.strip("`*_#\"'“”‘’ ").split())
    title = title.rstrip(".!?;:,–—-").strip()
    if len(title) < 2:
        return fallback

    words = title.split()
    if len(words) > _GENERATED_TITLE_MAX_WORDS:
        title = " ".join(words[:_GENERATED_TITLE_MAX_WORDS])
    return _fit_title(title, ellipsis=False)


def _fit_title(value: str, *, ellipsis: bool) -> str:
    if len(value) <= AUTOMATIC_TITLE_MAX_CHARS:
        return value

    reserve = 1 if ellipsis else 0
    prefix = value[: AUTOMATIC_TITLE_MAX_CHARS - reserve]
    word_boundary = prefix.rsplit(" ", 1)[0].rstrip(" ,.;:-")
    if len(word_boundary) >= AUTOMATIC_TITLE_MAX_CHARS // 2:
        prefix = word_boundary
    prefix = prefix.rstrip()
    return f"{prefix}…" if ellipsis else prefix


__all__ = [
    "AUTOMATIC_TITLE_MAX_CHARS",
    "fallback_conversation_title",
    "normalize_generated_title",
]
