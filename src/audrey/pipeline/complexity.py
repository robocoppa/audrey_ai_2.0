"""Complexity gate — token-count check against `COMPLEXITY_TOKEN_THRESHOLD`.

Prompts above the threshold skip fast path and go straight to deep panel,
because a long paste almost always benefits from multi-draft synthesis over
a single-model answer.

Uses tiktoken's `cl100k_base` as a universal-ish tokenizer. Ollama models
use their own tokenizers, but for a rough "is this a big prompt?" gate
cl100k_base is accurate enough (within ~15%) and cheap.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache

import tiktoken


# Lines starting with `>` (Audrey banner markup) or a bare `---`
# (banner-to-body / footer-to-body separator) are UI cruft, not part of
# the model's actual response. They are emitted by `pipeline/banners.py`
# into the streamed assistant content; when OWUI sends that assistant
# text back as conversation history, the cruft inflates the complexity
# gate's view. Strip both before counting tokens on assistant messages.
def _strip_audrey_markup(text: str) -> str:
    out_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(">"):
            continue
        if stripped == "---":
            continue
        out_lines.append(line)
    return "\n".join(out_lines)


@lru_cache(maxsize=1)
def _encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def _iter_text_parts(content: object) -> Iterable[str]:
    """Yield the text of a message's `content`, in order.

    Messages carry `content` as either a plain string or a multimodal list of
    parts (`{"type": "text", "text": ...}`, image parts, …). This yields the
    string for a `str` content, or each part's `text` for a list — and nothing
    at all for an image part or any unexpected shape (the token gate treats an
    unreadable message as contributing nothing). Both the token counter and the
    OWUI-task sniffer walk content the same way; this is the one place that
    knows its shape.
    """
    if isinstance(content, str):
        yield content
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                yield part["text"]


def _count_message_tokens(m: dict, enc: tiktoken.Encoding) -> int:
    role = m.get("role")
    n = 0
    for text in _iter_text_parts(m.get("content")):
        if role == "assistant":
            text = _strip_audrey_markup(text)
        n += len(enc.encode(text))
    return n


def count_tokens(messages: list[dict]) -> int:
    """Sum token counts across every message's `content`. System + tool messages count too."""
    enc = _encoder()
    return sum(_count_message_tokens(m, enc) for m in messages)


def count_tokens_by_role(messages: list[dict]) -> dict[str, int]:
    """Per-role token sums; unknown/missing roles bucket under `"other"`."""
    enc = _encoder()
    totals: dict[str, int] = {}
    for m in messages:
        role = m.get("role") or "other"
        totals[role] = totals.get(role, 0) + _count_message_tokens(m, enc)
    return totals


def count_last_user_tokens(messages: list[dict]) -> int:
    """Tokens in the most recent `role: "user"` message.

    Phase 6a breakdown signal — distinguishes "what's being asked right now"
    from accumulated conversation history. Returns 0 if no user message.
    """
    enc = _encoder()
    for m in reversed(messages):
        if m.get("role") == "user":
            return _count_message_tokens(m, enc)
    return 0


# OWUI utility tasks (Title Generation, Tags, Follow Up, Autocomplete, Retrieval
# Query rewrite, Image Prompt, Web Search Query) all submit a single user
# message that opens with this exact header. They are issued by OWUI itself,
# not by a human asking a question, and they ship the entire chat history
# bundled into the body — which trips Audrey's complexity gate to deep even
# though the work is a short-output utility task. Detect by prefix and force
# fast in both the streaming and non-streaming gates.
_OWUI_TASK_PREFIX = "### Task:"


def is_owui_task_request(messages: list[dict]) -> bool:
    """True when the latest user message is an OWUI-generated utility prompt."""
    for m in reversed(messages):
        if m.get("role") == "user":
            first_text = next(_iter_text_parts(m.get("content")), None)
            if first_text is None:
                return False
            return first_text.lstrip().startswith(_OWUI_TASK_PREFIX)
    return False


def is_complex(messages: list[dict], *, threshold: int) -> tuple[bool, int]:
    """Returns `(complex?, token_count)`. Complex prompts route to deep panel."""
    n = count_tokens(messages)
    return n >= threshold, n


def has_deep_intent(messages: list[dict], phrases: list[str]) -> bool:
    """True when the latest user message asks for depth, by phrase match.

    The token-count gate (`is_complex`) catches *long* prompts but misses the
    "short prompt, large task" case — e.g. a 30-token "compare the top 3 X,
    concise but thorough, think hard." Those read as trivial to a length gate
    yet genuinely want multi-draft synthesis. This matches explicit depth cues
    ("think hard", "in depth", "step by step", …) in the latest user turn so
    `audrey_auto` can escalate them to deep regardless of length.

    Substring match on the lowercased user text — phrases are written as the
    literal cue to look for. Empty `phrases` disables the check (returns
    False), so the feature is off unless `complexity.deep_intent_phrases` is
    configured. Only the latest user turn is inspected, mirroring
    `is_owui_task_request`: a depth cue three turns ago shouldn't force every
    later short follow-up to deep.
    """
    if not phrases:
        return False
    for m in reversed(messages):
        if m.get("role") == "user":
            text = " ".join(_iter_text_parts(m.get("content"))).lower()
            return any(p.lower() in text for p in phrases if p)
    return False


__all__ = [
    "count_last_user_tokens",
    "count_tokens",
    "count_tokens_by_role",
    "has_deep_intent",
    "is_complex",
    "is_owui_task_request",
]
