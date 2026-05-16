"""Complexity gate — token-count check against `COMPLEXITY_TOKEN_THRESHOLD`.

Prompts above the threshold skip fast path and go straight to deep panel,
because a long paste almost always benefits from multi-draft synthesis over
a single-model answer.

Uses tiktoken's `cl100k_base` as a universal-ish tokenizer. Ollama models
use their own tokenizers, but for a rough "is this a big prompt?" gate
cl100k_base is accurate enough (within ~15%) and cheap.
"""

from __future__ import annotations

from functools import lru_cache

import tiktoken


@lru_cache(maxsize=1)
def _encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def _count_message_tokens(m: dict, enc: tiktoken.Encoding) -> int:
    c = m.get("content")
    if isinstance(c, str):
        return len(enc.encode(c))
    # Multimodal content (list of parts) — only text parts contribute;
    # image bytes don't have meaningful token counts at this layer.
    if isinstance(c, list):
        n = 0
        for part in c:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                n += len(enc.encode(part["text"]))
        return n
    return 0


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
            content = m.get("content")
            if isinstance(content, str):
                return content.lstrip().startswith(_OWUI_TASK_PREFIX)
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        return part["text"].lstrip().startswith(_OWUI_TASK_PREFIX)
            return False
    return False


def is_complex(messages: list[dict], *, threshold: int) -> tuple[bool, int]:
    """Returns `(complex?, token_count)`. Complex prompts route to deep panel."""
    n = count_tokens(messages)
    return n >= threshold, n


__all__ = [
    "count_last_user_tokens",
    "count_tokens",
    "count_tokens_by_role",
    "is_complex",
    "is_owui_task_request",
]
