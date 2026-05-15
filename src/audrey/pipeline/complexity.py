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


def is_complex(messages: list[dict], *, threshold: int) -> tuple[bool, int]:
    """Returns `(complex?, token_count)`. Complex prompts route to deep panel."""
    n = count_tokens(messages)
    return n >= threshold, n


__all__ = ["count_tokens", "count_tokens_by_role", "is_complex"]
