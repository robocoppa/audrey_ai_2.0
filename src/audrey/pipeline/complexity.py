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


def count_tokens(messages: list[dict]) -> int:
    """Sum token counts across every message's `content`. System + tool messages count too."""
    enc = _encoder()
    total = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            total += len(enc.encode(c))
        # Multimodal content (list of parts) — we only count the text parts;
        # image bytes don't have meaningful token counts at this layer.
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    total += len(enc.encode(part["text"]))
    return total


def is_complex(messages: list[dict], *, threshold: int) -> tuple[bool, int]:
    n = count_tokens(messages)
    return n >= threshold, n


__all__ = ["count_tokens", "is_complex"]
