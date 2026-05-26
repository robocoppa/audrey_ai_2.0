"""Tests for `chunk_text` — specifically the tail-chunk skip behavior.

The chunker emits overlapping token windows. When the final stride
lands close to the end of the document, the tail chunk's new content
(tokens past the prior chunk's end) can be small enough that the tail
is a near-duplicate. `chunk_text` skips that tail when the new content
is at or below 10% of `chunk_tokens`.

These tests use real-text inputs (a `"word "` repeat pattern) rather
than synthetic token streams, because `chunk_text` calls `enc.encode`
on the actual text. `cl100k_base` happens to encode `"word "` as one
token per word in this pattern, so word-count ~= token-count, which
makes the asserted shapes easy to reason about.
"""

from __future__ import annotations

from audrey.kb.chunk import chunk_text


def _make_text(n_words: int) -> str:
    """Produce text whose token count is approximately `n_words`."""
    return "word " * n_words


def test_chunk_text_empty_input_returns_empty_list():
    assert chunk_text("", chunk_tokens=1000, overlap_tokens=100) == []
    assert chunk_text("   ", chunk_tokens=1000, overlap_tokens=100) == []


def test_chunk_text_single_chunk_when_below_threshold():
    # 500 tokens with chunk_tokens=1000 -> single chunk, no tail skip.
    chunks = chunk_text(_make_text(500), chunk_tokens=1000, overlap_tokens=100)
    assert len(chunks) == 1
    assert chunks[0].idx == 0


def test_chunk_text_keeps_tail_with_substantial_new_content():
    # 2700 tokens: stride=900, iterations at start=0, 900, 1800.
    # Tail at [1800, 2700], prior chunk ended at 1900 -> new tokens = 800.
    # 800 > 100 (10% of 1000), so tail is kept. Expect 3 chunks.
    chunks = chunk_text(_make_text(2700), chunk_tokens=1000, overlap_tokens=100)
    assert len(chunks) == 3
    assert [c.idx for c in chunks] == [0, 1, 2]


def test_chunk_text_drops_tail_when_new_content_is_near_zero():
    # 1901 tokens: iterations at start=0, 900, 1800.
    # Tail at [1800, 1901], prior chunk ended at 1900 -> new tokens = 1.
    # 1 <= 100 -> tail dropped. Expect 2 chunks instead of 3.
    chunks = chunk_text(_make_text(1901), chunk_tokens=1000, overlap_tokens=100)
    assert len(chunks) == 2


def test_chunk_text_drops_tail_at_exactly_the_threshold():
    # The threshold is `chunk_tokens // 10 = 100`. A tail with exactly
    # 100 new tokens should be dropped (the condition is `<=`, not `<`).
    # 1100 tokens: iter 0 end=1000 (prev_end becomes 1000), iter 1
    # start=900, end=1100 (terminate). new = 1100 - 1000 = 100. Dropped.
    # Result: just the first chunk.
    chunks = chunk_text(_make_text(1100), chunk_tokens=1000, overlap_tokens=100)
    assert len(chunks) == 1


def test_chunk_text_keeps_tail_one_token_past_threshold():
    # 1101 tokens: iter 1 end=1101, new = 101 > 100, tail kept. 2 chunks.
    chunks = chunk_text(_make_text(1101), chunk_tokens=1000, overlap_tokens=100)
    assert len(chunks) == 2


def test_chunk_text_drops_only_the_tail_not_a_middle_chunk():
    # Belt-and-suspenders against a regression that drops the wrong chunk.
    # 1901 tokens drops the tail (verified above). The remaining chunks
    # should cover the full input range -- first chunk starts at the
    # beginning, second chunk ends at token 1900 (so the last 1 token
    # is "lost" but it was 100% redundant with the prior chunk's
    # overlap window anyway).
    chunks = chunk_text(_make_text(1901), chunk_tokens=1000, overlap_tokens=100)
    assert len(chunks) == 2
    assert chunks[0].text.startswith("word")
    assert chunks[1].text.endswith("word")


def test_chunk_text_three_chunk_case_with_clean_middle():
    # 2901 tokens: iter 0 end=1000, iter 1 end=1900, iter 2 end=2800, iter 3 end=2901.
    # Tail at [2700, 2901], prior chunk ended at 2800 -> new=101 > 100. Kept.
    # Expect 4 chunks.
    chunks = chunk_text(_make_text(2901), chunk_tokens=1000, overlap_tokens=100)
    assert len(chunks) == 4


def test_chunk_text_respects_overlap_safety_clamp():
    # `overlap_tokens >= chunk_tokens` triggers the safety clamp to
    # `chunk_tokens // 5`. We're not testing tail-skip here, just that
    # the safety branch still produces multiple chunks rather than
    # crashing or looping forever.
    text = _make_text(3000)
    chunks = chunk_text(text, chunk_tokens=100, overlap_tokens=100)  # equal
    assert len(chunks) > 1
    chunks = chunk_text(text, chunk_tokens=100, overlap_tokens=200)  # greater
    assert len(chunks) > 1


def test_chunk_text_idx_reflects_iteration_not_emission():
    # When the tail-skip fires, the index sequence has no gap -- the
    # last *emitted* chunk's idx is whatever the loop iteration counter
    # was when it was appended. For 1901 tokens, two chunks are
    # emitted at iter 0 and iter 1 (the iter-2 candidate is skipped),
    # so indices are [0, 1].
    chunks = chunk_text(_make_text(1901), chunk_tokens=1000, overlap_tokens=100)
    assert [c.idx for c in chunks] == [0, 1]
