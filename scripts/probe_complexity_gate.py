#!/usr/bin/env python3
"""Replay Audrey's complexity gate against archived chat turns.

WHAT THIS DOES

Audrey routes every chat turn through a complexity gate
(`src/audrey/pipeline/complexity.py`) that sums tiktoken tokens across
every message and compares the total to `complexity.token_threshold`
(default 500). Anything at or above the threshold goes to deep mode.

Phase 6 testing surfaced a suspected bug: tool conversations bloat the
message list with `role: "tool"` results, pushing follow-up turns past
the threshold even when the user's intent is identical to a turn that
routed fast a moment earlier.

This script replays the gate against the chat archive SQLite to
quantify the effect. For every assistant turn in the archive, it
reconstructs the message list as it would have looked when classify
ran (everything in the conversation up to and including the user
message that triggered the assistant turn) and computes:

    current      - tiktoken sum across all messages   (today's behavior)
    proposed_B   - tiktoken sum skipping role:"tool"  (Option B fix)

Then it tabulates how many turns would flip from deep to fast under
Option B, with a per-user breakdown and a token-count histogram for
sizing Option A (a threshold raise).

WHAT THIS DOES NOT DO

- It does not modify Audrey, the archive, or any config. Pure read-only.
- It does not attempt to classify or route the turn. Only the complexity
  gate is replayed.
- It does not include the system prompt or tool definitions that audrey
  would have prepended at runtime — those are not stored in the archive.
  The numbers will be lower than what audrey actually saw, by a roughly
  constant offset per turn. The *relative* delta between `current` and
  `proposed_B` is unaffected.

USAGE

    # Copy the archive off Unraid first:
    docker compose cp custom-tools:/data/chat_archive.sqlite /tmp/chat_archive.sqlite

    # Default threshold (500), all users:
    python3 scripts/probe_complexity_gate.py /tmp/chat_archive.sqlite

    # Per-user breakdown for a specific threshold:
    python3 scripts/probe_complexity_gate.py /tmp/chat_archive.sqlite --threshold 500 --per-user

    # Dump the would-flip turns (for Step 4 eyeball pass):
    python3 scripts/probe_complexity_gate.py /tmp/chat_archive.sqlite --dump-flipped

OUTPUT

Three sections:

  1. Aggregate counts (turns total, deep today, deep under B, flipped).
  2. Histogram of `current` token counts (bucket size 250).
  3. Optional per-user breakdown.
  4. Optional list of flipped turns (conversation_id, message_id,
     user-prompt preview) for the eyeball pass — PII-bearing, do not
     commit this output.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

try:
    import tiktoken
except ImportError:
    print(
        "tiktoken not installed. Run: pip install tiktoken",
        file=sys.stderr,
    )
    sys.exit(2)


DEFAULT_THRESHOLD = 500
HISTOGRAM_BUCKET = 250


def _encoder():
    return tiktoken.get_encoding("cl100k_base")


def _count(messages: list[tuple[str, str]], *, skip_roles: set[str]) -> int:
    """Sum tokens across (role, content) pairs, skipping the given roles.

    Mirrors `pipeline/complexity.py:count_tokens` for the `str` content
    path. Archive content is always a string (no multimodal parts), so
    the list-of-parts branch from the runtime impl is not needed here.
    """
    enc = _encoder()
    total = 0
    for role, content in messages:
        if role in skip_roles:
            continue
        if isinstance(content, str):
            total += len(enc.encode(content))
    return total


def _bucket(n: int) -> int:
    return (n // HISTOGRAM_BUCKET) * HISTOGRAM_BUCKET


def replay(db_path: Path, *, threshold: int) -> dict:
    """Walk the archive in conversation order and replay the gate.

    Returns aggregate stats plus a list of flipped turns.

    Each "turn" here is one user message in the archive. The
    conversation history *up to and including* that user message is
    what classify would have seen at runtime, so that's what we count.
    Assistant and tool messages that came after this user message do
    not contribute to this turn's gate decision.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"archive not found: {db_path}")

    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    # Pull every message in conversation order. We walk conversations
    # one at a time and emit one "turn" per user message.
    rows = cur.execute(
        """
        SELECT conversation_id, user, role, content, created_at, message_id
        FROM messages
        ORDER BY conversation_id, created_at, message_id
        """
    ).fetchall()

    turns_total = 0
    turns_deep_today = 0
    turns_deep_under_b = 0
    flipped: list[dict] = []
    histogram: dict[int, int] = defaultdict(int)
    per_user: dict[str, dict[str, int]] = defaultdict(
        lambda: {"turns": 0, "deep_today": 0, "deep_under_b": 0, "flipped": 0}
    )

    # Group by conversation, walking chronologically inside each.
    current_conv = None
    history: list[tuple[str, str]] = []
    for r in rows:
        conv_id = r["conversation_id"]
        if conv_id != current_conv:
            current_conv = conv_id
            history = []

        history.append((r["role"], r["content"] or ""))

        # Only user messages trigger a classify/complexity decision.
        if r["role"] != "user":
            continue

        cur_tokens = _count(history, skip_roles=set())
        b_tokens = _count(history, skip_roles={"tool"})

        deep_today = cur_tokens >= threshold
        deep_under_b = b_tokens >= threshold

        turns_total += 1
        if deep_today:
            turns_deep_today += 1
        if deep_under_b:
            turns_deep_under_b += 1
        if deep_today and not deep_under_b:
            preview = (r["content"] or "")[:120].replace("\n", " ")
            flipped.append({
                "conversation_id": conv_id,
                "message_id": r["message_id"],
                "user": r["user"],
                "current": cur_tokens,
                "proposed_b": b_tokens,
                "preview": preview,
            })

        histogram[_bucket(cur_tokens)] += 1

        u = per_user[r["user"]]
        u["turns"] += 1
        if deep_today:
            u["deep_today"] += 1
        if deep_under_b:
            u["deep_under_b"] += 1
        if deep_today and not deep_under_b:
            u["flipped"] += 1

    con.close()

    return {
        "threshold": threshold,
        "turns_total": turns_total,
        "turns_deep_today": turns_deep_today,
        "turns_deep_under_b": turns_deep_under_b,
        "flipped_count": len(flipped),
        "flipped": flipped,
        "histogram": dict(sorted(histogram.items())),
        "per_user": dict(per_user),
    }


def _pct(part: int, whole: int) -> str:
    if whole == 0:
        return "0.0%"
    return f"{100.0 * part / whole:.1f}%"


def render(stats: dict, *, per_user: bool, dump_flipped: bool) -> None:
    t = stats["threshold"]
    tot = stats["turns_total"]
    today = stats["turns_deep_today"]
    under_b = stats["turns_deep_under_b"]
    flipped = stats["flipped_count"]

    print(f"complexity-gate replay (threshold={t})")
    print("=" * 50)
    print(f"turns total           : {tot}")
    print(f"deep today            : {today:>5}  ({_pct(today, tot)})")
    print(f"deep under Option B   : {under_b:>5}  ({_pct(under_b, tot)})")
    print(f"flipped (deep->fast)  : {flipped:>5}  ({_pct(flipped, today)} of deep)")
    print()

    print("histogram of `current` token counts")
    print("-" * 50)
    if stats["histogram"]:
        max_count = max(stats["histogram"].values())
    else:
        max_count = 1
    for bucket, count in stats["histogram"].items():
        bar_width = int(40 * count / max_count) if max_count else 0
        marker = "*" if bucket >= t else " "
        print(f"  {bucket:>5}-{bucket + HISTOGRAM_BUCKET - 1:>5} {marker} {'#' * bar_width:<40} {count}")
    print(f"  (marker * indicates bucket is at or above threshold {t})")
    print()

    if per_user:
        print("per-user breakdown")
        print("-" * 50)
        print(f"  {'user':<40} {'turns':>6} {'deep':>6} {'flip':>6}")
        for user, u in sorted(stats["per_user"].items()):
            print(
                f"  {user:<40} "
                f"{u['turns']:>6} "
                f"{u['deep_today']:>6} "
                f"{u['flipped']:>6}"
            )
        print()

    if dump_flipped:
        print("flipped turns (PII — do not commit this output)")
        print("-" * 50)
        for f in stats["flipped"]:
            print(
                f"  {f['user']:<25} "
                f"cur={f['current']:>4} B={f['proposed_b']:>4} "
                f"conv={f['conversation_id'][:8]} "
                f"msg={f['message_id'][:8]} "
                f"| {f['preview']}"
            )


def main() -> int:
    p = argparse.ArgumentParser(
        description="Replay Audrey's complexity gate against archived chat turns."
    )
    p.add_argument("archive", type=Path, help="path to chat_archive.sqlite")
    p.add_argument(
        "--threshold",
        type=int,
        default=DEFAULT_THRESHOLD,
        help=f"complexity.token_threshold to replay (default {DEFAULT_THRESHOLD})",
    )
    p.add_argument(
        "--per-user",
        action="store_true",
        help="show per-user breakdown",
    )
    p.add_argument(
        "--dump-flipped",
        action="store_true",
        help="list flipped turns with user-prompt previews (PII-bearing)",
    )
    args = p.parse_args()

    try:
        stats = replay(args.archive, threshold=args.threshold)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    render(stats, per_user=args.per_user, dump_flipped=args.dump_flipped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
