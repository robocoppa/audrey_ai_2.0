#!/usr/bin/env python3
"""Measure whether nomic's task prefixes change retrieval scores.

`nomic-embed-text` is trained for *asymmetric* retrieval: stored passages are
meant to carry a `search_document: ` prefix and queries a `search_query: ` one.
`kb/embed.py` sends raw text for both. This script measures what that costs,
so the decision to re-embed the whole KB rests on a number rather than on a
claim about how the model was trained.

It talks to Ollama directly — no Audrey, no Qdrant, nothing stored. Run it
from the LAPTOP; Ollama publishes 11434 on the box.

    python3 scripts/embed_prefix_probe.py \\
        --query "and watch us play some baseball" \\
        --passage "$(sed -n '5,15p' some-transcript.txt)"

Reading the output: the gap between the two cosines is the entire finding. If
prefixed scores materially higher on the *same* text, the current setup is
leaving retrieval quality on the table and the fix is a re-embed. If they are
within noise, the prefixes are not the problem and the search should continue
elsewhere.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request

DEFAULT_HOST = "http://192.168.1.11:11434"
DEFAULT_MODEL = "nomic-embed-text"


def embed(host: str, model: str, texts: list[str], timeout: int = 60) -> list[list[float]]:
    if not host.startswith(("http://", "https://")):
        raise ValueError(f"host must be http:// or https://, got {host!r}")
    request = urllib.request.Request(  # noqa: S310 - scheme checked above
        host.rstrip("/") + "/api/embed",
        data=json.dumps({"model": model, "input": texts}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - scheme checked above
        return json.loads(response.read())["embeddings"]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--query", required=True)
    parser.add_argument("--passage", required=True,
                        help="a chunk that genuinely contains the answer")
    parser.add_argument("--decoy", default=None,
                        help="unrelated text; the margin over this is what "
                             "retrieval actually depends on")
    args = parser.parse_args(argv)

    plain = embed(args.host, args.model, [args.query, args.passage])
    tagged = embed(args.host, args.model, [
        f"search_query: {args.query}",
        f"search_document: {args.passage}",
    ])

    plain_score = cosine(plain[0], plain[1])
    tagged_score = cosine(tagged[0], tagged[1])

    print(f"model: {args.model}   host: {args.host}")
    print(f"query:   {args.query[:70]!r}")
    print(f"passage: {args.passage[:70]!r}")
    print()
    print(f"  no prefixes (what Audrey does today) : {plain_score:.4f}")
    print(f"  with nomic task prefixes             : {tagged_score:.4f}")
    print(f"  delta                                : {tagged_score - plain_score:+.4f}")

    if args.decoy:
        plain_d = embed(args.host, args.model, [args.query, args.decoy])
        tagged_d = embed(args.host, args.model, [
            f"search_query: {args.query}",
            f"search_document: {args.decoy}",
        ])
        pd = cosine(plain_d[0], plain_d[1])
        td = cosine(tagged_d[0], tagged_d[1])
        print()
        print("  against the decoy (should score LOW):")
        print(f"    no prefixes : {pd:.4f}   margin over decoy: {plain_score - pd:+.4f}")
        print(f"    prefixed    : {td:.4f}   margin over decoy: {tagged_score - td:+.4f}")
        print()
        # The margin is what matters, not the absolute score. A floor can be
        # moved; an embedding that puts the right answer and an unrelated one
        # the same distance from the query cannot be rescued by tuning.
        print("  the margin is the finding — a floor can be re-tuned, a")
        print("  collapsed margin cannot.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
