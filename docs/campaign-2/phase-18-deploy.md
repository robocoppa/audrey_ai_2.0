# Campaign 2 Phase 18 — Concurrent tool-server discovery

A small startup optimization: discover all configured tool servers in
parallel instead of one after another. Low impact today (Audrey runs a single
`custom-tools` server), but it future-proofs multi-server setups and removes a
sequential `await`-in-a-loop.

**No runtime-path change — startup only.**

## What it does

`discover_all` looped over the configured server URLs and `await`ed
`discover_one` for each, so every server blocked the next. This phase fires
all the `discover_one` calls concurrently with `asyncio.gather`, then folds
the results into the registry **in `server_urls` order** so the existing
"later names win on collision" contract is preserved.

```text
before:  for url in servers: tools = await discover_one(url)   # serial
after:   results = await gather(*[discover_one(url) for url in servers])
         for url, tools in zip(servers, results): fold in order
```

## Why this exists

Discovery happens at boot (and on `POST /v1/tools/rediscover`). With one
server the difference is nil, but the sequential shape is a latent cost the
moment a second tool server is added — each unreachable server would add its
full timeout to boot serially. `discover_one` already returns `[]` on any
error and never raises, so `gather` needs no `return_exceptions=True` and the
failure handling is unchanged.

## What's in scope

- **[`src/audrey/tools/discovery.py`](../../src/audrey/tools/discovery.py)** —
  `discover_all` rewritten to gather concurrently and fold in input order.
  Per-server and final-total log lines preserved. `discover_one` unchanged.
- **[`tests/test_discovery.py`](../../tests/test_discovery.py)** — a two-server
  collision test asserting the later-listed server still wins after the
  reorder (the one behavior the change could threaten).

## Behavior invariant

- **Single server** (today's reality): byte-identical — one coro, same result,
  same registry.
- **Multiple servers**: collision resolution is unchanged because results are
  still folded in `server_urls` order; only the *network fetch* now overlaps.
  `gather` preserves input order in its result list, so the zip stays aligned.

## What's NOT in scope

- No change to *what* gets discovered (tag filtering, `$ref` inlining,
  keyword scrubbing are all untouched).
- No change to the rediscover route's contract.

## Deploy on Unraid

No config or custom-tools change. From `/mnt/user/appdata/audrey_ai_2.0`:

```
docker compose up -d --build audrey-ai
docker compose logs -f audrey-ai
```

## Verification

Hermetic (laptop): full suite green; the new collision test passes; ruff clean
on `discovery.py`.

Live, on the box:

1. Boot logs still show the tools discovered exactly as before:

   ```
   docker logs audrey-ai 2>&1 | grep -E "discovery:"
   ```

   Expect the same per-server line and the same `total N tool(s) registered`
   line (7 tools incl. `chat_history_search`, per last verified stack state).

2. If `tools=0` after boot (custom-tools was late), the admin rediscover route
   still rehydrates:

   ```
   curl -X POST http://localhost:<port>/v1/tools/rediscover   # admin-gated
   ```

## What this unblocks

Adding a second tool server later won't serialize boot-time discovery. Closes
Item 3 of the 2026-06-23 optimization review (`optimization-pass-plan.md`),
which had originally been deferred until a second server existed — pulled
forward because the change is trivial and self-contained.
