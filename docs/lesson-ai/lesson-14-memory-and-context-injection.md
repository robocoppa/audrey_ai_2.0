# Lesson 14 — Per-user context: identity, memory, and history

**Estimated time:** 45-60 minutes if you keep
[`auth.py`](../../src/audrey/auth.py),
[`pipeline/memory.py`](../../src/audrey/pipeline/memory.py),
[`pipeline/context.py`](../../src/audrey/pipeline/context.py),
[`pipeline/chat_archive.py`](../../src/audrey/pipeline/chat_archive.py), and
[`tools-server/chat_archive.py`](../../tools-server/chat_archive.py) open.

**Goal:** by the end of this lesson, you can answer
*"who is the user, and what does Audrey know about them before the model
sees a single word?"*

Earlier lessons traced the request from FastAPI route to model call,
but glossed over the fact that almost every pipeline function takes a
`user_id` parameter. Where does it come from? Who guarantees it's
real? And what becomes possible once it exists? This lesson opens
that surface.


## 1. Context

A model's reply depends on its prompt. If you ask "what's the latest
version of BTRFS?" with nothing else in context, the answer comes from
training cutoff. If you ask "remind me what city I'm in?" with no
prior conversation in the prompt, the model has no way to answer —
the training data didn't include you.

Audrey solves this by **establishing a user identity at the door** and
then prepending small system messages before the user's turn that the
model wouldn't otherwise have. Identity is the gating fact:

```text
identity         - who the request is from — the gating fact
  └─ memory recall   - this user's durable facts, recalled when relevant
  └─ chat archive    - this user's earlier conversations, search-on-demand
```

Without identity, the other two have nothing to do. With it, every
per-user surface in the system reads from one canonical source: the
storage namespace that `require_user` resolves for every request. The
rest of this lesson is what falls out of that decision.

(One non-identity-gated injection also runs on every request: a
datetime system message — "today is 2026-05-27T14:32:00-07:00, treat
this as the present moment." Two functions in
[`pipeline/context.py`](../../src/audrey/pipeline/context.py), no I/O,
no skip path; the instruction defuses models that would otherwise
reason about the timestamp as data. The node wrapper at
[`graph.py:257`](../../src/audrey/pipeline/graph.py#L257) prepends it
to `state["messages"]`. That's the whole subsystem; the rest of the
lesson is per-user.)


## 2. Read-along

### 2.1 Establishing the user — `require_user`

Memory and the chat archive both key on a per-user value. Where does it
come from? Open [`auth.py`](../../src/audrey/auth.py).

HTTP is stateless: every request arrives with no memory of the last one,
so "who is this?" has to be answered again on every request. The answer
comes from a **credential** the request carries, which the server checks.
Audrey accepts two, as Lesson 4 described:

- **A browser request** passes through Cloudflare Access, which adds a
  signed `Cf-Access-Jwt-Assertion` header once the person has signed in.
  Audrey checks that signature itself, against Cloudflare's published keys.
- **A bot or script** sends `Authorization: Bearer aud_pat_…`, a personal
  access token Audrey issued. Audrey looks it up in its own database.

Anything else is refused with 401. A bad Access assertion is refused
outright: Audrey never falls back to another credential on the same request.

#### What `require_user` looks like at the route

[`require_user`](../../src/audrey/auth.py#L308) is a FastAPI
**dependency**: a function whose return value FastAPI injects into a route
handler automatically. The chat route declares it at
[`routes/openai/routes.py:121`](../../src/audrey/routes/openai/routes.py#L121):

```python
async def chat_completions(
    payload: ChatCompletionRequest,
    request: Request,
    me: AuthedUser = Depends(require_user),
):
```

The `me: AuthedUser = Depends(require_user)` line reads as: "Before running
the route body, call `require_user(request)`. If it raises `HTTPException`,
return that error to the client. Otherwise, bind its return value to `me`."
The route never sees a credential or a 401 path; authentication lives in one
dependency.

Both credentials resolve to one Audrey account, and `require_user` returns
an `AuthedUser` for it. The field that matters here is `email`, and its name
is misleading: it holds the account's **storage namespace**, the private key
that memory, uploads and the chat archive file this person's data under. For
an account that existed before Audrey kept its own accounts, the namespace is
the email it had then; a newer account gets an opaque `ns_…` id. Either way it
never changes, even when the person's email does, which is why data is keyed
on it. A later lesson on identity covers accounts, tokens and namespaces in
full.

An account can exist without being usable: a new account waits for an admin
to approve it, and an admin can disable one. `require_user` refuses an
inactive account with 403, not 401, because Audrey knows who it is; the
account just isn't allowed in. Admin-only routes go one step further with
[`require_admin`](../../src/audrey/auth.py#L456), which runs `require_user`
first and then returns 403 unless the account is an admin.

#### What the route does with `me`

Once `require_user` returns, the route has a trusted `AuthedUser` and
uses `me.email` everywhere a `user_id` is needed. Notice this guard
at [`routes/openai/routes.py:178`](../../src/audrey/routes/openai/routes.py#L178):

```python
# Identity comes from the Authorization header via require_user, NOT from
# payload.user (OpenAI-spec passthrough, trusted for nothing).
if payload.user and payload.user != me.email:
    log.debug("chat.completions: payload.user=%r ignored (auth user=%r)", ...)
```

The OpenAI chat-completions spec includes an optional `user` field in
the request body. Audrey accepts it in the schema (for client
compatibility) but **never trusts it for identity** — a client could
put anything there. The header-resolved `me.email` is the only thing
that drives memory, archive, and quota scoping. A mismatch is logged
once for drift-debugging, then ignored.

### 2.2 Memory recall

Open [`pipeline/memory.py`](../../src/audrey/pipeline/memory.py).
`memory_recall` is the second graph node, right after `datetime`. Its
job is to look up durable per-user facts and prepend them as another
system message.

`recall_for_request` at
[`memory.py:60`](../../src/audrey/pipeline/memory.py#L60) opens with
four skip guards before any network call:

```text
not user_id                          - anonymous request, nothing to recall
registry missing memory_search       - tool offline, dispatch would fail
empty query (last_user_text)         - nothing to search with
query > MAX_QUERY_CHARS              - clamp; embedding signal dilutes past ~500 chars
```

Each returns `[]` without raising. **A best-effort feature must not
break the pipeline.** If recall can't run, the rest of the request
runs without recall — that's the intended posture.

The interesting move is what the recall code does *with* the search
call. It could open `httpx.AsyncClient()` and POST to `/memory_search`
directly. It doesn't. It builds an Ollama tool-call shape and reuses
`dispatch_one` from the ReAct loop:

```python
call = {"function": {"name": MEMORY_SEARCH_TOOL,
                     "arguments": {"user": user_id, "query": query, ...}}}
result = await dispatch_one(http, registry, call, ...)
```

Why? **Errors come back as data, not exceptions.** The dispatcher
already knows how to turn network errors, timeouts, 4xx, 5xx, and
unknown-tool failures into `ToolResult(is_error=True)`. Recall just
checks `result.is_error` and degrades to `[]` — no try/except wrapper,
no `httpx.RequestError` handling. Same posture for malformed JSON
bodies and missing `results` arrays.

Hits become a system message via
[`memory_system_message`](../../src/audrey/pipeline/memory.py#L121):

```text
[Relevant memories from previous conversations with this user:]
1. (favorite_color) blue
2. (city) Portland
Use these facts if they're relevant to the user's question. Ignore
irrelevant ones without mentioning them.
```

The trailing instruction is load-bearing — without it, models tend to
greet the user with "I see you're in Portland!" when the question is
about Python. Each `value` is truncated at 400 chars so the recall
block stays under roughly a kilobyte total.

**The `{user_id}` substitution.** Memory writes happen via the model
calling the `memory_store` tool. For the model to know when, the
composer adds a hint from
[`prompts.py:470`](../../src/audrey/pipeline/prompts.py#L470) telling
it to use `tags="user:{user_id}"`. At injection time the placeholder
is replaced with the real email. If the substitution didn't happen,
every entry would store under the literal string `user:{user_id}` and
recall would never find anything for any real user. The hint is only
added when `user_id` is non-empty — anonymous requests can't write to
memory anyway.

The composer at
[`prompts.py:599`](../../src/audrey/pipeline/prompts.py#L599) pins the
slot order: incoming system messages first, then task-role prompt,
then memory hint, then chat-history search guidance. The chat-history
guidance only appears when `chat_history_search` is in the registry —
which brings us to the next subsystem.


### 2.3 The chat archive — capture side

Memory holds durable facts the user states explicitly ("my favorite
color is blue"). The archive holds **every conversation Audrey has had
with this user**, searchable by the model when it needs to recall
something said in passing.

Two halves: Audrey captures responses and ships them to custom-tools;
custom-tools stores them in SQLite and indexes them in Qdrant. Start
with the capture side at
[`pipeline/chat_archive.py`](../../src/audrey/pipeline/chat_archive.py).

The streaming route is the tricky case. It yields SSE frames as fast
as the model produces them, but the archive needs the full assistant
content as one string. The naive approach — accumulate in the route
handler — would mean every streaming branch (fast, deep, ReAct,
non-tool) gets its own accumulator. That's bug-bait.

`StreamCollector` at
[`chat_archive.py:142`](../../src/audrey/pipeline/chat_archive.py#L142)
solves it with a passthrough generator wrapper. Usage from the route
handler — `collector = StreamCollector()`, then `async for frame in
collector.wrap(generator): yield frame`, then read `collector.text`
after the stream ends.

`wrap()` iterates the source generator, yields every frame *unchanged*,
and on the side parses `choices[0].delta.content` out of each SSE
frame to accumulate the assistant text. The route handler never has to
know about accumulation; it just wraps the generator and reads
`collector.text` after the stream ends.

Two details:

- **Banner frames are also `delta.content`-shaped.** The
  `> _Thinking_` / `> _Planning_` progress banners arrive as content
  deltas, indistinguishable from real synth output at the frame level. The
  fast streaming branch can wrap the model stream in `StreamCollector`. The
  deep-streaming branch instead keeps banners out of the archive by
  accumulating the answer body manually in `final_content` inside
  `_stream_deep_with_banners`
  ([`routes/openai/pipeline.py:735`](../../src/audrey/routes/openai/pipeline.py#L735)).
- **`partial=True` on client disconnect.** `wrap()` catches
  `CancelledError` from the source generator and sets
  `self.partial = True` before re-raising. Cancellation happens when
  a `/v1` client hangs up mid-response. The half-finished answer
  is still worth archiving — but the `partial` flag tells the search
  side that it's not the full reply, useful when the model later
  recalls "what I said to you last time."


### 2.4 The chat archive — conversation id

Capture is one half. Storing a turn needs a `conversation_id`, so search
results can be stitched back into threads. A turn from the browser app
already has one, because it belongs to an Audrey conversation. A turn from a
`/v1` client is harder: the API has no standard conversation field. So
[`resolve_conversation_id`](../../src/audrey/pipeline/chat_archive.py#L74)
walks a ladder and returns at the first hit:

```text
1. a top-level chat_id or conversation_id          - a client that sends one
2. either id under the request's metadata          - another common placement
3. either id in a message's metadata, newest first
4. sha256(user, first user turn)                   - deterministic fallback
5. a fresh random id                               - last resort
```

Steps 1 to 3 cover clients that track conversations themselves and say so,
wherever they put the id. Step 4 is the load-bearing fallback for clients
that send none. It hashes only the authenticated user and the first user
turn, so a client that resends the whole growing conversation on every
request keeps getting the same id from its first turn on, and its turns
still stitch into one thread. Step 5 fires only when there's no user turn
at all.

The route resolves this once, before the streaming and non-streaming paths
split, at
[`routes/openai/routes.py:276`](../../src/audrey/routes/openai/routes.py#L276), then
threads `conversation_id` through both the streaming and non-streaming
paths so capture and archive write agree.


### 2.5 The chat archive — write side

Once a turn finishes, Audrey ships the user message + assistant reply
to the tools server for indexing. The writer is
[`ChatArchiveClient.archive_turn`](../../src/audrey/pipeline/chat_archive.py#L502),
called **once per request, after the assistant content is known.**

"After the assistant content is known" is the key constraint. For a
non-streaming request, the full reply is built inside the pipeline
and returned as one chunk — the archive call fires after the graph
finishes, just before the route hands the JSON response back to the
client (see [`routes/openai/pipeline.py:200`](../../src/audrey/routes/openai/pipeline.py#L200)).
For a streaming-deep request, the reply is *only* fully known once
the SSE stream has been fully emitted — so the archive call lives at
the very end of `_stream_deep_with_banners` (see
[`routes/openai/pipeline.py:1104`](../../src/audrey/routes/openai/pipeline.py#L1104)),
using the `final_content` string accumulated from synthesizer deltas. Two call
sites, two different "the content is now known" moments, one writer.

Both call sites share the same posture: **never raise out of the
chat path.** The archive is a best-effort observability surface, not
a source of truth — losing one turn's archive entry is annoying but
not user-visible, while raising an exception during the response hand-
off would either 500 the request or cut the stream short for a user
who already got their answer. So `archive_turn` catches every HTTP
and timeout error internally, logs a warning, and bumps a fail
counter. The chat path continues as if nothing happened.

Why archive *after* the response is built rather than concurrently
with it? Two reasons. First, the archive needs the final assistant
content — there's nothing to write until the model has finished, so
interleaving wouldn't save much wall time. Second, by keeping archive
strictly downstream of response generation, an archive timeout or
HTTP error can't accidentally interfere with the generation logic —
the two concerns are sequenced cleanly. The cost is that the user
waits an extra round-trip (bounded by the 5-second
`ChatArchiveClient` timeout) before the request fully completes; in
practice the tools server lives on the same Docker network and
responds in milliseconds.

Three skip conditions short-circuit early to a `result="skipped"`
metric and no HTTP call:

```text
no user_id              - anonymous request, nothing to archive
no host_url             - chat_history_search not in registry → no host
both contents empty     - nothing to write
```

The `host_url` check deserves a closer look. Audrey doesn't have a hardcoded
custom-tools URL for the archive endpoint. Instead it looks up the
`chat_history_search` tool in the registry and reads its server URL —
**the same custom-tools instance that hosts the search tool hosts the
internal write endpoint.** If the search tool isn't published, archive
is implicitly disabled. One registry membership gates both
directions; the model can't search what the route can't write.

The actual write hits an internal route at
`/chat_history/archive` on the tools server, which sits outside the
OpenAPI tool surface (it's not model-callable — only Audrey's archive
client calls it).

Now switch to [`tools-server/chat_archive.py`](../../tools-server/chat_archive.py).
[`ChatArchiveStore.archive_turn`](../../tools-server/chat_archive.py#L736)
does three things in order:

1. **SQLite write** — user turn + assistant turn into `messages`, plus
   a row in `archive_chunks` per Q+A pair.
2. **Embed each chunk** via `nomic-embed-text` through Ollama.
3. **Qdrant upsert** of the embedded chunks, then `UPDATE
   archive_chunks SET indexed_at = ...` to mark them indexed.

Steps 2-3 can fail without losing data. If embedding or Qdrant errors,
the chunk row stays in SQLite with `indexed_at IS NULL`, and the stats route
reports that as `chunks_unindexed`. SQLite is still the source of truth and
Qdrant is still the index, but there is not yet an automatic reindex worker for
those rows; search will not see the missed chunk until an operator repairs or
re-archives it.
**Idempotency.** Every id is derived deterministically:

```python
message_id = sha256(user|conv|role|content|minute_bucket)[:32]
chunk_id   = sha256(conv|user_msg_id|asst_msg_id|chunk_idx)[:32]
```

The minute bucket in `message_id` means a retry within the same minute
hashes to the same id and `INSERT OR IGNORE` collapses it. The chunk id
ties to the message ids, so re-archiving the same Q+A pair upserts in
place rather than duplicating. Idempotency falls out of id derivation,
not lock-based concurrency control.

**Why chunk Q+A pairs together?** Search returns snippets. A snippet
that's only the assistant turn loses the "what was I asking" context
— the user types "what about the second one?" and the matching answer
is meaningless without the prior question. So
[`build_chunks`](../../tools-server/chat_archive.py#L270) concatenates
`User: ... \nAssistant: ...` into the chunk text. Long pairs split at
sentence boundaries with overlap, same chunking shape as the KB
ingest pipeline (covered in the KB-ingest lesson).

**Why a tool, not auto-injection?** Memory recall runs *every*
request, before the model sees the prompt. The chat archive does not.
The model decides when to search it via the `chat_history_search`
tool. The reasoning is in
[`prompts.py:501`](../../src/audrey/pipeline/prompts.py#L501):

```python
"Use `chat_history_search` only when the user references something "
"they previously discussed with you, or when answering requires a "
"specific prior decision. Do not call it for ordinary "
"personalization or to repeat back recent context — it returns "
"short snippets per call and burns context every time."
```

Auto-injecting archive results on every request would burn context for
no signal on the 95% of requests that don't need it. Letting the model
opt in keeps the cost where the value is.


### 2.6 The streaming bypass — two paths, one set of building blocks

One last fact to surface. The graph nodes for datetime and memory
recall are wrappers; the *building blocks* (`datetime_system_message`,
`recall_for_request`, `compose_system_messages`) are plain functions
in `pipeline/context.py` and `pipeline/memory.py`. That's because the
streaming-deep route bypasses the graph and calls them directly at
[`routes/openai/pipeline.py:1539`](../../src/audrey/routes/openai/pipeline.py#L1539).

| | Non-streaming | Streaming-deep |
|---|---|---|
| Lives in | [`graph.py:257, 273`](../../src/audrey/pipeline/graph.py#L257) | [`routes/openai/pipeline.py:1539`](../../src/audrey/routes/openai/pipeline.py#L1539) |
| Datetime | `node_datetime` | direct `datetime_system_message()` call |
| Recall | `node_memory_recall` | direct `recall_for_request()` call |
| Composer | `compose_system_messages(...)` | `compose_system_messages(...)` |
| Archive | post-graph in handler | inside `_stream_deep_with_banners` |

The streaming route needs to interleave SSE progress frames
(`> _Planning_`, `> _Dispatching panel_`) with the work, which means
it can't just run the graph node-by-node and yield the result. So it
imports the same building blocks and calls them at the right points
in its own SSE timeline. Two call sites, one set of helpers — if a
future change adds a third pre-classify injection, **both call sites
need the update**, or the new context is silently missing from one
path.


## 3. Comprehension questions

For each scenario below, sketch your answer before reading the
discussion. Operational judgment, not trivia.

**1. Someone signs in through Cloudflare Access without trouble, but every
chat request gets a 403. What's going on?**

Cloudflare Access proved who they are; Audrey's own account state decides
whether they may use it. Their account is inactive: either it's new and
still waiting for an admin to approve it, or an admin disabled it.
`require_user` resolves the account and refuses an inactive one with 403,
not 401, because Audrey knows exactly who is asking. Nothing about their
data changes. Once an admin activates the account, the next request works.

**2. A user reports "I told Audrey my birthday is in July, but the
next day it didn't remember." Trace the read and write paths.**

Memory writes happen via the model calling `memory_store` (a tool),
not via Audrey writing directly. Two failure modes are common:

- **The model never called `memory_store`.** Check the request logs
  for `tool_calls`. Causes: the model wasn't tool-capable, or
  `memory_store` wasn't in the registry, or the model decided the
  fact wasn't "durable enough." The store-hint at injection time
  improves the success rate but doesn't force it.
- **The model called it but with the wrong tag.** If the `{user_id}`
  substitution didn't happen (user_id was empty at hint-injection
  time), the entry stored under literal `user:{user_id}` and
  `memory_search` can't find it. Check the stored tag against the
  current `me.email`.

**3. An API client starts a new conversation by sending the same opening
question it sent yesterday, and it sends no conversation id. Does the archive
treat this as a continued conversation or a new one?**

A continued one. With no id from the client, step 4 of
[`resolve_conversation_id`](../../src/audrey/pipeline/chat_archive.py#L74)
hashes the authenticated user and the first user turn. Same user, same
opening question, same hash, so the new turn stitches into yesterday's
thread. That's the price of the fallback: "same opening turn" is the
stitching signal. A client that wants separate threads should send its own
`chat_id` or `conversation_id`. Browser turns are unaffected, because each
belongs to its own Audrey conversation.

**4. The archive write times out. Does the user see an error?**

No, and the turn isn't lost either. The response path doesn't call the
sidecar: it commits a small row to a local SQLite outbox,
[`ChatArchiveQueue`](../../src/audrey/pipeline/chat_archive.py#L569), and
moves on. A background worker delivers queued rows to the sidecar and keeps
retrying failed ones. A tools-server outage delays archive coverage for
those turns; once the sidecar is back, the worker catches up.

**5. Anonymous request (no auth header) hits the chat route. Walk what
breaks and where.**

It doesn't reach the pipeline at all. `Depends(require_user)` runs
first; a request with neither a Cloudflare Access assertion nor a bearer
token gets 401 before the route body executes. The chat route has no anonymous path — the
schema's `payload.user` field is OpenAI-spec passthrough, never
trusted for identity.

(If you're tracing this in tests where the dependency is mocked: the
pipeline functions individually handle empty `user_id` gracefully —
memory recall skips, archive client skips, datetime still runs. But
the real request can't get there.)


## When you're ready for the next lesson

This lesson opened the per-user surface — auth establishing identity,
memory recalling durable facts, the chat archive capturing
conversation history. The next lesson backs up a layer: when multiple
users are hammering the same instance, what keeps one user's deep
request from monopolizing the GPU and starving everybody else? Fair
scheduling, the in-flight slot cap, and the round-robin guard are the
upcoming subject.
