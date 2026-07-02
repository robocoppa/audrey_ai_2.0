# Lesson 13 — Per-user context: identity, memory, and history

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
per-user surface in the system reads from one canonical source: an
`AuthedUser.email` resolved by `require_user` once per request and
cached for 30 seconds. The rest of this lesson is what falls out of
that decision.

(One non-identity-gated injection also runs on every request: a
datetime system message — "today is 2026-05-27T14:32:00-07:00, treat
this as the present moment." Two functions in
[`pipeline/context.py`](../../src/audrey/pipeline/context.py), no I/O,
no skip path; the instruction defuses models that would otherwise
reason about the timestamp as data. The node wrapper at
[`graph.py:139`](../../src/audrey/pipeline/graph.py#L139) prepends it
to `state["messages"]`. That's the whole subsystem; the rest of the
lesson is per-user.)


## 2. Read-along

### 2.1 Establishing the user — `AuthedUser` and `require_user`

Memory and the chat archive both key on a per-user id. Where does that
id come from? Open [`auth.py`](../../src/audrey/auth.py).

#### What "identity" actually means over HTTP

HTTP is stateless. Every request arrives at the server with no memory
of any previous request. So "who is this user?" has to be re-answered
on every single request, somehow. The mechanism is a **credential the
client attaches to each request** — typically in a header — that the
server can validate.

Audrey's clients (Open WebUI and the upload page) attach a credential
called a **JWT** (JSON Web Token) — a string the user gets after
logging in, which they hand back on every subsequent request to prove
they're still the same person.

The client attaches the JWT in an HTTP header called `Authorization`,
using a convention called **bearer-token auth**:

```text
Authorization: Bearer eyJhbGciOiJIUzI1NiIs...<long opaque string>...
```

The word "bearer" is literal — "whoever bears (carries) this token is
treated as the identity it represents." That's both the strength
(simple) and the risk (if the token leaks, anyone holding it *is* the
user). Bearer tokens are the current standard for API auth.

#### Why Audrey doesn't validate the token itself

Most apps own their user database and validate tokens themselves.
Audrey does not. There's a separate service — **Open WebUI (OWUI)** —
which is the chat frontend the user logs into. OWUI owns the user
accounts, issues the JWTs, and is the authority on which JWTs are
real. Audrey is a backend that OWUI talks to.

If Audrey re-implemented JWT validation, it would have to share a
signing secret with OWUI, stay in sync with OWUI's user lifecycle
(activations, role changes, deletions), and keep its own user table
consistent with OWUI's. All of that would add unnecessary complexity.

The cleaner move is to **delegate**: on each request, Audrey forwards
the token to OWUI and asks "is this real, and if so, who?" OWUI's
answer is the source of truth. This pattern is called an **auth
probe** or **token introspection**. The slogan: *OWUI says who;
Audrey just remembers the answer briefly.*

#### What `require_user` looks like at the route

[`require_user`](../../src/audrey/auth.py#L126) is a FastAPI
**dependency**. A dependency in FastAPI is a function whose return
value gets injected into a route handler automatically. The chat
route declares it at
[`routes/openai/routes.py:87`](../../src/audrey/routes/openai/routes.py#L87):

```python
async def chat_completions(
    payload: ChatCompletionRequest,
    request: Request,
    me: AuthedUser = Depends(require_user),
):
```

The `me: AuthedUser = Depends(require_user)` line reads as: "before
running the route body, call `require_user(request)`. If it raises
`HTTPException`, return that error to the client. Otherwise, bind its
return value to the parameter `me`." From the route body's
perspective, `me` simply exists; the route never sees the JWT, the
OWUI probe, the cache, or the 401 path. Auth has been pushed entirely
into one dependency.

The return type, `AuthedUser`, is a small dataclass at
[`auth.py:64`](../../src/audrey/auth.py#L64):

```python
@dataclass(slots=True)
class AuthedUser:
    email: str       # canonical user id — never replace with a database id
    role: str        # "user" | "admin"
    owui_id: str
```

The load-bearing field is `email`. Per-user memory keys on it,
per-user uploads key on it, the chat archive keys on it. The comment
on the dataclass says so explicitly because a future refactor might
be tempted to add a numeric `id` field instead — every place that
hands `user_id=me.email` would silently break.

`owui_id` is OWUI's internal user id (kept for debugging / future
admin features), and `role` is `"user"` or `"admin"` — used by the
admin-only routes via a companion dependency,
[`require_admin`](../../src/audrey/auth.py#L157), which calls
`require_user` first then additionally checks `me.role == "admin"`
and returns 403 if not. (401 means "we don't know who you are"; 403
means "we know who you are and you're not allowed.")

#### Inside `require_user`, step by step

Three things happen on the way to `me`:

1. **Header parse.** The dependency reads the `Authorization` header.
   It accepts only the bearer shape — `Authorization: Bearer <token>`
   — and rejects anything else with 401. OWUI also sets an `HttpOnly`
   `token` cookie on its own same-origin responses, but Audrey
   **ignores cookies entirely**. The reason is **CSRF** (Cross-Site
   Request Forgery): a browser will automatically attach cookies to
   any request to the cookie's domain, including requests originated
   by a malicious third-party page the user happens to be visiting.
   If Audrey accepted cookie-borne tokens, that malicious page could
   silently make authenticated requests on the user's behalf.
   Requiring the explicit `Authorization` header closes that path —
   third-party JavaScript can't read or set arbitrary headers on
   cross-origin requests by default.

2. **Cache check.** Asking OWUI on every single request would be
   expensive — a dashboard with 20 widgets re-polling means 20 OWUI
   probes per refresh. So `require_user` keeps a per-token cache: if
   the same token was successfully resolved within the last 30
   seconds, return the cached `AuthedUser` without calling OWUI. The
   cache is a plain Python dict mapping token → `(timestamp,
   AuthedUser)`, with an opportunistic sweep of expired entries when
   the dict grows past 1024. For Audrey's dozens-of-users scale,
   that's plenty — no LRU eviction, no Redis, just a dict.

   The 30-second TTL is the freshness-vs-load tradeoff: long enough
   to absorb burst traffic, short enough that role/permission changes
   in OWUI propagate within half a minute.

3. **OWUI probe.** On a cache miss, `require_user` makes an HTTP
   request: `GET http://open-webui:8080/api/v1/auths/`, with the
   user's bearer token forwarded in *its* `Authorization` header. (We
   are, in a sense, impersonating the user against OWUI to ask "is
   this you?") OWUI replies with a JSON body containing the user's
   `id`, `email`, `role`, and other fields. Audrey extracts what it
   needs into `AuthedUser` and caches the result.

   The probe accepts only `role in {"user", "admin"}`. OWUI returns
   200 OK for `pending` users — accounts that exist but haven't been
   activated by an admin — but Audrey explicitly rejects those at
   [`auth.py:113`](../../src/audrey/auth.py#L113), returning 401 with
   "Account not activated." This is **fail-closed**: an unknown role
   string (a future OWUI version's new state, a garbled response)
   also fails, on the principle that auth code should refuse what it
   doesn't understand rather than guess.

   Failure modes: 401 from OWUI → 401 to the client (token is bad).
   Anything else (500, timeout, connection refused) → 502 to the
   client, which distinguishes "you're not logged in" from "auth
   itself is broken." Timeout is 5 seconds; OWUI and Audrey live on
   the same Docker network, so anything slower than that means OWUI
   is actually down, not slow.

#### What the route does with `me`

Once `require_user` returns, the route has a trusted `AuthedUser` and
uses `me.email` everywhere a `user_id` is needed. Notice this guard
at [`routes/openai/routes.py:109`](../../src/audrey/routes/openai/routes.py#L109):

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

#### Cache invalidation

The 30-second TTL means most state changes in OWUI propagate within
half a minute on their own. But there's one case where waiting is
wrong: an admin deletes or bans a user in OWUI, and that user's
already-cached `AuthedUser` shouldn't keep working for 30 more
seconds. OWUI v0.9.x doesn't emit user-deletion webhooks, so Audrey
exposes two admin endpoints in
[`routes/admin.py`](../../src/audrey/routes/admin.py) for explicit
eviction:

- `POST /v1/admin/auth/clear` — drops every cached entry. Useful
  after an OWUI config or version change.
- An email-scoped variant powered by
  [`clear_auth_cache_for_email`](../../src/audrey/auth.py#L182) —
  drops just one user's entries (a user can have multiple tokens
  across devices) without disturbing others.

In short: **OWUI says who; Audrey caches the answer for 30 seconds,
unless an admin says otherwise.**


### 2.2 Memory recall

Open [`pipeline/memory.py`](../../src/audrey/pipeline/memory.py).
`memory_recall` is the second graph node, right after `datetime`. Its
job is to look up durable per-user facts and prepend them as another
system message.

`recall_for_request` at
[`memory.py:59`](../../src/audrey/pipeline/memory.py#L59) opens with
four skip guards before any network call:

```text
not user_id                          - anonymous request, nothing to recall
registry missing memory_search       - tool offline, dispatch would fail
empty query (last_user_text)         - prefix-only OWUI utility tasks, etc.
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
[`memory_system_message`](../../src/audrey/pipeline/memory.py#L105):

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
[`prompts.py:118`](../../src/audrey/pipeline/prompts.py#L118) telling
it to use `tags="user:{user_id}"`. At injection time the placeholder
is replaced with the real email. If the substitution didn't happen,
every entry would store under the literal string `user:{user_id}` and
recall would never find anything for any real user. The hint is only
added when `user_id` is non-empty — anonymous requests can't write to
memory anyway.

The composer at
[`prompts.py:202`](../../src/audrey/pipeline/prompts.py#L202) pins the
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
[`chat_archive.py:116`](../../src/audrey/pipeline/chat_archive.py#L116)
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
  ([`routes/openai/pipeline.py:492`](../../src/audrey/routes/openai/pipeline.py#L492)).
- **`partial=True` on client disconnect.** `wrap()` catches
  `CancelledError` from the source generator and sets
  `self.partial = True` before re-raising. Cancellation happens when
  the user closes the OWUI tab mid-response. The half-finished answer
  is still worth archiving — but the `partial` flag tells the search
  side that it's not the full reply, useful when the model later
  recalls "what I said to you last time."


### 2.4 The chat archive — conversation id

Capture is one half. Persisting needs a `conversation_id` so search
results stitch into threads. OWUI sends a `chat_id` in the request
body, but Audrey doesn't get to pick where OWUI puts it — different
versions have shipped different shapes, and a future OWUI release
could move the field again.

[`resolve_conversation_id`](../../src/audrey/pipeline/chat_archive.py#L57)
walks a five-step ladder, returning at the first hit:

```text
1. raw_payload["chat_id"]                  - current OWUI fast path
2. raw_payload["metadata"]["chat_id"]      - speculative nested variant
3. messages[-1]["metadata"]["chat_id"]     - older OWUI shape
4. sha256(user, first 6 message contents)  - deterministic derive
5. fresh-<uuid4>                           - last resort
```

On the OWUI version Audrey runs against today, **step 1 wins on every
request**. Steps 2 and 3 exist because earlier OWUI releases placed
`chat_id` under `metadata` or attached it to the last message instead
of the top level; keeping both checks means an OWUI downgrade or a
future upgrade that flips the field location doesn't break archive
stitching without an Audrey change.

Step 4 is the genuinely load-bearing fallback. It fires when *no*
OWUI version supplies a `chat_id` — for example, a future OWUI rewrite
that drops the field entirely, or a non-OWUI client hitting the route
directly. The deterministic hash means a second request in the same
conversational context still lands on the same `conversation_id`
(same first six messages → same hash), so thread stitching survives
even without metadata. The six-message prefix is long enough to pin
a thread, short enough that the id stays stable as the conversation
grows.

Step 5 only fires when there are no messages and no user — a true
edge case (anonymous request with empty history).

The route resolves this once before pipeline branching at
[`routes/openai/routes.py:136`](../../src/audrey/routes/openai/routes.py#L136), then
threads `conversation_id` through both the streaming and non-streaming
paths so capture and archive write agree.


### 2.5 The chat archive — write side

Once a turn finishes, Audrey ships the user message + assistant reply
to the tools server for indexing. The writer is
[`ChatArchiveClient.archive_turn`](../../src/audrey/pipeline/chat_archive.py#L216),
called **once per request, after the assistant content is known.**

"After the assistant content is known" is the key constraint. For a
non-streaming request, the full reply is built inside the pipeline
and returned as one chunk — the archive call fires after the graph
finishes, just before the route hands the JSON response back to the
client (see [`routes/openai/pipeline.py:134`](../../src/audrey/routes/openai/pipeline.py#L134)).
For a streaming-deep request, the reply is *only* fully known once
the SSE stream has been fully emitted — so the archive call lives at
the very end of `_stream_deep_with_banners` (see
[`routes/openai/pipeline.py:818`](../../src/audrey/routes/openai/pipeline.py#L818)),
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
[`ChatArchiveStore.archive_turn`](../../tools-server/chat_archive.py#L356)
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
[`build_chunks`](../../tools-server/chat_archive.py#L160) concatenates
`User: ... \nAssistant: ...` into the chunk text. Long pairs split at
sentence boundaries with overlap, same chunking shape as the KB
ingest pipeline (covered in the KB-ingest lesson).

**Why a tool, not auto-injection?** Memory recall runs *every*
request, before the model sees the prompt. The chat archive does not.
The model decides when to search it via the `chat_history_search`
tool. The reasoning is in
[`prompts.py:305`](../../src/audrey/pipeline/prompts.py#L305):

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
[`routes/openai/pipeline.py:1161`](../../src/audrey/routes/openai/pipeline.py#L1161).

| | Non-streaming | Streaming-deep |
|---|---|---|
| Lives in | [`graph.py:139, 155`](../../src/audrey/pipeline/graph.py#L139) | [`routes/openai/pipeline.py:1161`](../../src/audrey/routes/openai/pipeline.py#L1161) |
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

**1. A request arrives with a valid bearer token, but Audrey returns
401 with "Account not activated." OWUI shows the user as logged in.
What's going on?**

OWUI's `role` is `pending` — the JWT is valid, but the user hasn't
been activated by an admin in OWUI. Audrey fails them closed at
[`auth.py:113`](../../src/audrey/auth.py#L113); the allowed set is
`{"user", "admin"}` only. The 30-second cache means this state lingers
for half a minute after the admin activates them in OWUI. To force
immediate eviction, hit `POST /v1/admin/auth/clear` (or its
email-scoped variant) — otherwise the next probe after TTL expiry
picks up the new role.

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

**3. A user opens a fresh OWUI tab, types the same opening question
they asked yesterday. Does the archive treat this as a continued
conversation or a new one?**

It depends on whether OWUI sends `chat_id`. If yes (steps 1-3 of
[`resolve_conversation_id`](../../src/audrey/pipeline/chat_archive.py#L57)),
new tab → new chat_id → new conversation. If OWUI omits the field
(step 4 fallback), the deterministic hash over `(user, first 6
message contents)` will match yesterday's hash and the new request
stitches into yesterday's conversation. That's deliberate — the goal
of step 4 is exactly to survive missing `chat_id` metadata — but it
means "same opening turns" is the stitching signal, not "same tab."

**4. The archive write times out. Does the user see an error?**

No. The post-response archive call is best-effort:
[`ChatArchiveClient.archive_turn`](../../src/audrey/pipeline/chat_archive.py#L216)
catches every `httpx.HTTPError` and `TimeoutError`, logs them, and
increments `chat_archive_writes_total{result="fail"}`. The response is
already on its way back to the user — the archive write happens *after*
the response is produced. Net effect: a transient tools-server outage
loses archive coverage for that turn, but chat keeps working. The next
turn's archive write will succeed independently.

**5. Anonymous request (no auth header) hits the chat route. Walk what
breaks and where.**

It doesn't reach the pipeline at all. `Depends(require_user)` runs
first; missing or malformed `Authorization` raises 401 before the
route body executes. The chat route has no anonymous path — the
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
