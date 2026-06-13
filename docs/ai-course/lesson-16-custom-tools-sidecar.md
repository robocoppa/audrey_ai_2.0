# Lesson 16 — The custom-tools sidecar

Throughout this course Audrey has been *calling tools* — `web_search`,
`kb_search`, `memory_store`, `chat_history_search`. Lesson 9 showed the
ReAct loop that decides to call them; Lesson 10 showed the wire protocol.
But every one of those calls crosses a network hop to a service we've
treated as a black box. This lesson opens it.

That service is the **custom-tools sidecar**: a small, separate FastAPI
app, in its own container, that exposes Audrey's entire tool surface. It is
the last subsystem the course hasn't covered. By the end you should be able
to add a tool, fix one, and reason about which side of the wire a failure
lives on.

## 1. Context

### 1.1 What this is, and why it's separate

The sidecar is **not part of Audrey**. It's a second process:
[`tools-server/app.py`](../../tools-server/app.py) is its own
`FastAPI(...)` instance, with its own dependencies, its own lifespan, and
its own container in `compose.yaml`. Audrey talks to it over HTTP on the
internal network, the same way it talks to Ollama or Qdrant.

Why split it out at all, instead of putting the tools inside Audrey?

- **Dependency isolation.** The sidecar pulls in the Brave Search SDK, the
  Qdrant client, and `aiosqlite`. None of those belong in the orchestrator,
  whose job is routing and pipeline control. Keeping them out of Audrey's
  process keeps Audrey's image and import graph lean.
- **Failure isolation.** A tool that wedges — a hung Brave request, a
  Qdrant timeout — degrades *one tool*, not the whole orchestrator. Audrey
  catches the HTTP failure and the rest of the pipeline keeps running.
- **It's the tool *registry*, discovered at boot.** Audrey doesn't hardcode
  what tools exist. At startup it reads the sidecar's `/openapi.json`,
  turns each route into a callable tool definition, and hands those to the
  model. (That discovery handshake is Lesson 9's territory; here we're on
  the *other* end of it — the thing being discovered.)

That last point is the one to hold onto: **the sidecar's HTTP routes
*are* the model's tools.** Add a route, and — if you shape it right — the
model gains a tool. That's the whole contract, and §2 unpacks it.

> **The `tools=0` boot race.** Because discovery happens once at Audrey
> startup, if the sidecar isn't healthy yet when Audrey boots, Audrey comes
> up advertising *zero* tools. `compose.yaml` orders Audrey's start after
> the sidecar's healthcheck to avoid this; the admin rediscover route is
> the manual recovery if it happens anyway. You saw this from Audrey's side
> in earlier lessons — the sidecar's `/health` route ([app.py:218](../../tools-server/app.py#L218))
> is what that healthcheck hits.

### 1.2 The four files

| File | Role |
|---|---|
| [`app.py`](../../tools-server/app.py) | The FastAPI app: every tool route, its request/response schemas, the lifespan that wires up clients. |
| [`brave.py`](../../tools-server/brave.py) | The Brave Search client behind `web_search` — caching + retry. |
| [`db.py`](../../tools-server/db.py) | The Qdrant-backed durable memory store behind `memory_*`. |
| [`settings.py`](../../tools-server/settings.py) | Env-driven config (Pydantic Settings). |

Plus [`chat_archive.py`](../../tools-server/chat_archive.py), the sidecar
half of the chat archive. Its write path (`archive_turn`, chunking) was
covered in [Lesson 13](lesson-13-memory-and-context-injection.md); here we
cover the read/maintenance side (search, prune, stats).

## 2. Read-along

### 2.1 The anatomy of a tool route

Every tool the model can call is one FastAPI route, and they all share the
same four-part shape. `web_search` is the cleanest example. Its request
schema is at [app.py:117](../../tools-server/app.py#L117):

```python
class WebSearchRequest(BaseModel):                      # ① request schema
    query: Annotated[str, Field(min_length=1, max_length=500, ...)]
    count: Annotated[int, Field(ge=1, le=10, ...)] = 5
```

and its route decorator + handler at [app.py:225](../../tools-server/app.py#L225):

```python
@app.post(
    "/web_search",
    operation_id="web_search",                          # ② tool name
    response_model=WebSearchResponse,                   # ③ response schema
    summary="Search the web via Brave Search API",      # ④ model-facing docs
    description="Query the public web for current information. ...",
)
async def web_search(req: WebSearchRequest) -> WebSearchResponse:
    ...
```

Each part has a job, and three of the four are *for the model*, not for you:

1. **The request schema** becomes the tool's JSON-Schema parameters. The
   `Field` constraints (`min_length`, `le=10`) are advertised to the model
   *and* enforced by FastAPI — a model that sends `count: 50` gets a 422
   before your handler runs.
2. **`operation_id`** is the literal tool name the model calls. Without it,
   FastAPI auto-generates an ugly name from the function + path; with it,
   the OpenAPI→Ollama-tool converter produces a clean `web_search`. **This
   is load-bearing** — change it and you rename the tool, breaking any
   prompt or memory that referenced the old name.
3. **`response_model`** shapes what comes back. It also documents the return
   shape in `/openapi.json`, though the model mostly cares about the call.
4. **`summary` / `description`** are the model-facing documentation. The
   `description` is sent to the model as the tool's description on *every
   request that advertises this tool*. This is prompt real estate — it's
   where you tell the model *when* to use the tool, not just what it does.
   (Look at `chat_history_search`'s description at
   [app.py:426](../../tools-server/app.py#L426): half of it is "use only
   when…" — actively steering the model away from over-calling it.)

That's the contract. Get all four right and the model gains a working tool;
get `operation_id` wrong and the tool is misnamed; get the `description`
wrong and the model calls it at the wrong times.

### 2.2 Concept spotlight: hiding routes from the model

Not every route should be a tool. The chat archive has three routes the
model must **never** call — writing a turn, pruning old data, reading
stats. Those are for Audrey's archive client and the admin operator only.

The mechanism is one flag ([app.py:477](../../tools-server/app.py#L477)):

```python
@app.post("/chat_history/archive", include_in_schema=False, tags=["internal"])
async def chat_history_archive(req: ArchiveTurnRequest) -> dict[str, Any]:
    ...
```

`include_in_schema=False` keeps the route out of `/openapi.json`. Since
Audrey builds the model's tool list *from* `/openapi.json`, a route that
isn't in the schema is invisible to discovery — the model literally cannot
see it, so it cannot call it. The route still works over HTTP for anyone
who knows the path (Audrey's archive client posts to `/chat_history/archive`
after every turn), but it's not a tool.

This is the sidecar's privilege boundary: **schema-visible routes are
tools; `include_in_schema=False` routes are internal plumbing.** When you
add a route, that one flag decides which side of the line it's on.

### 2.3 The three kinds of tool

The six model-facing tools look uniform from §2.1, but underneath they fall
into three groups with very different implementations:

| Kind | Tools | What the handler does |
|---|---|---|
| **Proxy** | `kb_search`, `kb_image_search` | Forwards back to Audrey's `/v1/kb/query`. |
| **External** | `web_search` | Calls a third-party API (Brave) via `brave.py`. |
| **Stateful** | `memory_*`, `chat_history_search` | Reads/writes Qdrant + SQLite via `db.py` / `chat_archive.py`. |

The **proxy** kind is worth pausing on because it's counterintuitive: the
sidecar calls *back into Audrey*. Here's the whole of `kb_search`
([app.py:269](../../tools-server/app.py#L269)), which is representative:

```python
async def kb_search(req: KBSearchRequest) -> KBSearchResponse:
    client: httpx.AsyncClient = app.state.audrey      # httpx client → Audrey
    payload: dict[str, Any] = {"query": req.query, "top_k": req.top_k}
    if req.user:
        payload["user"] = req.user                    # merge user uploads
    try:
        r = await client.post("/v1/kb/query", json=payload)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Audrey KB unreachable: {e}")
    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    body = r.json()
    return KBSearchResponse(query=req.query, results=body.get("results", []))
```

That's the entire handler. It does three things: build the payload, POST it
to Audrey's `/v1/kb/query`, and relay the response. The `httpx` client it
uses (`app.state.audrey`) was constructed in the lifespan, pointed at
Audrey's base URL.

Why the round-trip back into Audrey? The knowledge base — Qdrant
collections, the CLIP/embedding machinery — lives on Audrey's side
(Lessons 11–12). The sidecar doesn't own the KB; it only *exposes* it as a
model-callable tool. So the full call path for one KB search is:

```text
model → Audrey (ReAct loop) → sidecar (kb_search) → Audrey (/v1/kb/query) → Qdrant
```

Audrey calls out to the sidecar, which calls back into Audrey. It looks
circular, but the two hops do different jobs: the first turns "the model
wants to search" into "a tool call," and the second does the actual
retrieval. The sidecar is a thin adapter between those two — it's what lets
the model *name* a capability that physically lives in the orchestrator.

The error handling tells you where a failure lives. Two distinct cases:

- **`httpx.RequestError` → 502** ([app.py:277](../../tools-server/app.py#L277)).
  The sidecar couldn't *reach* Audrey at all — connection refused, DNS,
  timeout. The bug is Audrey-side or network, not the sidecar.
- **Audrey answered with `>= 400` → relay that status verbatim.** Audrey's
  KB route itself rejected the query (bad payload, KB not ready). The
  sidecar passes the status straight through rather than masking it.

So a 502 from `kb_search` means "the proxy works, its upstream doesn't,"
while any other 4xx is Audrey's own verdict on the query. `kb_image_search`
([app.py:300](../../tools-server/app.py#L300)) is the same shape against
`/v1/kb/query/image`, with one extra guard: it requires exactly one of
`query` / `image_url` / `image_b64` and 422s if the model sends none.

The next two sections take the External and Stateful kinds in depth.

### 2.4 `brave.py`: an external API client, done carefully

[`brave.py`](../../tools-server/brave.py) wraps the Brave Search API. It's
small but it models how to call a rate-limited, paid third-party service
without abusing it. Two mechanisms carry the weight.

**A TTL cache** ([brave.py:119](../../tools-server/brave.py#L119)). The
client keeps an in-memory `OrderedDict` keyed by `(query, count)`, each
entry stamped with a `time.monotonic()` expiry. The read path is small but
does three things at once:

```python
async def _cache_get(self, key):
    async with self._cache_lock:                 # ① one writer at a time
        entry = self._cache.get(key)
        if entry is None:
            return None
        expires_at, results = entry
        if time.monotonic() >= expires_at:       # ② expired → evict + miss
            self._cache.pop(key, None)
            return None
        self._cache.move_to_end(key)             # ③ mark recently-used (LRU)
        return results
```

① The `asyncio.Lock` matters because the sidecar serves concurrent
requests: two coroutines could otherwise mutate the `OrderedDict` mid-read.
② Expiry is checked lazily, on read — there's no background sweeper; a stale
entry simply misses and gets refetched. ③ `move_to_end` makes the dict an
LRU: the write path (`_cache_put`) calls `popitem(last=False)` to drop the
*least* recently used entry when the cache exceeds `cache_max_entries`, so
memory is bounded regardless of how many distinct queries come through.

The point of all this is the free tier: Brave's quota is small, and a
24h TTL means a query asked twice in a day costs one API call. The cache
is keyed on `(query, count)` because a different `count` is a genuinely
different result set.

**Retry with backoff** ([brave.py:97](../../tools-server/brave.py#L97)).
The actual fetch wraps the HTTP call in `tenacity`'s `AsyncRetrying`:

```python
async for attempt in AsyncRetrying(
    retry=retry_if_exception_type((BraveRateLimitError, httpx.HTTPStatusError)),
    stop=stop_after_attempt(4),                  # at most 4 tries
    wait=wait_exponential(multiplier=1, min=1, max=15),   # 1s, 2s, 4s, …
    reraise=True,
):
    with attempt:
        resp = await _do()
```

`_do()` raises `BraveRateLimitError` on a 429 and `raise_for_status()` on
other HTTP errors; both are in the `retry=` set, so a transient failure
backs off and tries again — 1s, then 2s, then 4s, capped at 15s. The
exponential wait is what makes this *polite*: hammering a rate-limited API
with immediate retries just deepens the limit. If all four attempts fail,
`reraise=True` lets the last exception out, which the client converts to a
`BraveRateLimitError`. The `web_search` handler catches that and returns a
503 ([app.py:240](../../tools-server/app.py#L240)).

So from the model's seat, a rate-limited Brave looks like a tool that
returned an error — the ReAct loop (Lesson 9) sees the failed tool result
and can apologize, answer from training, or move on. The request doesn't
die; one tool degraded. That's the failure-isolation payoff of §1.1 made
concrete.

> **Teaching aside — where the count limit actually binds.** Three layers
> touch `count`: the request schema caps it at 10
> ([app.py:119](../../tools-server/app.py#L119)), the cache key clamps to
> 20, and `_fetch` passes it straight to Brave. They disagree — but it
> doesn't matter, because the schema cap is the *real* ceiling: FastAPI
> rejects `count > 10` with a 422 before any other layer sees it. The
> lower-level clamps are belt-and-suspenders for a hypothetical caller that
> bypasses the schema. The lesson: when several layers validate the same
> value, the *outermost enforced* one is the contract; the inner ones are
> defense in depth.

### 2.5 `db.py`: the durable memory store

[`db.py`](../../tools-server/db.py) backs `memory_store` / `memory_recall`
/ `memory_search`. It is Qdrant-backed, embedded with `nomic-embed-text`
(the same 768-d text embedder the KB uses — Lesson 11). A few design
choices make it correct and worth understanding.

**Deterministic point ids** ([db.py:61](../../tools-server/db.py#L61)).
Each memory's Qdrant point id is `uuid5(NAMESPACE_URL, f"{user}|{key}")` —
a *deterministic* hash of the (user, key) pair. So re-storing the same
(user, key) produces the *same* id, and the upsert overwrites rather than
duplicating. This is how `memory_store` is idempotent: store "favorite
language: Python" twice and you have one point, not two.

**What actually gets embedded** ([`_embedding_text`, db.py:66](../../tools-server/db.py#L66)).
A memory is a `(key, value, tags)` triple, but you can't embed a dict — you
need one string to send to `nomic-embed-text`. The store builds it as
`f"{key}: {value} [tags: {stripped_tags}]"`, and the construction is
deliberate:

- **`key` goes first** so short queries get a lexical hook. A query like
  "favorite language" should match a memory keyed `favorite_language` even
  if the value text is sparse.
- **The `user:<id>` tag is *stripped* before embedding.** It's scope
  metadata, not semantic content — embedding "user:alice" would add noise
  that pulls unrelated memories together just because they share an owner.
  Topic tags (`topic:hardware`) stay in, because those *are* meaningful
  signal.

So the user tag does double duty: it's pulled out for the exact-match
*filter* (below) and pulled out of the *embedding*. Same token, two
removals, two different reasons.

**`recall` and `search` are different operations.** Both are scoped to a
user, but they answer different questions:

| | `recall(key, user)` | `search(query, user, top_k)` |
|---|---|---|
| Question | "what's stored under *this exact key*?" | "what memories are *about this*?" |
| Mechanism | Qdrant **scroll** — payload filter, no vector | **vector search** + payload filter + threshold |
| Returns | one entry (newest if duplicates) | up to `top_k` ranked entries |

`recall` ([db.py:229](../../tools-server/db.py#L229)) is a pure payload
lookup — it never embeds anything, just scrolls for points where `key` and
`user` both match. `search` ([db.py:258](../../tools-server/db.py#L258))
embeds the query, runs a cosine vector search filtered to the user, and
drops anything below `MEMORY_SIMILARITY_THRESHOLD`. That threshold is set
*tight* (0.5) on purpose: a memory false-positive is injected into the
prompt as "a fact about the user," so a wrong hit actively misleads the
model — worse than no hit. (Contrast the chat archive's looser threshold in
§2.7, where the stakes are lower.)

**User scoping that can't be fooled by substrings.** Memories are
per-user; the scope must be exact, or one user's memory leaks into
another's. The free-form `tags` string carries `user:<id>`, but filtering
on a substring of `tags` is fragile — `user:al` would match `user:alice`.
So at write time the store *extracts* the user id
([`_parse_user`, db.py:53](../../tools-server/db.py#L53)) and duplicates it
into a dedicated `user` payload field with a keyword index. Every read
filters on that exact field ([recall, db.py:229](../../tools-server/db.py#L229);
[search, db.py:258](../../tools-server/db.py#L258)). The docstrings on both
read methods carry the same warning: never relax the `user` filter.

**The model never supplies `user`.** This is the key safety property, and
it connects to a fixed audit finding. The `user` field is required in the
schemas, but the model is told (in the field descriptions) that *Audrey
fills it in automatically*. That's because Audrey's dispatch layer
overrides the `user` argument with the authenticated pipeline user before
the call ever reaches the sidecar — the `_USER_SCOPED_TOOLS` mechanism from
Lesson 9. A model can't write to another user's memory even if it tries to,
because Audrey rewrites the `user` argument regardless of what the model
emitted. The sidecar trusts that the `user` it receives is already
authenticated.

**Legacy migration on boot** ([db.py:144](../../tools-server/db.py#L144)).
An earlier version of memory used SQLite. On first startup, if a legacy
`memory.db` exists, `init()` reads every row, embeds it, upserts to Qdrant,
then renames the file to `memory.db.migrated`. It's idempotent — the rename
means a second boot finds no `memory.db` and does nothing. This is a
one-shot data migration that lives in the application's startup path rather
than a separate script, which is fine for a single-replica sidecar.

### 2.6 `settings.py` and the lifespan: wiring it together

[`settings.py`](../../tools-server/settings.py) is Pydantic Settings —
every knob is an env var with a default (see Lesson 2 for the pattern, and
Lesson 5 for Audrey's larger version). The interesting part is what
happens at *load*: a validator that refuses to start with a bad config.

```python
@model_validator(mode="after")
def _check_chunk_overlap(self) -> Settings:
    if self.chat_archive_chunk_overlap_chars >= self.chat_archive_chunk_max_chars:
        raise ValueError("CHAT_ARCHIVE_CHUNK_OVERLAP_CHARS ... must be less than ...")
    return self
```

([settings.py:56](../../tools-server/settings.py#L56)) The chat archive's
text splitter steps through long text by `max_chars - overlap`. If overlap
were ever set `>=` max_chars, that step would be zero or negative and the
splitter would crash — but only later, on a write large enough to need
splitting. The validator turns a *latent runtime crash* into an *immediate,
legible boot failure*: misconfigure the two env vars and the sidecar
refuses to start, naming both knobs. This is the "fail fast at boot"
principle — surface a config error at the earliest possible moment, not on
some unlucky request hours later.

The **lifespan** ([app.py:47](../../tools-server/app.py#L47)) is where the
clients are born. On startup it constructs the `BraveClient`, the
`MemoryStore`, an httpx client pointed at Audrey (for the proxy tools), and
the `ChatArchiveStore`; calls each one's `init()`; and hangs them on
`app.state`. Every handler then reaches them via `app.state.brave`,
`app.state.memory`, and so on. On shutdown the `finally` block closes each
client. This is the standard FastAPI lifespan shape (Lesson 5 §startup
covered Audrey's): construct expensive, long-lived clients *once*, share
them across requests, close them cleanly.

### 2.7 The chat archive's read side

[Lesson 13](lesson-13-memory-and-context-injection.md) covered how a turn
gets written — `archive_turn`, the Q+A-pair chunking, the embed-and-upsert.
The sidecar exposes three more operations on top of that store:

- **`search`** ([chat_archive.py:491](../../tools-server/chat_archive.py#L491))
  backs the `chat_history_search` tool. It embeds the query, vector-searches
  Qdrant filtered by `user` (plus an optional `created_at` date range), and
  returns snippet-first hits. Its similarity threshold is set *looser* than
  durable memory's — chat-history recall is human-triggered ("did we talk
  about X?"), so a few marginal hits are acceptable, whereas a
  `memory_search` false positive would poison the prompt with a wrong
  "fact about the user." Same machinery, different threshold, for a
  principled reason.
- **`prune`** ([chat_archive.py:550](../../tools-server/chat_archive.py#L550))
  applies retention: if `retention_days > 0`, it deletes Qdrant points
  *first*, then the SQLite rows older than the cutoff. (Vectors before
  source, so a crash mid-prune leaves recoverable source rows, never
  orphaned vectors pointing at deleted text.) Reached only via the admin
  route.
- **`stats`** ([chat_archive.py:604](../../tools-server/chat_archive.py#L604))
  returns row counts including `chunks_unindexed` — chunks in SQLite that
  never made it into Qdrant (an embed/upsert failure at write time). A
  non-zero value is the signal that the index has drifted from the source
  and needs a reindex.

These three are the `include_in_schema=False` internal routes from §2.2 —
the model can't call them; Audrey's admin surface and archive client can.

## 3. Comprehension questions

**1. You add a new route `@app.post("/summarize_url")` to the sidecar and
redeploy, but the model never calls it. List three things that could be
wrong.**

Three independent failure points, all from §2.1–§2.2. (a) A missing or
wrong `operation_id` — without it FastAPI auto-generates an unusable name,
so even if the tool is discovered the model can't call it cleanly (contrast
the explicit `operation_id="web_search"` at
[`app.py:227`](../../tools-server/app.py#L227)). (b) `include_in_schema=False`
left on (the flag at [`app.py:493`](../../tools-server/app.py#L493) on the
internal routes), which hides the route from `/openapi.json`, so discovery
never sees it. (c) Discovery already ran: Audrey reads `/openapi.json` *once
at startup*, so a route added after that boot is invisible until you hit the
admin rediscover route or restart Audrey — the `tools=0`-style staleness from
§1.1.

**2. `kb_search` returns a 502 to the model. Whose fault — the sidecar or
Audrey?**

Audrey's (or the network between them), not the sidecar. `kb_search` is a
*proxy* tool ([`app.py:269`](../../tools-server/app.py#L269)): its handler
reaches *back into* Audrey's `/v1/kb/query` via the
`app.state.audrey` httpx client. A 502 means that upstream call failed — the
sidecar is up and serving, but the Audrey KB endpoint it depends on isn't
answering. This is the proxy kind's signature from §2.3: a failure here lives
on the *other* side of the wire than the tool's name suggests.

**3. Why can the model call `chat_history_search` but not
`chat_history/prune`?**

`chat_history_search` ([`app.py:427`](../../tools-server/app.py#L427)) is a
normal route — visible in `/openapi.json`, so discovery turns it into a tool.
`chat_history/prune` ([`app.py:493`](../../tools-server/app.py#L493)) is
declared `include_in_schema=False`, so it never appears in the schema Audrey
reads — the model has no name to call. The route still works over HTTP for the
admin operator who knows the path; it's just not a *tool*. This is the §2.2
privilege boundary: schema-visible routes are tools, `include_in_schema=False`
routes are internal plumbing.

**4. A user reports seeing another user's stored memory in a recall. Where
do you look first?**

The `user` payload filter, in two places. First, `recall` and `search` must
filter on the exact `user` keyword field — see the `FieldCondition(key="user", …)`
in `recall` ([`db.py:241`](../../tools-server/db.py#L241)) and `search`
([`db.py:279`](../../tools-server/db.py#L279)) — never a substring of `tags`.
Second, confirm Audrey's dispatch is still overriding the model-supplied `user`
argument with the *authenticated* user: the `_USER_SCOPED_TOOLS` membership
check at [`dispatch.py:130`](../../src/audrey/tools/dispatch.py#L130) is what
forces that override. If the model's own `user` value ever reached the store
unmodified, scoping would be defeated — a model could read another account by
guessing its id.

**5. Brave starts rate-limiting you mid-conversation. What does the model
see, and what keeps it from taking down the whole request?**

After the retry budget exhausts, `brave.py` raises `BraveRateLimitError`
([`brave.py:92`](../../tools-server/brave.py#L92)), and the `web_search`
handler converts it to a 503
([`app.py:241`](../../tools-server/app.py#L241)). The model doesn't see a
crash — it sees a *failed tool result*, which the ReAct loop (Lesson 9) feeds
back like any other: the model can apologize, try a different approach, or
answer without the web. That's the failure-isolation payoff of the separate
process from §1.1 — one tool degraded, the request lives.

**6. You set `CHAT_ARCHIVE_CHUNK_OVERLAP_CHARS=3000` with the default
`MAX_CHARS=2500`. What happens, and when?**

The sidecar refuses to start. The `Settings` validator
([`settings.py:57`](../../tools-server/settings.py#L57)) raises at boot
because overlap ≥ max_chars, naming both env vars in the message. Before this
validator existed, the bad config would have crashed *later* — on the first
archive write large enough to trigger a hard split, where the chunk step
`max_chars - overlap` goes ≤ 0 — a much harder failure to trace back to its
cause. This is fail-fast at boot turning a lurking runtime bug into an obvious
config error (the §2.5 pattern).

## That's it for the course

That was the last subsystem — nice work seeing the course through to the
end. The course now traces a request from the
public route (Lesson 15), through classification and routing (Lesson 7),
deep mode and the tool/ReAct loop (Lessons 8–9), the model and KB layers
(Lessons 6, 11–12), per-user context (Lesson 13), and fair scheduling
(Lesson 14) —
and out to the tools that live on the other side of the wire, here. You
have seen every load-bearing file in Audrey and the sidecar that serves it.

The remaining way to deepen this knowledge isn't another lesson — it's
maintenance: when a bug surfaces or a feature is needed, you now have the
map to find the right file and the *why* behind its shape.
