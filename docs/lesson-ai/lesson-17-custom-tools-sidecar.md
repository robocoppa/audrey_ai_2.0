# Lesson 17 — The custom-tools sidecar

Throughout this course Audrey has been *calling tools* — `web_search`,
`kb_search`, `memory_store`, `chat_history_search`. Lesson 10 showed the
ReAct loop that decides to call them; Lesson 11 showed the wire protocol.
But every one of those calls crosses a network hop to a service we've
treated as a black box. This lesson opens it.

That service is the **custom-tools sidecar**: a small, separate FastAPI
app, in its own container, that serves the tools Audrey's models call. By
the end you should be able to add a tool, fix one, and reason about which
side of the wire a failure lives on.

## 1. Context

### 1.1 What this is, and why it's separate

The sidecar is **not part of Audrey**. It's a second process:
[`tools-server/app.py`](../../tools-server/app.py) is its own
`FastAPI(...)` instance, with its own dependencies, lifespan and container.
Audrey talks to it over HTTP on the internal network, the same way it talks
to Ollama or Qdrant. Why split it out?

- **Dependency isolation.** What a tool needs stays with the tool. The
  HTML-to-text extractor behind `web_fetch`, for one, exists only in the
  sidecar's image.
- **Failure isolation.** A tool that wedges — a hung search, a Qdrant
  timeout — degrades *one tool*, not the orchestrator. Inside the sidecar
  the idea repeats one level down: A backend that fails takes out the tools
  that need it, never the process. Even web search is optional; with no
  search provider configured, only `web_search` is missing (§2.4).
- **It's the tool *registry*, kept live.** Audrey doesn't hardcode which
  tools exist. It reads the sidecar's `/openapi.json` at startup and again
  every minute, and turns each declared tool route into a tool definition
  for the model. The same document reports which of the sidecar's
  components are working, so a tool whose backend is down stops being
  offered until the backend recovers. (Lesson 10 covered discovery from
  Audrey's side; this is the other end.)

That last point is the one to hold onto: **the sidecar's schema-visible
POST routes tagged `tools` are the model's tools**, as long as Audrey holds
a declaration for each (§2.1). Health and internal routes, and routes
hidden from OpenAPI, remain service plumbing.

> **Startup order.** `compose.yaml` starts Audrey only after the sidecar
> passes its healthcheck, which calls the
> [`health`](../../tools-server/app.py#L550) route. If the first discovery
> still comes back empty or partial, Audrey retries every few seconds for
> about two minutes, then settles into its once-a-minute refresh. Lesson 5
> walked that loop.

### 1.2 The files

| Job | File | What it holds |
|---|---|---|
| The app | [`app.py`](../../tools-server/app.py), [`settings.py`](../../tools-server/settings.py) | The core tool routes and their schemas, the lifespan, the capability check; environment-driven settings |
| Search and the web | [`brave.py`](../../tools-server/brave.py), [`searxng.py`](../../tools-server/searxng.py), [`fetch.py`](../../tools-server/fetch.py) | The two search providers behind `web_search`; the guarded page fetcher behind `web_fetch` |
| State the sidecar owns | [`db.py`](../../tools-server/db.py), [`chat_archive.py`](../../tools-server/chat_archive.py) | Durable memory in Qdrant; the chat archive, with SQLite as its source and Qdrant as its index |
| | [`text_embedding.py`](../../tools-server/text_embedding.py), [`memory_backup.py`](../../tools-server/memory_backup.py) | The pinned embedding transport both stores use; scheduled memory snapshots |
| Health | [`capabilities.py`](../../tools-server/capabilities.py) | Which components work right now, and the supervisor that keeps that true |
| Proxies back to Audrey | [`evidence_tools.py`](../../tools-server/evidence_tools.py), [`reference_tools.py`](../../tools-server/reference_tools.py) | Evidence tools (tables, PDF pages, video frames, past conversations) and reference-library tools, which forward to Audrey's own routes |
| Local compute | [`calculator.py`](../../tools-server/calculator.py) | The bounded calculator, built on a core shared with Audrey |

This lesson reads the app, search, memory and health. `fetch.py` gets a
short look (§2.5); a later lesson on outbound requests reads it in full.
The chat archive's write path was covered in
[Lesson 14](lesson-14-memory-and-context-injection.md), and its read side
is §2.8. The proxy tools belong with the lessons on files and reference
libraries.

## 2. Read-along

### 2.1 The anatomy of a tool route

Every tool the model can call is one FastAPI route here plus one
declaration in Audrey. `web_search` is the cleanest example; its route is
at [app.py:666](../../tools-server/app.py#L666):

```python
@app.post(
    "/web_search",
    operation_id="web_search",                          # ② tool name
    response_model=WebSearchResponse,                   # ③ response schema
    tags=["tools"],                                     # ④ discovery-visible tag
    summary="Search the web using the configured Brave and/or SearXNG providers",
    description="Query the public web for current information. ...",  # ⑤ model-facing docs
)
async def web_search(req: WebSearchRequest) -> WebSearchResponse:  # ① request schema
    _require_capabilities("web_search")                 # ⑥ capability check
    ...
```

Each part has a job, and most of them are for the model or the discovery
machinery, not for you:

1. **The request schema**,
   [`WebSearchRequest`](../../tools-server/app.py#L261), becomes the tool's
   JSON-Schema parameters. Its `Field` constraints are advertised to the
   model *and* enforced by FastAPI: A model that sends `count: 50` gets a
   422 before your handler runs.
2. **`operation_id`** is the literal tool name, and the key Audrey looks the
   declaration up by. **It's load-bearing**: Change it and you rename the
   tool, orphaning its declaration and breaking any prompt or skill that
   names it.
3. **`response_model`** shapes what comes back and documents it in
   `/openapi.json`.
4. **`tags=["tools"]`** marks intent. Discovery only considers POST
   operations tagged `tools`.
5. **`summary` / `description`** are model-facing documentation, sent on
   *every request that offers the tool*. This is prompt real estate, where
   you tell the model *when* to use the tool. (The description on
   [`chat_history_search`](../../tools-server/app.py#L980) is half "use only
   when…", steering the model away from over-calling it.)
6. **The capability check** is the handler's first line. It names the
   components the route can't work without, and
   [`_require_capabilities`](../../tools-server/app.py#L230) refuses the
   call if any is down, with a 503 that lists them and a `Retry-After`
   header. "Can't run right now" is a different answer from "ran and
   failed".
7. **The declaration** lives in Audrey, in
   [`TOOL_DECLARATIONS`](../../src/audrey/tools/discovery.py#L100): how the
   person's identity is bound into a call, and which components the tool
   depends on. Three entries, from
   [discovery.py:123](../../src/audrey/tools/discovery.py#L123):

```python
_declare("web_search", dependencies=("web_search",)),
_declare("web_fetch", dependencies=("web_fetch",)),
_declare(
    "kb_search",
    user_scope=ToolUserScope.ARGUMENT,
    dependencies=("audrey_kb", "qdrant", "text_embedding"),
),
```

Discovery checks each `tools` route against its declaration. A `user`
field the declaration doesn't bind, or a binding the schema can't carry, is
a policy error, and a `tools` route with *no* declaration rejects every tool
from the server, not just the new one. Lesson 10 explained why failing
closed is right; the working rule is that a declaration ships with its route
or before it, never after.

The check and the declaration name the same components on purpose, because
they do different jobs. The declaration decides whether the tool is
*offered*: Discovery leaves out any tool whose dependency the sidecar
reports down. The check decides whether a *call* is served, which matters in
the gap before Audrey notices a failure (§2.7). When a route gains a
dependency, both lists gain it.

### 2.2 Concept spotlight: hiding routes from the model

Not every route should be a tool. The chat archive has three routes the
model must **never** call — writing a turn, pruning old data, reading
stats. They're for Audrey's archive client and the admin operator only, and
the mechanism is one flag ([app.py:1491](../../tools-server/app.py#L1491)):

```python
@app.post("/chat_history/archive", include_in_schema=False, tags=["internal"])
async def chat_history_archive(req: ArchiveTurnRequest) -> dict[str, Any]:
    ...
```

`include_in_schema=False` keeps the route out of `/openapi.json`. Since
Audrey builds the model's tool list from that document, the model cannot see
the route, so it cannot call it. The route still works over HTTP for anyone
who knows the path; Audrey's archive client posts to it after every turn.
When you add a route, that one flag decides which side of the line it's on.

Hiding a route isn't protecting it, though, and the tool routes aren't
protected either. Many of them take a `user` value and trust it, which is
safe only because nothing outside the internal network can reach the
sidecar. It publishes no port (the comment on
[`custom-tools`](../../compose.yaml#L135) in `compose.yaml` says why),
Audrey is its only caller, and Audrey fills in `user` itself before every
call (§2.6). The routes that export or delete a person's data add a second
lock, a check of Audrey's service token
([`_require_internal_service`](../../tools-server/app.py#L1203)). The rule
that follows: Never publish the sidecar's port, not even for debugging.

### 2.3 The four kinds of tool

The model-facing tools look uniform from §2.1, but underneath they fall
into four groups:

| Kind | Tools | What the handler does |
|---|---|---|
| **Proxy** | `kb_search`, `kb_image_search`, `list_my_files`, `get_file_text`, the evidence and reference-library tools | Forwards to one of Audrey's own routes |
| **External** | `web_search`, `web_fetch` | Reaches the outside web: a search provider (§2.4) or a model-chosen page (§2.5) |
| **Stateful** | `memory_*`, `chat_history_search` | Reads and writes state the sidecar owns, in Qdrant and SQLite (§2.6, §2.8) |
| **Local compute** | `calculate` | Evaluates an expression in-process, with nothing behind it |

The **proxy** kind is the largest, and the counterintuitive one: The
sidecar calls *back into Audrey*. Here's the core of
[`kb_search`](../../tools-server/app.py#L767):

```python
async def kb_search(req: KBSearchRequest) -> KBSearchResponse:
    _require_capabilities("audrey_kb", "qdrant", "text_embedding")
    client: httpx.AsyncClient = app.state.audrey      # built in the lifespan
    ...                                               # payload: query, top_k, user
    try:
        r = await client.post("/v1/kb/query", json=payload)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Audrey KB unreachable: {e}")
    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    ...                                               # relay results and any notice
```

The knowledge base — its collections, embedders and upload records — lives
on Audrey's side (Lessons 12–13). The sidecar doesn't own it; it *exposes*
it as a tool. So one KB search runs:

```text
model → Audrey (ReAct loop) → sidecar (kb_search) → Audrey (/v1/kb/query) → Qdrant
```

It looks circular, but the hops do different jobs: The first turns "the
model wants to search" into a tool call, and the second does the retrieval.

**How Audrey trusts the second hop.** `/v1/kb/query` also serves people,
and for them it searches as whoever is signed in. The sidecar's client
carries Audrey's service token in an `X-Audrey-Service-Token` header, and
[`kb_query`](../../src/audrey/routes/kb.py#L435) honors the `user` in the
body only from a caller holding that token; anyone else searches as
themselves. So the chain holds end to end: Audrey's dispatch puts the
authenticated person in `user`, the sidecar relays it, and Audrey accepts it
only from the sidecar.

The status code tells you where a failure lives. A **503 naming
components** is the sidecar's own capability check, refusing before it
tries. A **502** means the sidecar couldn't reach Audrey at all: a refused
connection, DNS, a timeout. **Any other error status** is Audrey's KB route
rejecting the query, relayed as is. `kb_image_search` has the same shape
against `/v1/kb/query/image`, and the other proxies follow the same pattern
against their own Audrey routes.

### 2.4 Web search: two providers, either one enough

`web_search` can use two providers: Brave, a paid API with a key and a
monthly quota, and SearXNG, a self-hosted search aggregator with no key and
no per-query cost. Either one is enough. The sidecar's
[`lifespan`](../../tools-server/app.py#L87) builds a client only for a
provider that's configured, and with neither, it marks the `web_search`
capability unavailable and starts everything else. A missing search key is
a reason to lose search, not a reason to lose memory and the knowledge base
with it.

**Picking a provider.** The handler picks a primary and a fallback for each
query ([app.py:694](../../tools-server/app.py#L694)):

```python
if brave is None:
    if searxng is None:
        raise HTTPException(status_code=503, ...)
    primary, other = searxng, None
elif settings.web_search_alternate and searxng is not None and _prefer_searxng(req.query):
    primary, other = searxng, brave
else:
    primary, other = brave, searxng
```

A single configured provider takes every query. With both, and alternation
on (the default), [`_prefer_searxng`](../../tools-server/app.py#L557)
hashes the normalized query with `blake2b` and lets one bit pick. The hash
is stable, unlike Python's per-process salted `hash()`, so a query goes to
the same provider restart after restart: Repeats hit that provider's cache,
and no shared counter is needed. The split halves the paid provider's quota
use, and it decorrelates the two, so a bad minute on one provider no longer
sinks every search a deep panel fires at once.

**Falling back.**
[`_search_with_fallback`](../../tools-server/app.py#L603) asks the other
provider when the primary fails with a recoverable error (Brave's quota, a
rate limit or an upstream failure, or any SearXNG error) and when it
succeeds with *no results*:

| Primary | The other provider | The model gets |
|---|---|---|
| results | not asked | the primary's results |
| no results | results | the other's results |
| no results | no results, an error, or none configured | an empty list, with a 200 |
| a recoverable error | answers | whatever the other returned |
| a recoverable error | an error, or none configured | a 503 |

Two principles sit in that table. **An empty result isn't proof that
nothing exists**: An aggregator that returns nothing is often being
throttled by its own upstream engines, so empty earns a second opinion. And
**an empty result isn't an error**: When both providers looked and found
nothing, the model gets an honest empty list, not a 503. "I looked and found
nothing" and "I couldn't look" are different answers. (Memory search draws
the same line, §2.6.)

**Each client protects itself**, in its own way, because the providers
differ:

| | Brave ([`brave.py`](../../tools-server/brave.py)) | SearXNG ([`searxng.py`](../../tools-server/searxng.py)) |
|---|---|---|
| Cache | a day by default, keyed on query and count, because `count` is sent upstream | short, keyed on the query alone, because `count` only slices the answer locally; an empty answer is never cached |
| Retries | rate limits and HTTP errors, with exponential backoff; an exhausted quota isn't retried, so the fallback starts at once | transport and HTTP errors, with a shorter backoff, plus one extra try for an empty answer |

Both caches are LRU-bounded. SearXNG's extra try for an empty answer, and
its refusal to cache one, come from the first principle: An empty answer is
often transient, and a cached one would starve every identical query until
the entry expired.

From the model's seat, a search outage is a tool that returned an error.
The ReAct loop (Lesson 10) feeds it back like any other result, and the
model answers from what it has or says it couldn't search. One tool
degraded; the request lives.

### 2.5 `fetch.py`: when the model picks the URL

`web_search` returns a title, a URL and a short snippet: enough to cite, not
to *read*. `web_fetch` ([`fetch.py`](../../tools-server/fetch.py)) opens one
page and returns its main text. That makes it the sidecar's most
security-sensitive route, because **the model supplies the URL**, and the
fetcher runs on the internal network next to Qdrant and Ollama. Steering it
at those services is **SSRF** (server-side request forgery), and `fetch.py`
is a stack of guards against it:

| Guard | Stops |
|---|---|
| scheme allowlist (http/https) | `file://`, `gopher://` |
| resolve the host once, require every address to be public, connect to that address | `qdrant:6333`, `127.0.0.1`, `2130706433`, and a name that re-resolves inward between check and connect |
| follow redirects by hand, re-checking each hop | `evil.com` → `302` → `qdrant:6333` |
| size caps on the download and the decompressed body, a content-type gate, timeouts, a concurrency cap | oversized, compressed-bomb, slow or binary responses, and bursts |

Two rows close classic holes: Followed automatically, a redirect could land
inside after the first URL passed the check, and a second DNS lookup at
connect time could answer "internal" after the check saw "public" (DNS
rebinding). A later lesson on outbound requests reads the file in full.

### 2.6 `db.py`: the durable memory store

[`db.py`](../../tools-server/db.py) backs `memory_store`, `memory_recall`
and `memory_search`. Memories live in Qdrant, embedded with the model
pinned in the shared config's `text_embedding` profile, the same one
Audrey's knowledge base uses for text.

**The vector space is pinned, and checked.** Lesson 3 introduced the idea;
this is the enforcement. Every embedding call,
[`embed_text`](../../tools-server/text_embedding.py#L37), checks the
installed model's digest against the pinned one, before embedding and again
after. And [`_ensure_collection`](../../tools-server/db.py#L226) refuses a
collection whose size, distance or recorded identity doesn't match the
profile. Vectors from two models still compare numerically, but the numbers
mean nothing; refusing, and asking for a reindex, beats searching quietly
wrong.

**Writes.** A memory is a `(key, value, tags)` triple. Its point id,
[`_point_id`](../../tools-server/db.py#L66), is `uuid5` over `user|key`, so
storing the same key again overwrites instead of duplicating. The text that
gets embedded, from [`_embedding_text`](../../tools-server/db.py#L88), is
`"{key}: {value} [tags: …]"`: the key first, as a lexical hook for short
queries, and the `user:` tag stripped, because it's scope metadata, not
meaning.

**Reads.** `recall` is an exact-key payload lookup that never embeds
anything. [`search`](../../tools-server/db.py#L611) first tries the whole
query as a key, treating spaces, underscores and hyphens as one separator
([`_literal_key_variants`](../../tools-server/db.py#L71)), because a model
that saved `favorite_language` may later search for "favorite language".
Only when no key matches does it run a vector search, above a similarity
floor from the same profile. The floor is high on purpose: A memory hit is
injected into the prompt as a fact about the person, so a wrong hit misleads
the model, and missing a memory is the cheaper mistake.

**Scope: exact, and purge-aware.** At write time the store copies the user
id out of the `user:` tag into its own keyword-indexed field, because a
substring filter on `tags` would let `user:al` match `user:alice`. Every
read filters through [`_user_filter`](../../tools-server/db.py#L119):

```python
def _user_filter(user: str, *, visible_after: str = "") -> qm.Filter:
    ...
    must: list[qm.FieldCondition] = [
        qm.FieldCondition(key="user", match=qm.MatchValue(value=user)),
    ]
    if visible_after:
        must.append(qm.FieldCondition(
            key="updated_at",
            range=qm.DatetimeRange(gt=visible_after),
        ))
    return qm.Filter(must=must)
```

`visible_after` is the person's purge cutoff, recorded in the chat archive's
database when someone purges their stored data. Older memories stay hidden
from then on, even while the deletion is still working through, and newer
ones are new activity, so they show. A memory with no timestamp fails the
range check and stays hidden: The filter fails closed. The cutoff living in
the archive is why the memory reads declare `chat_archive_source` as a
dependency.

**The model never controls the scope.** Audrey's dispatch binds identity
exactly as each declaration says
([dispatch.py:280](../../src/audrey/tools/dispatch.py#L280)):

```python
if user_id and spec.user_scope is ToolUserScope.TAGS:
    args["tags"] = _force_user_tag(str(args.get("tags") or ""), user_id)
elif user_id and spec.user_scope is ToolUserScope.ARGUMENT:
    args["user"] = user_id
```

For argument-scoped tools such as `memory_search`, the model's `user` is
replaced with the authenticated person; for the tag-scoped `memory_store`,
any `user:` tag the model wrote is replaced. A scoped tool with no
authenticated caller isn't called at all. The sidecar trusts the scope it
receives because, as §2.2 showed, Audrey is its only caller.

**"I couldn't look" isn't "you have none."** When the embedder fails,
[`memory_search`](../../tools-server/app.py#L903) answers 503, not an empty
list, which would claim there's nothing to recall. Auto-recall treats the
error as skip-and-continue, so a turn never breaks; the skip just stops
being silent.

**Backups.**
[`MemoryBackupScheduler`](../../tools-server/memory_backup.py#L384) takes a
snapshot of the memory collection at startup and then daily by default, and
no chat waits for it. Each bundle also copies the chat archive's database,
because the purge cutoffs live there: Restored memories without them could
show what a person had purged. Checking a bundle is a separate operator step
that restores it into a disposable collection, verifies every point's owner
and id, and deletes it; nothing can restore over the live collection.

### 2.7 Startup, supervision and shutdown

[`settings.py`](../../tools-server/settings.py) is Pydantic Settings (Lesson
2 has the pattern). A validator,
[`_check_chunk_overlap`](../../tools-server/settings.py#L133), refuses two
contradictions at boot: an archive chunk overlap at or above the chunk size,
which would crash a large write hours later, and a size cap the code doesn't
enforce, which would silently do nothing. It's Lesson 5's rule in a second
process: Fail fast when running would be a lie, and degrade when a useful
subset still works.

The [`lifespan`](../../tools-server/app.py#L87) builds the long-lived
clients once: the configured search clients, the memory store, an `httpx`
client for Audrey that carries the service token, and the chat archive with
its maintenance worker. It hangs them on `app.state` and starts two
background jobs, the backup scheduler and the capability supervisor.

**Capabilities.** [`capabilities.py`](../../tools-server/capabilities.py)
keeps a small registry of named components. Stateless ones such as
`web_fetch` or `audrey_kb` start available, because their clients work as
soon as they exist. Stateful ones (`qdrant`, `memory`, the two embedders and
the archive) start pending, and become available only when their own checks
pass. The [`CapabilitySupervisor`](../../tools-server/capabilities.py#L108)
runs a pass at startup and then every few seconds, checking each component
on its own so one failure never blocks another. The telling branch is
Qdrant failing ([capabilities.py:178](../../tools-server/capabilities.py#L178)):

```python
try:
    await self._memory.probe_qdrant(timeout_s=self._probe_timeout_s)
except Exception as exc:  # noqa: BLE001 — backend clients raise a wide tree
    self.registry.set_unavailable("qdrant", "connection_failed")
    # A restart may replace collections while the service is down.
    # Recovery must revalidate their pinned vector-space identity.
    self._memory_ready = False
    self._archive_index_ready = False
    self.registry.set_unavailable("memory", "backend_unavailable")
    self.registry.set_unavailable("chat_archive", "backend_unavailable")
    ...
    return
```

Losing Qdrant marks the components that live in it unavailable, and also
*forgets* that their collections were checked. A Qdrant that comes back may
not hold what it held before, so recovery re-runs §2.6's identity check
before anything is marked available again.

**Two speeds.** A failed component is retried on every pass. A healthy one
is rechecked on a slow timer, five minutes by default. The recheck exists
because a component that was healthy at startup can break later, and a
readiness flag nobody re-examines turns into a lie. The timer is slow
because rechecking an embedder means running it, which costs GPU time.
History's embedder is checked separately from memory's: It can be a
different model, and a warm memory embedder proves nothing about it.

**Publishing the state.** FastAPI normally caches `/openapi.json`; the
sidecar swaps in [`_dynamic_openapi`](../../tools-server/app.py#L212),
which rebuilds it per request and attaches the registry's snapshot as
`x-audrey-capabilities`. Audrey's discovery reads the snapshot alongside
each declaration's dependencies: A tool whose dependency is down stays in
Audrey's registry, marked unavailable, and isn't offered. Audrey refreshes
once a minute ([`_maintain_tool_discovery`](../../src/audrey/main.py#L460)),
and each refresh also re-evaluates skills, so a skill whose tools vanished
shows as degraded. The loop, its fast retries after a cold start, and an
admin's manual rediscovery all go through one function under one lock, so
an older snapshot can't land after a newer one. And an empty snapshot never
replaces the live list: Empty can mean a restarting sidecar or a rejected
catalogue, so it's inconclusive, while an explicit "unavailable" still
withdraws its tools.

**Propagation is eventual.** If Qdrant drops, the sidecar notices within
seconds and Audrey stops offering the affected tools within about a minute.
If a healthy embedder breaks, the sidecar notices at its next recheck, so
the worst case is about six minutes. In between, a call meets the capability
check, which answers 503 and names the component. Withdrawal keeps the model
from being offered a tool that can't work; the request-time check keeps the
window honest.

**Shutdown** stops the background jobs first (backups, then the supervisor,
which also stops the archive's maintenance worker), then closes the clients,
skipping any search client that was never built. It's the rule Lesson 5
showed for Audrey: Stop the work before the clients it uses.

### 2.8 The chat archive's read side

[Lesson 14](lesson-14-memory-and-context-injection.md) covered how a turn
gets written. Three more operations sit on top of that store:

- **`search`** ([`search`](../../tools-server/chat_archive.py#L1161)) backs
  `chat_history_search`. It runs ranked keyword matching over the SQLite
  source first, which also finds chunks still waiting to be indexed, then
  fills the remaining slots with semantic hits, filtered by `user`, the
  purge cutoff and an optional date range. Keyword hits carry no score and
  still come back when the embedder is down. The semantic floor, from
  history's own embedding profile, favors recall: Lookup is human-triggered
  ("did we talk about X?") and the answer checks the excerpts, whereas a
  memory false positive becomes a "fact" in the prompt.
- **`prune`** ([`prune`](../../tools-server/chat_archive.py#L2084)) applies
  retention through an outbox: a bounded batch is deleted from Qdrant first,
  and the SQLite rows are finalized only once Qdrant has acknowledged.
  Vectors go before source, so a crash midway leaves recoverable rows, never
  vectors pointing at deleted text. A maintenance pass does this every few
  minutes; the admin route runs one on demand.
- **`stats`** ([`stats`](../../tools-server/chat_archive.py#L2305)) returns
  row counts, including `chunks_unindexed`: chunks that haven't made it into
  Qdrant yet. Maintenance retries them, so a count that stays high means
  indexing keeps failing.

The last two sit behind the internal routes from §2.2.

## 3. Comprehension questions

**1. You add a route `/summarize_url` to the sidecar, tagged `tools` and
with its own `operation_id`, and redeploy the sidecar. The model never calls
it. What could be wrong?**

Start with Audrey's log. If the route has no entry in
[`TOOL_DECLARATIONS`](../../src/audrey/tools/discovery.py#L100), discovery
rejects the sidecar's whole catalogue and logs an error naming the route. A
running Audrey treats that empty answer as inconclusive and keeps its last
good tool list, so nothing else looks broken, but the new tool never appears
and availability stops updating. On its next restart, Audrey starts with no
sidecar tools at all. Without that error, check that the route is in the
schema (§2.2) and that no dependency in its declaration is reported down,
which keeps a declared tool listed but unoffered (§2.7). Waiting isn't the
fix: A correctly declared route appears within about a minute.

**2. `kb_search` returns a 502 to the model. Whose fault — the sidecar or
Audrey?**

Audrey's, or the network between them. `kb_search` is a *proxy* tool
([`kb_search`](../../tools-server/app.py#L767)) that reaches back into
Audrey's `/v1/kb/query`, and a 502 means that upstream call got no answer;
the sidecar itself is up. Compare a 503 naming `qdrant` or
`text_embedding`: That's the sidecar's own capability check, refusing before
it tries. This is the proxy kind's signature from §2.3: A failure can live
on the other side of the wire from the one the tool's name suggests.

**3. Qdrant restarts and is unreachable for two minutes. Which tools leave
the model's list, which keep working, and what happens when it's back?**

Within seconds the supervisor's probe fails, and the sidecar marks `qdrant`,
`memory` and the archive's index unavailable (§2.7). Within about a minute,
Audrey's refresh withdraws every tool whose declaration depends on one of
those: the two knowledge-base searches, the three memory tools,
`chat_history_search` and the reference search. Web search, the calculator
and the file tools stay on offer, because their declarations don't name
Qdrant; a call that slips into the gap gets a 503 naming the component. When
Qdrant answers again, the supervisor re-checks the collections' vector-space
identity before marking them available, and Audrey's next refresh offers the
tools again. Nothing restarts, and nobody rediscovers by hand.

**4. A person reports seeing someone else's stored memory in a recall. Where
do you look first?**

At the scope, in each of its three layers. In the store, every read must
filter on the exact `user` field through
[`_user_filter`](../../tools-server/db.py#L119), never a substring of
`tags`. In Audrey's dispatch, `memory_recall` and `memory_search` must still
be declared argument-scoped, so the model's `user` is overwritten (§2.6); a
declaration that stopped binding `user` while the schema still carries it
would be refused at discovery. And in the network: The routes trust the
`user` they're sent, so if anything besides Audrey can reach the sidecar,
for example through a published port, anyone there can name any account.

**5. Brave starts answering with 429s in the middle of a research run, and
SearXNG is configured. What does the model see?**

Usually nothing at all. For queries that hash to Brave as primary, Brave's
client retries with backoff and gives up, the handler falls back to
SearXNG, and the model gets SearXNG's results; queries that hash to SearXNG
never touch Brave. An exhausted quota (a 402) skips the retries, so the
fallback starts at once. Only if SearXNG fails too does the model get a 503,
which the ReAct loop treats as a failed tool result, and the run goes on.
Without SearXNG configured, the 429s reach the model as 503s once Brave's
retries run out.

## That's the course so far

This is the last lesson published so far, and the sidecar is a fitting place
to pause: It's the far end of the request path. You've followed a message from
the browser and the `/v1` door (Lessons 4 and 16), through routing (Lesson 8),
the model layer (Lessons 6 and 7), deep and research modes (Lesson 9), the
tool loop (Lessons 10 and 11), the knowledge base and per-user context
(Lessons 12–14) and fair scheduling (Lesson 15), out to the tools this lesson
read. Lessons on the application around the pipeline are on the way.
