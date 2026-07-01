# Lesson 3 — Foundations III: the satellite libraries

**Estimated time:** 50-70 minutes if you read carefully. You can split
it across sessions — sections are self-contained.

**Goal:** know what each *satellite library* — the ones Audrey calls
out to rather than is structured around — does, what problem it
solves, and recognize its shape when you see it. After this lesson
the foundations tour is done and we can start walking real code.

**What this lesson covers:**

1. [httpx](#1-httpx) — the async HTTP client
2. [Vector search: Qdrant + embedding models](#2-vector-search-qdrant--embedding-models)
3. [Prometheus metrics](#3-prometheus-metrics)
4. [pytest](#4-pytest)


## 1. httpx

A modern async-aware HTTP client for Python. Same role as the classic
`requests` library — make HTTP calls to other services — but built to
play nicely with `async`/`await`.

### Why Audrey needs an async HTTP client

Audrey is a *gateway*. It rarely does the actual work itself; mostly
it forwards requests to other services and waits for replies. In one
chat completion, Audrey typically makes:

- **A GET to OWUI** (`http://open-webui:8080/api/v1/auths/`) to
  validate the bearer token. Fast on the local network, but still an
  outbound HTTP call Audrey must await.
- **A GET to Ollama for `/api/tags`** at startup, to find out which
  models are loaded.
- **One or more POSTs to Ollama's `/api/chat`** to actually generate
  tokens. Each can take 10s–3min for cloud models.
- **POSTs to custom-tools** (`http://custom-tools:8001/web_search`,
  `/kb_search`, etc.) when the model decides to call a tool.
- **POSTs to Qdrant** for vector searches.
- **GETs to image CDNs** when the user asks "search the KB for an
  image like this URL."

Most of the time spent in a chat completion is **Audrey waiting for
something else to come back**. That's exactly the situation Lesson 1
§1 made the case for handling with `async`/`await` — but the async
machinery only pays off if every I/O call along the way is *also*
async-aware. A single sync HTTP call buried in an `async def` blocks
the whole event loop and erases the benefit.

That's why httpx specifically: It's the mainstream library that gives
you `await client.post(...)` cleanly, so all the outbound calls in
the list above stay non-blocking. The classic `requests` library would
work *functionally* — the bytes would arrive — but every call would
freeze the event loop for its full duration, and the concurrency story
from Lesson 1 collapses.

### The shape

```python
import httpx

async def fetch():
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get("https://example.com")
        return response.json()
```

Why `AsyncClient` and not just `httpx.get`? The client object holds
connection-pool state — TCP connections, TLS sessions — that can be
reused across multiple requests. Reusing a client is meaningfully
faster than spinning up a fresh connection per call. Use the client
inside an `async with` so the connection pool gets cleaned up when
you're done.

### A real example from Audrey

Here's `_probe_owui` from [`src/audrey/auth.py`](../../src/audrey/auth.py),
which is what runs every time a user makes a chat-completion request
and we need to verify their bearer token. Slightly simplified:

```python
async def _probe_owui(owui_url: str, token: str) -> AuthedUser:
    """Validate a token by asking OWUI who owns it. Raises on failure."""
    url = f"{owui_url.rstrip('/')}/api/v1/auths/"
    try:
        async with httpx.AsyncClient(timeout=5.0) as http:
            r = await http.get(url, headers={"Authorization": f"Bearer {token}"})
    except httpx.HTTPError as e:
        # Network error, DNS failure, timeout — OWUI is broken or unreachable.
        # Return 502 so the client knows it's a backend problem, not their fault.
        raise HTTPException(status_code=502, detail="Auth backend unreachable.") from e

    if r.status_code == 401:
        # OWUI rejected the token.
        raise HTTPException(status_code=401, detail="Token rejected by OWUI.")
    if r.status_code >= 400:
        # OWUI itself errored — log and surface as 502.
        raise HTTPException(status_code=502, detail=f"Auth probe failed ({r.status_code}).")

    body = r.json()
    return AuthedUser(
        email=body["email"],
        role=body["role"].lower(),
        owui_id=body["id"],
    )
```

Read top to bottom. The whole function is one HTTP round trip
wrapped in error handling that translates network-level outcomes
into HTTP-level outcomes for our own caller. Key things to notice:

- `async with httpx.AsyncClient(timeout=5.0) as http:` — opens a
  client with a 5-second hard timeout. The `async with` guarantees
  the connection pool is cleaned up even if the request raises.
- `await http.get(...)` — yields control to the event loop while
  OWUI thinks. Other users' requests can make progress here.
- `raise HTTPException(...)` — FastAPI's way of saying "stop here,
  return this status code to the client." See Lesson 2 §1.

### Useful httpx methods

- `await client.get(url, headers=..., params=..., timeout=...)`
- `await client.post(url, json={...}, headers=...)` — `json=` makes
  httpx serialize the dict and set `Content-Type: application/json`
  for you.
- `await client.stream("GET", url)` — opens the response and lets you
  read its body in chunks instead of all at once. Audrey uses this
  in [`kb/embed.py`](../../src/audrey/kb/embed.py) when fetching a
  user-supplied image URL: Stream the bytes, count them as they
  arrive, abort if the file claims to be a 50 GB "image."

### Errors

- `response.raise_for_status()` — turns 4xx/5xx into an
  `httpx.HTTPStatusError`. Many Audrey call sites use this so a bad
  response becomes an exception you can catch.
- `httpx.TimeoutException` — the request didn't complete within the
  client's timeout.
- `httpx.ConnectError` / `httpx.NetworkError` — couldn't even reach
  the server. All of these subclass `httpx.HTTPError`, so a single
  `except httpx.HTTPError` catches everything network-related.

### Where you'll see it in Audrey

- [`src/audrey/models/ollama.py`](../../src/audrey/models/ollama.py) —
  the entire Ollama client. Every chat completion in the system
  ultimately goes through one of its `chat()` / `chat_stream()`
  methods, which are httpx calls under the hood.
- [`src/audrey/auth.py`](../../src/audrey/auth.py) — OWUI token
  probe (the example above).
- [`src/audrey/tools/dispatch.py`](../../src/audrey/tools/dispatch.py) —
  dispatching tool calls to the custom-tools server.
- [`src/audrey/kb/embed.py`](../../src/audrey/kb/embed.py) — image
  URL fetch (with the SSRF guards and byte cap).


## 2. Vector search: Qdrant + embedding models

Audrey's knowledge base stores **embeddings** (numeric vectors) of
text chunks and images, and answers queries via **vector similarity
search**. Three pieces:

### What's an embedding?

A neural network that turns a piece of text (or an image) into a
fixed-length array of floats. The geometry of these vectors encodes
semantic meaning: Similar concepts produce vectors near each other in
space; unrelated concepts produce vectors far apart.

Concretely:

- `nomic-embed-text` produces 768-dimensional float vectors from
  text. Run via Ollama.
- `clip-ViT-B-32` (CLIP) produces 512-dimensional vectors from
  images **or** from text — both end up in the same vector space, so
  you can search images using a text query.

You don't need to understand how the network works. Treat it as: text
in → vector out. Two vectors are "close" if their cosine similarity
is near 1; "far" if near 0 or negative.

### What's Qdrant?

A vector database. Stores points (each = a vector + a payload
dictionary), indexes them so you can find the N nearest vectors to
a query vector quickly, and supports metadata filters (e.g. "find the
nearest 5 vectors *that also have payload.user = 'alice@example.com'*").

You speak to Qdrant over HTTP via the `qdrant-client` Python library.
Audrey wraps it in [`src/audrey/kb/qdrant.py`](../../src/audrey/kb/qdrant.py).

### Why Audrey needs vector search

Audrey's KB has a large amount of **unique text sources** producing
indexed chunks across many different topic subdirectories. The user
types something like "what's the difference between a flathead and
a Phillips screwdriver" and expects an answer grounded in those
documents — even though the word "Phillips" might not appear in the
most relevant chunk (which is titled "the origin of the slot-head
fastener").

Two non-options:

- **SQL `LIKE` (or full-text search).** Doesn't capture semantic
  similarity. "Phillips screwdriver" would miss any chunk that uses
  "cross-recess driver" or "JIS bit." Synonyms, paraphrases, related
  concepts — all invisible.
- **Brute-force cosine in Python.** Fine for the small case (a
  thousand or so vectors); falls apart well before the KB's actual
  scale. Each query would have to compute every stored vector's
  similarity against the query — possible, but adds non-trivial
  latency to every KB-touching request, and the cost grows linearly
  with collection size.

A vector database indexes the *geometry* of the embedding space, so
a query is sub-100ms regardless of collection size. Qdrant
specifically is what Audrey uses; a full sweep across every stored
point takes only a handful of seconds, so per-query cost is a small
fraction of that.

The metadata-filter capability is also load-bearing. Per-user upload
collections (`kb_user_text_*`) are indexed alongside global ones, and
the filter is what keeps user A's uploads from leaking into user B's
queries.

### How they fit together in Audrey

```
text content ──► nomic-embed-text ──► 768-d vector ──► Qdrant.upsert(point)
                                                             │
                                                             ▼
                                                       kb_text collection

text query ──► nomic-embed-text ──► 768-d query vector ──► Qdrant.search(top_k=5)
                                                                  │
                                                                  ▼
                                                          5 nearest hits + payloads
```

Same flow for images, but with CLIP and the `kb_images` collection.

### Where you'll see it in Audrey

- [`src/audrey/kb/embed.py`](../../src/audrey/kb/embed.py) —
  `TextEmbedder`, `ImageEmbedder`.
- [`src/audrey/kb/qdrant.py`](../../src/audrey/kb/qdrant.py) — the
  Qdrant client wrapper.
- [`src/audrey/kb/ingest.py`](../../src/audrey/kb/ingest.py) — the
  full ingest flow.


## 3. Prometheus metrics

A standard for instrumenting an application with **counters**,
**gauges**, and **histograms** so an external scraper (also called
Prometheus) can collect them periodically and store them as time
series.

### The three metric types

- **Counter** — only goes up (or resets to 0 on restart). "Number of
  requests served." `requests_total.inc()`.
- **Gauge** — goes up and down. "Current cache size." `gauge.set(5)`
  / `.inc()` / `.dec()`.
- **Histogram** — records a distribution of values. "Request latency
  in seconds." `latency_seconds.observe(0.42)`. Internally it's a
  bunch of counters bucketed by value range.

Each metric has a name (`audrey_requests_total`) and optional
**labels** (`method="POST"`, `path="/v1/chat/completions"`). Labels
let you slice the data — but each unique label combination becomes a
separate time series, so high-cardinality labels (like user emails)
explode storage. Audrey is careful about label cardinality
([`metrics.py`](../../src/audrey/metrics.py) has notes).

### Why Audrey needs Prometheus

Audrey runs N concurrent generations across **5 virtual models** with
real cloud-billing implications when deep panel runs. Operationally
you need to be able to answer questions like:

- **"Is the deep panel slower today than last week?"** —
  `audrey_model_seconds_bucket` over time, aggregated by model.
- **"Is fair-gate actually fair?"** — per-user request counts and
  wait times. A real round-robin starvation bug shipped at one
  point and got debugged after the fact; the metric is what would
  have *caught* it if the test hadn't.
- **"Which tool errors most often?"** —
  `audrey_tool_calls_total{tool, status}` is what cross-checks the
  per-worker tool-summary footer the streaming response embeds.
- **"Is anyone using virtual model X?"** — request counts per virtual
  model. Shapes which models are worth keeping warm.
- **"Did Ollama just regress?"** — model-call latency histograms,
  alertable when the p99 jumps.

Without metrics, all of those questions get answered by `grep` on
container logs, which is both slow and lossy (logs rotate). With
Prometheus + Grafana, the same questions are panel queries that
stay accurate over weeks of history.

The four alert rules Audrey ships are all built on these metrics —
they're how the system pages you when something silently breaks (a
model goes offline, the fair-gate queue grows unboundedly, etc.).

### How exposition works

Your app exposes a `/metrics` endpoint that returns the current values
in a specific text format:

```
# HELP audrey_requests_total Total requests served
# TYPE audrey_requests_total counter
audrey_requests_total{method="POST",path="/v1/chat/completions"} 142
```

The Prometheus server scrapes this endpoint every N seconds and
stores each line as a data point. Grafana then queries Prometheus to
draw graphs.

### Where you'll see it in Audrey

- [`src/audrey/metrics.py`](../../src/audrey/metrics.py) — every
  metric definition.
- [`src/audrey/main.py`](../../src/audrey/main.py) — the `/metrics`
  endpoint.
- Throughout the codebase: `pipeline_total.labels(...).inc()`,
  `model_seconds.observe(elapsed)`, etc. — instrumentation sites.
- [`monitoring/`](../../monitoring/) — Prometheus + Grafana compose
  files (separate from the audrey-ai container).



## 4. pytest

The standard Python test framework. Discovers files matching
`test_*.py`, finds functions matching `test_*`, runs them, reports
results.

### The shape

```python
# tests/test_math.py
def test_addition():
    assert 2 + 2 == 4

def test_division_by_zero():
    import pytest
    with pytest.raises(ZeroDivisionError):
        1 / 0
```

Run with `pytest tests/`. Output tells you what passed, what failed,
and shows the offending line for failures.

### Why Audrey needs pytest

Audrey ships with a hermetic test suite (no live Ollama / Qdrant /
OWUI required), wall-clock target sub-second for the whole run.
Several of those tests are *regression guards* for specific bugs
that already bit us in production:

- **`test_two_user_round_robin_skips_last_granted`** — guards the
  fair-gate scheduler. A naive round-robin could starve a user who
  fired three back-to-back requests if a second user joined
  mid-flight. The test reconstructs that scenario with
  `asyncio.Event`s for ordering and asserts user B's request slips
  into slot 2, not slot 4. Without this test, a future refactor
  could silently reintroduce the starvation.
- **`test_authed_user_has_no_id_field`** — guards against an
  `AttributeError` that bit when an early version of the auth wiring
  reached for `me.id` instead of `me.email`. A future maintainer
  adding an `.id` field to `AuthedUser` will hit a deliberate
  failing test that says "this was removed on purpose; if you really
  need it back, delete this test and add a real one." The bug is
  more visible as a refusal to re-add the field than as a runtime
  crash months later.
- **`test_kb_embed_ssrf.py`** — parametrized over a grid of IP
  ranges and URL schemes — guards the SSRF check on the
  user-supplied image URL endpoint. Without it, a regression that
  re-allows requests to internal addresses (loopback, Docker
  subnet, link-local AWS metadata) goes undetected until someone
  exploits it.

Without a test framework, those would be ad-hoc scripts that nobody
runs. With `pytest tests/ -q` in the dev workflow, the whole battery
runs in well under a second — fast enough that it's painless to run
before every commit.

A second thing pytest specifically buys you: parametrization. The
`_BREVITY_CUES` test in `test_reflect.py` covers every entry in the
cue tuple individually — one test definition, one assertion, but
pytest treats each cue as its own reported test case. Without
parametrization, it'd either be one test that catches "any cue
works" without knowing *which* broke, or a stack of nearly-identical
hand-written tests.

### Useful features

- **You write tests using Python's built-in `assert` keyword.** Just
  `assert x == y`. There's nothing pytest-specific to learn here —
  any expression that's truthy passes, any expression that's falsy
  fails the test. When an assertion fails, pytest doesn't just say
  "AssertionError" and quit; it inspects both sides of the comparison
  and prints them so you can see *what* didn't match — e.g. for
  `assert response.status == 200`, the failure output shows both the
  actual status code and the expected `200` side by side. Other
  Python testing frameworks (notably the standard-library `unittest`
  module) make you call methods like `self.assertEqual(x, y)` to get
  the same nice diff; pytest's trick is that it rewrites your plain
  `assert` statements at import time to inject that diagnostic
  behavior, so you don't have to.
- **Fixtures** — reusable test setup. `@pytest.fixture` decorates a
  function; tests that take that function's name as a parameter
  receive its return value:
  ```python
  @pytest.fixture
  def db():
      conn = make_test_db()
      yield conn
      conn.close()

  def test_query(db):
      assert db.query("SELECT 1") == [(1,)]
  ```
- **Parametrization** — run the same test against many inputs:
  ```python
  @pytest.mark.parametrize("input,expected", [(2, 4), (3, 9), (4, 16)])
  def test_square(input, expected):
      assert input ** 2 == expected
  ```
  Each input becomes its own reported test case.

### Async tests

Audrey uses `pytest-asyncio` (a plugin) so async tests just work:

```python
async def test_fetch():
    result = await some_async_function()
    assert result == "expected"
```

The plugin's `asyncio_mode = "auto"` config (in `pyproject.toml`)
means any `async def test_*` is auto-detected — you don't need
`@pytest.mark.asyncio`.

### Where you'll see it in Audrey

- [`tests/`](../../tests/) — the full hermetic test suite (no live
  Ollama / Qdrant / OWUI required). Run with
  `.venv/bin/pytest tests/ -q`.
- The test suite itself is what later lessons use to demonstrate
  testing patterns.



## That's the foundation

That's the whole toolbox — three lessons, eleven concepts. None of
those sections is the whole story for any single library; each one
has a documentation site you could spend a day on. The point of the
foundations was to get you to the place where every piece of Audrey
you read from here on lands somewhere familiar.


[Lesson 4 — The request lifecycle, end-to-end](lesson-04-request-lifecycle.md)
is where we stop touring libraries and start walking real code. We
trace one chat-completion request from "OWUI sends POST" to "user
sees the answer," touching every major component without going deep
on any. Think of it as the map; later lessons fill in each region.
