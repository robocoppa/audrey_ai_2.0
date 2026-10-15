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

- **Now and then, a GET to Cloudflare** for the public keys that verify
  a browser's sign-in. Audrey caches them, so most requests skip this,
  but when the cache runs out it's an outbound call Audrey must await.
- **A GET to Ollama's `/api/tags`**, which lists the installed models.
  Audrey uses it to check that Ollama is up and to build the list of
  models a person may pick.
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

When you use Audrey in a browser, Cloudflare Access adds a signed
token to each request, and Audrey checks the signature with
Cloudflare's *public* keys before trusting it. Those keys are
published as JSON at a fixed URL on your Cloudflare team's domain,
and Cloudflare changes them from time to time, so Audrey fetches
them over HTTP and caches them. (What the keys are for, and how the
signature check works, belongs to the lesson on identity later in
the course.) Here's the fetch,
[`_fetch_keys`](../../src/audrey/identity/cloudflare_access.py#L155),
with the part that converts each key into a usable object left out:

```python
async def _fetch_keys(self) -> dict[str, jwt.PyJWK]:
    try:
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            response = await client.get(self.jwks_url)
            response.raise_for_status()
            payload: Any = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise CloudflareAccessUnavailableError(
            "Cloudflare Access signing keys are unavailable"
        ) from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("keys"), list):
        raise CloudflareAccessUnavailableError("Cloudflare Access JWKS is malformed")
    ...
```

Read top to bottom. The whole function is one HTTP round trip,
wrapped in error handling that turns every way the network can let
you down into one error Audrey understands. Key things to notice:

- `async with httpx.AsyncClient(timeout=self.timeout_s) as client:` —
  opens a client with a hard timeout (five seconds by default). The
  `async with` guarantees the connection pool is cleaned up even if
  the request raises.
- `await client.get(...)` — yields control to the event loop while
  Cloudflare answers. Other users' requests can make progress here.
- `response.raise_for_status()` — turns a 4xx or 5xx answer into an
  exception, so a bad status lands in the same `except` as a network
  failure.
- `except (httpx.HTTPError, ValueError)` — one clause catches all
  three ways this can fail: the network (`httpx.HTTPError` covers
  timeouts, refused connections and DNS failures), a bad status (the
  exception `raise_for_status` raised is an `httpx.HTTPError` too),
  and a body that isn't JSON (`response.json()` raises a
  `ValueError`).
- `raise CloudflareAccessUnavailableError(...) from exc` — Audrey's
  own exception type. Callers don't need to know about httpx; they
  need to know "the keys are unavailable." `from exc` keeps the
  original error attached for the log.

Translating errors at the boundary pays off one layer up. The
identity code catches `CloudflareAccessUnavailableError` and answers
`502 Bad Gateway` ("our upstream is broken"), while a token that
fails its check gets `401 Unauthorized` ("you're not signed in"). A
person seeing those two codes knows whether to sign in again or to
wait.

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
- [`src/audrey/identity/cloudflare_access.py`](../../src/audrey/identity/cloudflare_access.py) —
  the signing-key fetch (the example above).
- [`src/audrey/tools/dispatch.py`](../../src/audrey/tools/dispatch.py) —
  dispatching tool calls to the custom-tools server.
- [`src/audrey/kb/embed.py`](../../src/audrey/kb/embed.py) — image
  URL fetch (with the SSRF guards and byte cap).


## 2. Vector search: Qdrant + embedding models

Audrey's knowledge base stores **embeddings** (numeric vectors) of
text chunks and images, and answers queries via **vector similarity
search**. Three pieces:

### What's an embedding?

An **embedding** is a list of numbers (a **vector**) that stands for
what a piece of text or an image means. An embedding model reads the
input and produces the list, always the same length. You don't need to
know how the model works inside.

Think of each list as an address on a map of meanings. The model has
learned to give text about the same thing nearby addresses. "My car
won't start" lands close to "engine trouble", though they share no
words, and far from "a bread recipe". The map just has far more
directions than a street map: hundreds of numbers per address instead
of two.

That turns searching by meaning into measuring distance. Audrey
converts the question the same way, then finds the stored lists closest
to it. Closeness is scored with **cosine similarity**, which runs from
near 1 for very similar meanings down to around 0 for unrelated ones.

- `nomic-embed-text` (run by Ollama) turns text into 768 numbers.
- `clip-ViT-B-32` (CLIP) turns images into 512 numbers, and can put
  text on the same map: A query like "a black dog" lands near photos
  of black dogs, which is how a written question finds a picture.

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

Given `tests/`, pytest finds and runs both functions, then reports what
passed and what failed, showing the offending line for each failure.

### Why Audrey needs pytest

Audrey ships with a hermetic test suite: it needs no live Ollama,
Qdrant or custom-tools, and the whole run takes a minute or two on a
laptop. Several of those tests are *regression guards* for specific bugs
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
runs in a minute or two, and a single test file in seconds, so it's
easy to run before every commit.

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

- [`tests/`](../../tests/) — the full hermetic test suite: no live
  Ollama, Qdrant or custom-tools required.



## That's the foundation

That's the whole toolbox — three lessons, eleven concepts. None of
those sections is the whole story for any single library; each one
has a documentation site you could spend a day on. The point of the
foundations was to get you to the place where every piece of Audrey
you read from here on lands somewhere familiar.


[Lesson 4 — The platform and the request lifecycle](lesson-04-request-lifecycle.md)
is where we stop touring libraries and start walking real code. We
trace one message from the moment you press Send in the browser to
the moment the answer finishes on screen, and see how a bot calling
the API reaches the same pipeline, touching every major component
without going deep on any. Think of it as the map; later lessons fill
in each region.
