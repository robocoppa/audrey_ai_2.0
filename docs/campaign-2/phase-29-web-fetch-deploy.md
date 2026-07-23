# Campaign 2 Phase 29 — web_fetch page-opener

A new custom-tools endpoint, `web_fetch`, that opens one http(s) URL and returns
its main readable text. It gives research workers a way to read *past* a search
snippet — where the exact dates, version numbers, specs, and direct quotes
actually live — and turns the recurring hallucinated `web_fetch ❌` into a real,
succeeding tool.

**custom-tools change — needs `--build custom-tools`, not audrey-ai.** (audrey-ai
still re-discovers the tool; see step 6.)

## What it does

`web_search` returns only title + url + snippet (~150 chars). Models trained in
search-then-read harnesses kept inventing a page-opener (`web_fetch` / `read_url`)
and failing as `unknown_tool` — glm-5.2 did it on every research run, regardless
of prompt wording. This makes `web_fetch` real:

```text
POST /web_fetch  {"url": "https://…", "max_chars": 6000}
  → fetch (SSRF-guarded) → trafilatura.extract(html) → {"url": final, "text": …}
```

Extraction is [trafilatura](https://github.com/adbar/trafilatura) — purpose-built
article extraction that strips nav/boilerplate. We call **only**
`trafilatura.extract(html_string)`, never `trafilatura.fetch_url` (which would do
its own unguarded HTTP request and bypass every guard below).

## Why this exists

A researcher that finds the perfect arXiv abstract in a search result could cite
the URL but never read the paper — every research answer was built from snippets
plus model priors. The open thread (PROJECT_STATE) noted a failed fetch dropping a
source to `SOURCES:none`; models want a page-opener and reach for one whether or
not we offer it.

## Safety — this is the whole reason it wasn't built sooner

`web_fetch` fetches a **model-chosen URL from inside `ollama-net`** — i.e. a
model-steerable HTTP client on the LAN. Guards, defense-in-depth on top of the
auth chat-completions already requires (all in
[`tools-server/fetch.py`](../../tools-server/fetch.py)):

- **scheme allowlist** — http/https only; no `file://`, `gopher://`, etc.
- **`_is_unsafe_address`** — ported from `kb/embed.py`; rejects any host whose
  DNS resolves to a private / loopback / link-local / reserved address, and
  unresolvable hosts. Blocks `qdrant:6333`, `127.0.0.1:*`, cloud metadata.
- **manual per-hop redirect revalidation** — the guard that made this hard.
  Automatic redirect-following is the classic bypass: `evil.example` returns
  `302 → http://qdrant:6333` and the initial-host check never sees the internal
  target. So `follow_redirects=False` and `_validate_url` re-runs on every hop.
- **2 MB byte cap** streamed — a multi-GB body can't OOM the container.
- **content-type gate** — only HTML/plain text is extracted; a binary is rejected
  before trafilatura sees it.
- **15s timeout** — under the 30s tool-dispatch ceiling, so a slow page reports
  "fetch timed out" instead of surfacing as a bare dispatch timeout.

**Known gap** (identical to the existing image path): DNS rebinding. We validate
the host's resolved IPs, but httpx re-resolves at connect time. The real fix
(connect to the validated IP, pass hostname via SNI) is out of scope; the bar is
an attacker controlling DNS for a domain a research query is induced to fetch.

## What's in scope

- **[`tools-server/fetch.py`](../../tools-server/fetch.py)** — new: the fetcher +
  guards + trafilatura extraction.
- **[`tools-server/app.py`](../../tools-server/app.py)** — new `/web_fetch`
  endpoint (`operation_id="web_fetch"`), 422-on-failure so the reason reaches the
  model.
- **[`tools-server/pyproject.toml`](../../tools-server/pyproject.toml)** —
  `trafilatura>=1.12` (real custom-tools dep; test-only at the repo root, so the
  audrey-ai image never ships it). `uv.lock` regenerated.

## What's NOT in scope

- **No prompt change.** `web_fetch` is left to OpenAPI auto-discovery, NOT named
  in any role prompt. Naming a tool measurably changes call behavior (the
  2026-07-22 "page"→"source" A-B-A), so whether to actively steer workers toward
  it is a deliberate, separately-measured change. Ship the tool, let discovery
  surface it, measure organic use first.
- **No Dockerfile edit.** Layer 1 compiles deps from `tools-server/pyproject.toml`
  (picks up trafilatura + its cp312 wheels); Layer 2's `COPY tools-server/*.py`
  grabs `fetch.py`. Exactly what the Phase 5 wheel-packaging was built for.

## Deploy on Unraid

From `/mnt/user/appdata/audrey_ai_2.0`:

```
docker compose up -d --build custom-tools
```

Host ports are unpublished (2026-07-18 security review), and the app containers
have **no curl** — all probes below run **inside `custom-tools`** via `python3`
(present; it's a Python service) against `localhost:8001`.

## Verification

Hermetic (laptop): **799 pytests pass** (+21 in `test_web_fetch.py`); ruff clean;
hard cite baseline held at 11 drift / 0 broken.

Live, on the box — tiers 1–5 test the tool in isolation (seconds, no models). **If
step 3 fails, stop and do not proceed — the SSRF surface is open.**

**0. Code landed.**

```
docker exec custom-tools sh -c 'test -f /app/fetch.py && echo present || echo MISSING'
docker exec custom-tools python3 -c "import trafilatura; print(trafilatura.__version__)"
```

Want `present` and a version. A trafilatura import error means Layer 1 didn't
pick up the dep — the container won't serve.

**1. Discovery — the endpoint is registrable.**

```
docker exec custom-tools python3 -c "import httpx; d=httpx.get('http://localhost:8001/openapi.json',timeout=10).json(); print('web_fetch' in [m.get('operationId') for p in d['paths'].values() for m in p.values()])"
```

Want `True`.

**2. Happy path — real page, real text.**

```
docker exec custom-tools python3 -c "
import httpx
r = httpx.post('http://localhost:8001/web_fetch', json={'url':'https://en.wikipedia.org/wiki/Attention_Is_All_You_Need','max_chars':600}, timeout=60)
b = r.json(); print('status', r.status_code); print('final_url', b.get('url')); print(repr(b.get('text','')[:200]))
"
```

Want `200`, a final URL, and readable prose — no HTML tags, no nav.

**3. SSRF — internal hosts blocked (the load-bearing test).**

```
docker exec custom-tools python3 -c "
import httpx
for u in ['http://qdrant:6333/','http://127.0.0.1:8001/health','http://169.254.169.254/latest/meta-data/']:
    r = httpx.post('http://localhost:8001/web_fetch', json={'url':u}, timeout=30)
    print(u,'->',r.status_code, r.json().get('detail','')[:60])
"
```

All three want `422` with `private/internal address`. The third is the
cloud-metadata endpoint. **Any 200 here is a guard failure — stop.**

**4. Scheme and non-HTML gates.**

```
docker exec custom-tools python3 -c "
import httpx
for u in ['file:///etc/passwd','https://arxiv.org/pdf/1706.03762']:
    r = httpx.post('http://localhost:8001/web_fetch', json={'url':u}, timeout=30)
    print(u,'->',r.status_code, r.json().get('detail','')[:70])
"
```

`file://` → `422 … only http/https`. The PDF → `422 … not readable text` (rejected
on content-type before the body downloads).

**5. Truncation honored.** `max_chars` has a floor of **500** (`ge=500` on the
request model) — anything lower is a 422 validation error, not a truncated fetch.
Verify by eyeballing the two returned lengths. Keep this command free of `<`, `>`,
and non-ASCII: comparison operators inside a `docker exec … -c "…"` string are a
shell-redirection hazard, and a stray `…` is an encoding hazard — either can make
the whole command silently no-op over SSH (see
`docs/reference/bash-linux-best-practices.md`). Plain `for` + plain `print`:

```
docker exec custom-tools python3 -c "
import httpx
u='https://en.wikipedia.org/wiki/Euclid'
for mc in (500, 6000):
    r=httpx.post('http://localhost:8001/web_fetch',json={'url':u,'max_chars':mc},timeout=60)
    print('max_chars', mc, 'status', r.status_code, 'textlen', len(r.json().get('text','')))
"
```

Want both `status 200`, with the 500-cap `textlen` ~513 (500 + the appended
marker) and the 6000-cap `textlen` much larger. A `422` here means `max_chars`
was below the 500 floor.

**6. End-to-end — Audrey registers it.** custom-tools was rebuilt, but audrey-ai
discovered tools at *its* last startup, so it must re-read:

```
docker compose restart audrey-ai
docker compose logs --since 2m audrey-ai | grep -iE "tools=|discover"
```

Want the tool count go 7 → **8**. Then a research run is the only thing that shows
whether workers *use it well*: after one, the case where glm-5.2 kept inventing
`web_fetch` should now log success —

```
docker compose logs --since 30m audrey-ai | grep "dispatch: web_fetch"
```

want `dispatch: web_fetch ok`, not `unknown tool 'web_fetch'`.

## What this unblocks

Research workers can read a page, not just its snippet — the fix for the
`SOURCES:none`-on-failed-fetch thread and the recurring invented `web_fetch ❌`.
Whether to prompt-steer workers toward it is a follow-up, A-B-A-gated on the
citation-quality metric (PROJECT_STATE "Prompt wording").

**Follow-up:** `docs/lesson-ai/lesson-16` (custom-tools sidecar) predates this
tool — it needs a `web_fetch` section, and its `app.py` line cites shifted ~46
lines from the insertion (advisory drift only; hard baseline still 11/0).
