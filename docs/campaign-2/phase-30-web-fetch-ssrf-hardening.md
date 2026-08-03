# Campaign 2 Phase 30 — web_fetch SSRF hardening

Remediation of the 2026-07-22 SSRF audit of the `web_fetch` page-opener (shipped in
Phase 29). Four commits (A→D) hardening the model-steerable HTTP client against
internal-service reach and resource exhaustion. **custom-tools change — deploy with
`--build custom-tools`.**

**Status as of 2026-07-23: COMPLETE — all four commits deployed and box-verified.**
A/B/C/D all live on the box. On-box verification (inside `custom-tools`):
- **Guard (all commits):** `qdrant:6333`, `127.0.0.1`, `169.254.169.254` → 422
  `private/internal` — the pin refactor did NOT regress the SSRF block.
- **D pinning (the deployed module):** stubbed resolver → outgoing `connect_host`
  is the vetted IP `93.184.216.34` while `Host`/`sni_hostname` are `example.com`.
- **D happy path (real TLS egress):** `https://en.wikipedia.org/wiki/Euclid` → 200
  + readable text, so pinned-IP + SNI passes real cert verification on the box.

827 hermetic tests green, ruff clean. **Open followup (logged below, non-blocking):**
`_resolve_safe` returns IPv4+IPv6 but D pins to `ips[0]` only, dropping the
multi-address fallback anyio gave pre-D — harmless on this IPv4-first box, a
robustness gap on IPv6-first hosts.

### Followup — pin with per-IP fallback (non-blocking robustness)

D pins to `ips[0]`. If `getaddrinfo` ever returns an unreachable IP first (e.g. a
broken IPv6 on an IPv6-capable host), the fetch fails with no retry, where pre-D
anyio would have tried the next address. Fix: iterate the **vetted** IPs, attempt
each until one connects (every IP is already validated by `_resolve_safe`, so
trying them in turn stays safe). Not biting on the current box (glibc sorts IPv4
first on an IPv4-only container, observed `172.66.147.243` / `208.80.153.224` as
`ips[0]`). Own commit + deploy when picked up.

Scope: `tools-server/fetch.py` (the `web_fetch` page-opener) + its wiring in
`tools-server/app.py`. Tests in `tests/test_web_fetch.py`.

---

## Why this exists

`web_fetch` is a model-steerable HTTP client running inside `ollama-net`: a
research worker picks the URL, so it is a deliberate SSRF surface. Egress from
the container is unrestricted and internal services (`qdrant:6333`, `ollama`,
the tools-server itself) are reachable on the LAN. **Threat model:** a research
query induced (e.g. via prompt injection in fetched content) to fetch an
attacker-controlled URL. Win condition to disprove: "reach something internal,
or exhaust resources."

A security audit on 2026-07-22 tested the guards against the actual installed
libraries (httpx 0.28.1, trafilatura 2.0.0, lxml 6.1.0, zstandard 0.25.0,
brotli absent). The guards that were confirmed solid:

- scheme allowlist (http/https), verified
- resolved-IP judgment (`_is_unsafe_address` via `getaddrinfo`) — blocks direct
  internal URLs and every IP-encoding bypass tried (mapped IPv6, int/hex/octal,
  public-DNS→loopback); `getaddrinfo` IDNA-normalizes the same way httpx does,
  so unicode-dot tricks resolve identically and are blocked
- manual per-hop redirect re-validation — scheme-relative `//qdrant`, `file://`,
  `gopher://`, and CRLF Locations all caught
- URL parser: `urllib.parse` (guard) and `httpx.URL` (fetch) agree on host for
  userinfo/backslash/double-`@` tricks — no differential bypass
- **trafilatura is XXE-safe**: `extract(html_string)` routes through libxml2's
  HTML parser, which never resolves DOCTYPE entities. A `file://` canary entity
  is left literal (no local-file read), billion-laughs returns empty instantly.

## Findings and status

| # | Finding | Severity | Fix | Status |
|---|---------|----------|-----|--------|
| 1 | DNS-rebinding TOCTOU is a live SSRF bypass (guard resolves; httpx re-resolves at connect, per hop) | **High** | Commit D | **done (deployed 2026-07-23)** |
| 2 | Decompression cap is transient-unbounded — cap counts decompressed bytes (good) but only after a full chunk is materialized; httpx decoders have no `max_length` | **High** | Commit B | **done (deployed 2026-07-23)** |
| 3 | No wall-clock deadline (15 s is per-op, resets per read; redirects multiply) + no concurrency cap → coroutines orphan past Audrey's 30 s dispatch give-up | **Med-High** | Commit C | **done (deployed 2026-07-23)** |
| 4 | Validation seam has uncaught-exception → HTTP 500 paths (`httpx.InvalidURL` not an `HTTPError`; `urlparse`/`.port` raise `ValueError`), bypassing the model-safe error contract | **Low-Med** | Commit A | **done (deployed 2026-07-23)** |
| 5 | XXE / external entity / billion-laughs | — | verified safe; locked by regression test | **done (deployed 2026-07-23)** |
| 6 | URL parser differential / IDNA / redirect re-validation | — | verified safe (one note: https→http downgrade permitted) | no action |

Evidence measurements are in the appendix.

---

## Remediation commits

Land order A → B → C → D (rising risk; D is the biggest refactor). Each is
independent. Commit messages follow the repo's bare `<type>: <summary>` form.

### Commit A — validation-seam robustness + missing tests  ✅ DONE (deployed 2026-07-23)

`fix(tools): convert web_fetch URL-parse errors to clean 422s`

- `_validate_url`: wrap `urlparse` + `.hostname` + `.port` in
  `try/except ValueError → FetchError("the URL is malformed…")`. Catches the bad
  IPv6 literal (`http://[::1`, raises at the `urlparse` call) and out-of-range
  port (`.port` access) — the latter also kills the connect-time `ExceptionGroup`.
- Fetch loop except clause: add `httpx.InvalidURL` and `ValueError` alongside
  `httpx.HTTPError`. `InvalidURL` is **not** an `HTTPError` subclass, so control
  chars in the initial URL or a redirect `Location` (`\t`, `\n`, `\x00`, CRLF)
  now become a clean `FetchError` instead of a 500. Raises before any socket
  opens, so still hermetic.
- Tests (16 new, all green): malformed-URL fuzz → `FetchError`; XXE + billion-
  laughs regression; endpoint boundary via `TestClient` (Pydantic 1–2000 char
  url / 500–20000 max_chars → 422, and malformed-string URL → 422 not 500).

Note left for later: `app.py:435` uses `status.HTTP_422_UNPROCESSABLE_ENTITY`,
which is deprecated in this FastAPI (rename to `…_CONTENT`). Pre-existing, out of
scope for this work.

### Commit B — bound decompression  (Finding 2, High)  ✅ DONE (deployed 2026-07-23)

`fix(tools): bound web_fetch decompression to the byte cap`

Shipped in `tools-server/fetch.py`:

- New `_decompress_bounded(raw, encoding)` decodes per `Content-Encoding` with
  output capped at `_BYTE_CAP + 1`, so a bomb raises instead of ballooning:
  - `""`/`identity` → passthrough
  - `gzip`/`x-gzip`/`deflate` → `zlib.decompressobj(wbits)` with
    `decompress(raw, _BYTE_CAP + 1)`; `unconsumed_tail` non-empty ⇒ over cap.
    wbits tried `47` (auto gzip/zlib) then `-15` (raw deflate).
  - `zstd` → `zstandard…stream_reader(BytesIO(raw)).read(_BYTE_CAP + 1)` (guarded
    import; if zstandard absent, `zstd` falls through to rejection).
  - brotli / unknown / **multiple** encodings → `FetchError`, never raw to trafilatura.
- `_read_capped` now iterates `resp.aiter_raw()` (undecoded wire bytes) under the
  byte cap, then calls `_decompress_bounded`. This caps both the raw download and
  the decoded result — peak is O(`_BYTE_CAP`) regardless of ratio.
- Tests (6 new): gzip + zstd bombs → `FetchError("cap")` with `tracemalloc` peak
  under 24 MB (measured 4.4 MB / 2.1 MB, vs 219 / 419 MB pre-fix); gzip + raw-
  deflate happy paths decode; brotli and multi-encoding rejected. Also added an
  `_AsyncBytes`/`_resp` test helper so MockTransport serves real streams (required
  now that we read `aiter_raw()`).

### Commit C — overall deadline + concurrency cap  (Finding 3, Med-High)  ✅ DONE (deployed 2026-07-23)

`fix(tools): add an overall deadline and concurrency cap to web_fetch`

Shipped in `tools-server/fetch.py` + `tools-server/settings.py`:

- `fetch_readable` is now a thin wrapper:
  `async with asyncio.timeout(_OVERALL_DEADLINE_S), _FETCH_SEMAPHORE:` around the
  real body (renamed `_fetch_readable`). `TimeoutError` → `FetchError("fetch
  exceeded the 25s deadline")`. Hard wall-clock stop over redirects + reads +
  extraction; external cancellation (dispatch give-up) still propagates as
  `CancelledError`, not masked as our timeout.
- Module-level `asyncio.Semaphore(_MAX_CONCURRENT_FETCHES)` acquired **inside** the
  deadline, so a saturated pool waits it out and fails as a clean timeout, not a
  hang. (Safe at import: an unsaturated `Semaphore` never touches the event loop,
  so it doesn't bind to one — reused fine across per-test loops.)
- Both constants config-surfaced: `WEB_FETCH_OVERALL_DEADLINE_S` (**25.0**) and
  `WEB_FETCH_MAX_CONCURRENT` (**8**) in `settings.py`.
- Skipped the optional shared-`AsyncClient`/`Limits` refactor — the semaphore
  already bounds concurrent clients (≤8), so sockets are bounded; per-call clients
  keep the diff focused.
- Tests (2 new, all green): patched-deadline slow fetch → `FetchError` near the
  deadline (not the 15 s per-op / not a hang); semaphore(2) + 6 concurrent fetches
  → peak in-flight **exactly 2** (cap holds AND concurrency actually happens).

### Commit D — pin to validated IP + SNI  (Finding 1, High) — the SSRF closer  ✅ DONE (deployed 2026-07-23)

`fix(tools): pin web_fetch to the validated IP to close DNS rebinding`

Shipped in `tools-server/fetch.py`. httpcore 1.0.9 honors the `sni_hostname`
request extension, so we connect to a specific IP while preserving TLS SNI/cert
verification — confirmed by a real HTTPS fetch (example.com + Wikipedia both 200,
no TLS error).

- `_is_unsafe_address(host) -> bool` replaced with `_resolve_safe(host) -> list[str]`
  (+ a pure `_ip_is_unsafe` classifier): one `getaddrinfo(type=SOCK_STREAM)`,
  validate **every** returned IP, `FetchError` on any bad IP / resolution failure /
  empty result, else the deduped vetted IPs. `_validate_url` now only checks
  scheme/shape and **returns the hostname** — it no longer resolves.
- Each request is pinned to `ips[0]`:
  - `httpx.URL(current).copy_with(host=ips[0])` → socket target is the IP (IPv6
    auto-bracketed by httpx); original port/path/query preserved.
  - `Host` header → the exact value httpx would send for the original URL, taken
    from `client.build_request("GET", target).headers["host"]` (handles default-port
    stripping + IPv6 brackets — no hand-rolled authority logic).
  - `extensions={"sni_hostname": hostname}` → TLS SNI + cert verification on the
    real hostname; the socket only ever reaches the vetted IP.
  - `client.build_request(...)` + `client.send(req, stream=True, follow_redirects=False)`,
    body closed in a `finally`.
- Resolve+validate+pin re-runs on **every** redirect hop; `current` stays the human
  (hostname) URL so the reported `final_url` and redirect joins are unchanged.
- "KNOWN GAP: DNS rebinding" caveat + the `_is_unsafe_address` bullet deleted from
  the module docstring; the guard bullet now describes resolve-once-and-pin.
- Tests: `test_request_is_pinned_to_validated_ip` asserts the outgoing connect host
  is the pinned IP while `Host`/`sni_hostname` carry the real name; the redirect and
  direct-guard tests were reworked onto `_resolve_safe` (internal IP / redirect
  target / unresolvable host all raise). 49 web_fetch tests, all green.

---

## Defaults chosen (flag to change)

- Overall deadline **25 s**; max concurrent fetches **8**; both module constants
  surfaced to config.
- Decompression: keep gzip/deflate/zstd with bounded decode; **reject** br and
  unknown encodings.
- Saturated concurrency pool → fail as timeout (no separate 429).

## Post-deploy status

- Tool-layer suite: **827 pass**, ruff clean; no vl/image path touched, so no image
  smoke needed.
- All four deployed via `docker compose up -d --build custom-tools`; the on-box 422
  guard battery, the pinning proof against the deployed module, and the real-TLS
  happy path are in the status block at the top.
- `docs/PROJECT_STATE.md` records the completion; the Phase-29 doc's stale "Known
  gap: DNS rebinding" line was corrected when D landed.

---

## Smoke testing

Two tiers for every commit:

- **Hermetic** — `pytest tests/test_web_fetch.py` (offline, deterministic; the
  boundary + guard tests double as smokes). Always run this first.
- **Functional (laptop)** — hit the real code path over the ASGI stack. Uses
  `starlette.testclient.TestClient(app.app)` **without** a `with` block, which
  skips the lifespan so no DB/qdrant is needed. Import seam:
  `import sys; sys.path.insert(0, "tools-server")`.
- **On-box (post-deploy)** — `custom-tools` has **no host port** and **no
  `curl`** in the image, but it has `python` and listens on `:8001` on
  `ollama-net`. So smoke from inside the container after
  `docker compose up -d --build custom-tools`. Always start with the deploy-state
  check — `custom-tools` code is flat at `/app/*.py`:

  ```bash
  docker exec custom-tools grep -c "<a string the new code added>" fetch.py   # 0 = redeploy didn't take
  ```

  and POST via python (urllib raises `HTTPError` for 4xx/5xx, so catch and print
  the code):

  ```bash
  docker exec custom-tools python -c '
  import urllib.request, urllib.error, json
  req = urllib.request.Request("http://127.0.0.1:8001/web_fetch",
      data=json.dumps({"url":"<test url>"}).encode(),
      headers={"content-type":"application/json"}, method="POST")
  try: r = urllib.request.urlopen(req, timeout=30); print(r.status, r.read()[:200].decode(errors="replace"))
  except urllib.error.HTTPError as e: print("status", e.code, e.read()[:200].decode(errors="replace"))
  '
  ```

### Commit A — validation seam (500 → 422)  ✅ verified 2026-07-22

Observable change: malformed URLs that used to 500 now return a model-safe 422.
The laptop smoke below is fully offline (every case fails at validation or
request-build; no egress) and exits non-zero on any 500:

```bash
.venv/bin/python - <<'PY'
import sys; sys.path.insert(0, "tools-server")
import app
from starlette.testclient import TestClient
c = TestClient(app.app); bad = 0
for label, payload in [
    ("malformed IPv6",   {"url": "http://[::1"}),
    ("tab control char", {"url": "http://8.8.8.8\t/"}),
    ("null byte",        {"url": "http://8.8.8.8/\x00"}),
    ("port out of range",{"url": "http://8.8.8.8:99999999/"}),
    ("internal qdrant",  {"url": "http://qdrant:6333/collections"}),
    ("bad scheme",       {"url": "file:///etc/passwd"}),
    ("empty url",        {"url": ""}),
]:
    r = c.post("/web_fetch", json=payload); bad += r.status_code == 500
    print(f"{label:18} {r.status_code}{'  <<< 500!' if r.status_code==500 else ''}")
print("PASS" if not bad else f"FAIL — {bad} returned 500"); sys.exit(1 if bad else 0)
PY
```

To see the "before": `git stash`, rerun (four cases flip to 500), `git stash pop`.

On-box after deploy: deploy-check string `the URL is malformed`; POST
`{"url":"http://[::1"}` → expect `status 422`. Optional happy path (egress is open
on the box): `{"url":"https://example.com/"}` → `200` + text.

### Commit B — bounded decompression (memory can't balloon)  ✅ verified on laptop 2026-07-23

Measured peak: gzip 4.4 MB, zstd 2.1 MB (vs 219 / 419 MB pre-fix). Deployed
2026-07-23 (serving a bomb on `ollama-net` not run — hermetic + laptop sufficient).

Hermetic: the gzip + zstd bomb tests assert a `tracemalloc` peak under a small
threshold (the point is the *transient* is bounded, not just the final total).

Functional (laptop) — serve a bomb through the real `fetch_readable` via
`MockTransport` and confirm it raises `FetchError` with a bounded peak. **The
body MUST be served as a `stream=`, not `content=`:** Commit B now reads
`resp.aiter_raw()`, and `httpx.Response(..., content=gz)` pre-sets `_content` and
marks the stream consumed, so `aiter_raw()` raises `StreamConsumed` before any
guard runs (this is why the test file carries the `_AsyncBytes`/`_resp` helper).

```bash
.venv/bin/python - <<'PY'
import sys, gzip, asyncio, tracemalloc; sys.path.insert(0, "tools-server")
import httpx, fetch
from fetch import fetch_readable, FetchError

class _AsyncBytes(httpx.AsyncByteStream):        # serve a real stream (aiter_raw)
    def __init__(self, data): self._data = data
    async def __aiter__(self): yield self._data

fetch._is_unsafe_address = lambda _h: False
gz = gzip.compress(b"A" * (100*1024*1024))                      # 100MB -> ~100KB
def h(req):
    return httpx.Response(200, headers={"content-type":"text/html","content-encoding":"gzip"}, stream=_AsyncBytes(gz))
async def go():
    tracemalloc.start()
    try: await fetch_readable("http://e.example/b", max_chars=6000, transport=httpx.MockTransport(h)); print("FAIL: no error")
    except FetchError as e:
        peak = tracemalloc.get_traced_memory()[1]/1e6
        print(f"{e} | peak={peak:.0f}MB", "-> PASS" if peak < 16 else "-> FAIL (peak too high)")
asyncio.run(go())
PY
```

Pre-Commit-B this peaks ~219MB (gzip) / ~419MB (zstd); post-fix it must stay
near the cap. Repeat with `zstandard.ZstdCompressor().compress(...)` and
`content-encoding: zstd`. **Verified 2026-07-23** — gzip peak 4 MB, zstd peak
2 MB, both `-> PASS`. On-box: only meaningful if you can serve a bomb on
`ollama-net` (throwaway container) — the hermetic + laptop smoke is sufficient.

### Commit C — overall deadline + concurrency cap  ✅ verified on laptop 2026-07-23

Deadline smoke fired at **25.0s** (real default, `< 30s` ceiling); default deadline
25.0 / max concurrent 8 confirmed. Concurrency cap covered hermetically (peak == 2).

Hermetic: a `MockTransport` handler that `await asyncio.sleep(deadline + 5)` must
raise `FetchError` in ~deadline, not hang to the per-op timeout.

Functional (laptop) — time a slow response and confirm the wall-clock stop:

```bash
.venv/bin/python - <<'PY'
import sys, asyncio, time; sys.path.insert(0, "tools-server")
import httpx, fetch
from fetch import fetch_readable, FetchError
fetch._is_unsafe_address = lambda _h: False
async def slow(req): await asyncio.sleep(60); return httpx.Response(200, content=b"<html/>")
async def go():
    t = time.monotonic()
    try: await fetch_readable("http://e.example/s", max_chars=6000, transport=httpx.MockTransport(slow))
    except FetchError as e: print(f"{e} in {time.monotonic()-t:.1f}s", "-> PASS" if time.monotonic()-t < 30 else "-> FAIL")
asyncio.run(go())
PY
```

Concurrency: fire ~20 of the above concurrently with `asyncio.gather` and confirm
in-flight count is bounded by the semaphore (they finish in batches, not all at
once) and none exceed the deadline. On-box slowloris needs a drip server; skip
unless investigating a real orphan.

### Commit D — pin to validated IP (DNS rebinding closed)  ✅ verified on laptop 2026-07-23

Hermetic (the key one, `test_request_is_pinned_to_validated_ip`): stubbed resolver
returns a **public** IP; the MockTransport handler captured `request.url.host ==
93.184.216.34` (pinned IP), `Host == example.com`, `sni_hostname == example.com` —
a post-validation rebind cannot move the socket. Real HTTPS fetch confirmed pinned
IP + SNI still passes cert verification (example.com + Wikipedia, both 200). Guard
smoke confirmed internal literals (127.0.0.1, 169.254.169.254, ::1, 10.0.0.5) all
raise `private/internal` end-to-end through `fetch_readable`.

Live adversarial (optional, needs a controlled rebinding domain — e.g. a name
with TTL 0 alternating a public IP and 127.0.0.1): POST that URL repeatedly; every
attempt must 422 (`private/internal`), never connect to loopback. Without a
rebinding domain this isn't reproducible on demand — the hermetic assertion is the
real proof. On-box: also re-run the Commit A internal-host smoke
(`{"url":"http://qdrant:6333/collections"}` → 422) to confirm the refactor didn't
regress the basic guard.

---

## Appendix — audit evidence (measured 2026-07-22)

- **Decompression:** gzip 100 KB → 105 MB body, `tracemalloc` peak **219 MB**;
  zstd 6.3 KB → 210 MB body (ratio 32,676:1), peak **419 MB**. Both ultimately
  raise `page body exceeds the 2 MB cap` (aggregate protection holds), but the
  spike happens first. httpx `SUPPORTED_DECODERS = [identity, gzip, deflate, zstd]`
  because zstandard is installed; `GZipDecoder.decode` calls
  `decompressor.decompress(data)` with no `max_length`.
- **XXE:** `file://` canary entity → not expanded, `&xxe;` left literal, secret
  never read. `http://qdrant:6333` network entity → no callout. Billion-laughs →
  empty output in 0.000 s.
- **Parser differential:** urllib vs httpx agree on host across userinfo /
  backslash / double-`@` cases; `getaddrinfo("127。0。0。1")` → `127.0.0.1`
  (IDNA-normalized) → blocked. Control-char URLs → httpx `InvalidURL`
  (pre-Commit-A: uncaught 500).
- **Validation seam (pre-Commit-A):** `http://8.8.8.8\t/`, `\n`, `\x00`,
  `#\r\n…` → `InvalidURL` (500); `http://[::1` → `ValueError` (500, inside the
  gate); `http://8.8.8.8:99999999/` → `ExceptionGroup` (500). All now → `FetchError`.
- **Pin-to-IP viability:** httpcore 1.0.9 connection reads
  `sni_hostname = request.extensions.get("sni_hostname")` → `server_hostname`.
