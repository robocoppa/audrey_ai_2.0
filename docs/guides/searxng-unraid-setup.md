# SearXNG on Unraid — setup for Audrey's `web_search` fallback

SearXNG is the self-hosted meta-search instance Audrey's `web_search` falls back
to when Brave is quota-exhausted (402) or rate-limited. It's a **prebuilt public
image** — nothing to build, just run it via the Unraid Docker UI (same pattern as
ollama/qdrant). No API key, no per-query cost.

## 1. Add the container (Unraid Docker UI → Add Container)

| Field | Value |
|-------|-------|
| **Name** | `searxng` |
| **Repository** | `searxng/searxng:latest` |
| **Network** | `ollama-net` — the SAME network custom-tools/audrey-ai/qdrant/ollama use. This box addresses every service by container name over this network, so SearXNG must join it for `custom-tools` to reach it as `http://searxng:8080`. |
| **Port** | host `8088` → container `8080` (host port only needed for the LAN `curl` check below; the container-name path uses 8080 directly) |
| **Path / volume** | host `/mnt/user/appdata/searxng` → container `/etc/searxng` (holds `settings.yml`) |

Env vars:

| Variable | Value |
|----------|-------|
| `SEARXNG_BASE_URL` | `http://192.168.1.11:8088/` |
| `SEARXNG_SECRET` | any long random string (or set `secret_key` in settings.yml) |

> **Network note:** this box reaches every internal service by container name over
> `ollama-net` (`AUDREY_URL=http://audrey-ai:8000`, `QDRANT_URL=http://qdrant:6333`,
> etc.) — no host-IP URLs. SearXNG follows the same pattern, hence
> `SEARXNG_URL=http://searxng:8080` in step 3. `ollama-net` is `external: true` in
> compose (created outside it), so just attach the searxng container to it.

Start it once. SearXNG writes a default `settings.yml` into
`/mnt/user/appdata/searxng/` on first run.

## 2. Enable the JSON API (REQUIRED — off by default)

Audrey calls SearXNG's JSON endpoint (`/search?format=json`). It is **disabled by
default** and returns **403** without this. Edit
`/mnt/user/appdata/searxng/settings.yml`:

```yaml
search:
  formats:
    - html
    - json      # ← add this line
```

(While you're in there, confirm `server.secret_key` is set to something non-empty
— SearXNG refuses to start with the placeholder.)

Restart the `searxng` container.

## 3. Point custom-tools at it

Add to the **custom-tools** container's env (Unraid → custom-tools → Edit):

```
SEARXNG_URL=http://searxng:8080
```

(Container name + internal port 8080, over `ollama-net` — matches how
custom-tools reaches audrey-ai/qdrant/ollama. Use the host-IP form
`http://192.168.1.11:8088` only if you chose NOT to put searxng on `ollama-net`.)

Rebuild/restart custom-tools. The startup log should now read
`searxng=http://searxng:8080` instead of `searxng=UNSET`.

## 4. Verify

Direct check (from anywhere on the LAN):

```bash
curl 'http://192.168.1.11:8088/search?q=test&format=json' | head -c 300
```

Should return JSON with a `results` array. If you get **403**, the JSON format
isn't enabled (step 2). If **connection refused**, the container/port is wrong.

Then through Audrey — send one `audrey_research` request and check custom-tools logs:

```bash
docker logs custom-tools 2>&1 | grep "SearXNG returned"
```

Expect `SearXNG returned N results` with **N > 0** (the abandoned DDG attempt
returned 0 — that's the regression this fixes).

## 5. Enable throttle-resistant general engines (REQUIRED — defaults are NOT enough)

**The default engine set is not reliable for grounding.** A general web query only
routes to *general-category* engines, and SearXNG's defaults leave the general
category resting almost entirely on three engines that actively fight scrapers:
**brave**, **google cse**, and **startpage**. All three rate-limit or CAPTCHA a
self-hosted instance under normal load, and when they do, `web_search` returns
**zero results** — the "trusted fallback" silently fails exactly when Brave (the
primary) is also capped. (Diagnosed 2026-07-15: a real research query returned 0
results with `unresponsive_engines = [brave: too many requests, google cse: too
many requests, startpage: Suspended: CAPTCHA]`, while 80 engines were "enabled" —
but the other 77 are category-locked, e.g. arxiv/pubmed/github, and never serve a
general query.)

The fix is to enable **independent, throttle-resistant general engines** that
don't fight self-hosted instances. With `use_default_settings: true` (the normal
setup), your `settings.yml` `engines:` block is an **override layer** on top of
SearXNG's built-in defaults — you only list the general-web engines whose
`disabled:` flag you want to change. Everything else keeps its default (so the
category engines — arxiv/pubmed/github/etc. — stay on automatically; don't touch
them).

**The verified-good set on this box (2026-07-15):**

```yaml
engines:
  # general web — enable independent engines that don't block self-hosted instances
  - name: duckduckgo
    disabled: false      # ✅ workhorse
  - name: mojeek
    disabled: false      # ✅ independent crawler + own index
  - name: mwmbl
    disabled: false      # ✅ independent community index
  - name: brave
    disabled: false      # works when its quota resets; harmless (just skipped) when capped
  # explicitly OFF — proven to fail on this box, they only clutter unresponsive_engines
  - name: startpage
    disabled: true       # CAPTCHA (Google proxy — Google CAPTCHAs it)
  - name: qwant
    disabled: true       # access denied on this instance
  # keep these off too (default was on; images/news aren't general grounding)
  - name: duckduckgo images
    disabled: true
  - name: duckduckgo news
    disabled: true
```

**This is a measured result, not a guess.** Enabling this set and restarting took
a real research query from **0 → 86 results**, with `unresponsive_engines` down to
just `[brave: too many requests]` (the expected quota cap, which recovers). ddg +
mojeek + mwmbl all verified working — three independent general engines, so no
single point of failure. qwant (`access denied`) and startpage (`CAPTCHA`) were
verified dead here and left off.

Restart the `searxng` container after editing — **config changes do NOT take
effect until restart** (common gotcha: editing the file, then probing the still-
old running config and seeing no change).

> **Verify the engines actually work on your version** (no `curl` inside app
> containers — they don't have it; use `python3` from custom-tools):
> ```bash
> docker exec custom-tools python3 -c "import urllib.request,json,urllib.parse; \
>   q=urllib.parse.quote('transformer attention mechanism'); \
>   d=json.load(urllib.request.urlopen('http://searxng:8080/search?q='+q+'&format=json',timeout=15)); \
>   print('results:',len(d.get('results',[]))); print('unresponsive:',d.get('unresponsive_engines',[]))"
> ```
> Expect **results in the double digits** and DDG/Mojeek/Qwant NOT in
> `unresponsive`. If any of them appears there with an error, that engine is
> broken on your SearXNG version (DDG occasionally breaks when DuckDuckGo changes
> their API) — leave it off and try alternatives (`mwmbl`, `yep`, `yandex`).
> A `200` with an empty `results` array is not "healthy": always check the count
> in the body, not the HTTP status.

## Notes

- **No Dockerfile / `docker build`** — SearXNG is third-party, pulled prebuilt.
  It lives on the Unraid UI like ollama/qdrant, NOT in `compose.yaml` (which
  scopes to audrey-ai + custom-tools only).
- SearXNG's `settings.yml` is **not tracked in this repo** — it lives only on the
  box at `/mnt/user/appdata/searxng/`. This guide is the tracked record of what it
  should contain; keep it in sync when you change the box's engine set.
- **Why so many "enabled" engines still fails:** engine count is misleading. What
  matters is how many *general-category*, *throttle-resistant* engines answer a
  plain query. See step 5 — that's the number to keep healthy.
