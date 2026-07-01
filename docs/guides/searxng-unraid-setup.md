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

## Notes

- **No Dockerfile / `docker build`** — SearXNG is third-party, pulled prebuilt.
  It lives on the Unraid UI like ollama/qdrant, NOT in `compose.yaml` (which
  scopes to audrey-ai + custom-tools only).
- SearXNG aggregates many engines; if results are thin, its own `settings.yml`
  lets you enable/disable engines, but the defaults are fine for grounding.
